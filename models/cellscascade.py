# Cells Cascade Classifier - PREDICT ONLY
# This class provides interface for predicting a cascade of classifiers
import copy
import functools
import glob
import logging
import math
import os

import numpy as np
import tabulate
import yaml
from collections import OrderedDict

from models.cellscls import CellsClassifier
from core.labels import apply_labels_remap

BATCH_SIZE = 256


def model_instances_from_dir(models_root_dir, model_names):
    """
    load models in subdirs of models_root_dir, or the directory itself.
    sub dirs are recognized as model dirs if they contain model.yml file.

    model_names: an optional list of strings. only models with matching names are
                 loaded.

    A dictionary is returned with the following structure:
    { model_name: CellsClassifier instance }
    """
    if not os.path.exists(models_root_dir):
        raise RuntimeError(f'models root dir does not exist: {models_root_dir}')

    sub_dirs = glob.glob(f'{models_root_dir}/*')
    sub_dirs = list(filter(lambda d: os.path.exists(f'{d}/model.yml'), sub_dirs))

    if len(sub_dirs) == 0:
        if os.path.exists(f'{models_root_dir}/model.yml'):
            sub_dirs = [models_root_dir]
    else:
        logging.info(f'found {len(sub_dirs)} model dirs at: {models_root_dir}')

    models_dict = {}
    for model_dir in sub_dirs:
        model_cfg_yaml = f'{model_dir}/model.yml'
        cells_model = CellsClassifier.from_yaml(model_cfg_yaml)
        model_name = cells_model.get_model_name()

        if model_name not in model_names:
            logging.warning(f'ignoring model: {model_name}')
            continue

        if model_name in models_dict:
            raise RuntimeError(f'model with name {model_name} already loaded')

        cells_model.set_model_dir(model_dir)
        models_dict[model_name] = cells_model

    missing_model_names = set(model_names) - set(models_dict.keys())
    if missing_model_names:
        logging.error(f'missing models: {missing_model_names}')
        raise RuntimeError('missing models')

    return models_dict


class CellsCascade:

    # use the same 'factory' API of CellsModel when directory is the input
    @classmethod
    def from_yaml(cls, yaml_file):
        with open(yaml_file, 'r') as f:
            model_cfg = yaml.safe_load(f)

        return cls(model_cfg=model_cfg)

    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.model_dir = model_cfg['model_dir']

        # load cascade classifier models
        model_names = model_cfg['models']
        self.models_dict = model_instances_from_dir(self.model_dir, model_names)

        # keras models are loaded in a separate call. this indicates that state
        self.keras_models_loaded = False

        self.flat_label_map = apply_labels_remap(label_mapping=self.model_cfg['label_mapping'],
                                                 label_remap=self.model_cfg['label_remap'])

    def load_trained_model(self, model_dir=None, log_prefix='', model_filename='model.h5'):
        for model_name, cells_model_instance in self.models_dict.items():
            cells_model_instance.load_trained_model(log_prefix=log_prefix, model_dir=model_dir,
                                                    model_filename=model_filename)

        self.keras_models_loaded = True

    def get_models_dict(self):
        return self.models_dict

    def set_model_dir(self, model_dir):
        self.model_dir = model_dir

    def get_input_image_shape(self):
        """Input image size is the image size that the keras model expects as input."""

        return self.model_cfg['input_image_size']

    def get_level_scale_div(self):
        if 'level_scale_div' not in self.model_cfg:
            return None
        level_scale_div = int(self.model_cfg['level_scale_div'])

        if not math.log2(level_scale_div).is_integer():
            raise ValueError(f'level_scale_div must be a power of 2: {level_scale_div}')

        return level_scale_div

    def get_model_res(self):
        """
        Returns the model resolution
        """
        return float(self.model_cfg['model_res'])

    def get_label_mapping(self):
        return self.model_cfg['label_mapping']

    def get_label_remap(self):
        return self.model_cfg['label_remap']

    def get_model_dir(self):
        return self.model_dir

    def get_model_name(self):
        return self.model_cfg['model_name']

    def get_cascade_params(self):
        return self.model_cfg['cascade_params']

    def get_cascade_models(self):
        return self.model_cfg['models']

    def get_flat_label_mapping(self, str_labels=False):
        if not str_labels:
            return self.flat_label_map

        # flat_label_map holds a map of label -> class_num, after label_remap (label -> label) is applied.
        flat_label_map = apply_labels_remap(label_mapping=self.model_cfg['label_mapping'],
                                            label_remap=self.model_cfg['label_remap'],
                                            str_labels=str_labels)
        return flat_label_map

    def get_model_config(self):
        """
        returns a copy of the model config.
        :return:
        """
        return copy.deepcopy(self.model_cfg)

    def get_predict_fn(self, params=None):
        """
        Returns prediction function tied to this cells model instance.
        predict_fn(image_source, rois)
        image_source: scan image source
        params: dictionary of parameters to pass to the prediction function
        """

        # check if model is loaded
        if not self.keras_models_loaded:
            raise RuntimeError('keras model is not loaded.')

        # default params
        _params = {'batch_size': BATCH_SIZE}

        # update with args, if given
        if params is not None:
            _params.update(params)

        fn = functools.partial(predict_fn, cells_model=self, params=_params)
        return fn


def predict_fn(cells_model, image_source, rois, params):
    models = cells_model.get_models_dict()
    cascade_params = cells_model.get_cascade_params()
    cascade_model_names = cells_model.get_cascade_models()

    batch_size = params['batch_size']
    debug_save_to = params.get('debug_save_to', None)
    debug_image_ext = 'png'

    classifier_names = models.keys()
    assert sorted(classifier_names) == sorted(cascade_model_names)

    logging.info(f'cascade classification for models: {",".join(classifier_names)}')
    logging.info(f'cascade classification prediction batch size is: {batch_size}')

    scan_id = image_source.infer_scan_id_from_src()
    if not scan_id:
        scan_id = 'unknown_scan_id'

    collaterals = None

    # initialize data structure for cascade classification
    # key is input roi tuple with first 5 elements: x, y, w, h, label_str (input)
    # value is a tuple of (classes_od, log_str),
    # where:
    #   classes_od: ordered dict with the following as key/value:
    #      key: class string
    #      value: confidence (percent) of that class as given by its updating model.
    #   log_str: string containing a log of how the classification decision was reached.
    classification_map = {}

    # roi_t -> debug data (list)
    classification_dbg = {}

    for roi in rois:
        roi_t = tuple(roi[0:5])

        classes_od = OrderedDict({'cell': (100.0, "detection")})
        cls_log = [f'{roi_t} roi initialized to "cell".\n']

        classification_map[roi_t] = (classes_od, cls_log)
        x, y, w, h, lbl = roi_t
        classification_dbg[roi_t] = [f'{scan_id},{x},{y},{w},{h}']

    for model_name in cascade_model_names:
        model = models[model_name]
        model_cascade_params = cascade_params[model_name]
        logging.info(f'classifying with model: {model_name}')

        cls_debug_save_to = None
        if debug_save_to:
            cls_debug_save_to = f'{debug_save_to}/{model_name}'
            if not os.path.exists(cls_debug_save_to):
                os.mkdir(cls_debug_save_to)

        params = {'debug_save_to': cls_debug_save_to, 'debug_image_ext': debug_image_ext, 'cache_rois': True,
                  'batch_size': batch_size}
        if collaterals and 'cached_rois_map' in collaterals:
            params['cached_rois_map'] = collaterals['cached_rois_map']

        # obtain prediction function for the current model
        single_predict_fn = model.get_predict_fn(params=params)

        # this set contains the 'pass filter' for input labels
        model_input_label_set = set(model_cascade_params['model_input_labels'])

        # this set contains the 'reject filter' for input labels
        if 'ignore_input_classes' in model_cascade_params:
            ignore_input_classes_set = set(model_cascade_params['ignore_input_classes'])
        else:
            ignore_input_classes_set = set()

        if 'probability_thresholds' in model_cascade_params:
            probability_thresholds_map = model_cascade_params['probability_thresholds']
        else:
            probability_thresholds_map = {}

        # list of model classes, their indices are class numbers
        model_classes = model_cascade_params['classes']

        # for every class not present in probability threshold map,
        # initialize the threshold to 0.0 (i.e. no threshold)
        for class_name in model_classes:
            if class_name not in probability_thresholds_map:
                probability_thresholds_map[class_name] = 0.0

        if 'output_label_map' in model_cascade_params:
            output_label_map = model_cascade_params['output_label_map']
        else:
            output_label_map = {}

        selected_rois = []
        for roi_t in classification_map:
            classes_od, cls_log = classification_map[roi_t]
            roi_cls_set = set(classes_od.keys())

            roi_prerequisite_set = roi_cls_set & model_input_label_set
            if not roi_prerequisite_set:
                classification_dbg[roi_t].append(f'-{model_name}:no_prereq')
                if cls_log:
                    cls_log.append(f'model {model_name}: rejected update due to prerequisite '
                                   f'missing: required: {model_input_label_set}, '
                                   f'cell has: {roi_cls_set}.')
                continue

            roi_skip_update_set = roi_cls_set & ignore_input_classes_set
            # non empty skip_update set causes current roi to skip updates
            # from the current model
            if roi_skip_update_set:
                classification_dbg[roi_t].append(f'-{model_name}:no_update')
                if cls_log:
                    cls_log.append(f'model {model_name}: rejected update due to match on '
                                   f'ignore class: to ignore: {ignore_input_classes_set}, '
                                   f'cell has: {roi_cls_set}.')
                continue

            # if none of the above filter acts on this roi, it is added to the
            # selected rois for prediction list.
            selected_rois.append(roi_t)

        # run prediction
        if len(selected_rois) == 0:
            logging.info(f'no cells to run predict on for model: {model_name}')
            continue

        _, collaterals = single_predict_fn(image_source=image_source, rois=selected_rois)
        results_map = collaterals['results_map']

        # post prediction class updates
        for roi_t in results_map:
            classes_od, cls_log = classification_map[roi_t]
            pred_cls_str, pred_confidence = results_map[roi_t]

            if pred_confidence < probability_thresholds_map[pred_cls_str]:
                classification_dbg[roi_t].append(f'-{model_name}:{pred_cls_str}:thres:{pred_confidence:.1f}<'\
                                                 f'{probability_thresholds_map[pred_cls_str]:.1f}')
                if cls_log:
                    cls_log.append(f'model {model_name}: rejected update due to probability '
                                   f'thresholding: minimum required: '
                                   f'{probability_thresholds_map[pred_cls_str]}%, '
                                   f'cell has: {pred_confidence:.3f}%.')
                continue

            # mapping on the output label may be configured
            if pred_cls_str in output_label_map:
                roi_class_str = output_label_map[pred_cls_str]
                if cls_log:
                    cls_log.append(f'model {model_name}: output map applied: '
                                   f'{pred_cls_str} -> {roi_class_str}')
            else:
                roi_class_str = pred_cls_str

            # at this point, no more filters / mappings. update this model's
            # "final" decision for the ROI.
            classes_od[roi_class_str] = (pred_confidence, model_name)
            classification_dbg[roi_t].append(f'+{model_name}:{roi_class_str}:{pred_confidence:.1f}')
            if cls_log:
                cls_log.append(f'model {model_name}: updated classification to: '
                               f'{roi_class_str}, confidence: {pred_confidence:.3f}%')

    # gather final classification results
    result_labels = []
    for roi_t in classification_map:
        classes_od, cls_log = classification_map[roi_t]
        cls_str, _ = list(classes_od.items())[-1]
        classification_dbg[roi_t].insert(1, cls_str)
        result_labels.append((*roi_t[0:4], cls_str))

        if cls_log:
            cls_log.append(f'final classification: {cls_str}.')

    if debug_save_to:
        cls_log_file = f'{debug_save_to}/cascade-classification.log'
        with open(cls_log_file, 'w') as f:
            for roi_t in classification_map:
                cls_log = classification_map[roi_t][1]
                log_str = f'roi: {roi_t}:\n\t'
                log_str += "\n\t".join(cls_log) + '\n---\n'
                f.write(log_str)

            logging.info(f'saved cascade decision log to: {cls_log_file}')

        with open(f'{debug_save_to}/cascade_summary.log', 'w') as f:
            f.write(tabulate.tabulate(list(classification_dbg.values()), tablefmt='plain'))

    pyramid_res = image_source.get_resolution()
    results = {'scan_id': scan_id,
               'pyramid_resolution': pyramid_res,
               'labels': result_labels,
               'ROIs': []}

    return results, None
