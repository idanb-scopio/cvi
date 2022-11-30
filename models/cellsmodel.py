import copy
import logging
import math
import yaml
import tensorflow as tf
from core.labels import apply_labels_remap
from models.model_ops.learn_rate_factory import learn_rate_schedule_generator
from models.losses.weighted_classification_loss import WeightedClassificationLoss


class CellsModel:

    @classmethod
    def from_yaml(cls, yaml_file):
        with open(yaml_file, 'r') as f:
            model_cfg = yaml.safe_load(f)
        return cls(model_cfg=model_cfg)

    def __init__(self, model_cfg):
        model_cfg = copy.deepcopy(model_cfg)

        # convert mappings to lower case
        model_cfg['label_remap'] = {k.lower(): v.lower() for k, v in model_cfg['label_remap'].items()}
        model_cfg['label_mapping'] = {k.lower(): v for k, v in model_cfg['label_mapping'].items()}

        self.model_cfg = model_cfg

        # flat_label_map holds a map of label -> class_num, after label_remap (label -> label) is applied.
        self.flat_label_map = apply_labels_remap(label_mapping=self.model_cfg['label_mapping'],
                                                 label_remap=self.model_cfg['label_remap'])

        # trained keras model, used in prediction
        self.trained_model = None
        self.model_dir = None

    def get_flat_label_mapping(self, str_labels=False):
        if not str_labels:
            return self.flat_label_map

        # flat_label_map holds a map of label -> class_num, after label_remap (label -> label) is applied.
        flat_label_map = apply_labels_remap(label_mapping=self.model_cfg['label_mapping'],
                                            label_remap=self.model_cfg['label_remap'],
                                            str_labels=str_labels)
        return flat_label_map

    def get_label_mapping(self):
        return self.model_cfg['label_mapping']

    def get_label_remap(self):
        return self.model_cfg['label_remap']

    def get_num_classes(self):
        return len(self.model_cfg['label_mapping'])

    def get_level_scale_div(self):
        if 'level_scale_div' not in self.model_cfg:
            return None
        level_scale_div = int(self.model_cfg['level_scale_div'])

        if not math.log2(level_scale_div).is_integer():
            raise ValueError(f'level_scale_div must be a power of 2: {level_scale_div}')

        return level_scale_div

    def get_scale_down_factor(self):
        scale_down_factor = int(self.model_cfg['scale_down_factor'])
        return scale_down_factor

    def get_image_shape_in_dataset(self):
        return self.model_cfg['image_shape_in_dataset']

    def calc_image_shape_in_dataset(self):
        """
        Calculate the image shape (WxHxC) as represented in the dataset. The dataset representation of
        the input image takes into account, for instance, the margin needed to do 45 degrees rotations
        in the image augmentation stage (without leaving black margins).
        :return: shape (W, H, C) tuple of the input image as it is saved in the TFRecord dataset.
        """
        shape = self.model_cfg['input_image_size']

        image_aug_params = self.model_cfg.get('training_image_augmentations', None)

        # shape values are integers
        shape = [int(e) for e in shape]

        for i in range(2):
            x = shape[i]

            # take into account rotations up to 45 degrees, without leaving black margins
            size_tfr = math.ceil(x * math.sqrt(2.0))

            if image_aug_params:

                if 'max_shift' in image_aug_params:
                    size_tfr = size_tfr + image_aug_params['max_shift'] * 2

            # ensure each dimension is an even number
            if not size_tfr % 2 == 0:
                size_tfr += 1

            # update the shape
            shape[i] = size_tfr

        self.model_cfg['image_shape_in_dataset'] = shape
        return shape

    def get_image_aug_params(self):
        return self.model_cfg.get('training_image_augmentations', None)

    def get_input_image_shape(self):
        """Input image size is the image size that the keras model expects as input."""

        return self.model_cfg['input_image_size']

    def get_model_name(self):
        try:
            name = self.model_cfg['model_name']
        except KeyError:
            name = self.model_cfg['name']
            logging.warning('"name" field in model.yml is deprecated.')
        return name

    def get_model_type(self):
        return self.model_cfg['model_type']

    def generate_examples_map(self, image_source, labels, model_res, thread_count=None):
        """
        Given an input source (pyramid data) and labels - array of tuples: (x, y, w, h, label_str), return a
        dictionary which maps a label to it's feature and target label.

        :return: dictionary of: key=label (x, y, w, h, label_tr) -> example (format is different for specific instances
                                                                             of this subclass)
        """
        raise RuntimeError('not implemented in base class')

    def set_model_res(self, model_res):
        """
        Set the model resolution of the input image. Resolution is determined at build stage by the user, or
        auto inferred from the distribution of input resolutions.
        :param model_res: floating point value: mm / pixel. examples: alpha = 0.000133, X100 = 0.0002016.
        """
        if 'model_res' in self.model_cfg:
            raise ValueError('model_res already exists')    # for debugging purposes

        self.model_cfg['model_res'] = model_res

    def get_model_res(self):
        """
        Returns the model resolution
        """
        return float(self.model_cfg['model_res'])

    def set_balancing_usage(self, use_balancing=False):

        self.model_cfg['use_balancing'] = use_balancing

    def get_balancing_usage(self):

        use_balancing = self.model_cfg.get('use_balancing', False)
        return use_balancing

    def set_epoch_examples_num(self, epoch_examples_num):
        """
        Set the epoch examples number - may be estimated when using probabilistic dataset examples rejection resample.
        :param epoch_examples_num: (estimated) number of example
        """

        self.model_cfg['epoch_examples_num'] = epoch_examples_num

    def get_epoch_examples_num(self):

        epoch_examples_num = self.model_cfg.get('epoch_examples_num', None)

        return epoch_examples_num

    def get_model_config(self):
        """
        returns a copy of the model config.
        :return:
        """
        return copy.deepcopy(self.model_cfg)

    def get_model_config_yaml(self):
        """
        Returns a YAML format (string) of model config.
        """
        return yaml.dump(self.model_cfg, default_flow_style=None)

    def set_model_dir(self, model_dir):
        self.model_dir = model_dir

    def get_model_dir(self):
        return self.model_dir

    def load_trained_model(self, model_dir=None, log_prefix='', model_filename='model.h5', load_weights_only=False):
        if not model_dir:
            model_dir = self.model_dir
        if not model_dir:
            raise ValueError('missing model dir')

        keras_model_file = f'{model_dir}/{model_filename}'
        logging.info(f'{log_prefix}loading model: {keras_model_file}')
        if load_weights_only:
            self.trained_model = self.get_keras_model(training=False)
            self.trained_model.load_weights(keras_model_file, by_name=True)
            pass
        else:
            self.trained_model = tf.keras.models.load_model(keras_model_file,
                                                        custom_objects={'WeightedClassificationLoss':
                                                                        WeightedClassificationLoss})

    def get_trained_model(self):
        return self.trained_model

    def save_predict_model(self, output_dir, print_summary=False, best_output_model_file=None):
        return

    def get_train_model_weights_suffix(self):
        suffix = ''
        return suffix

    def get_checkpoint_callback(self, output_model_file, save_best_only=False):

        """
        :param output_model_file: checkpoint output model file
        :param save_best_only: save best model with respect to the evaluated metric
        :return: checkpoint Keras callback
        """

        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(output_model_file, save_best_only=save_best_only)

        return checkpoint_cb

    def get_learn_rate_schedule_fn(self):

        model_cfg = self.get_model_config()
        learn_rate_config = model_cfg.get('learn_rate', None)
        learn_rate_schedule = learn_rate_schedule_generator(learn_rate_config)
        if learn_rate_schedule:
            lr_schedule_fn = learn_rate_schedule.get_scheduler_fn()
        else:
            lr_schedule_fn = None

        return lr_schedule_fn

    def use_train_input_format_for_eval(self):
        return False
