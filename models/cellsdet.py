import functools
import logging
import tensorflow as tf
import numpy as np
import cv2

from models.cellsmodel import CellsModel
from core.labels import create_label_images
from core.utils import read_rois_from_pyramid, bytes_feature, split_rois, expand_to_multiples_of, \
    get_bbox_from_center_points
from core.postprocess import label_image_regions, filter_duplicate_detections
import lib.pyramiddata
from lib.dsutils import get_gaussian_image, save_image
from lib.debugutils import reg_image_to_rgb, mark_rectangle, draw_cross
from models.keras.det_net_1 import det_net_1
from models.keras.det_net_2 import det_net_2
from core.imageaug import RandomImageAugmentation


# default parameters for prediction. those can be override by supplying
# the optional 'params' dictionary to get_prediction_fn() method.

# tile size when the input ROI is split into tiles. tile is a square.
# params key: 'tile_size'
TILE_SIZE = 1024

# overlap in pixels between adjacent tiles. overlap must be big enough to
# contain the detection object plus spare area
# params key: 'tile_overlap:
TILE_OVERLAP = 100

# results are returned as a bounding box. this parameter specifies the bounding
# box size, such that the detection point is at its center. this usually is
# set to the size that is the input size for classification.
# params key: 'label_bbox_size'
LABEL_BBOX_SIZE = (140, 140)

# debug colors list
DEBUG_COLOR_LIST = [(0, 255, 0), (255, 255, 0), (255, 128, 0), (255, 0, 0)]

# Euclidean distance between points to filter duplicates in tiles overlap areas
DUPLICATES_THRESHOLD = 30

# output can be saved as json in the same format as dataset json:
# detections will be presented as labels around the center. this size is the width, height of the label rectangle.
# it should match the input size for the classifiers so that pdsview can show what data is going to be classified
# in the next step.
LABEL_RECT_SIZE_PIXELS = 140

# default label for a detection
LABEL_STR = 'cell'


class CellsDetector(CellsModel):

    def __init__(self, model_cfg):
        super().__init__(model_cfg=model_cfg)

        # cached gaussian image
        self.gaussian_image = None

    def generate_single_example(self, image_source, label, model_res, **kwargs):
        """
        Generate single example for image detection.

        To generate a single example in cells detection, it's neighbour locations must be
        known at this point. Since we get only a single label as input here, an additional
        parameter which is called scaled_centers_map is added. It contains a dictionary:
        key: (x, y) center point, scaled (both level_scale_div, and resize_factor).
        value: 0 or 1 (no cell/cell)
        """
        if 'scaled_centers_map' not in kwargs:
            raise ValueError('scaled_centers_map arg is required')
        scaled_centers_map = kwargs['scaled_centers_map']

        source_image_size = self.calc_image_shape_in_dataset()[0]
        if source_image_size != self.get_image_shape_in_dataset()[1]:
            raise ValueError('currently only square ROIs are supported')

        if self.gaussian_image is None:
            g_size = int(self.model_cfg['gaussian_size'])
            self.gaussian_image = get_gaussian_image(g_size)

        interpolation_type = self.model_cfg.get('build_interpolation_type', 'linear')

        rois_map = read_rois_from_pyramid(image_source=image_source,
                                          rois=[label],
                                          cells_model=self,
                                          model_res=model_res,
                                          use_bulk_read=False,
                                          thread_count=None,
                                          interpolation_type=interpolation_type)

        # resize_factor is needed for target label images creation
        if image_source.is_same_resolution(model_res):
            resize_factor = None
        else:
            resize_factor = image_source.get_resolution() / model_res

        examples_map = create_label_images(rois_map=rois_map,
                                           flat_label_mapping=self.get_flat_label_mapping(),
                                           level_scale_div=self.get_level_scale_div(),
                                           source_image_size=source_image_size,
                                           scale_down_factor=self.get_scale_down_factor(),
                                           gaussian_image=self.gaussian_image,
                                           resize_factor=resize_factor,
                                           scaled_centers_map=scaled_centers_map)

        return examples_map[tuple(label[0:5])]

    def generate_examples_map(self, image_source, labels, model_res, thread_count=None):
        """
        Given an input source (pyramid data) and labels - array of tuples: (x, y, w, h, label_str), return a
        dictionary which maps a label to it's feature and target label.
        For Detector, the feature is the source numpy image, and the target label is a (downsized) guassians image.
        :return: dictionary of: key=label (x, y, w, h, label_tr) -> tuple (src numpy image, target numpy image),
                 where target numpy image is a uint8 mono image (downsized according to model configuration).
        """

        # create label image from the rois_map dictionary
        # this function returns a dictionary like rois_map with values of a tuple: (source image, label image)
        # for each label (dict keys).
        source_image_size = self.calc_image_shape_in_dataset()[0]
        if source_image_size != self.get_image_shape_in_dataset()[1]:
            raise ValueError('currently only square ROIs are supported')

        if self.gaussian_image is None:
            g_size = int(self.model_cfg['gaussian_size'])
            self.gaussian_image = get_gaussian_image(g_size)

        # read images from image source according to the input labels and model configuration
        rois_map = read_rois_from_pyramid(image_source=image_source,
                                          rois=labels,
                                          cells_model=self,
                                          model_res=model_res)

        # resize_factor is needed for target label images creation
        if image_source.is_same_resolution(model_res):
            resize_factor = None
        else:
            resize_factor = image_source.get_resolution() / model_res

        examples_map = create_label_images(rois_map=rois_map,
                                           flat_label_mapping=self.get_flat_label_mapping(),
                                           level_scale_div=self.get_level_scale_div(),
                                           source_image_size=source_image_size,
                                           scale_down_factor=self.get_scale_down_factor(),
                                           gaussian_image=self.gaussian_image,
                                           resize_factor=resize_factor)

        return examples_map

    def serialize_example(self, example, id_str):
        image, label_image = example
        feature = {
            "image": bytes_feature(image),
            "label_image": bytes_feature(label_image),
            "id": bytes_feature(id_str)
        }

        tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
        return tf_example.SerializeToString()

    def get_parser_fn(self, **kwargs):
        image_shape = self.get_image_shape_in_dataset()
        scale_down_factor = self.get_scale_down_factor()
        label_image_shape = (image_shape[0] // scale_down_factor,
                             image_shape[1] // scale_down_factor,
                             1)
        parser_fn = functools.partial(tfrecord_parser_fn,
                                      image_shape=image_shape,
                                      label_image_shape=label_image_shape)
        return parser_fn

    def get_example_preview_image(self, example, sub_image_shape=None, fill_color=(60, 0, 0)):
        """
        Given a parsed example of CellsDetection data type, return a preview image. Preview
        images are used in dspreview for debug/validation purposes and are the src and target
        images.
        :param example: parsed example tensor
        :param sub_image_shape: shape of each sub image in the preview image.
                                If None, then returned shape is the src image
                                shape.
        :param fill_color: RGB values of color to fill inside the gaps between preview_shape and
                           src shape
        :return: preview image (RGB numpy array)
        """

        # convert from tensor (eager)
        input_dict = example[0]
        target = example[1]
        src_img = np.squeeze(input_dict["image"].numpy())
        lbl_img = np.squeeze(target.numpy())
        src_h, src_w = src_img.shape[0], src_img.shape[1]
        lbl_h, lbl_w = lbl_img.shape[0], lbl_img.shape[1]

        # resize label image
        scale = self.get_scale_down_factor()
        if src_h != scale * lbl_h or src_w != scale * lbl_w:
            raise ValueError('dimensions error')

        lbl_img = cv2.resize(reg_image_to_rgb(reg_image=lbl_img, channel=-1),
                             dsize=(scale * lbl_h, scale * lbl_w))

        if sub_image_shape:
            s_img_h, s_img_w = sub_image_shape
            if s_img_h < src_h or s_img_w < src_w:
                raise ValueError(f'preview shape must be >= src shape on every dimension. '
                                 f'preview={sub_image_shape}, src={src_img.shape}')
        else:
            s_img_h = src_h
            s_img_w = src_w

        preview_img = np.full(shape=(s_img_h, 2*s_img_w, 3), dtype='uint8', fill_value=fill_color)
        gap = (s_img_h - src_h) // 2

        preview_img[gap:gap+src_h, gap:gap+src_w, :] = src_img
        preview_img[gap:gap+src_h, s_img_w+gap:s_img_w+gap+src_w, :] = lbl_img

        return preview_img

    def get_preview_image_shape(self, sub_image_shape=None):
        """
        Returns the shape of a single preview image. Preview shape for detector is 2x the width
        of the sub image shape.
        :param sub_image_shape: if given, the output is calculated taking this shape as the base
                                size of the preview shape.
        :return: tuple (h, w, ch)
        """
        if not sub_image_shape:
            shape = list(self.get_image_shape_in_dataset())
        else:
            shape = list(sub_image_shape)

        shape[1] = 2*shape[1]
        return tuple(shape)

    def get_inputs(self, training):

        input_image_size = (None, None, 3)

        inputs = {
            'image': tf.keras.layers.Input(shape=input_image_size, name='image', dtype=tf.float32)
        }

        return inputs

    def get_outputs(self, inputs, training):

        det_model_name = self.model_cfg.get('det_net_type', 'det_net_1')
        if det_model_name == 'det_net_1':
            det_fn = det_net_1
        elif det_model_name == 'det_net_2':
            det_fn = det_net_2
        else:
            raise ValueError(f'Illegal det model name {det_model_name}')

        outputs = det_fn(inputs=inputs['image'], dropout=None)

        return outputs

    def get_losses(self):

        losses = 'mse'

        return losses

    def get_metrics(self, training):

        metrics = ['mse']

        return metrics

    def get_keras_model(self, training=False, compile=True):
        """
        Returns a keras model for training.
        """

        inputs = self.get_inputs(training)
        outputs = self.get_outputs(inputs, training)
        loss = self.get_losses()
        metrics = self.get_metrics(training)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='det')
        if compile:
            model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss=loss, metrics=metrics)

        return model

    def get_checkpoint_callback(self, output_model_file, save_best_only=False):

        """
        :param output_model_file: checkpoint output model file
        :param save_best_only: save best model with respect to the evaluated metric
        :return: checkpoint Keras callback
        """

        if save_best_only:
            checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=output_model_file,
                                                               save_best_only=save_best_only)
            return checkpoint_cb

        # modify model filename with epoch and val loss data
        # path parts
        p = output_model_file.split('.')
        if not len(p) == 2:
            raise ValueError(f'invalid output_model_file chars: {output_model_file}')
        prefix = p[0]
        suffix = p[1]
        output_model_file = prefix + '-{epoch:03d}.' + suffix
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=output_model_file)

        return checkpoint_cb

    def get_preprocess_crop_fn(self):
        """
        returns a function which preprocesses the input data and returns the center
        cropped version of the dataset. Map this on a Dataset instance to get the
        correct input for validation set and preview.
        """
        # training features (image) target size
        target_size = int(self.model_cfg['input_image_size'][0])

        # label image target size
        target_label_size = target_size // int(self.model_cfg['scale_down_factor'])

        preprocess_fn = functools.partial(keras_crop_preprocess,
                                          target_size=target_size,
                                          target_label_size=target_label_size)

        return preprocess_fn

    def get_preprocess_augmentations_class(self):
        """
        returns a class (Keras layer), callable, which is mapped on tf.data.Dataset instance
        in training, to produce image augmentations.
        :return:
        """
        image_aug_params = self.get_image_aug_params()

        max_shift = image_aug_params.get('max_shift', 0)
        rotation_angle_multiples = image_aug_params.get('rotation_angle_multiples', 90)
        max_rotation_angle = image_aug_params.get('max_rotation_angle', 0)

        # the ratio between X and Y images (Y is smaller by scale_down_factor)
        scale_down_factor = self.get_scale_down_factor()

        p_aug_inst = RandomImageAugmentation(max_shift=max_shift,
                                             max_rotation_angle=max_rotation_angle,
                                             angle_multiples=rotation_angle_multiples,
                                             augment_y=True,
                                             y_scale_down_factor=scale_down_factor)

        return p_aug_inst.augment

    def get_predict_fn(self, params=None):
        """
        Returns prediction function tied to this cells model instance.
        predict_fn(image_source, rois)
        image_source: scan image source
        params: dictionary of parameters to pass to the prediction function
        """

        # check if model is loaded
        trained_model = self.get_trained_model()
        if not trained_model:
            raise RuntimeError('keras model is not loaded.')

        # default params
        _params = {'tile_size': TILE_SIZE,
                   'tile_overlap': TILE_OVERLAP,
                   'label_bbox_size': LABEL_BBOX_SIZE,
                   'label_str': LABEL_STR}

        # update with args, if given
        if params is not None:
            _params.update(params)

        fn = functools.partial(predict_fn, cells_model=self, params=_params)
        return fn


# TFRecord parser function for cells detection
def tfrecord_parser_fn(record_bytes, image_shape, label_image_shape):
    example = tf.io.parse_single_example(
        # data
        record_bytes,

        # schema
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "label_image": tf.io.FixedLenFeature([], tf.string),
            "id": tf.io.FixedLenFeature([], tf.string)
        })
    image_ser = example["image"]
    image = tf.io.parse_tensor(image_ser, tf.uint8)
    image = tf.cast(image, dtype='float32')
    image = tf.ensure_shape(image, shape=image_shape, name='src_image')

    label_ser = example["label_image"]
    label = tf.io.parse_tensor(label_ser, tf.uint8)
    label = tf.cast(label, dtype='float32')
    label = tf.expand_dims(label, -1)
    label = tf.ensure_shape(label, shape=label_image_shape, name='label_image')

    id_str_ser = example["id"]
    id_str = tf.io.parse_tensor(id_str_ser, tf.string)

    features = {"image": image, "id": id_str}

    return features, label


def keras_crop_preprocess(inputs, label, target_size, target_label_size):

    x = inputs['image']
    y = label

    # shallow copy of the dataset dict (TFRecord fields)
    modified_inputs = inputs.copy()

    # NOTE: this assumes that width == height for image and image_label
    x_size = target_size
    y_size = target_label_size

    x = tf.keras.layers.CenterCrop(x_size, x_size)(x)
    y = tf.keras.layers.CenterCrop(y_size, y_size)(y)

    modified_inputs['image'] = x

    return modified_inputs, y


def predict_single_roi(cells_model, image_source, roi, params, roi_idx=0):
    level_scale_div = cells_model.get_level_scale_div()

    sdf = cells_model.get_scale_down_factor()

    # scale down ROI coordinates from full res to detection's lower res
    scaled_roi = [e // level_scale_div for e in roi]

    # calculate resize_factor from model resolution and input resolution
    model_res = cells_model.get_model_res()

    # resize_factor is needed for target label images creation. resizing by a factor of resize_factor
    # brings the image source the the model resolution.
    if image_source.is_same_resolution(model_res):
        resize_factor = 1.0
    else:
        resize_factor = image_source.get_resolution() / model_res

    tile_size = params['tile_size']
    tile_overlap = params['tile_overlap']

    # ROI may need to be split to multiple parts (with overlaps) to reduce
    # input size that can fit the GPU memory. Each ROI in the list is represented
    # (x, y, tile_size, tile_size) ROI tuple. Results are joined after the
    # running model.predict()
    tiled_rois_indexed = split_rois(scaled_roi, tile_size, tile_overlap)
    tiled_rois = tiled_rois_indexed.values()
    tiled_rois = sorted(tiled_rois, key=lambda elem: elem[2] * elem[3], reverse=True)

    debug_save_to = params.get('debug_save_to', None)
    marked_image = None
    if debug_save_to:

        # full roi image is saved only under certain size
        if scaled_roi[2] <= 65000 and scaled_roi[3] <= 65000:
            marked_image = lib.pyramiddata.read_roi_from_pyramid(image_source=image_source,
                                                                 roi=scaled_roi,
                                                                 level_scale_div=level_scale_div,
                                                                 resize_factor=None)

            for tiled_roi in tiled_rois:
                local_tile = (tiled_roi[0] - scaled_roi[0], tiled_roi[1] - scaled_roi[1], tiled_roi[2], tiled_roi[3])
                tile_x, tile_y = tiled_roi[0:2]
                mark_rectangle(image=marked_image, rect=local_tile, color=(0, 0, 255), text=f'{tile_x},{tile_y}',
                               line_width=1)

            mark_rectangle(marked_image, rect=[40, 40, 140, 80], color=[60, 60, 0],
                           text=f'scaled down by: {level_scale_div}',
                           draw_border=False, font_scale=1.0)

            save_image(marked_image, f'{debug_save_to}/full-roi_{roi_idx}-tiled.jpg', is_bgr=True)

    # global_detections = []

    model = cells_model.get_trained_model()

    tile_idx = 0
    global_detections_map = {}
    for tile_indices in tiled_rois_indexed:
        tiled_roi = tiled_rois_indexed[tile_indices]
        tile_x, tile_y, tile_w, tile_h = tiled_roi[0], tiled_roi[1], tiled_roi[2], tiled_roi[3]
        logging.debug(f'[{tile_idx+1}/{len(tiled_rois_indexed)}] read image data (size: {tile_w}x{tile_h})')

        # load image data from source, in lower resolution (level_scale_div) with a possible fixed resize for diffs
        # between model res and pyramid res (resize_factor)
        image_data = lib.pyramiddata.read_roi_from_pyramid(image_source=image_source,
                                                           roi=tiled_roi,
                                                           level_scale_div=level_scale_div,
                                                           resize_factor=resize_factor)

        # input data needs to be able to divide several times by half without encountering odd number.
        # therefore an expansion of the image takes place to a high enough multiple of 2.
        # logging.debug('expand')
        expanded_image = expand_to_multiples_of(img=image_data, multiple=16, fill_value=(255, 255, 255))
        if expanded_image.shape != image_data.shape:
            logging.debug(f'image expanded: {image_data.shape} -> {expanded_image.shape}')

        # convert to tensor
        # logging.debug('convert to tensor')
        t_im = tf.convert_to_tensor(expanded_image)
        t_im = tf.reshape(t_im, (1,) + expanded_image.shape)

        logging.debug('predict')
        predict_result = model.predict(t_im)
        predict_result = np.squeeze(predict_result)

        # logging.debug('find local max')
        local_max = label_image_regions(predict_result)

        localized_detections = []
        for i in range(local_max.shape[0]):
            # relative coordinates in the input image
            yc_rel, xc_rel = local_max[i, 0], local_max[i, 1]
            localized_detections.append((xc_rel, yc_rel))

        if debug_save_to:
            rgb_predict_result = reg_image_to_rgb(predict_result)
            for detection in localized_detections:
                thickness = 1
                draw_cross(rgb_predict_result, detection, color=(0, 255, 0), d=1, thickness=1)
                draw_cross(expanded_image, (2 * detection[0], 2 * detection[1]),
                           color=(0, 255, 0), d=34, thickness=thickness)
            save_image(expanded_image, f'{debug_save_to}/tile-src-x{tile_x}-y{tile_y}.jpg', is_bgr=True)
            save_image(rgb_predict_result, f'{debug_save_to}/tile-det-x{tile_x}-y{tile_y}.jpg')

        # detection in src res (full / level_scale_div). up scaled by factor of scale_down_factor (sdf) 2
        # from NN's reg image output
        detections_in_src_res = [(sdf * lx, sdf * ly) for lx, ly in localized_detections]

        # detection in src res, rescale by 1/resize_factor (model res/pyramid res)
        detections_in_src_res_before_resize = [(lx/resize_factor, ly/resize_factor) for lx, ly in detections_in_src_res]

        # detections in src res, global coordinates, scaled.
        scaled_global_detections = [(tx + tile_x, ty + tile_y) for tx, ty in detections_in_src_res_before_resize]

        # detections in full res, global coordinated (unscaled)
        # unscaled_global_detections = [(int(level_scale_div * tx), int(level_scale_div * ty))
        #                              for tx, ty in scaled_global_detections]

        global_detections_map[tile_indices] = scaled_global_detections
        # global_detections += unscaled_global_detections
        tile_idx += 1

    global_detections_map = filter_duplicate_detections(global_detections_map, dist_threshold=25)

    global_detections = []
    for detections in global_detections_map.values():
        if not detections:
            continue

        # scale back to top level scale
        top_level_scaled_detections = [(int(level_scale_div * tx), int(level_scale_div * ty))
                                       for tx, ty in detections]
        global_detections += top_level_scaled_detections

    if debug_save_to and marked_image is not None:
        marked_image_with_duplicates = np.copy(marked_image)
        d_map = {}
        for detection in global_detections:
            if detection not in d_map:
                d_map[detection] = 1
            else:
                d_map[detection] += 1

            color_idx = min(d_map[detection] - 1, len(DEBUG_COLOR_LIST) - 1)
            color = DEBUG_COLOR_LIST[color_idx]
            dx = (detection[0] - roi[0]) // level_scale_div
            dy = (detection[1] - roi[1]) // level_scale_div
            draw_cross(img=marked_image_with_duplicates, center=(dx, dy), color=color, d=24, thickness=1)
        mark_rectangle(marked_image_with_duplicates, rect=[40, 40, 140, 80], color=(60, 60, 0),
                       text=f'scaled down by: {level_scale_div}',
                       draw_border=False, font_scale=1.0)
        save_image(marked_image_with_duplicates, f'{debug_save_to}/full-roi_{roi_idx}-marked.jpg', is_bgr=True)

    # results are center points (x,y)
    return global_detections


def predict_fn(cells_model, image_source, rois, params):
    # collaterals are returned along with the results. can be used to transfer
    # metadata (such as cached ROI image data) between calls to predict_fn of
    # different models
    collaterals = {}

    total_detected_labels = []
    total_rois = []
    for roi_idx, roi in enumerate(rois):

        # run predict (inference). result is a list of the detected center points (x,y)
        # of the detected cells.
        center_points = predict_single_roi(cells_model, image_source, roi[0:4], params, roi_idx=roi_idx)

        # transform the center points to label format: (x_topleft, y_topleft, width, height, default label)
        bbox_size = params['label_bbox_size']
        label_str = params['label_str']
        detected_labels = get_bbox_from_center_points(centers=center_points,
                                                      bbox_size=bbox_size,
                                                      label_str=label_str)

        total_detected_labels += detected_labels

        # for debug viewing (pdsview), an ROI description string is added
        if len(roi) >= 5:
            roi_with_label = roi
        else:   # roi length is 4, containing no label string
            roi_with_label = [*roi, f'ROI idx {roi_idx}']

        total_rois.append(roi_with_label)

    scan_id = image_source.infer_scan_id_from_src()
    if not scan_id:
        scan_id = 'unknown_scan_id'

    pyramid_res = image_source.get_resolution()
    results = {'scan_id': scan_id,
               'pyramid_resolution': pyramid_res,
               'labels': total_detected_labels,
               'ROIs': total_rois}
    return results, collaterals
