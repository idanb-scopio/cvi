import functools
import logging
import numpy as np
import tensorflow as tf

import lib.pyramiddata
from models.cellsmodel import CellsModel
from core.utils import read_rois_from_pyramid, bytes_feature, int64_feature
from core.labels import create_class_debug_titles
from lib.debugutils import mark_rectangle
from lib.mosaicsaver import MosaicSaver
from models.keras.resnet import res_net
from core.imageaug import RandomImageAugmentation

from models.losses.loss_factory import classification_losses_generator
from models.losses.identity_loss import IdentityLoss
from models.layers.features_center_loss_layer import FeaturesCenterLossLayer

# ** prediction default parameters **

# batch size is the amount of images to put in a batch for prediction.
# larger usually means more efficient, but too large will hit an out of memory
# GPU issues and degrade performance.
# params key: 'batch_size'
BATCH_SIZE = 256


class CellsClassifier(CellsModel):

    def __init__(self, model_cfg):
        super().__init__(model_cfg=model_cfg)

        # cache for class debug titles dictionary
        self.use_feature_loss = self.is_features_loss()
        self.class_debug_titles = None

    def generate_single_example(self, image_source, label, model_res, **kwargs):
        """Generate single examples for image classification."""

        source_image_size = self.calc_image_shape_in_dataset()[0]
        if source_image_size != self.get_image_shape_in_dataset()[1]:
            raise ValueError('currently only square ROIs are supported')

        interpolation_type = self.model_cfg.get('build_interpolation_type', 'linear')

        rois_map = read_rois_from_pyramid(image_source=image_source,
                                          rois=[label],
                                          cells_model=self,
                                          model_res=model_res,
                                          use_bulk_read=False,
                                          thread_count=None,
                                          interpolation_type=interpolation_type)

        label_mapping = self.get_flat_label_mapping()
        class_num = label_mapping[label[4]]
        image = rois_map[tuple(label[0:5])]
        return image, class_num

    def generate_examples_map(self, image_source, labels, model_res, thread_count=None):
        """
        Given an input source (pyramid data) and labels - array of tuples: (x, y, w, h, label_str), return a
        dictionary which maps a label to it's feature and target label.
        For Classifier, the feature is the source numpy image, and the target label is an int representing a class
        index from 0 to N-1 (N number of classes).
        :return: dictionary of: key=label (x, y, w, h, label_tr) -> tuple (src numpy image, int).
        """

        source_image_size = self.calc_image_shape_in_dataset()[0]
        if source_image_size != self.get_image_shape_in_dataset()[1]:
            raise ValueError('currently only square ROIs are supported')

        label_mapping = self.get_flat_label_mapping()

        # read images from image source according to the input labels and model configuration
        interpolation_type = self.model_cfg.get('build_interpolation_type', 'linear')
        rois_map = read_rois_from_pyramid(image_source=image_source,
                                          rois=labels,
                                          cells_model=self,
                                          model_res=model_res,
                                          use_bulk_read=True,
                                          thread_count=thread_count,
                                          interpolation_type=interpolation_type)

        examples_map = {}
        for lbl, src_im in rois_map.items():
            class_num = label_mapping[lbl[4].lower()]
            examples_map[lbl] = (src_im, class_num)

        return examples_map

    def serialize_example(self, example, id_str):

        image, class_num = example
        feature = {
            "image": bytes_feature(image),
            "class": int64_feature(class_num),
            "id": bytes_feature(id_str)
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example.SerializeToString()

    def get_parser_fn(self, **kwargs):
        training = kwargs.get('training', False)
        return_one_hot = kwargs.get('return_one_hot', True)

        num_classes = self.get_num_classes()
        image_shape = self.get_image_shape_in_dataset()

        parser_fn = functools.partial(tfrecord_parser_fn,
                                      num_classes=num_classes,
                                      image_shape=image_shape,
                                      return_one_hot=return_one_hot,
                                      use_feature_loss=self.use_feature_loss and training)

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

        class_debug_titles = self.get_class_debug_titles()

        # convert from tensor (eager)
        input_dict = example[0]
        target = example[1]
        src_img = np.squeeze(input_dict["image"].numpy())
        class_num = target.numpy()
        class_title = class_debug_titles[class_num]

        src_h, src_w = src_img.shape[0], src_img.shape[1]

        if sub_image_shape:
            s_img_h, s_img_w = sub_image_shape
            if s_img_h < src_h or s_img_w < src_w:
                raise ValueError(f'preview shape must be >= src shape on every dimension. '
                                 f'preview={sub_image_shape}, src={src_img.shape}')
        else:
            s_img_h = src_h
            s_img_w = src_w

        preview_img = np.full(shape=(s_img_h, s_img_w, 3), dtype='uint8', fill_value=fill_color)
        gap = (s_img_h - src_h) // 2
        preview_img[gap:gap+src_h, gap:gap+src_w, :] = src_img

        # add class title "class_num: class_str" to the image. keep an offset to allow space for mosaic saver indices
        mark_rectangle(image=preview_img,
                       rect=(0, 0, s_img_w, s_img_h),
                       text=f'        {class_title}',
                       color=(0, 200, 0),
                       line_width=1,
                       draw_border=False)

        return preview_img

    def get_class_debug_titles(self):
        if self.class_debug_titles is None:
            label_mappings = self.get_label_mapping()
            self.class_debug_titles = create_class_debug_titles(label_mappings, truncate_text=10)

        return self.class_debug_titles

    def get_preview_image_shape(self, sub_image_shape=None):
        """
        Returns the shape of a single preview image. Preview shape for classifier is just the sub image shape.
        :param sub_image_shape: if given, the output is calculated taking this shape as the base
                                size of the preview shape.
        :return: tuple (h, w, ch)
        """
        if not sub_image_shape:
            shape = self.get_image_shape_in_dataset()
        else:
            shape = sub_image_shape

        return shape

    def is_features_loss(self):

        use_feature_loss = False
        if 'loss' in self.model_cfg:
            features_loss_key = self.model_cfg['loss'].get('feature_loss', None)
            use_feature_loss = features_loss_key is not None
        return use_feature_loss

    def get_features_loss_config(self):
        assert self.model_cfg.get('loss', None)
        features_loss_key = self.model_cfg['loss'].get('feature_loss', None)
        loss_config = self.model_cfg['loss'].get(features_loss_key, None)
        return loss_config

    def get_inputs(self, training):

        input_image_size = self.model_cfg['input_image_size']

        inputs = {
            'image': tf.keras.layers.Input(shape=input_image_size, name='image', dtype=tf.float32)
        }
        if self.use_feature_loss and training:
            num_classes = self.get_num_classes()
            inputs['label'] = tf.keras.layers.Input(shape=[num_classes], name='label', dtype=tf.float32)

        return inputs

    def get_features_layer_loss(self, features, inputs):

        center_loss_config = self.get_features_loss_config()
        assert center_loss_config

        loss_layer_input = {
            'features': features,
            'one_hot_labels': inputs['label']
        }

        features_layer_loss = FeaturesCenterLossLayer(**center_loss_config)
        center_loss = features_layer_loss(inputs=loss_layer_input)

        return center_loss

    def get_outputs(self, inputs, training):

        num_classes = self.get_num_classes()

        if self.use_feature_loss and training:
            features, probabilities = res_net(inputs['image'], num_classes=num_classes)
            features_loss = self.get_features_layer_loss(features, inputs)
            outputs = {
                'probabilites': probabilities,
                'features_loss': features_loss
            }
        else:
            _, outputs = res_net(inputs['image'], num_classes=num_classes)

        return outputs

    def get_losses(self, training):

        loss_config = self.model_cfg.get('loss', None)

        loss_fn = classification_losses_generator(loss_config)

        if self.use_feature_loss and training:
            losses = {
                'probabilites': loss_fn,
                'features_loss': IdentityLoss()
            }
        else:
            losses = loss_fn

        return losses

    def get_metrics(self, training):

        if self.use_feature_loss and training:
            metrics = {
                'probabilites': ['accuracy'],
                'features_loss': ['mae']
            }
        else:
            metrics = ['accuracy']

        return metrics

    def get_keras_model(self, training=False):

        inputs = self.get_inputs(training)
        outputs = self.get_outputs(inputs, training)
        loss = self.get_losses(training)
        metrics = self.get_metrics(training)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='res_net')
        model.compile(optimizer='adam', loss=loss, metrics=metrics)

        return model

    def save_predict_model(self, output_dir, print_summary=False, best_output_model_file=None):

        if not self.use_feature_loss:
            return

        train_model_file =  best_output_model_file if best_output_model_file else f'{output_dir}/model_tr.h5'
        keras_model = self.get_keras_model(training=False)
        keras_model.load_weights(train_model_file, by_name=True)
        predict_model_file = f'{output_dir}/model.h5'
        keras_model.save(filepath=predict_model_file, include_optimizer=False)
        with open(f'{output_dir}/predict_model_summary.txt', 'w') as f:
            keras_model.summary(print_fn=lambda x: f.write(x + '\n'))
        if print_summary:
            keras_model.summary()

    def get_train_model_weights_suffix(self):
        suffix = ''
        if self.use_feature_loss:
            suffix = '_tr'
        return suffix

    def get_checkpoint_callback(self, output_model_file, save_best_only=False):

        """
        :param output_model_file: checkpoint output model file
        :param save_best_only: save best model with respect to the evaluated metric
        :return: checkpoint Keras callback
        """

        metric_name = 'probabilities_accuracy' if self.use_feature_loss else 'accuracy'
        if save_best_only:
            checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=output_model_file,
                                                               save_best_only=True,
                                                               verbose=1,
                                                               monitor=f'val_{metric_name}',
                                                               mode='max')

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

    def use_train_input_format_for_eval(self):

        train_input_format = self.use_feature_loss

        return train_input_format

    def get_preprocess_crop_fn(self):
        """
        returns a function which preprocesses the input data and returns the center
        cropped version of the dataset. Map this on a Dataset instance to get the
        correct input for validation set and preview.
        """
        # training features (image) target size
        target_size = int(self.model_cfg['input_image_size'][0])

        train_preprocess_fn = functools.partial(keras_crop_preprocess,
                                                target_size=target_size)

        return train_preprocess_fn

    def get_preprocess_augmentations_class(self):
        """
        returns a class (Keras layer), callable, which is mapped on tf.data.Dataset instance
        in training, to produce image augmentations.
        :return:
        """
        image_aug_params = self.get_image_aug_params()
        if not image_aug_params:
            logging.warning('there are no training image augmentation parameters in model configuration. defaults'
                            ' will be used.')
            image_aug_params = {}

        max_shift = image_aug_params.get('max_shift', 0)
        rotation_angle_multiples = image_aug_params.get('rotation_angle_multiples', 90)
        max_rotation_angle = image_aug_params.get('max_rotation_angle', 0)

        p_aug_inst = RandomImageAugmentation(max_shift=max_shift,
                                             max_rotation_angle=max_rotation_angle,
                                             angle_multiples=rotation_angle_multiples)

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
        _params = {'batch_size': BATCH_SIZE}

        # update with args, if given
        if params is not None:
            _params.update(params)

        fn = functools.partial(predict_fn, cells_model=self, params=_params)
        return fn


# function for parsing TFRecord data
def tfrecord_parser_fn(record_bytes, num_classes, image_shape, return_one_hot=True,
                       use_feature_loss=False):
    schema = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "class": tf.io.FixedLenFeature([], tf.int64),
            "id": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(record_bytes,  # data
                                         schema)        # schema

    image_ser = example["image"]
    image = tf.io.parse_tensor(image_ser, tf.uint8)
    image = tf.cast(image, dtype='float32')
    image = tf.ensure_shape(image, shape=image_shape, name='src_image')

    # for training, one_hot vector is returned for each class number
    # for preview, the class number itself is returned
    if return_one_hot:
        label = tf.one_hot(tf.cast(example["class"], tf.int32), num_classes)
    else:
        label = example["class"]

    id_str_ser = example["id"]
    id_str = tf.io.parse_tensor(id_str_ser, tf.string)

    in_features = {'image': image, 'label': label, 'id': id_str}
    targets = label

    if use_feature_loss:
        in_features['label'] = label
        targets = {
            'probabilites': label,
            'features_loss': example["class"]
        }

    return in_features, targets


def keras_crop_preprocess(inputs, label, target_size):

    # NOTE: this assumes that width == height for image and image_label
    x_size = target_size
    center_crop_fn = tf.keras.layers.CenterCrop(x_size, x_size)

    modified_inputs = inputs.copy()
    modified_inputs['image'] = center_crop_fn(inputs['image'])

    return modified_inputs, label


def read_rois(cells_model, image_source, rois, thread_count=None):
    # calculate resize_factor from model resolution and input resolution
    model_res = cells_model.get_model_res()

    # resize_factor is needed for target label images creation. resizing by a factor of resize_factor
    # brings the image source the the model resolution.
    if image_source.is_same_resolution(model_res):
        resize_factor = 1.0
    else:
        resize_factor = image_source.get_resolution() / model_res

    # input image shape is the shape of the input image expected by the keras model.
    input_image_shape = cells_model.get_input_image_shape()

    logging.info(f'reading {len(rois)} rois from pyramid ...')
    rois_map = lib.pyramiddata.read_rois_from_pyramid(rois=rois,
                                                      image_source=image_source,
                                                      target_size=input_image_shape[0:2],
                                                      resize_factor=resize_factor,
                                                      thread_count=thread_count)

    return rois_map


def predict_fn(cells_model, image_source, rois, params):
    logging.debug('cells classifier: predict_fn start')
    # collaterals are returned along with the results. can be used to transfer
    # metadata (such as cached ROI image data) between calls to predict_fn of
    # different models
    collaterals = {}

    # convert to a 4D shape with the first dimension acting as running index between the
    # images.
    input_image_shape = cells_model.get_input_image_shape()

    debug_save_to = params.get('debug_save_to', None)
    debug_ext = params.get('debug_image_ext', 'jpg')
    debug_mosaic = None

    if debug_save_to:
        debug_mosaic = MosaicSaver(sub_image_shape=input_image_shape,
                                   mosaic_w=10000,
                                   mosaic_h=10000,
                                   output_dir=debug_save_to,
                                   image_ext=debug_ext)

    logging.debug('cells classifier: reading ROIs')
    if 'cached_rois_map' in params:
        # when using cached_rois_map, it's the responsibility of the caller to
        # ensure that all rois in the input arg 'rois' are indeed inside the
        # cached rois map.
        rois_map = params['cached_rois_map']
    else:
        # read roi data from image source. image data is resized if model_res != image_res
        # data is returned as a map: roi -> numpy image
        rois_map = read_rois(cells_model, image_source, rois, thread_count=16)

    # initialize the 4D input array
    n_rois = len(rois)
    rois_array = np.empty(shape=(n_rois, *input_image_shape), dtype='uint8')

    # copy image data to the 4D array
    logging.debug('cells classifier: copy to 4D array')
    for idx, bbox_roi in enumerate(rois):
        rois_array[idx, :, :, :] = rois_map[tuple(bbox_roi)]

    logging.debug('cells classifier: convert to tensor')
    rois_tensor = tf.convert_to_tensor(rois_array)

    batch_size = params['batch_size']
    trained_model = cells_model.get_trained_model()
    logging.info(f'classifying {n_rois} cells. batch_size={batch_size}')

    # run keras model predict(). This returns an array of shape (n_rois, n_classes)
    # containing probabilities
    raw_predict_res = trained_model.predict(rois_tensor, batch_size=batch_size)
    logging.debug('cells classifier: argmax-ing')
    predicted_class_num = np.argmax(raw_predict_res, axis=1)
    predicted_confidence = np.max(raw_predict_res, axis=1)

    label_mapping = cells_model.get_label_mapping()
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}
    results_map = {}
    logging.debug('cells classifier: building results map')
    for idx, bbox_roi in enumerate(rois):
        results_map[tuple(bbox_roi)] = (reverse_label_mapping[predicted_class_num[idx]],
                                        predicted_confidence[idx] * 100.0)

    if debug_save_to:
        # get nicely formatted '<class number>: <class string>' map for class number
        class_debug_titles = cells_model.get_class_debug_titles()
        debug_summary_str = ''

        for idx, bbox_roi in enumerate(rois):
            # the input image is marked and therefore copied
            debug_image = rois_map[tuple(bbox_roi)].copy()

            class_num = predicted_class_num[idx]
            confidence = predicted_confidence[idx]
            debug_text = f'{class_debug_titles[class_num]} {100 * confidence:.1f}%'

            mark_rectangle(debug_image, rect=(0, 20, debug_image.shape[1], debug_image.shape[0]),
                           text=debug_text, draw_border=False, color=(0, 180, 0), line_width=1, font_scale=0.4)

            debug_mosaic.add_image(debug_image)
            debug_summary_str += f'{idx}: {str(bbox_roi)}\n'

        debug_mosaic.save()
        with open(f'{debug_save_to}/index_origin.txt', 'w') as f:
            f.write(debug_summary_str)

    if params.get('cache_rois', False):
        collaterals['cached_rois_map'] = rois_map

    collaterals['probabilities_matrix'] = raw_predict_res
    collaterals['results_map'] = results_map

    # summarize results as labels array
    logging.debug('cells classifier: building results json')
    labels = []

    for roi in results_map:
        x, y, w, h = roi[0], roi[1], roi[2], roi[3]
        labels.append([x, y, w, h, results_map[roi][0]])

    scan_id = image_source.infer_scan_id_from_src()
    if not scan_id:
        scan_id = 'unknown_scan_id'

    pyramid_res = image_source.get_resolution()
    results = {'scan_id': scan_id,
               'pyramid_resolution': pyramid_res,
               'labels': labels,
               'ROIs': []}

    logging.debug('cells classifier: end of predict_fn')
    return results, collaterals

