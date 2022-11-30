import math
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import random

DEFAULT_SEED = 42
DEFAULT_MAX_SHIFT = 0
DEFAULT_MAX_ROTATION_ANGLE = 360
DEFAULT_ANGLE_MULTIPLES = 90

# multiplication vec
MIN_FACTOR_RGB = [0.82, 0.9, 0.9]
MAX_FACTOR_RGB = [1.3, 1.2, 1]
# hue
HUE_SHIFT_SCALE = 0.016
MAX_HUE_SHIFT = 0.08
# saturation
SATURATION_SCALE = 0.25
MAX_SATURATION_FACTOR = 2.
MIN_SATURATION_FACTOR = 0.4
# value (hsv)
HSV_VALUE_FACTOR_SCALE = 0.12
MAX_HSV_VALUE_FACTOR = 2.
MAX_HSV_VALUE_SHIFT = 0.
HSV_VALUE_SHIFT_SCALE = 0.1
# contrast
COLOR_CONTRAST_SCALE = 0.06
MAX_COLOR_CONTRAST = 1.3
# color shift
COLOR_SHIFT_SCALE = 0.04
MAX_COLOR_SHIFT = 0.2
# gamma correction
GAMMA_SCALE = 0.04


def color_augment_images(x, colors):
    # contrast
    r, g, b = tf.split(x, num_or_size_splits=3, axis=3)
    r = tf.clip_by_value(tf.reduce_mean(r) + (r - tf.reduce_mean(r)) * colors['contrast_factor'], 0, 255)
    g = tf.clip_by_value(tf.reduce_mean(g) + (g - tf.reduce_mean(g)) * colors['contrast_factor'], 0, 255)
    b = tf.clip_by_value(tf.reduce_mean(b) + (b - tf.reduce_mean(b)) * colors['contrast_factor'], 0, 255)
    x = tf.concat([r, g, b], axis=3, name='concat')

    # gamma
    # x = tf.image.adjust_gamma(x, gamma=colors['gamma'], gain=1) # had problems with tf params
    x = tf.clip_by_value(tf.math.pow(x, colors['gamma']), 0, 255)

    # hsv
    x = x / 255.
    x_hsv = tf.image.rgb_to_hsv(x)
    h, s, v = tf.split(x_hsv, num_or_size_splits=3, axis=3)
    h = tf.math.floormod(h + (1 + colors['hue_shift']), 1.)
    s = tf.clip_by_value(s * colors['saturation_factor'], 0, 1.)
    v = tf.clip_by_value(v * colors['value_factor'] + colors['value_shift'], 0, 1.)
    new_hsv = tf.concat([h, s, v], axis=3, name='concat')
    x = tf.image.hsv_to_rgb(new_hsv)
    x *= 255.

    # multiplication vector and shift in rgb
    r, g, b = tf.split(x, num_or_size_splits=3, axis=3)
    r = tf.clip_by_value(r * colors['multvec'][0] + colors['color_shift'], 0, 255)
    g = tf.clip_by_value(g * colors['multvec'][1] + colors['color_shift'], 0, 255)
    b = tf.clip_by_value(b * colors['multvec'][2] + colors['color_shift'], 0, 255)
    x = tf.concat([r, g, b], axis=3, name='concat')

    return x


def augment_images(x, angles, shifts, colors=None):
    # colors
    if colors:
        x = color_augment_images(x, colors)

    # rotate
    angles_rad = angles / 180.0 * math.pi   # convert: deg -> rad
    x = tfa.image.rotate(x, angles_rad, fill_mode='constant', fill_value=0.0, interpolation='bilinear')

    # translate
    x = tfa.image.translate(x, shifts)

    return x


def prev_augment_images(x, angles, shifts):
    image_height = tf.cast(tf.shape(x)[1], tf.dtypes.float32)[None]
    image_width = tf.cast(tf.shape(x)[2], tf.dtypes.float32)[None]

    # build rotation affine transform
    t_rotate = tfa.image.angles_to_projective_transforms(angles=angles,
                                                         image_width=image_width,
                                                         image_height=image_height)

    # build translation affine transform
    t_shift = tfa.image.translations_to_projective_transforms(translations=shifts)

    # compose: rotation then translation transforms
    t_transform = tfa.image.compose_transforms([t_rotate, t_shift])

    x_output = tfa.image.transform(x, t_transform, fill_mode='constant')

    return x_output


class RandomImageAugmentation:

    def __init__(self,
                 max_shift=DEFAULT_MAX_SHIFT,
                 max_rotation_angle=360,
                 angle_multiples=DEFAULT_ANGLE_MULTIPLES,
                 augment_y=False,
                 y_scale_down_factor=None):

        self.max_shift = max_shift
        self.max_rotation_angle = max_rotation_angle

        if angle_multiples == 0:
            raise ValueError(f'angle multiples must be a positive number')

        self.angle_multiples = angle_multiples

        self.augment_y = augment_y
        if self.augment_y:
            if not y_scale_down_factor:
                raise ValueError('y_scale_down_factor must be provided.')
            self.y_scale_down_factor = y_scale_down_factor
        else:
            self.y_scale_down_factor = None

    # augmentation function. map this onto a tf.data.Dataset
    def augment(self, inputs, y):

        # deal with one image or batch of images of dims: (H, W, 3)
        x = inputs['image']

        # ensure X's tensor rank is 4: (batch_size, H, W, 3)
        x_shape = tf.shape(x)
        if len(x_shape) == 3:
            orig_rank = 3
            x = tf.expand_dims(x, axis=0)
            if self.augment_y:
                y = tf.expand_dims(y, axis=0)
            bs = 1
        else:
            orig_rank = 4
            bs = x_shape[0]

        # random translation (shift) in X, Y
        if self.max_shift == 0:
            # no shift
            shifts = np.zeros(shape=(bs, 2), dtype='float32')
        else:
            # boundary value for random function (excluded from provided values)
            b_val = self.max_shift
            shifts = tf.cast(tf.random.uniform(shape=(bs, 2), minval=-b_val, maxval=b_val+1,
                                               dtype=tf.dtypes.int32),
                             tf.dtypes.float32)

        # random rotations by degrees
        if self.max_rotation_angle == 0:
            # no rotation
            angles = np.zeros(shape=(bs,), dtype='float32')
        else:
            num_angles = int(self.max_rotation_angle // self.angle_multiples)
            random_k = tf.cast(tf.random.uniform(shape=(bs,), minval=0, maxval=num_angles, dtype=tf.dtypes.int32),
                               tf.dtypes.float32)
            angles = random_k * self.angle_multiples

        # random colors params
        # todo: consider using non symmetrical normal distribution for multiplication vector
        multvec = [1, 1, 1]
        for ind in range(3):
            multvec[ind] = tf.clip_by_value(tf.random.normal(shape=(bs, 1), mean=1.0,
                                                             stddev=(MAX_FACTOR_RGB[ind] - MIN_FACTOR_RGB[ind]) / 6,
                                                             dtype=tf.dtypes.float32),
                                            MIN_FACTOR_RGB[ind], MAX_FACTOR_RGB[ind])
        hue_shift = tf.clip_by_value(tf.random.normal(shape=(bs, 1), mean=0.0, stddev=HUE_SHIFT_SCALE,
                                                      dtype=tf.dtypes.float32), -MAX_HUE_SHIFT, MAX_HUE_SHIFT)
        saturation_factor = tf.clip_by_value(tf.random.normal(shape=(bs, 1), mean=1.0, stddev=SATURATION_SCALE,
                                             dtype=tf.dtypes.float32), MIN_SATURATION_FACTOR, MAX_SATURATION_FACTOR)
        value_factor = tf.clip_by_value(tf.random.normal(shape=(bs, 1), mean=1.0, stddev=HSV_VALUE_FACTOR_SCALE,
                                        dtype=tf.dtypes.float32), 1. / MAX_HSV_VALUE_FACTOR, MAX_HSV_VALUE_FACTOR)
        value_shift = tf.clip_by_value(tf.random.normal(shape=(bs, 1), mean=0.0, stddev=HSV_VALUE_SHIFT_SCALE,
                                       dtype=tf.dtypes.float32), -MAX_HSV_VALUE_SHIFT, MAX_HSV_VALUE_SHIFT)
        contrast_factor = tf.clip_by_value(tf.random.normal(shape=(bs, 1), mean=1.0, stddev=COLOR_CONTRAST_SCALE,
                                           dtype=tf.dtypes.float32), 1. / MAX_COLOR_CONTRAST, MAX_COLOR_CONTRAST)
        color_shift = 255 * tf.clip_by_value(tf.random.normal(shape=(bs, 1), mean=0.0, stddev=COLOR_SHIFT_SCALE,
                                             dtype=tf.dtypes.float32), -MAX_COLOR_SHIFT, MAX_COLOR_SHIFT)
        gamma = tf.clip_by_value(tf.random.normal(shape=(bs, 1), mean=1.0, stddev=GAMMA_SCALE / 3,
                                 dtype=tf.dtypes.float32), 1 - GAMMA_SCALE, 1 + GAMMA_SCALE)

        # colors_dict = dict(multvec=[1, 1, 1], color_shift=0, contrast_factor=1, gamma=gamma,
        #                    hue_shift=0, saturation_factor=1, value_factor=1, value_shift=0)
        colors = dict(multvec=multvec, color_shift=color_shift, contrast_factor=contrast_factor,
                      hue_shift=hue_shift, saturation_factor=saturation_factor,
                      value_factor=value_factor, value_shift=value_shift, gamma=gamma)
        x = augment_images(x, angles=angles, shifts=shifts, colors=colors)

        # when Y is also an image
        if self.augment_y:
            y_shifts = shifts / self.y_scale_down_factor

            y = augment_images(y, angles=angles, shifts=y_shifts)

        if orig_rank == 3:
            x = tf.squeeze(x, axis=[0])

            if self.augment_y:
                y = tf.squeeze(y, axis=[0])

        modified_inputs = inputs.copy()
        modified_inputs['image'] = x

        return modified_inputs, y

