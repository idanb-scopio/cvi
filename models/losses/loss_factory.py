from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.losses.weighted_classification_loss import WeightedClassificationLoss
import tensorflow as tf


def classification_losses_generator(loss_config):
    """
     Generator function for Keras Loss classes
    """

    loss_type = 'categorical_cross_entropy' if (loss_config is None) else loss_config['loss_type']
    if loss_type == 'categorical_cross_entropy':
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
    elif loss_type == 'weighted_categorical_cross_entropy':
        loss_fn = WeightedClassificationLoss(**loss_config['weighted_categorical_cross_entropy'])
    else:
        raise ValueError('loss type `{}` is not supported.'.format(loss_type))

    return loss_fn
