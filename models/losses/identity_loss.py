from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops


class IdentityLoss(tf.keras.losses.Loss):
    """
      Helper loss for usage with loss layer, expects a loss as prediction and returns it
    """

    def __init__(self, **config):
        """
        :param config: Not used
        """

        self.config = {}
        super(IdentityLoss, self).__init__(name='identity_loss')

    def get_config(self):

        return self.config

    def call(self, y_true, y_pred):

        """
        Loss call override

        :param y_true: Required by the interface but not used
        :param y_pred: A loss from loss layer (>=0)
        :return: The input prediction as the loss
        """

        loss_pred = tf_ops.convert_to_tensor_v2(y_pred)
        loss = tf.reduce_mean(loss_pred)

        return loss
