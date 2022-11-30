from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.keras.utils import losses_utils


class WeightedClassificationLoss(tf.keras.losses.Loss):
    """
      Sample loss weighted categorical cross entropy Keras Loss class
      Supports per target class or per (target_class, pred_class) weights
    """

    def __init__(self, **loss_config):

        self.config = copy.deepcopy(loss_config)
        assert len(self.config['class_weights']) == self.config['num_classes']
        assert len(self.config['class_weights_matrix']) == (self.config['num_classes'] ** 2)

        self.use_per_class_weights = self.config.get('use_per_class_weights', False)
        self.class_weights = tf.constant(self.config['class_weights'], dtype=tf.float32)
        self.class_weights_matrix = tf.constant(self.config['class_weights_matrix'], dtype=tf.float32)
        self.num_classes = tf.constant(self.config['num_classes'], dtype=tf.int32)
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False,
                                                               reduction=losses_utils.ReductionV2.NONE)
        super(WeightedClassificationLoss, self).__init__(name='weighted_categorical_cross_entropy')

    def get_config(self):

        return self.config

    def calc_per_class_weights(self, onehot_labels):

        """
        Sample weights using per target_class weights

        :param onehot_labels
        :return: weights
        """

        labels = tf.argmax(onehot_labels, axis=-1)
        weights = tf.gather(self.class_weights, labels, name='per_class_weights')

        return weights

    def calc_label_pred_class_weights(self, probabilities, onehot_labels):
        """
        Sample weights using per (target_class, pred_class) matrix weights

        :param logits
        :param onehot_labels
        :return: weights
        """

        tiled_probabilities = tf.tile(probabilities, [1, self.num_classes])
        tiled_probabilities = tf.reshape(tiled_probabilities, [-1, self.num_classes, self.num_classes],
                                         name='tiled_probabilities')
        tiled_labels = tf.tile(onehot_labels, [1, self.num_classes])
        tiled_labels = tf.reshape(tiled_labels, [-1, self.num_classes, self.num_classes])
        tiled_labels = tf.transpose(tiled_labels, [0, 2, 1], name='tiled_labels')
        prob_matrix = tf.multiply(tiled_probabilities, tiled_labels, name='prob_matrix')
        class_weight_matrix = tf.reshape(self.class_weights_matrix, [1, self.num_classes, self.num_classes],
                                         name='class_weight_matrix')
        weights_matrix = tf.multiply(prob_matrix, class_weight_matrix, name='weights_matrix')
        weights = tf.reduce_sum(weights_matrix, axis=[1, 2], name='weights')
        weights = tf.stop_gradient(weights, name='const_weights')
        return weights

    def call(self, y_true, y_pred):

        """
        Loss call override

        :param y_true: Labels
        :param y_pred: Predictions - softmax results
        :return: loss
        """
        probabilities = tf_ops.convert_to_tensor_v2(y_pred)
        onehot_labels = tf.cast(y_true, y_pred.dtype)

        if self.use_per_class_weights:
            weights = self.calc_per_class_weights(onehot_labels)
        else:
            weights = self.calc_label_pred_class_weights(probabilities, onehot_labels)

        cross_entropy_loss = self.loss_fn(y_true=onehot_labels, y_pred=probabilities, sample_weight=weights)
        cross_entropy_loss_mean = tf.reduce_mean(cross_entropy_loss, name='batch_cross_entropy')

        return cross_entropy_loss_mean
