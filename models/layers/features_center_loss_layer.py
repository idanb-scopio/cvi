from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.ops import state_ops as tf_state_ops


class FeaturesCenterLossLayer(tf.keras.layers.Layer):

    """
    Center loss as Keras Layer class with loss results.
    Requires flat 1D features and one-hot labels as training inputs
    """

    def __init__(self, **config):

        """
        :param config: center loss configuration
        """

        self.config = copy.deepcopy(config)
        assert len(self.config['center_loss_weights_list']) == self.config['num_classes']

        self.num_classes = tf.constant(self.config['num_classes'], dtype=tf.int32)
        config_update_factor = self.config.get('update_factor', float(0.1))
        self.update_factor = tf.constant(config_update_factor, dtype=tf.float32)
        self.center_loss_weights_list = tf.constant(self.config['center_loss_weights_list'], dtype=tf.float32)
        self.loss_weight = tf.constant(self.config.get('loss_weight', 1.0), dtype=tf.float32)
        self.sqr_ofs = tf.constant(float(1.), dtype=tf.float32)

        super(FeaturesCenterLossLayer, self).__init__(name='features_center_loss')

    def get_config(self):

        return self.config

    def build(self, input_shape):

        """
        Build the layer non-trainable features centers variables, and set features shape related data

        :param input_shape: call inputs shape
        """

        self.num_features = tf.constant(input_shape['features'].as_list()[1], dtype=tf.int32)
        self.center_loss_weights = tf.tile(tf.expand_dims(self.center_loss_weights_list, axis=-1), [1, self.num_features])
        self.centers_shape = [self.num_classes, self.num_features]
        self.features_centers = self.add_weight(name='features_centers',
                                                shape=self.centers_shape,
                                                dtype=tf.float32,
                                                initializer='zeros',
                                                aggregation= tf.VariableAggregation.MEAN,
                                                trainable=False)

    def call(self, inputs):

        """
        :param inputs: deep layers features batch for which the center loss applies, and the batch
                       one hot labels targets
        :return: center loss
        """

        # Gather inputs
        in_features = tf_ops.convert_to_tensor_v2(inputs['features'])
        onehot_labels = tf.cast(inputs['one_hot_labels'], in_features.dtype)
        labels = tf.argmax(onehot_labels, axis=-1)

        # Input feature centers for the input labels and difference to input features value
        in_features_centers = tf.gather(self.features_centers, labels, name='in_features_centers')
        diff = in_features_centers - in_features

        # Update feature centers
        update = self.update_factor * diff
        indices = tf.expand_dims(labels, axis=-1)
        new_feature_centers = tf.tensor_scatter_nd_sub(tensor=self.features_centers, indices=indices, updates=update)
        assigned_val = tf_state_ops.assign(self.features_centers, new_feature_centers)

        # Calculate weighted center loss
        loss_vals = 2 * self.sqr_ofs * (tf.math.sqrt(float(1.) + tf.square(diff) / self.sqr_ofs) - float(1.))
        weights = tf.gather(self.center_loss_weights, labels, name='features_centers_weights')
        weighted_loss_vals = tf.multiply(self.loss_weight * weights, loss_vals, name='features_centers_weighted_loss_vals')
        center_loss = tf.reduce_mean(weighted_loss_vals, axis=-1, name='features_center_loss')

        return center_loss





