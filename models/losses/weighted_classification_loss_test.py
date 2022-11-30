from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from models.losses.weighted_classification_loss import WeightedClassificationLoss
import tensorflow as tf

def test_weighted_classification_loss():
    num_classes = 3
    batch_size = 2
    config = {
        'num_classes': num_classes,
        'use_per_class_weights': False,
        'class_weights': [1., 1.3, 1.6],
        'class_weights_matrix': [1., 2.5, 3.5,
                                 2., 1., 2.5,
                                 3., 2., 1.]
    }

    loss_fn = WeightedClassificationLoss(**config)

    scores = np.random.rand(batch_size, num_classes) * 10
    factor = 1. / (np.sum(scores, axis=-1, keepdims=True) + 1e-9)
    probabilities = (scores * factor).astype('float32')
    labels = np.random.randint(low=0, high=3, size=batch_size, dtype='int32')

    tf_labels = tf.constant(labels, dtype=tf.int32)
    tf_y_true = tf.one_hot(indices=tf_labels, depth=num_classes)
    tf_y_pred = tf.constant(probabilities, dtype=tf.float32)
    tf_loss_val = loss_fn(y_true=tf_y_true, y_pred=tf_y_pred)
    loss_val = tf_loss_val.numpy()
    print('Loss : {}'.format(loss_val))


if __name__ == '__main__':
    test_weighted_classification_loss()
