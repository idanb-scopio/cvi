from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from models.layers.features_center_loss_layer import FeaturesCenterLossLayer
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def test_features_center_loss():


    num_classes = 3
    iters = 800
    batch_size = 8

    config = {
        'num_classes': num_classes,
        'update_factor': 0.02,
        'center_loss_weights_list': [float(1.), float(1.2), float(0.8)]
    }

    layer_loss = FeaturesCenterLossLayer(**config)

    loc_1 = [0., 0.]
    cov = [[2., 0.], [0., 2.]]
    loc_2 = [0., 5.]
    loc_3 = [5., 0.]
    loc_list = [loc_1, loc_2, loc_3]
    iter_vals = []
    for idx, loc in enumerate(loc_list):
        iter_vals.append(np.random.multivariate_normal(mean=loc, cov=cov, size=iters))
    iter_labels = np.random.randint(low=0, high=3, size=iters, dtype='int32')
    iter_preds = np.array(iter_vals)

    loss_vals = []
    features_vals = []
    centers_vals = []
    for i in range(num_classes):
        features_vals.append([])
        centers_vals.append([])
    for batch_idx in range(int(iters / batch_size)):
        ind = np.arange(batch_idx * batch_size, (batch_idx + 1) * batch_size, dtype='int32')
        labels = iter_labels[ind]
        tf_labels = tf.constant(labels, dtype=tf.int32)
        one_hot_labels = tf.one_hot(indices=tf_labels, depth=num_classes)
        features = tf.constant(iter_preds[labels, ind, :], dtype=tf.float32)
        layer_inputs = {
            'features': features,
            'one_hot_labels': one_hot_labels
        }
        tf_loss_val = layer_loss(inputs=layer_inputs)

        batch_losses = tf_loss_val.numpy()
        loss = np.mean(batch_losses)
        loss_vals.append(loss)
        np_centers = layer_loss.features_centers.numpy()
        label_list = labels.tolist()
        np_features = features.numpy()
        for idx, label in enumerate(label_list):
            features_vals[label].append(np_features[idx, :])
        for idx in range(num_classes):
            centers_vals[idx].append(np_centers[idx, :])

        print('iter : %d' % batch_idx)


    fig1 = plt.figure(1, figsize=(7, 6))
    fig1.clf()
    ax1 = fig1.add_subplot(111)
    start_plt = 0
    cols = ['r', 'g', 'b']
    for idx in range(num_classes):
        features_arr = np.array(features_vals[idx], dtype='float32')
        centers_arr = np.array(centers_vals[idx], dtype='float32')
        col = cols[idx % 3]
        ax1.plot(features_arr[ start_plt:, 0], features_arr[ start_plt:, 1], col + '.', markersize=1)
        ax1.plot(centers_arr[ start_plt:, 0], centers_arr[ start_plt:, 1], col + '+', markersize=4)
        ax1.plot(centers_arr[ start_plt:, 0], centers_arr[ start_plt:, 1], col)
        plt.show(block=False)
        pass
    plt.show(block=True)


if __name__ == '__main__':

    test_features_center_loss()
