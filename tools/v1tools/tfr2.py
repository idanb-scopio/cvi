#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import numpy as np
import cv2
import tensorflow as tf


def npimage_feature(value):
    """Returns a bytes_list from a string / byte."""
    np_ser = tf.io.serialize_tensor(value).numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[np_ser]))


def create_example(image_path, label_img_path):
    im = cv2.imread(image_path)
    im_label = cv2.imread(label_img_path)

    # take a single channel and downsize to 48x48
    im_label_mono = im_label[:, :, 0]
    w, h = im_label_mono.shape[0:2]
    assert w == 96 and h == 96
    w_half, h_half = w // 2, h // 2
    im_label_resized = cv2.resize(im_label_mono, dsize=(w_half, h_half))

    feature = {
        "image": npimage_feature(im),
        "label_image": npimage_feature(im_label_resized)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def gen_train_image_list(images_list_file):
    with open(images_list_file, 'r') as f:
        png_list = [line.strip() for line in f.readlines()]

    dataset_files_list = []
    for f_name in png_list:
        prefix = f_name[:-14]
        input_image = f'{prefix}_net_train.png'
        label_image = f'{prefix}_target_train.png'

        if not os.path.isfile(input_image) or not os.path.isfile(label_image):
            logging.error(f'missing file(s) on {prefix}')
            continue

        dataset_files_list.append((input_image, label_image))

    return dataset_files_list


def create_tfrecord(dataset_files_list, tfrec_file):
    size = len(dataset_files_list)

    logging.info(f'writing TFRecord file: {tfrec_file}')
    tfr_options = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter(tfrec_file) as writer:

        print('\n' + ' '*80 + '\r')
        for idx, example_pair in enumerate(dataset_files_list):
            X_image_file = example_pair[0]
            Y_label_file = example_pair[1]

            example = create_example(X_image_file, Y_label_file)
            writer.write(example.SerializeToString())
            print(f'[{idx+1}/{size}] writing to {tfrec_file}\r', end='', flush=True)


def tfrec_decode_fn(record_bytes):
    return tf.io.parse_single_example(
        # data
        record_bytes,

        # schema
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "label_image": tf.io.FixedLenFeature([], tf.string)
        })


if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG,
                        format='[%(asctime)s.%(msecs)03d] -%(levelname).1s- : %(message)s',
                        datefmt='%d-%m-%Y %H:%M:%S')

    #png_list_file = '/mnt/ssd/mickey/train_pngs.txt'
    png_list_file = '/srv/mickey/wbcdet1/train_pngs.txt'

    tfrec_dir = '/mnt/ssd/mickey/tfrecord3'
    if not os.path.isdir(tfrec_dir):
        os.mkdir(tfrec_dir)

    tfrec_file = f'{tfrec_dir}/dataset.tfrec'

    dataset_files_list = gen_train_image_list(png_list_file)

    # split
    N = len(dataset_files_list)
    n_train = int(0.7 * N)
    n_val = int(0.2 * N)
    n_test = N - n_train - n_val
    assert n_train + n_val + n_test == N

    create_tfrecord(dataset_files_list[:n_train],                f'{tfrec_dir}/train.tfrec')
    create_tfrecord(dataset_files_list[n_train:(n_train+n_val)], f'{tfrec_dir}/val.tfrec')
    create_tfrecord(dataset_files_list[n_train+n_val:],          f'{tfrec_dir}/test.tfrec')

    # print('\n' + ' '*80 + '\r')
    # counter = 0
    # for batch in tf.data.TFRecordDataset([tfrec_file]).map(tfrec_decode_fn):
    #     image_ser = batch["image"]
    #     image = tf.io.parse_tensor(image_ser, tf.uint8)
    #     counter += 1
    #     print(f'{counter} samples decoded... \r', end='', flush=True)

    # print(f'processed {counter} samples.')
