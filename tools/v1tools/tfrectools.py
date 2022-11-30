#!/usr/bin/env python3

import argparse
import glob
import logging
import os
import sys
import tensorflow as tf


WBC_FEATURE_DESCRIPTION = {
    "database_id": tf.io.FixedLenFeature([], tf.int64),
    "hardness_level": tf.io.FixedLenFeature([], tf.float32),
    "image_raw": tf.io.FixedLenFeature([], tf.string),
    "lab_name": tf.io.FixedLenFeature([], tf.string),
    "sample": tf.io.FixedLenFeature([], tf.string),
    "samples": tf.io.FixedLenFeature([], tf.string),
    "scan_id": tf.io.FixedLenFeature([], tf.int64),
    "session": tf.io.FixedLenFeature([], tf.int64),
    "slide_id": tf.io.FixedLenFeature([], tf.int64),
    "species": tf.io.FixedLenFeature([], tf.string),
    "top_left_x": tf.io.FixedLenFeature([], tf.int64),
    "top_left_y": tf.io.FixedLenFeature([], tf.int64),
    "train_rate_class": tf.io.FixedLenFeature([], tf.string)
}


def _parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, WBC_FEATURE_DESCRIPTION)


logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] -%(levelname).1s- : %(message)s',
                    datefmt='%d-%m-%Y %H:%M:%S')

parser = argparse.ArgumentParser()
parser.add_argument('tfrec_path', type=str, help='json dataset file')
args = vars(parser.parse_args())

tfrec_path = args['tfrec_path']

if os.path.isfile(tfrec_path):
    tfrec_files = [tfrec_path]
elif os.path.isdir(tfrec_path):
    tfrec_files = glob.glob(f'{tfrec_path}/**/*.tfrecords', recursive=True)
else:
    sys.exit(f'invalid path for tfrecord files/dir: {tfrec_path}')

if len(tfrec_files) == 0:
    sys.exit(f'empty tfrecord files list for path: {tfrec_path}')

logging.info(f'parsing {len(tfrec_files)} tfrecord files.')

raw_dataset = tf.data.TFRecordDataset(tfrec_files)
parsed_dataset = raw_dataset.map(_parse_function)

stats = {}
for parsed_record in parsed_dataset:
    labeled_class = parsed_record["train_rate_class"].numpy()
    if labeled_class in stats:
        stats[labeled_class] += 1
    else:
        stats[labeled_class] = 1

print(stats)
