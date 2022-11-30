#!/usr/bin/env python3
#
# Dataset Preview tool: generate preview images of a given dataset

import argparse
import logging
import os
import sys

import numpy as np
import tensorflow as tf
from core.dataset import TFRDataset
from models.cellsfactory import model_instance_from_yaml
from lib.mosaicsaver import MosaicSaver


# GPU should not be allocated for the preview tool
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

# number of mosaic images to create by default
DEFAULT_MOSAIC_IMAGES = 1


def gen_preview(output_dir, dataset, cells_model, show_types, limit, tag='train'):
    # input image size as is in the tfrecord dataset
    uncropped_shape = cells_model.get_image_shape_in_dataset()

    # cropped input image size, as required by the NN input
    cropped_shape = cells_model.get_input_image_shape()

    sub_image_shape = None
    show_datasets = []

    if 'dataset' in show_types:
        show_datasets.append(dataset)

        # dataset shape of input image is larger than cropped/augmented version
        sub_image_shape = uncropped_shape

    preprocess_crop_fn = cells_model.get_preprocess_crop_fn()
    if 'cropped' in show_types:
        # get the center crop using the preprocess crop function
        dataset_cropped = dataset.map(preprocess_crop_fn)
        show_datasets.append(dataset_cropped)

    if 'augmented' in show_types:
        preprocess_augmentations = cells_model.get_preprocess_augmentations_class()

        # map data augmentation class (callable) onto the dataset. image transformations are
        # done on the CPU, so num_parallel_calls is used to reduce runtime
        dataset_aug = dataset.map(preprocess_augmentations, num_parallel_calls=tf.data.AUTOTUNE)

        # after image augmentation get the center crop using the preprocess crop function
        dataset_aug = dataset_aug.map(preprocess_crop_fn)

        show_datasets.append(dataset_aug)

    if not sub_image_shape:
        # if not assigned, this means it's either cropped or augmented. both have the
        # same size: cropped
        sub_image_shape = cropped_shape

    # sub input images shape width and height
    sub_image_h, sub_image_w = sub_image_shape[0], sub_image_shape[1]

    assert len(show_datasets) > 0
    joined_dataset = tf.data.Dataset.zip(tuple(show_datasets))

    n = len(show_datasets)

    # single type of preview shape. preview shape may include label image (detection) or not (classification)
    single_preview_shape = cells_model.get_preview_image_shape(sub_image_shape)

    # extend preview img_shape to account of all requested show types
    # extension happens on the width (2nd) dimension
    multi_preview_shape = (single_preview_shape[0], n * single_preview_shape[1], single_preview_shape[2])

    mosaic_saver = MosaicSaver(sub_image_shape=multi_preview_shape,
                               mosaic_w=20000, mosaic_h=20000,
                               output_dir=output_dir,
                               tag=tag,
                               save_meta=True)

    pics_per_mosaic = mosaic_saver.get_max_sub_images()
    n_samples = pics_per_mosaic * limit

    logging.info(f'saving {pics_per_mosaic} images in each mosaic image')

    for zipped_examples in joined_dataset.take(n_samples):

        # for each zipped example, create a preview image to be filled by sub images
        preview_img = np.empty(shape=multi_preview_shape, dtype='uint8')

        id_str = ''
        for idx, example in enumerate(zipped_examples):
            sub_img = cells_model.get_example_preview_image(example, sub_image_shape=(sub_image_h, sub_image_w))
            x_start = idx * single_preview_shape[1]
            x_end = x_start + single_preview_shape[1]
            preview_img[:, x_start:x_end, :] = sub_img
            id_str = example[0]["id"].numpy().decode('utf8')

        mosaic_saver.add_image(preview_img, meta=id_str)

    mosaic_saver.save()


def main():
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] -%(levelname).1s- : %(message)s',
                        datefmt='%d-%m-%Y %H:%M:%S')

    parser = argparse.ArgumentParser()
    parser.add_argument('build_dir', type=str, help='build dir containing data and configs')
    parser.add_argument('--save-to', type=str, help='save compare figures to dir')
    parser.add_argument('--limit', type=int, help='limit number of image mosaics. '
                        f'set to 0 for the full dataset. default: {DEFAULT_MOSAIC_IMAGES}')

    parser.add_argument('--show-dataset', action='store_true', help='display unmodified dataset images (TFRecord)')
    parser.add_argument('--show-cropped', action='store_true', help='display cropped images (NN input size)')
    parser.add_argument('--show-augmented', action='store_true', help='display augmented images (training NN input)')
    parser.add_argument('--show-all', action='store_true', help='show all available modes')
    args = vars(parser.parse_args())

    build_dir = args['build_dir']

    print(f'loading dataset from build dir: {build_dir}')
    model_cfg_yaml = f'{build_dir}/model.yml'
    if not os.path.exists(model_cfg_yaml):
        sys.exit(f'model config not found: {model_cfg_yaml}')

    # obtain model instance
    cells_model = model_instance_from_yaml(model_cfg_yaml)

    output_dir = args['save_to']
    if not output_dir:
        output_dir = f'{build_dir}/preview'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    show_types = []
    if args['show_all']:
        show_types = ['dataset', 'cropped', 'augmented']
    else:
        if args['show_dataset']:
            show_types.append('dataset')
        if args['show_cropped']:
            show_types.append('cropped')
        if args['show_augmented']:
            show_types.append('augmented')

    # default
    if len(show_types) == 0:
        show_types = ['dataset']

    parser_fn = cells_model.get_parser_fn(return_one_hot=False)
    train_ds_dir = f'{build_dir}/data-train'
    val_ds_dir = f'{build_dir}/data-val'
    train_dataset = TFRDataset(train_ds_dir, cells_model=cells_model, sampled=True).get_dataset()
    train_dataset = train_dataset.map(parser_fn)

    val_dataset = TFRDataset(val_ds_dir, cells_model=cells_model, sampled=True).get_dataset()
    if val_dataset:
        val_dataset = val_dataset.map(parser_fn)

    if args['limit']:
        limit = args['limit']
    else:
        limit = DEFAULT_MOSAIC_IMAGES

    gen_preview(output_dir, train_dataset, cells_model, show_types, limit, tag='train')

    if val_dataset:
        gen_preview(output_dir, val_dataset, cells_model, show_types, limit, tag='val')


if __name__ == '__main__':
    main()
