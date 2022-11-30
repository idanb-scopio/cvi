#!/usr/bin/env python3

import argparse
import logging
import os
import shutil
import sys
from datetime import datetime

import tensorflow as tf
from models.cellsfactory import model_instance_from_yaml
from core.dataset import TFRDataset


# larger batch size requires mode GPU memory
DEFAULT_BATCH_SIZE_PER_REPLICA = 128

DEFAULT_NUMBER_OF_EPOCHS = 150


def get_stats_str(stats, label_mapping):
    labels_stats = stats
    header_format = '{:<10s}{:<15s}{:<8s}\n'
    row_format = '{:<10d}{:<15s}{:<8d}\n'

    stats_str = header_format.format('class id', 'label name', 'count')
    stats_str += header_format.format('========', '==========', '=====')

    reverse_mapping = {v: k for k, v in label_mapping.items()}

    for class_num in range(len(label_mapping)):
        label_name = reverse_mapping[class_num]
        count = labels_stats[label_name]
        stats_str += row_format.format(class_num, label_name, count)

    return stats_str


def get_dataset(dataset_dir, cells_model, training=False):
    """
    Returns a tf.data.Dataset instance from a dataset dir containing TFRecord files.
    training: True/False if dataset is for training. The parsed input is returned in a manner suited for training
              (for instance, onehot vectors instead of class numbers)
    """
    dataset = TFRDataset(dataset_dir, cells_model=cells_model, sampled=True).get_dataset()
    if not dataset:
        logging.warning(f'empty dataset: {dataset_dir}')
        return None

    parser_fn = cells_model.get_parser_fn(training=training)
    dataset = dataset.map(parser_fn)

    return dataset


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(asctime)s.%(msecs)03d] -%(levelname).1s- : %(message)s',
                        datefmt='%d-%m-%Y %H:%M:%S')

    parser = argparse.ArgumentParser()
    parser.add_argument('build_dir', type=str, help='build dir containing data and configs')
    parser.add_argument('--output-dir', type=str, required=False, help='model output dir')
    parser.add_argument('--batch-size', type=int,
                        help=f'batch size (default: {DEFAULT_BATCH_SIZE_PER_REPLICA})')
    parser.add_argument('--no-multi-gpu', dest='multi_gpu', action='store_false')
    parser.add_argument('--epochs', type=int,
                        help=f'number of epochs to run (default: {DEFAULT_NUMBER_OF_EPOCHS})')

    parser.set_defaults(multi_gpu=True)
    args = vars(parser.parse_args())

    if args['batch_size']:
        batch_size_per_replica = args['batch_size']
    else:
        batch_size_per_replica = DEFAULT_BATCH_SIZE_PER_REPLICA

    build_dir = args['build_dir']
    if not os.path.exists(build_dir):
        sys.exit(f'build dir not found: {build_dir}')

    model_cfg_yaml = f'{build_dir}/model.yml'
    if not os.path.exists(model_cfg_yaml):
        sys.exit(f'model config not found: {model_cfg_yaml}')

    if args['output_dir']:
        output_dir = args['output_dir']
    else:
        output_dir = f'{build_dir}/trained_model'
        logging.info(f'model will be saved in: {output_dir}')

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if args['epochs']:
        epochs = args['epochs']
    else:
        epochs = DEFAULT_NUMBER_OF_EPOCHS

    # obtain model instance
    cells_model = model_instance_from_yaml(model_cfg_yaml)
    model_cfg = cells_model.get_model_config()

    if args['multi_gpu']:
        strategy = tf.distribute.MirroredStrategy()
        logging.info(f'multi gpu training: using {strategy.num_replicas_in_sync} devices.')
        with strategy.scope():
            keras_model = cells_model.get_keras_model(training=True)

        dist_dataset_options = tf.data.Options()
        dist_dataset_options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        n_devices = strategy.num_replicas_in_sync
    else:
        keras_model = cells_model.get_keras_model(training=True)
        dist_dataset_options = None
        n_devices = 1

    # batch size is affected by the number of available devices
    batch_size = batch_size_per_replica * n_devices

    parser_fn = cells_model.get_parser_fn(training=True)

    # load training dataset from TFRecord files
    train_ds_dir = f'{build_dir}/data-train'
    train_tfr = TFRDataset(train_ds_dir, cells_model=cells_model, sampled=True)
    if not train_tfr:
        logging.error(f'empty training dataset: {train_ds_dir}')
        sys.exit(1)
    train_dataset = train_tfr.get_dataset().map(parser_fn)
    steps_per_epoch = train_tfr.get_steps_per_epoch(batch_size=batch_size)

    # get and print training stats
    train_stats = train_tfr.get_dist()
    label_mapping = cells_model.get_label_mapping()
    train_stats_str = get_stats_str(train_stats, label_mapping)
    logging.info(f'\ntraining stats:\n{train_stats_str}\n')

    # load validation dataset (if exists)
    val_ds_dir = f'{build_dir}/data-val'
    val_tfr = TFRDataset(val_ds_dir, cells_model=cells_model, sampled=True)
    if val_tfr:
        # get and print training stats
        val_stats = val_tfr.get_dist()
        val_stats_str = get_stats_str(val_stats, label_mapping)
        logging.info(f'\nvalidation stats:\n{val_stats_str}\n')
    else:
        logging.warning(f'no validation set found in: {val_ds_dir}')

    if args['multi_gpu']:
        train_dataset = train_dataset.with_options(dist_dataset_options)

    model_res = float(model_cfg['model_res'])
    logging.info(f'model resolution: {model_res:.7f} mm/pixel')

    # get the preprocess crop function, which takes the larger image stored in the dataset,
    # and returns a center crop with the dimensions that match the NN input
    preprocess_crop_fn = cells_model.get_preprocess_crop_fn()

    # get the image augmentation class (callable) that can be mapped onto a dataset instance
    # this performs image augmentations (on the CPU)
    preprocess_augmentations = cells_model.get_preprocess_augmentations_class()

    # image augmentations are applied on the train dataset
    train_dataset = train_dataset.map(preprocess_augmentations, num_parallel_calls=tf.data.AUTOTUNE)

    # crop function returns the center crop of the (larger) image in the dataset.
    train_dataset = train_dataset.map(preprocess_crop_fn)

    # batch examples in the dataset
    train_dataset = train_dataset.batch(batch_size)

    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    val_dataset = None
    val_steps = None
    if val_tfr:
        val_dataset = val_tfr.get_dataset().map(parser_fn)
        val_steps = val_tfr.get_steps_per_epoch(batch_size=batch_size)
        val_dataset = val_dataset.map(preprocess_crop_fn)
        val_dataset = val_dataset.batch(batch_size)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    train_model_weights_suffix = cells_model.get_train_model_weights_suffix()
    output_model_file = f'{output_dir}/model{train_model_weights_suffix}.h5'
    with open(f'{output_dir}/model_summary.txt', 'w') as f:
        keras_model.summary(print_fn=lambda x: f.write(x + '\n'))

    # copy model yml config
    shutil.copy(model_cfg_yaml, f'{output_dir}/model{train_model_weights_suffix}.yml')

    callbacks = []

    # Model checkpoint callback
    checkpoint_cb = cells_model.get_checkpoint_callback(output_model_file, save_best_only=False)
    callbacks.append(checkpoint_cb)

    # Tensorboard file writer and callback
    log_dir = f'{output_dir}/logs/fit-{datetime.now().strftime("%d.%m.%Y-%H_%M_%S")}'
    # file_writer = tf.summary.create_file_writer(f'{log_dir}/data')
    # file_writer.set_as_default()
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks.append(tensorboard_cb)

    # Learn rate per epoch schedule callback
    learn_rate_schedule_fn = cells_model.get_learn_rate_schedule_fn()
    if learn_rate_schedule_fn:
        logging.info(f'using custom learning rate')
        learn_rate_cb = tf.keras.callbacks.LearningRateScheduler(schedule=learn_rate_schedule_fn, verbose=0)
        callbacks.append(learn_rate_cb)

    logging.info(f'training on {n_devices} devices, effective batch_size = {batch_size}')
    keras_model.fit(train_dataset,
                    validation_data=val_dataset,
                    callbacks=callbacks,
                    epochs=epochs,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=val_steps)

    print_summary = False
    if print_summary:
        keras_model.summary()
    cells_model.save_predict_model(output_dir, print_summary)

    logging.info('training done.')


if __name__ == '__main__':
    main()
