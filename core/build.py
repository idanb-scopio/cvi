#!/usr/bin/env python3
#
# Build: Dataset Build Process: transform JSON labels and image data to TFRecord files for training.
#

import argparse
import functools
import glob
import json
import logging
import multiprocessing
import os
import sys
import yaml

from lib.pyramiddata import apply_blacklist_res_swap

# reduce TF logger messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from core.labels import create_centers_map
from lib.scanimsrc import ScanImageSource, is_same_resolution
from models.cellsfactory import model_instance_from_yaml
from lib.lmanifest import LabelsManifest

# disable the use of GPU for build: multiprocess acceleration causes out of memory when using GPU.
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

# floating point error tolerance for model resolution
FP_RES_ERR = 1e-6

# input resolution mapping: for command line ease of use of model resolution override
INPUT_RES_MAP = {
    'x100': 0.0002016,
    'alpha': 0.000133,
}


def split(a, n):
    """
    split a list into n equal parts.
    :param a: list
    :param n: number of parts
    :returns list of lists
    >> list(split(range(11), 3))
       [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10]]
    """
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def split_manifests(manifests, n):
    """
    split the manifests in the manifests dict into n equal parts.
    returns n manifests dicts.
    """
    # holds a list of manifests dicts
    manifests_list = [dict() for i in range(n)]

    for cls_str in manifests:
        manifest = manifests[cls_str]

        # split the manifest (list of tuples) to n equal sizes
        # uneven split may cause empty manifests if there are fewer samples
        # than the size of n.
        spl_manifest = list(split(manifest, n))

        # assign the split manifests in the all-classes manifests dictionaries
        for i in range(n):
            manifests_list[i][cls_str] = spl_manifest[i]

    return manifests_list


def gen_scaled_centers_maps(json_files, cells_model):
    flat_label_mapping = cells_model.get_flat_label_mapping()
    level_scale_div = cells_model.get_level_scale_div()
    model_res = cells_model.get_model_res()

    # all scaled_center_map dicts, keyed by scan_id
    scaled_centers_maps = {}
    for jf in json_files:

        with open(jf, 'r') as f:
            scan_data = json.load(f)

        labels = scan_data['labels']
        scan_id = scan_data['scan_id']
        image_res = scan_data['pyramid_resolution']
        image_res = apply_blacklist_res_swap(image_res)

        if is_same_resolution(image_res, model_res):
            resize_factor = None
        else:
            resize_factor = image_res / model_res

        # generate the (x, y) -> class_num (1 or 0) dictionary for all labels
        scaled_centers_map = create_centers_map(labels, flat_label_mapping, level_scale_div, resize_factor)
        if scan_id in scaled_centers_maps:
            scaled_centers_maps[scan_id].update(scaled_centers_map)
        else:
            scaled_centers_maps[scan_id] = scaled_centers_map

    return scaled_centers_maps


def run_multiprocess(manifests, n_procs, output_dir, pyramid_basedir, cells_model, scan_id_res_map, proc_pool,
                     scaled_centers_maps):
    file_indices_list = [i for i in range(1, n_procs + 1)]

    manifests_list = split_manifests(manifests, n_procs)
    build_args = zip(file_indices_list, manifests_list)
    build_fn = functools.partial(build_dataset_tfrec_manifest,
                                 output_dir=output_dir,
                                 pyramid_basedir=pyramid_basedir,
                                 cells_model=cells_model,
                                 scan_id_res_map=scan_id_res_map,
                                 scaled_centers_maps=scaled_centers_maps)

    proc_pool.starmap(build_fn, build_args)


def build_dataset_tfrec_manifest(file_index,
                                 manifests,
                                 output_dir,
                                 pyramid_basedir,
                                 cells_model,
                                 scan_id_res_map,
                                 scaled_centers_maps):
    cached_image_sources = {}
    model_res = cells_model.get_model_res()

    # loop over class manifests dict
    for cls_str in manifests.keys():

        # skip empty manifests
        if not manifests[cls_str]:
            continue

        manifest = manifests[cls_str]

        # create tfrecord file
        cls_dir = f'{output_dir}/{cls_str}'
        if not os.path.exists(cls_dir):
            os.mkdir(cls_dir)

        tfrec_file = f'{cls_dir}/dataset-{file_index}.tfrec'
        logging.info(f'writing {len(manifest)} samples to {tfrec_file}')

        with tf.io.TFRecordWriter(tfrec_file) as tfr_w:

            # iterate over manifest entries
            for entry in manifest:
                scan_id, x, y, w, h, label_str = entry

                # obtain the image source object (pyramid + res info)
                if scan_id in cached_image_sources:
                    image_source = cached_image_sources[scan_id]
                else:
                    pyramid_dir = f'{pyramid_basedir}/{scan_id}'
                    pyramid_res = scan_id_res_map[scan_id]
                    image_source = ScanImageSource(image_source=pyramid_dir,
                                                   resolution=pyramid_res)
                    cached_image_sources[scan_id] = image_source

                # read example image from pyramid. result is (image, class_num) tuple
                if scaled_centers_maps:
                    scaled_centers_map = scaled_centers_maps[entry[0]]
                else:
                    scaled_centers_map = None
                example = cells_model.generate_single_example(image_source,
                                                              entry[1:6],
                                                              model_res,
                                                              scaled_centers_map=scaled_centers_map)

                # serialize the example into TF Record format
                id_str = ','.join(str(e) for e in entry[0:5])
                serialized_example = cells_model.serialize_example(example, id_str)

                # write example to TFRecord file
                tfr_w.write(serialized_example)


def build_dataset_manifest(json_files, pyramid_basedir, model_cfg_file, model_res, output_dir,
                           n_procs=1, **kwargs):
    """
    Build a TFRecord based dataset.
    :param json_files: list of JSON dataset files
    :param pyramid_basedir: dir containing pyramid scan subdirs (uuid named)
    :param model_cfg_file: model configuration .yaml file
    :param model_res: model resolution in mm/pixel. inputs which are in different res are resized to match.
    :param output_dir: build output dir. TFRecord files are placed in the proper subdirs under it.
    :param n_procs: number of parallel processes to run
    """

    train_dir = f'{output_dir}/data-train'
    val_dir = f'{output_dir}/data-val'

    # obtain model instance
    cells_model = model_instance_from_yaml(model_cfg_file)

    # set the model resolution
    cells_model.set_model_res(model_res)

    # calculate image size in the dataset, taking image augmentations into account
    cells_model.calc_image_shape_in_dataset()

    val_split = kwargs.get('val_split', 0.0)
    limit = kwargs.get('limit', None)

    # create labels manifest
    flat_mapping = cells_model.get_flat_label_mapping(str_labels=True)

    manifest_dir = f'{output_dir}/manifest'
    if not os.path.exists(manifest_dir):
        os.mkdir(manifest_dir)

    # initialize labels manifests. str -> "class str" label mapping is supplied.
    # label_mapping example: for smudge model (wbc/smudge/neg) "basophil" -> "wbc" mapping
    labels_manifest = LabelsManifest(label_mapping=flat_mapping)

    # create labels manifest from json files. also create the train/val split here.
    # labels that were not used are saved in the discarded dir.
    labels_manifest.create_from_files(json_files=json_files,
                                      val_split=val_split,
                                      discarded_dir=f'{manifest_dir}/discarded')

    # save manifest metadata
    labels_manifest.save_manifest_metadata(f'{output_dir}/manifest')

    # create centers maps if needed
    model_type = cells_model.get_model_type()
    if model_type == 'cells_detector':
        scaled_centers_maps = gen_scaled_centers_maps(json_files, cells_model)
    else:
        scaled_centers_maps = None

    # get (train, val) all classes manifest
    # label mapping is applied here according to the model's label_remap property
    train_manifests, val_manifests = labels_manifest.get_manifests(apply_mapping=True, limit=limit)

    # get scan_id to pyramid resolution map
    scan_id_res_map = labels_manifest.get_scan_id_res_map()

    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)

    if n_procs == 1:
        build_dataset_tfrec_manifest(manifests=train_manifests,
                                     output_dir=train_dir,
                                     pyramid_basedir=pyramid_basedir,
                                     cells_model=cells_model,
                                     scan_id_res_map=scan_id_res_map,
                                     scaled_centers_maps=scaled_centers_maps,
                                     file_index=1)

        build_dataset_tfrec_manifest(manifests=val_manifests,
                                     output_dir=val_dir,
                                     pyramid_basedir=pyramid_basedir,
                                     cells_model=cells_model,
                                     scan_id_res_map=scan_id_res_map,
                                     scaled_centers_maps=scaled_centers_maps,
                                     file_index=1)
    else:

        with multiprocessing.Pool(n_procs) as proc_pool:

            # build training - multiprocess
            run_multiprocess(manifests=train_manifests,
                             n_procs=n_procs,
                             output_dir=train_dir,
                             pyramid_basedir=pyramid_basedir,
                             cells_model=cells_model,
                             scan_id_res_map=scan_id_res_map,
                             proc_pool=proc_pool,
                             scaled_centers_maps=scaled_centers_maps)

            # build validation - multiprocess
            run_multiprocess(manifests=val_manifests,
                             n_procs=n_procs,
                             output_dir=val_dir,
                             pyramid_basedir=pyramid_basedir,
                             cells_model=cells_model,
                             scan_id_res_map=scan_id_res_map,
                             proc_pool=proc_pool,
                             scaled_centers_maps=scaled_centers_maps)

    # save model configuration. it includes the model_res parameter.
    output_model_yaml = f'{output_dir}/model.yml'
    logging.info(f'saving model config: {output_model_yaml}')
    with open(output_model_yaml, 'w') as f:
        f.write(cells_model.get_model_config_yaml())

    with open(f'{output_dir}/manifest/dist.yml', 'r') as f:
        dist = yaml.safe_load(f.read())

    with open(f'{train_dir}/dist.yml', 'w') as f:
        yaml.safe_dump(dist['train'], f)

    with open(f'{val_dir}/dist.yml', 'w') as f:
        yaml.safe_dump(dist['val'], f)

    logging.info('build done.')


def main():
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] -%(levelname).1s- : %(message)s',
                        datefmt='%d-%m-%Y %H:%M:%S')

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, help='dir containing JSON dataset files', required=True)
    parser.add_argument('--pyramid-basedir', type=str, help='top level dir for pyramid data.', required=True)
    parser.add_argument('--model-cfg', type=str, help='model config file', required=True)
    parser.add_argument('--output-dir', type=str, required=True,
                        help="output dir for build stage's data and collaterals")
    parser.add_argument('--workers', type=int, help='number of parallel processes (workers)', default=1)
    parser.add_argument('--model-res', type=str, help=f'override auto model resolution detection: '
                                                      f'{",".join(INPUT_RES_MAP.keys())}', required=True)
    parser.add_argument('--val-split', type=float, help=f'fraction of the samples to select for validation. '
                                                        'example: 0.2 for 20%% validation set. '
                                                        'default: no validation set (0.0).')
    parser.add_argument('--limit', type=int, help='limit the amount of samples written per tfrecord file (for debug)')
    args = vars(parser.parse_args())

    dataset_dir = args['dataset_dir']
    if not os.path.exists(dataset_dir):
        sys.exit(f'dataset dir not found: {dataset_dir}')

    pyramid_basedir = args['pyramid_basedir']
    if not os.path.exists(pyramid_basedir):
        sys.exit(f'pyramid basedir not found: {pyramid_basedir}')

    model_cfg_yaml = args['model_cfg']
    if not os.path.isfile(model_cfg_yaml):
        sys.exit(f'model config file not found: {model_cfg_yaml}')

    output_dir = args['output_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    json_files = glob.glob(f'{dataset_dir}/*.json')
    if len(json_files) == 0:
        sys.exit(f'unable to list json files in: {dataset_dir}')

    num_procs = args['workers']
    logging.info(f'using {num_procs} parallel processes')

    input_res = args['model_res'].lower()
    try:
        model_res = float(input_res)

        # if a floating point value is given, verify order of magnitude so that
        # input values other than mm/pixel (for instance um/pixel) can be caught
        min_res = min(INPUT_RES_MAP.values())
        if model_res > 10 * min_res or model_res < 0.1 * min_res:
            sys.exit(f'invalid input resolution order of magnitude. values should be in mm/pixel: {model_res}')

    except ValueError:
        if input_res not in INPUT_RES_MAP:
            sys.exit(f'unknown input resolution string shortcut: {input_res}')
        model_res = INPUT_RES_MAP[input_res]

    logging.info(f'using model resolution: {model_res:.7f} mm/pixel')
    if args['val_split']:
        val_split = args['val_split']
        logging.info(f'using validation set split: {val_split*100:.1f}%')
    else:
        val_split = 0.0

    if args['limit']:
        limit = args['limit']
    else:
        limit = None

    build_dataset_manifest(json_files=json_files,
                           pyramid_basedir=pyramid_basedir,
                           model_cfg_file=model_cfg_yaml,
                           model_res=model_res,
                           output_dir=output_dir,
                           n_procs=num_procs,
                           val_split=val_split,
                           limit=limit)


if __name__ == '__main__':
    main()
