#!/usr/bin/env python3
#
# Basic predict utility.

import argparse
import glob
import json
import logging
import os
import sys

import tensorflow as tf
from lib.scanimsrc import ScanImageSource
from models.cellsfactory import model_instance_from_yaml
from lib.pyramiddata import apply_blacklist_res_swap
from core.labels import get_center_point


# input resolution mapping: for command line ease of use.
INPUT_RES_MAP = {
    'x100': 0.0002016,
    'alpha': 0.000133,
    'ht': 0.000164137506078669,
}


def process_args(args):
    if args['limit_to_gpu']:
        gpu_id = args['limit_to_gpu']
        logging.warning(f'GPU usage will be limited to GPU{gpu_id}')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    params = {}
    input_src = args['input']

    model_dir = args['model_dir']
    model_cfg_yaml = f'{model_dir}/model.yml'
    if not os.path.exists(model_cfg_yaml):
        sys.exit(f'invalid model dir input: file not found: {model_cfg_yaml}')

    # obtain model instance
    cells_model = model_instance_from_yaml(model_cfg_yaml)
    cells_model.set_model_dir(os.path.dirname(os.path.abspath(model_cfg_yaml)))

    params['cells_model'] = cells_model

    # the case of a single ROI fed into command line as a string of "x,y,width,height"
    if args['roi']:

        roi = args['roi'].split(',')
        if len(roi) != 4:
            sys.exit(f'error in roi format: {roi}')

        roi = [int(e) for e in roi]

        if not args['input_res']:
            sys.exit('input resolution is required when --roi is used')

        input_res = args['input_res'].lower()
        try:
            resolution = float(input_res)

            # if a floating point value is given, verify order of magnitude so that
            # input values other than mm/pixel (for instance um/pixel) can be caught
            min_res = min(INPUT_RES_MAP.values())
            if resolution > 10 * min_res or resolution < 0.1 * min_res:
                sys.exit(f'invalid input resolution order of magnitude. values should be in mm/pixel: {resolution}')

        except ValueError:
            if input_res not in INPUT_RES_MAP:
                sys.exit(f'unknown input resolution string shortcut: {input_res}')
            resolution = INPUT_RES_MAP[input_res]

        try:
            image_source = ScanImageSource(image_source=input_src, resolution=resolution)
        except Exception as e:
            sys.exit(f'invalid image input source: {str(e)}')

        scan_id = image_source.infer_scan_id_from_src()
        params['rois'] = {scan_id: {'image_source': image_source,
                                    'rois': [roi]}}

    # the case where a single .json or dir containing jsons are used as input
    elif args['rois_from_labels'] or args['rois_from_json']:
        params['rois'] = {}

        if args['rois_from_labels']:
            json_input = args['rois_from_labels']
        else:
            json_input = args['rois_from_json']

        if not os.path.exists(json_input):
            sys.exit(f'json input not found: {json_input}')

        if os.path.isdir(json_input):
            json_files = glob.glob(f'{json_input}/*.json')
            if not json_files:
                sys.exit(f'no json files found: {json_input}')
        else:
            json_files = [json_input]

        for jf in json_files:
            with open(jf, 'r') as f:
                scan_data = json.load(f)

            scan_id = scan_data['scan_id']
            pyr_res = scan_data['pyramid_resolution']
            pyr_res_wa = apply_blacklist_res_swap(pyr_res)
            if pyr_res != pyr_res_wa:
                logging.warning(f'blacklisted resolution for scan id: {scan_id}: {pyr_res} fixed to: {pyr_res_wa}')
                pyr_res = pyr_res_wa

            if os.path.isdir(json_input):
                image_src = ScanImageSource(image_source=f'{input_src}/{scan_id}', resolution=pyr_res)
            else:
                image_src = ScanImageSource(image_source=input_src, resolution=pyr_res)

            if args['rois_from_labels']:
                labels = scan_data['labels']
                roi_w, roi_h = cells_model.get_input_image_shape()[0:2]

                # check for input label dimensions and adjust bbox if enabled
                for idx, label in enumerate(labels):
                    lbl_w, lbl_h = label[2:4]

                    if lbl_w != roi_w or lbl_h != roi_h:
                        if args['adjust_labels']:
                            cx, cy = get_center_point(*label[0:4])
                            lx = cx - roi_w // 2
                            ly = cy - roi_h // 2
                            adjusted_label = [lx, ly, roi_w, roi_h, label[4]]
                            labels[idx] = adjusted_label
                        else:
                            sys.exit(f'input label bbox size not the same as model input size. use --adjust-labels.')

                rois = labels
            else:
                rois = scan_data['ROIs']
            params['rois'][scan_id] = {'image_source': image_src,
                                       'rois': rois}
    else:
        sys.exit('invalid args combination.')

    output_dir = args['output_dir']
    if not output_dir:
        output_dir = os.getcwd()

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not os.path.exists(output_dir):
        sys.exit(f'output dir does not exist: {output_dir}')
    params['output_dir'] = output_dir

    debug_save_to = args['debug_save']
    if debug_save_to:

        # when no value is passed, use 'output_dir/debug folder'
        if debug_save_to == '#':
            debug_save_to = f'{output_dir}/debug'

        if not os.path.exists(debug_save_to):
            os.mkdir(debug_save_to)

    params['debug_save_to'] = debug_save_to
    params['limit_gpu_mem'] = args['limit_gpu_mem']

    return params


def main(raw_args=None):
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] -%(levelname).1s- : %(message)s',
                        datefmt='%d-%m-%Y %H:%M:%S')
    logging.debug('core.predict: start.')

    parser = argparse.ArgumentParser()
    roi_group = parser.add_mutually_exclusive_group(required=True)
    roi_group.add_argument('--roi', type=str, help='region of interest inside the pyramid', required=False)
    roi_group.add_argument('--rois-from-labels', type=str, help='take list of ROIs from a CVI JSON file: labels field',
                           required=False)
    roi_group.add_argument('--rois-from-json', type=str, help='take list of ROIs from a CVI JSON file: ROIs field',
                           required=False)
    parser.add_argument('--model-dir', type=str, help='dir containing the model', required=True)
    parser.add_argument('--input', type=str, help='input source (image/pyramid dir)', required=True)
    parser.add_argument('--input-res', type=str, help=f'input resolution in mm/pixel or a string shortcut from: '
                        f'{",".join(INPUT_RES_MAP.keys())}')
    parser.add_argument('--debug-save', nargs='?', const='#', type=str, help='dir to save debug images to')
    parser.add_argument('--output-dir', type=str, help='output dir for saving detections JSON')
    parser.add_argument('--enable-cpu', action='store_true', help='enable the use of CPU inference (such as when no'
                                                                  ' gpu is available).')
    parser.add_argument('--limit-gpu-mem', type=int, help='gpu memory limit, in megabytes (default: no limit)')
    parser.add_argument('--limit-to-gpu', type=int, help='use only gpu id supplied by this argument')
    parser.add_argument('--tile-size', type=int, help='detection tile size, in pixels. example: 1024')
    parser.add_argument('--model-filename', type=str, help='custom model.h5 filename (for eval flow)')
    parser.add_argument('--adjust-labels', action='store_true', help='when rois come from labels - adjust to model'\
                        ' input size')
    parser.add_argument('--load-weights-only', action='store_true', help='needed in some classification models')
    args = vars(parser.parse_args(raw_args))

    # process args: verify input arguments and create instances
    p_args = process_args(args)

    if len(tf.config.list_physical_devices('GPU')) == 0:
        if not args['enable_cpu']:
            sys.exit('Aborting: no GPU devices were found. To run anyway use --enable-cpu')

    if args['limit_gpu_mem']:
        memory_limit = args['limit_gpu_mem']
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                                    [tf.config.experimental.VirtualDeviceConfiguration(
                                                                        memory_limit=memory_limit
                                                                    )])
            logging.info(f'memory limit of {memory_limit} MB is set on: {gpus[0]}')
        else:
            sys.exit('unable to set gpu memory limit: no gpus were found.')

    load_weights_only = args['load_weights_only']
    cells_model = p_args['cells_model']
    rois_dict = p_args['rois']
    params = {}

    # set bounding box size if exists in model config
    model_cfg = cells_model.get_model_config()
    if 'detection_bounding_box_size' in model_cfg:
        params['label_bbox_size'] = model_cfg['detection_bounding_box_size']

    if 'detection_label_str' in model_cfg:
        params['label_str'] = model_cfg['detection_label_str']
    else:
        params['label_str'] = "cell"

    if args['tile_size']:
        tile_size = args['tile_size']
        params['tile_size'] = tile_size
        logging.info(f'using detection tile size of: {tile_size}x{tile_size}')

    if args['model_filename']:
        model_filename = args['model_filename']
    else:
        model_filename = 'model.h5'

    # load keras model
    cells_model.load_trained_model(model_filename=model_filename, load_weights_only=load_weights_only)

    for scan_id in rois_dict:
        image_source = rois_dict[scan_id]['image_source']
        rois = rois_dict[scan_id]['rois']
        pyr_res = image_source.get_resolution()
        logging.info(f'running prediction on {scan_id} @ {pyr_res:.7f} mm/pixel')

        if p_args['debug_save_to']:
            debug_save_to_scan_id = f'{p_args["debug_save_to"]}/{scan_id}'
            if not os.path.exists(debug_save_to_scan_id):
                os.mkdir(debug_save_to_scan_id)

            params = {'debug_save_to': debug_save_to_scan_id}

        # get prediction function
        predict_fn = cells_model.get_predict_fn(params=params)

        # prediction function returns a result and collaterals
        results, collaterals = predict_fn(image_source=image_source, rois=rois)

        scan_id = results['scan_id']
        output_dir = p_args['output_dir']
        json_file = f'{output_dir}/{scan_id}.json'
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=4)
        logging.info(f'predictions saved in: {json_file}')


if __name__ == '__main__':
    main()
