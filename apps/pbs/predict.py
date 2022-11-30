#!/usr/bin/env python3
#
# Peripheral Blood Smear (PBS) analysis tool: run prediction for cells detection and classify the
# results. Input is given as an image source (Pyramid / JPEG), output is a CVI JSON.

import argparse
import json
import logging
import os
import sys
import yaml

from models.cellsfactory import model_instance_from_yaml
from lib.scanimsrc import ScanImageSource


# input resolution mapping: for command line ease of use.
INPUT_RES_MAP = {
    'x100': 0.0002016,
    'alpha': 0.000133,
    'ht': 0.000164137506078669,
}


PBS_CONFIG_FILE = f'{os.path.dirname(os.path.abspath(__file__))}/config.yml'


def process_args(args):
    params = {}

    roi = None

    # the case of a single ROI fed into command line as a string of "x,y,width,height"
    if args['roi']:
        roi = args['roi'].split(',')
        if len(roi) != 4:
            sys.exit(f'error in roi format: {roi}')

        roi = [int(e) for e in roi]
        params['rois'] = [roi]

    # load ROIs from a JSON file. used to test against ground truth / labeled data.
    if args['rois_from_json']:
        json_file = args['rois_from_json']
        if not os.path.exists(json_file):
            sys.exit(f'file not found: {json_file}')

        with open(json_file, 'r') as f:
            json_data = json.load(f)

        try:
            rois = json_data['ROIs']
            logging.info(f'found {len(rois)} rois')
        except KeyError as e:
            sys.exit(f'error: {str(e)}')

        params['rois'] = rois

    output_dir = args['output_dir']
    if not output_dir:
        output_dir = os.getcwd()

    if not os.path.exists(output_dir):
        sys.exit(f'output dir does not exist: {output_dir}')
    params['output_dir'] = output_dir

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

    params['input_res'] = resolution

    # input pyramid / image source
    input_src = args['input']
    try:
        image_source = ScanImageSource(image_source=input_src, resolution=resolution)
    except Exception as e:
        sys.exit(f'invalid image input source: {str(e)}')

    params['image_source'] = image_source
    params['models_root'] = args['models_root']

    debug_save_to = args['debug_save_to']
    if debug_save_to:
        if not os.path.exists(debug_save_to):
            os.mkdir(debug_save_to)
    params['debug_save_to'] = debug_save_to

    return params


def main(raw_args=None):
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] -%(levelname).1s- : %(message)s',
                        datefmt='%d-%m-%Y %H:%M:%S')

    parser = argparse.ArgumentParser()
    roi_group = parser.add_mutually_exclusive_group(required=True)
    roi_group.add_argument('--roi', type=str, help='region of interest inside the pyramid', required=False)
    roi_group.add_argument('--rois-from-json', type=str, help='take list of ROIs from a CVI JSON file: ROIs field',
                           required=False)
    parser.add_argument('--models-root', type=str, help='dir containing models sub dirs', required=True)
    parser.add_argument('--input', type=str, help='input source (image/pyramid dir)', required=True)
    parser.add_argument('--input-res', type=str, help=f'input resolution in mm/pixel or a string shortcut from: '
                                                      f'{",".join(INPUT_RES_MAP.keys())}', required=True)
    parser.add_argument('--debug-save-to', type=str, help='dir to save debug images to')
    parser.add_argument('--output-dir', type=str, help='output dir for saving detections JSON')
    parser.add_argument('--use-multi-gpu', action='store_true', help='use multiple GPUs when available.')
    args = vars(parser.parse_args(raw_args))

    if not os.path.exists(PBS_CONFIG_FILE):
        sys.exit(f'unable to find PBS config file: {PBS_CONFIG_FILE}')

    with open(PBS_CONFIG_FILE, 'r') as f:
        pbs_config = yaml.safe_load(f.read())

    # process args: verify input arguments and create instances
    p_args = process_args(args)

    image_source = p_args['image_source']
    rois = p_args['rois']
    models_root = p_args['models_root']
    model_names = pbs_config['models']
    models = {}
    for model_name in model_names:

        model_dir = f'{models_root}/{model_name}'
        if not os.path.exists(model_dir):
            sys.exit(f'error: required model dir not found: {model_dir}')

        model_cfg_yaml = f'{model_dir}/model.yml'
        cells_model = model_instance_from_yaml(model_cfg_yaml)
        cells_model.set_model_dir(os.path.dirname(os.path.abspath(model_cfg_yaml)))

        # load keras model
        cells_model.load_trained_model()

        models[model_name] = cells_model

    # --- DETECTION --- #
    debug_save_to = p_args['debug_save_to']
    detection_debug_save_to = None
    if debug_save_to:
        detection_debug_save_to = f'{debug_save_to}/detection'
        if not os.path.exists(detection_debug_save_to):
            os.mkdir(detection_debug_save_to)

        logging.info(f'debug data for detection is saved to: {detection_debug_save_to}')

    # configure additional parameters for WBC predict function
    wbc_det_params = pbs_config['prediction_params']['wbc_det']
    wbc_det_params['debug_save_to'] = detection_debug_save_to

    logging.info(f'wbc_det params: {wbc_det_params}')

    # get prediction function
    predict_fn = models['wbc_det'].get_predict_fn(params=wbc_det_params)
    results, collaterals = predict_fn(image_source=image_source, rois=rois)

    if debug_save_to:
        output_json = f'{debug_save_to}/detection/{results["scan_id"]}.json'
        with open(output_json, 'w') as f:
            json.dump(results, f)
        logging.info(f'detection results saved to: {output_json}')

    output_dir = p_args['output_dir']
    json_file = f'{output_dir}/{results["scan_id"]}.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=4)
    logging.info(f'predictions saved in: {json_file}')

    # --- CLASSIFIERS --- #

    wbc_rois = results['labels']
    logging.info(f'detected {len(wbc_rois)} rois. running through 7-stage classifiers.')

    wbc_classifiers = models['classifiers']
    cls_params = pbs_config['prediction_params']['classifiers']
    logging.info(f'cascade classifiers params: {cls_params}')
    cls_predict_fn = wbc_classifiers.get_predict_fn(params=cls_params)

    # run prediction
    results, collaterals = cls_predict_fn(image_source=image_source, rois=wbc_rois)
    logging.info('classification done.')


if __name__ == '__main__':
    main()

