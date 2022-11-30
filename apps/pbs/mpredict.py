#!/usr/bin/env python3
import argparse
import json
import logging
import os
import sys
import time
import yaml
from core.mp.multipredictor import MultiPredictor


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

    params['input_resolution'] = resolution
    params['input_source'] = args['input']
    params['models_root'] = args['models_root']

    debug_save_to = args['debug_save_to']
    if debug_save_to:
        if not os.path.exists(debug_save_to):
            os.mkdir(debug_save_to)
    params['debug_save_to'] = debug_save_to

    return params


def main():
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] -%(levelname).1s- : %(message)s',
                        datefmt='%d-%m-%Y %H:%M:%S')

    parser = argparse.ArgumentParser()
    roi_group = parser.add_mutually_exclusive_group(required=True)
    roi_group.add_argument('--roi', type=str, help='region of interest inside the pyramid', required=False)
    roi_group.add_argument('--rois-from-json', type=str, help='take list of ROIs from a CVI JSON file: ROIs field',
                           required=False)
    parser.add_argument('--models-root', type=str, help='models root dir, with sub dirs as specified in the config',
                        required=True)
    parser.add_argument('--input', type=str, help='input source (image/pyramid dir)', required=True)
    parser.add_argument('--input-res', type=str, help=f'input resolution in mm/pixel or a string shortcut from: '
                                                      f'{",".join(INPUT_RES_MAP.keys())}', required=True)
    parser.add_argument('--debug-save-to', type=str, help='dir to save debug images to')
    parser.add_argument('--output-dir', type=str, help='output dir for saving detections JSON')
    args = vars(parser.parse_args())

    p_args = process_args(args)
    models_root = p_args['models_root']
    input_rois = p_args['rois']
    input_source = p_args['input_source']
    input_resolution = p_args['input_resolution']

    if not os.path.exists(PBS_CONFIG_FILE):
        sys.exit(f'unable to find PBS config file: {PBS_CONFIG_FILE}')

    with open(PBS_CONFIG_FILE, 'r') as f:
        pbs_config = yaml.safe_load(f.read())

    model_names = pbs_config['models']
    predictors = {}
    for model_name in model_names:

        model_dir = f'{models_root}/{model_name}'
        if not os.path.exists(model_dir):
            sys.exit(f'error: required model dir not found: {model_dir}')

        worker_params = pbs_config['worker_params'][model_name]
        prediction_params = pbs_config['prediction_params'][model_name]

        # create a MultiPredictor instance. models are not yet loaded.
        predictors[model_name] = MultiPredictor(model_dir=model_dir,
                                                worker_params=worker_params,
                                                prediction_params=prediction_params)

    wbc_detector = predictors['wbc_det']
    wbc_classifiers = predictors['classifiers']
    plt_detector = predictors['plt_det']

    wbc_detector.start()
    wbc_classifiers.start()

    # run wbc detection
    wbc_detector.predict(input_rois=input_rois, input_source=input_source, input_resolution=input_resolution)

    # wait for detection results
    detections = wbc_detector.get_results()

    output_json = f'wbc_det-{detections["scan_id"]}.json'
    with open(output_json, 'w') as f:
        json.dump(detections, f, indent=4)
    logging.info(f'detection results saved to: {output_json}')

    # run classifiers
    detection_rois = detections['labels']
    wbc_classifiers.predict(input_rois=detection_rois, input_source=input_source, input_resolution=input_resolution)

    # wait for results
    cls_results = wbc_classifiers.get_results()

    output_json = f'cascade-{cls_results["scan_id"]}.json'
    with open(output_json, 'w') as f:
        json.dump(cls_results, f, indent=4)
    logging.info(f'detection results saved to: {output_json}')


if __name__ == '__main__':
    main()
