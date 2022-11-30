#!/usr/bin/env python3
# simple test client for predsrv

import argparse
import json
import logging
import os
import sys
import time
import yaml
import datetime

from lib.zmqrr import JsonRRSocket, TimeoutException

# input resolution mapping: for command line ease of use.
INPUT_RES_MAP = {
    'x100': 0.0002016,
    'alpha': 0.000133,
    'ht': 0.000164137506078669,
}


PBS_CONFIG_FILE = f'{os.path.dirname(os.path.abspath(__file__))}/config.yml'


def process_args(args, model_name):
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
            if model_name == 'classifiers':
                rois = json_data['labels']
            else:
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

    if not os.path.exists(PBS_CONFIG_FILE):
        sys.exit(f'unable to find PBS config file: {PBS_CONFIG_FILE}')

    with open(PBS_CONFIG_FILE, 'r') as f:
        pbs_config = yaml.safe_load(f.read())

    model_names = pbs_config['models']

    parser = argparse.ArgumentParser()
    roi_group = parser.add_mutually_exclusive_group(required=True)
    roi_group.add_argument('--roi', type=str, help='region of interest inside the pyramid', required=False)
    roi_group.add_argument('--rois-from-json', type=str, help='take list of ROIs from a CVI JSON file: ROIs field',
                           required=False)
    parser.add_argument('--input', type=str, help='input source (image/pyramid dir)', required=True)
    parser.add_argument('--input-res', type=str, help=f'input resolution in mm/pixel or a string shortcut from: '
                                                      f'{",".join(INPUT_RES_MAP.keys())}', required=True)
    parser.add_argument('--debug-save-to', type=str, help='dir to save debug images to')
    parser.add_argument('--output-dir', type=str, help='output dir for saving detections JSON')
    parser.add_argument('--model', type=str, help=f'model server api. select from: {",".join(model_names)}',
                        required=True)
    args = vars(parser.parse_args())

    model_name = args['model'].lower()
    if model_name not in model_names:
        sys.exit(f'invalid pbs model: {model_name}. select from: {model_names}')

    p_args = process_args(args, model_name)
    print(p_args)

    server_endpoint = pbs_config['worker_params'][model_name]['server_endpoint']
    logging.info(f'connecting to endpoint: {server_endpoint}')

    server_zsock = JsonRRSocket()
    server_zsock.connect(server_endpoint)

    # check if server is up
    server_zsock.send({"operation": "status"})
    try:
        reply = server_zsock.recv(timeout_ms=3000)
    except TimeoutException:
        logging.error(f'prediction server timeout on endpoint: {server_endpoint}')
        sys.exit(1)

    status = reply['results']
    uptime_str = datetime.timedelta(seconds=int(status['uptime']))
    pid = status['pid']
    logging.info(f'{model_name} prediction server is up. pid: {pid}, uptime: {uptime_str}')

    operation = {"operation": "predict",
                 "input_source": p_args["input_source"],
                 "input_resolution": p_args["input_resolution"],
                 "rois": p_args["rois"]
                 }

    # send predict operation
    server_zsock.send(operation)
    t_start = time.time()
    reply = server_zsock.recv(timeout_ms=60000)
    t_end = time.time()
    if reply['status'] == 'ok':
        logging.info(f'prediction took: {t_end - t_start:.3f} seconds')
    else:
        logging.info(f'operation error: {reply}')
        sys.exit(1)

    results = reply['results']
    output_dir = p_args['output_dir']
    json_file = f'{output_dir}/{results["scan_id"]}.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=4)
    logging.info(f'predictions saved in: {json_file}')


if __name__ == '__main__':
    main()
