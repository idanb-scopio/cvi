#!/usr/bin/env python3
# PBS app - client server
import atexit
import argparse
import json
import logging
import os
import psutil
import shlex
import subprocess
import sys
import time
import yaml
from lib.zmqrr import JsonRRSocket, TimeoutException


# input resolution mapping: for command line ease of use.
INPUT_RES_MAP = {
    'x100': 0.0002016,
    'alpha': 0.000133,
    'ht': 0.0001641375,
}


PBS_CONFIG_FILE = f'{os.path.dirname(os.path.abspath(__file__))}/config.yml'
WORKER_SERVER_EXEC = f'{os.path.dirname(os.path.abspath(__file__))}/predict_server.py'

PID_FILE_BASEDIR = '/tmp'

launched_workers = []


def kill_launched_workers():
    # destroy launched workers
    if len(launched_workers) > 0:
        for p in launched_workers:
            logging.warning(f'killing launched worker (pid: {p.pid})')
            p.kill()


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

        rois = json_data['ROIs']
        logging.info(f'found {len(rois)} rois')

        params['rois'] = rois

    # the case of a single ROI fed into command line as a string of "x,y,width,height"
    plt_roi = None
    if args['plt_roi']:
        roi = args['plt_roi'].split(',')
        if len(roi) != 4:
            sys.exit(f'error in roi format: {roi}')

        roi = [int(e) for e in roi]
        params['plt_rois'] = [roi]

    # load ROIs from a JSON file. used to test against ground truth / labeled data.
    params['plt_rois'] = []
    if args['plt_rois_from_json']:
        json_file = args['plt_rois_from_json']
        if not os.path.exists(json_file):
            sys.exit(f'file not found: {json_file}')

        with open(json_file, 'r') as f:
            json_data = json.load(f)

        rois = json_data['ROIs']
        logging.info(f'found {len(rois)} rois')

        params['plt_rois'] = rois

    if args['plt_scale_roi']:
        scaled_rois = []
        for roi in params['plt_rois']:
            rx, ry, rw, rh = roi[0:4]
            scaled_rois.append([rx, ry, rw // 4, rh // 4, roi[4:]])

        params['plt_rois'] = scaled_rois

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

    if not os.path.exists(PBS_CONFIG_FILE):
        sys.exit(f'unable to find PBS config file: {PBS_CONFIG_FILE}')

    with open(PBS_CONFIG_FILE, 'r') as f:
        pbs_config = yaml.safe_load(f.read())

    parser = argparse.ArgumentParser()
    parser.add_argument('--models-root', type=str, help='dir containing models sub dirs', required=True)

    # ROI - white blood cells
    roi_group = parser.add_mutually_exclusive_group(required=True)
    roi_group.add_argument('--roi', type=str, help='region of interest inside the pyramid', required=False)
    roi_group.add_argument('--rois-from-json', type=str, help='take list of ROIs from a CVI JSON file: ROIs field',
                           required=False)

    # ROI - platelets
    plt_roi_group = parser.add_mutually_exclusive_group(required=False)
    plt_roi_group.add_argument('--plt-roi', type=str, help='ROI for platelet detection', required=False)
    plt_roi_group.add_argument('--plt-rois-from-json', type=str, help='take list of ROIs for platelet detection from '
                               'a CVI JSON file: ROIs field', required=False)

    parser.add_argument('--input', type=str, help='input source (image/pyramid dir)', required=True)
    parser.add_argument('--input-res', type=str, help=f'input resolution in mm/pixel or a string shortcut from: '
                                                      f'{",".join(INPUT_RES_MAP.keys())}', required=True)
    parser.add_argument('--debug-save-to', type=str, help='dir to save debug images to')
    parser.add_argument('--output-dir', type=str, help='output dir for saving detections JSON')
    parser.add_argument('--plt-scale-roi', action='store_true', help='scale down plt roi dimensions by 4')
    args = vars(parser.parse_args())

    p_args = process_args(args)
    print(p_args)

    model_names = pbs_config['models']
    models_root = p_args['models_root']
    if not os.path.isdir(models_root):
        sys.exit(f'models root dir not found: {models_root}')

    # check if workers are already up. if not, launch the missing ones
    for model_name in model_names:
        pid_file = f'{PID_FILE_BASEDIR}/{model_name}.pid'

        if os.path.exists(pid_file):
            with open(pid_file, 'r') as f:
                worker_pid = int(f.read())

            if psutil.pid_exists(worker_pid):
                logging.info(f'worker for {model_name} is up (pid: {worker_pid})')
                continue
            else:
                # erase stale worker pid file
                os.unlink(pid_file)

        model_dir = f'{models_root}/{model_name}'
        if not os.path.isdir(model_dir):
            sys.exit(f'model dir not found: {model_dir}')

        cmd = f'{WORKER_SERVER_EXEC} --model-dir {model_dir}'
        cmd_argv = shlex.split(cmd)
        logging.warning(f'launching worker: {cmd}')
        p = subprocess.Popen(cmd_argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        launched_workers.append(p)

    # when exiting, kill any workers that were launch by this program
    atexit.register(kill_launched_workers)

    # initialize sockets
    jz_sockets = {}
    for model_name in model_names:
        server_endpoint = pbs_config['worker_params'][model_name]['server_endpoint']
        logging.info(f'connecting to endpoint: {server_endpoint}')
        server_zsock = JsonRRSocket()
        server_zsock.connect(server_endpoint)
        jz_sockets[model_name] = server_zsock

    # run WBC det
    operation = {"operation": "predict",
                 "input_source": p_args["input_source"],
                 "input_resolution": p_args["input_resolution"],
                 "rois": p_args["rois"]
                 }
    # send predict operation
    jz_sockets['wbc_detection'].send(operation)
    t_start = time.time()

    # run plt det
    if p_args["plt_rois"]:
        operation = {"operation": "predict",
                     "input_source": p_args["input_source"],
                     "input_resolution": p_args["input_resolution"],
                     "rois": p_args["plt_rois"]
                     }
        jz_sockets['plt_detection'].send(operation)
    else:
        logging.warning('no platelet rois were given, skipping platelet run')

    # collect wbc det results
    reply = jz_sockets['wbc_detection'].recv(timeout_ms=60000)
    t_end = time.time()

    if reply['status'] == 'ok':
        logging.info(f'wbc detection took: {t_end - t_start:.3f} seconds')
    else:
        logging.info(f'operation error: {reply}')
        sys.exit(1)

    detection_results = reply['results']
    to_classify_rois = detection_results['labels']
    # run WBC det
    operation = {"operation": "predict",
                 "input_source": p_args["input_source"],
                 "input_resolution": p_args["input_resolution"],
                 "rois": to_classify_rois
                 }

    # send predict operation to classifiers
    jz_sockets['classifiers'].send(operation)
    t_start = time.time()

    # collect plt detection results
    if p_args["plt_rois"]:
        reply = jz_sockets['plt_detection'].recv(timeout_ms=60000)

        if reply['status'] == 'ok':
            logging.info(f'wbc plt done.')
        else:
            logging.info(f'operation error: {reply}')
            sys.exit(1)
        plt_results = reply['results']
    else:
        plt_results = {'labels': []}

    reply = jz_sockets['classifiers'].recv(timeout_ms=60000)
    t_end = time.time()

    if reply['status'] == 'ok':
        logging.info(f'wbc classification took: {t_end - t_start:.3f} seconds')
    else:
        logging.info(f'operation error: {reply}')
        sys.exit(1)

    results = reply['results']
    logging.info(f'found {len(results["labels"])} labels and {len(plt_results["labels"])} plts.')
    results['labels'] += plt_results['labels']

    output_dir = p_args['output_dir']
    json_file = f'{output_dir}/{results["scan_id"]}.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=4)
    logging.info(f'predictions saved in: {json_file}')


if __name__ == '__main__':
    main()
