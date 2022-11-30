#!/usr/bin/env python3
import argparse
import atexit
import json
import logging
import os
import sys
import time
import yaml
import zmq

from core.mp.multipredictor import set_visible_devices, set_gpu_mem_limit
from models.cellsfactory import model_instance_from_yaml
from lib.scanimsrc import ScanImageSource


class CommandErrorException(Exception):
    pass


DEFAULT_GPU_DEVICE = 0
DEFAULT_MEM_LIMIT = 0


# input resolution mapping: for command line ease of use.
INPUT_RES_MAP = {
    'x100': 0.0002016,
    'alpha': 0.000133,
    'ht': 0.000164137506078669,
}


PBS_CONFIG_FILE = f'{os.path.dirname(os.path.abspath(__file__))}/config.yml'

PID_FILE_BASEDIR = '/tmp'
clean_up_list = []


def create_pid_file(pid_file_path):

    with open(pid_file_path, 'w') as f:
        f.write(str(os.getpid()))


def atexit_cleanups():
    for f in clean_up_list:
        if os.path.exists(f):
            try:
                os.unlink(f)
            except Exception:
                pass


def run_predict(predict_fn, input_source, input_resolution, rois):
    image_source = ScanImageSource(image_source=input_source, resolution=input_resolution)
    logging.info('running prediction function.')
    results, collaterals = predict_fn(image_source=image_source, rois=rois)
    logging.info('prediction function done.')
    return results


def process_request(request, predict_fn, state):
    if 'operation' not in request:
        raise CommandErrorException('missing "operation" field')

    operation = request['operation']

    if operation == 'quit':
        return {}

    elif operation == 'status':
        uptime = time.time() - state['start_time']
        pid = os.getpid()
        return {'uptime': uptime, 'pid': pid}

    elif operation == 'predict':
        try:
            input_source = request['input_source']
            input_resolution = request['input_resolution']
            rois = request['rois']
        except KeyError as e:
            raise CommandErrorException(f'missing field: {str(e)}')

        results = run_predict(predict_fn, input_source, input_resolution, rois)
        return results
    else:
        raise CommandErrorException(f'unknown operation: {operation}')


def run_server_loop(cells_model, bind_addr, predict_fn):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(bind_addr)

    model_name = cells_model.get_model_name()
    logging.info(f'starting prediction server for model {model_name} on address: {bind_addr}')

    state = {'start_time': time.time()}
    quit_requested = False
    while not quit_requested:

        message = socket.recv()
        request = json.loads(message)
        logging.info(f'received request: {str(request)[:128]}')
        try:
            results = process_request(request, predict_fn, state)
            if results:
                reply_message = json.dumps({'status': 'ok', 'results': results}).encode('ascii')
                logging.info(f'sending reply: {str(reply_message)[:128]}')
                socket.send(reply_message)
            else:   # quit request
                reply_message = json.dumps({'status': 'ok'}).encode('ascii')
                logging.info(f'sending reply: {str(reply_message)[:128]}')
                socket.send(reply_message)
                quit_requested = True

        except CommandErrorException as e:
            reply_message = json.dumps({'status': 'error', 'reason': str(e)}).encode('ascii')
            logging.error(f'operation error. reply: {str(reply_message)[:128]}')
            socket.send(reply_message)

    logging.info('quitting server')
    socket.close()


def main():
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] -%(levelname).1s- : %(message)s',
                        datefmt='%d-%m-%Y %H:%M:%S')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, help='dir containing models sub dirs', required=True)
    args = vars(parser.parse_args())

    if not os.path.exists(PBS_CONFIG_FILE):
        sys.exit(f'unable to find PBS config file: {PBS_CONFIG_FILE}')

    with open(PBS_CONFIG_FILE, 'r') as f:
        pbs_config = yaml.safe_load(f.read())

    model_dir = args['model_dir']
    if not os.path.exists(model_dir):
        sys.exit(f'model dir does not exist: {model_dir}')

    model_cfg_yaml = f'{model_dir}/model.yml'
    cells_model = model_instance_from_yaml(model_cfg_yaml)
    cells_model.set_model_dir(os.path.dirname(os.path.abspath(model_cfg_yaml)))

    model_name = cells_model.get_model_name()
    if model_name not in pbs_config['models']:
        sys.exit(f'unrecognized model name: {model_name}')

    pid_file_path = f'{PID_FILE_BASEDIR}/{model_name}.pid'
    create_pid_file(pid_file_path)
    clean_up_list.append(pid_file_path)
    atexit.register(atexit_cleanups)

    worker_params = pbs_config['worker_params'][model_name]
    gpu_device = worker_params['gpu_device']
    gpu_mem_limit = worker_params['gpu_mem_limit']

    # worker device settings
    set_visible_devices(gpu_device)

    # once a single device is selected, it is always referred to as device 0
    set_gpu_mem_limit(gpu_mem_limit, 0)
    logging.info(f'worker parameters: device: {gpu_device}, memory limit: {gpu_mem_limit}')

    model_dir = args['model_dir']
    if not os.path.exists(model_dir):
        sys.exit(f'model dir does not exist: {model_dir}')

    # load keras model
    cells_model.load_trained_model()

    # get predict function
    prediction_params = pbs_config['prediction_params'][model_name]
    logging.info(f'prediciton params: {prediction_params}')
    predict_fn = cells_model.get_predict_fn(params=prediction_params)

    bind_addr = worker_params['server_endpoint']

    # add socket file (if ipc) to clean up list
    if bind_addr[0:6] == 'ipc://':
        clean_up_list.append(bind_addr[6:])

    run_server_loop(cells_model, bind_addr, predict_fn)


if __name__ == '__main__':
    main()

