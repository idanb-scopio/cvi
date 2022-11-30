import logging
import multiprocessing
import os
import random
import string
import sys
import threading
import time
import zmq
import tensorflow as tf

from models.cellsfactory import model_instance_from_yaml
from lib.scanimsrc import ScanImageSource


DEFAULT_WORKER_PARAMS = {
    'gpu_device': 0,        # default GPU device is GPU0. set to -1 for CPU based inference.
    'gpu_mem_limit': 0      # default GPU mem limit. 0 for none, meaning allow TF to allocate all GPU mem.
}


def random_string(length):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def set_visible_devices(device_num):
    """
    determine which GPU predict() will be running on.
    :param device_num: gpu number (0..MAX_GPUS-1), or -1 for CPU based inference.
    """
    if device_num < 0:
        logging.warning('Using CPU based inference.')
    else:
        logging.warning(f'GPU usage is limited to GPU{device_num}')

    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_num)


def set_gpu_mem_limit(memory_limit, device_num):
    if memory_limit == 0:
        return

    if device_num < 0:
        logging.warning('memory limit cannot be set for CPU based inference.')
        return

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_virtual_device_configuration(gpus[device_num],
                                                                [tf.config.experimental.VirtualDeviceConfiguration(
                                                                    memory_limit=memory_limit
                                                                )])
        logging.info(f'memory limit of {memory_limit} MB is set on: {gpus[0]}')
    else:
        raise RuntimeError('no GPUs were found')


def run_predict(job_request, predict_fn):
    input_src = job_request['input_source']     # pyramid dzi, or image file
    resolution = job_request['input_resolution']    # in mm_per_pixel
    rois = job_request['rois']  # list of [x_topleft, y_topleft, width, height]

    image_source = ScanImageSource(image_source=input_src, resolution=resolution)
    logging.info('running prediction function.')
    results, collaterals = predict_fn(image_source=image_source, rois=rois)
    logging.info('prediction function done.')
    return results


def merge_results(results, partial_res):
    if not results:
        results.update(partial_res)
        return

    if 'labels' in results and 'labels' in partial_res:
        results['labels'] += partial_res['labels']
        logging.info(f'merge_results: added {len(partial_res["labels"])} labels')
    if 'ROIs' in results and 'ROIs' in partial_res:
        results['ROIs'] += partial_res['ROIs']
        logging.info(f'merge_results: added {len(partial_res["ROIs"])} ROIs')


class MultiPredictor:

    # communication between processes is done using ZMQ IPC sockets
    SOCKETS_DIR = '/tmp/cvi_sockets'

    # receive timeout, in milliseconds
    RECV_TIMEOUT = 60000

    # heartbeat timeout, in seconds
    HB_TIMEOUT = 5000

    def __init__(self, model_dir, prediction_params=None, worker_params=None):
        self.context = zmq.Context()
        self.socket_job_sender = self.context.socket(zmq.PUSH)
        self.socket_res_collector = self.context.socket(zmq.PULL)

        # do not linger on a sender socket when the process goes down
        self.socket_job_sender.setsockopt(zmq.LINGER, 0)

        # set limit to results receive socket, to avoid hangs
        self.socket_res_collector.setsockopt(zmq.RCVTIMEO, MultiPredictor.RECV_TIMEOUT)

        # setup endpoint sockets
        if not os.path.exists(MultiPredictor.SOCKETS_DIR):
            os.mkdir(MultiPredictor.SOCKETS_DIR)
        random_id = random_string(8)

        self.job_sender_endpoint = f'ipc://{MultiPredictor.SOCKETS_DIR}/job_sender_socket-{random_id}'
        self.res_collector_endpoint = f'ipc://{MultiPredictor.SOCKETS_DIR}/res_collector_socket-{random_id}'
        self.heartbeat_endpoint = f'ipc://{MultiPredictor.SOCKETS_DIR}/heartbeat_socket-{random_id}'

        # bind sockets for the forked server to connect to
        self.socket_job_sender.bind(self.job_sender_endpoint)
        self.socket_res_collector.bind(self.res_collector_endpoint)

        if not prediction_params:
            prediction_params = {}
        if not worker_params:
            worker_params = {}

        self.worker_params = DEFAULT_WORKER_PARAMS.copy()
        self.worker_params.update(worker_params)
        self.prediction_params = prediction_params

        # initialize a model object. no model is loaded or tf function actually called
        # so at this point no memory/device allocation happens.
        model_cfg_yaml = f'{model_dir}/model.yml'
        self.model = model_instance_from_yaml(model_cfg_yaml)
        self.model.set_model_dir(os.path.dirname(os.path.abspath(model_cfg_yaml)))

        # server process handler
        self.server_process = None
        self.hb_monitor_thread = None
        self.hb_quit = False

    def get_log_prefix(self):
        pid = os.getpid()
        model_name = self.model.get_model_name()
        return f'{pid} {model_name}'

    def start(self):
        self.hb_monitor_thread = threading.Thread(target=self.heartbeat_main_loop, daemon=True)
        self.hb_monitor_thread.start()

        self.server_process = multiprocessing.Process(target=self.server_loop, daemon=True)
        self.server_process.start()

    def heartbeat_main_loop(self):
        log_pfx = self.get_log_prefix()
        logging.debug(f'[{log_pfx}] heartbeat_main_loop: thread started.')
        # heartbeat monitoring, main side
        socket_hb_req = self.context.socket(zmq.REQ)
        socket_hb_req.setsockopt(zmq.LINGER, 0)
        socket_hb_req.setsockopt(zmq.RCVTIMEO, MultiPredictor.HB_TIMEOUT)
        socket_hb_req.setsockopt(zmq.SNDTIMEO, 250)
        socket_hb_req.bind(self.heartbeat_endpoint)

        # wait for the forked process to start
        while self.server_process is None or not self.server_process.is_alive():
            logging.warning(f'[{log_pfx}] heartbeat_main_loop: waiting for server process to be up')
            time.sleep(1)

        logging.info(f'[{log_pfx}] heartbeat_main_loop: server process is alive')
        seq = 0

        while True:
            # logging.debug(f'heartbeat_main_loop: sending seq: {seq}')
            try:
                socket_hb_req.send_pyobj(seq)
            except zmq.error.ZMQError as e:
                logging.error(f'[{log_pfx}] heartbeat monitor: waiting for worker to connect (zmq: {str(e)})')
                time.sleep(0.5)
                continue

            try:
                recv_seq = socket_hb_req.recv_pyobj()
                if not recv_seq == seq:
                    logging.error(f'[{log_pfx}] heartbeat monitor: sequence mismatch: main expects {seq}, '
                                  f'server sent {recv_seq}')
            except zmq.error.ZMQError as e:
                logging.error(f'[{log_pfx}] heartbeat monitor: missed heartbeat (seq {seq}) [{str(e)}]')
            seq += 1
            time.sleep(1)

    def heartbeat_server_loop(self):
        log_pfx = self.get_log_prefix()
        logging.debug(f'[{log_pfx}] heartbeat_server_loop: thread started')

        # the server loop runs in its own process and therefore needs a separate zmq context
        socket_hb_rep = self.context.socket(zmq.REP)
        socket_hb_rep.setsockopt(zmq.LINGER, 0)
        socket_hb_rep.setsockopt(zmq.RCVTIMEO, MultiPredictor.HB_TIMEOUT)
        socket_hb_rep.connect(self.heartbeat_endpoint)

        while True:
            try:
                seq = socket_hb_rep.recv_pyobj()
                logging.debug(f'[{log_pfx}] server heartbeat monitor: received sequence: {seq}')
                socket_hb_rep.send_pyobj(seq)
            except zmq.error.ZMQError as e:
                logging.info(f'[{log_pfx}] heartbeat_server_loop: zmq error: {str(e)}')
                logging.fatal(f'[{log_pfx}] server heartbeat monitor: lost heartbeat from main program. exiting.')
                self.hb_quit = True
                sys.exit(1)

    def server_loop(self):
        log_pfx = self.get_log_prefix()

        # the forked process needs its own zmq context
        self.context = zmq.Context()

        # initialize server side sockets for receiving job requests and sending
        # back results
        job_socket = self.context.socket(zmq.PULL)
        res_socket = self.context.socket(zmq.PUSH)

        res_socket.setsockopt(zmq.LINGER, 0)
        job_socket.setsockopt(zmq.RCVTIMEO, 1000)

        # connect to endpoints
        job_socket.connect(self.job_sender_endpoint)
        res_socket.connect(self.res_collector_endpoint)

        hb_monitor_thread = threading.Thread(target=self.heartbeat_server_loop, daemon=True)
        logging.info(f'[{log_pfx}] server_loop: starting heartbeat thread')
        hb_monitor_thread.start()

        # initialize TF and load models
        logging.info(f'[{log_pfx}] server_loop: initializing')
        self.initialize()
        logging.info(f'[{log_pfx}] server_loop: initialization done')

        # get prediction function. it is of the format:
        # predict_fn(image_source, rois)
        predict_fn = self.model.get_predict_fn(params=self.prediction_params)

        results = {}
        # loop
        while not self.hb_quit:
            # job request is a json cvi. for detection, the 'labels' part should not
            # appear (it is ignored). For classification, 'labels' part is read and updated.

            # wait for incoming requests
            try:
                job_request = job_socket.recv_pyobj()
            except zmq.error.ZMQError as e:
                if e.errno == zmq.EAGAIN:
                    logging.debug(f'[{log_pfx}] server_loop: waiting for job (timeout interval)')
                    continue

            logging.debug(f'[{log_pfx}] predictor server received: {job_request}')

            if 'operation' not in job_request:
                logging.error(f'[{log_pfx}] missing "operation" field: {job_request}')

            operation = job_request['operation']
            if operation == 'get_results':
                # send and clear results
                logging.debug(f'[{log_pfx}] sending results: {results}')
                res_socket.send_pyobj(results)
                results = {}

            if operation == 'predict':
                partial_res = run_predict(job_request, predict_fn)

                # note: merge_results modifies the first dictionary (results)
                # with data from partial_res
                merge_results(results, partial_res)

    def initialize(self):
        log_pfx = self.get_log_prefix()

        # hardware related settings
        # select device on which to run the inference process: GPU id or -1 for CPU
        device_num = self.worker_params['gpu_device']
        set_visible_devices(device_num)

        # set gpu memory limit. if limit is 0 or CPU is selected, this is a no-op
        gpu_mem_limit = self.worker_params['gpu_mem_limit']
        set_gpu_mem_limit(gpu_mem_limit, device_num)

        if device_num < 0:
            device_num_str = 'cpu'
        else:
            device_num_str = f'gpu:{device_num}'

        if gpu_mem_limit > 0:
            mem_limit_str = f'{gpu_mem_limit} MB'
        else:
            mem_limit_str = 'no limit'

        logging.info(f'[{log_pfx}] initializing predictor on {device_num_str}, with memory limit: {mem_limit_str}')

        # load model(s) on GPU/CPU. here TensorFlow should initialize for the first time.
        self.model.load_trained_model(log_prefix=f'[{log_pfx}] ')

    def predict(self, input_rois, input_source, input_resolution):
        if self.server_process is None:
            raise RuntimeError('server process is not up')

        if not self.server_process.is_alive():
            raise RuntimeError('server process is down')

        request = {'operation': 'predict',
                   'rois': input_rois,
                   'input_source': input_source,
                   'input_resolution': input_resolution}

        logging.info(f'main: sending {str(request)[:120]}')
        self.socket_job_sender.send_pyobj(request)

    def get_results(self):
        if self.server_process is None:
            raise RuntimeError('server process is not up')

        if not self.server_process.is_alive():
            raise RuntimeError('server process is down')

        request = {'operation': 'get_results'}
        logging.debug(f'main: sending {request}')
        self.socket_job_sender.send_pyobj(request)
        reply = self.socket_res_collector.recv_pyobj()
        logging.debug(f'main: received: {reply}')
        return reply
