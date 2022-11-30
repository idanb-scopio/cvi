"""
 Title         : Core main utilities.
 Project       : Training

 File          : core_utils.py
 Author        : Shahar Karny
 Created       : 02/10/2017
-----------------------------------------------------------------------------
 Description :   Support auxiliary functions for the core.
-----------------------------------------------------------------------------
 Copyright (c) This model is the confidential and
 proprietary property of ScopioLabs Ltd and the possession or use of this
 file requires a written license from ScopioLabs Ltd.
------------------------------------------------------------------------------
 Modification history :

"""

import collections
import six

import tensorflow as tf

from tensorflow.python.platform import tf_logging as logging
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training import device_setter

import subprocess
import os
import json
import datetime
import copy

import core_path

import logging
logger = logging.getLogger(name='cv_tools')

def get_core_info():
    info_dict = {'Core Version': 'v4 with Band support and Binary training'}
    return info_dict


# def cv_log():
#     # Adding important messages to cv_log


class PredictionPing(session_run_hook.SessionRunHook):
    def __init__(self, number_of_test_steps, progress_manager):
        self.number_of_test_steps = number_of_test_steps
        self.current_prediction_step = 1

        self.progress_manager = progress_manager
        self.progress_manager.create_reporter(self.number_of_test_steps)

    # def begin(self):
    #     logging.info('Begining run...................')

    def after_create_session(self, session, coord):
        logging.info('\033[94mTensorFlow session is created, and graph is finalized.\033[0m')

    def before_run(self, run_context):  # pylint: disable=unused-argument
        logging.info('\033[94mRunning prediction step %d from %d\033[0m' % (
            self.current_prediction_step, self.number_of_test_steps))
        print('\033[94mRunning prediction step %d from %d\033[0m' % (
            self.current_prediction_step, self.number_of_test_steps))
        self.current_prediction_step += 1
        self.progress_manager.report_progress()

    # def after_run(self, run_context, run_values):  # pylint: disable=unused-argument
    #     logging.info('\033[94mAfter run %d from %d\033[0m' %(self.current_prediction_step, self.number_of_test_steps))


class ExamplesPerSecondHook(session_run_hook.SessionRunHook):
    """Hook to print out examples per second.

      Total time is tracked and then divided by the total number of steps
      to get the average step time and then batch_size is used to determine
      the running average of examples per second. The examples per second for the
      most recent interval is also logged.
    """

    def __init__(
            self,
            batch_size,
            every_n_steps=100,
            every_n_secs=None,
            number_of_total_steps=None,
            number_of_steps_per_epoch=None,
            number_of_epochs=None,
            progress_manager=None):
        """Initializer for ExamplesPerSecondHook.

          Args:
          batch_size: Total batch size used to calculate examples/second from
          global time.
          every_n_steps: Log stats every n steps.
          every_n_secs: Log stats every n seconds.
        """
        self.every_n_steps = every_n_steps
        if (every_n_steps is None) == (every_n_secs is None):
            raise ValueError('exactly one of every_n_steps'
                             ' and every_n_secs should be provided.')
        self._timer = basic_session_run_hooks.SecondOrStepTimer(
            every_steps=every_n_steps, every_secs=every_n_secs)

        self._step_train_time = 0
        self._total_steps = 0
        self._batch_size = batch_size

        self.number_of_total_steps = number_of_total_steps
        self.number_of_steps_per_epoch = number_of_steps_per_epoch
        self.number_of_epochs = number_of_epochs
        self.current_epoch = 0

    def after_create_session(self, session, coord):
        logging.info('\033[94mTensorFlow training session is created, and graph is finalized.\033[0m')

    def begin(self):
        self._global_step_tensor = training_util.get_global_step()
        if self._global_step_tensor is None:
            raise RuntimeError(
                'Global step should be created to use StepCounterHook.')

    def before_run(self, run_context):  # pylint: disable=unused-argument
        # logging.info('\033[94mRunning %d training steps\033[0m' % (self.every_n_steps))
        logging.info('\033[94mRunning 1 training step\033[0m')
        print('\033[94mRunning 1 training step\033[0m')
        return basic_session_run_hooks.SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        _ = run_context

        global_step = run_values.results
        if self._timer.should_trigger_for_step(global_step):
            elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(
                global_step)
            if elapsed_time is not None:
                steps_per_sec = elapsed_steps / elapsed_time
                self._step_train_time += elapsed_time
                self._total_steps += elapsed_steps

                average_examples_per_sec = self._batch_size * (
                        self._total_steps / self._step_train_time)
                current_examples_per_sec = steps_per_sec * self._batch_size
                # Average examples/sec followed by current examples/sec
                logging.info('%s: %g (%g), global step = %g', 'Average examples/sec',
                             average_examples_per_sec, current_examples_per_sec,
                             global_step)
                print('%s: %g (%g), global step = %g'%('Average examples/sec', average_examples_per_sec, current_examples_per_sec, global_step))

                # logging.info('%s %g', 'Session progress [%]:', 100. * (float(global_step) / self.number_of_total_steps))
                # logging.info('%s %g', 'Estimated time remaining [Min]:', ((self.number_of_total_steps - global_step) / steps_per_sec)/60)
                logging.info('Session progress [%%]: %7.4f , (%d / %d)' % (
                100. * (float(global_step) / self.number_of_total_steps), global_step, self.number_of_total_steps))
                print('Session progress [%%]: %7.4f , (%d / %d)' % (
                    100. * (float(global_step) / self.number_of_total_steps), global_step, self.number_of_total_steps))

                now = datetime.datetime.now()
                end_time_estimate = now + datetime.timedelta(0, (
                        (self.number_of_total_steps - global_step) / steps_per_sec))
                logging.info('%s %s', 'Expected end time:', end_time_estimate)
                print('%s %s'%('Expected end time:', end_time_estimate))


def local_device_setter(num_devices=1,
                        ps_device_type='cpu',
                        worker_device='/cpu:0',
                        ps_ops=None,
                        ps_strategy=None):
    if ps_ops == None:
        ps_ops = ['Variable', 'VariableV2', 'VarHandleOp']

    if ps_strategy is None:
        ps_strategy = device_setter._RoundRobinStrategy(num_devices)
    if not six.callable(ps_strategy):
        raise TypeError("ps_strategy must be callable")

    def _local_device_chooser(op):
        current_device = pydev.DeviceSpec.from_string(op.device or "")

        node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
        if node_def.op in ps_ops:
            ps_device_spec = pydev.DeviceSpec.from_string(
                '/{}:{}'.format(ps_device_type, ps_strategy(op)))

            ps_device_spec.merge_from(current_device)
            return ps_device_spec.to_string()
        else:
            worker_device_spec = pydev.DeviceSpec.from_string(worker_device or "")
            worker_device_spec.merge_from(current_device)
            return worker_device_spec.to_string()

    return _local_device_chooser


def gpu_selection(max_gpu_to_use):
    # Help function to select the GPU we are working with
    # Try to work with 'max_gpu_to_use', if possible.
    # Returns - number of GPU available for work and lowest free memory

    if max_gpu_to_use > 2:
        raise ValueError('Talk to shahar about using more than 2 GPUs')

    # Performing a check on number of available GPU for training
    gpu_query_cmd = r"nvidia-smi --query-gpu=count,memory.free --format=csv"

    GPU_id = subprocess.check_output(gpu_query_cmd, shell=True)
    # Start [arsing the NVIDIA-SMI result
    GPU_id = GPU_id.split('\n')

    try:
        free_memory_gpu0 = int(GPU_id[1].split(',')[1].split(' ')[1])
        gpu0_valid = True
    except Exception:
        free_memory_gpu0 = 0
        gpu0_valid = False
    try:
        free_memory_gpu1 = int(GPU_id[2].split(',')[1].split(' ')[1])
        gpu1_valid = True
    except Exception:
        free_memory_gpu1 = 0
        gpu1_valid = False

    # Decide which GPUs to work with out of the available GPUs:

    if not gpu0_valid and not gpu1_valid:
        raise ValueError('Did not detected any GPU available for training')

    if gpu0_valid and gpu1_valid:
        # use number of GPU as the user defined in the flags
        if max_gpu_to_use == 1:
            if free_memory_gpu1 > free_memory_gpu0:
                selected_GPU_id = '1'
                lowest_free_memory = free_memory_gpu1
            else:
                selected_GPU_id = '0'
                lowest_free_memory = free_memory_gpu0

            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = selected_GPU_id
        else:
            # Using both GPUs
            lowest_free_memory = min(free_memory_gpu0, free_memory_gpu1)

        return max_gpu_to_use, lowest_free_memory

    else:
        # Only 1 GPU is available - return this result with the free memory that is not 0
        lowest_free_memory = max(free_memory_gpu0, free_memory_gpu1)
        return 1, lowest_free_memory


def validate_tenvis_config_params_api(config_params=None, print_config = True):
    # Config params is a holds run related settings.
    #
    # This utility is used to validate and assign defaults to config param
    # passed by the training manager and the user at core-api
    if config_params is None:
        config_params = {}
        config_params['species'] = 'unknown'
        config_params['training_type'] = 'single_task_training'
        config_params['staining'] = 'unknown'
        config_params['reference_model_dir_list'] = None
        config_params['generate_training_data_from_local_db'] = True
        config_params['sessions_and_scans_dict_filename_path'] = None

        # Optional specific model configuration
        config_params['run'] = None
        config_params['data'] = None
        config_params['detection'] = None
        config_params['segmentation'] = None
        config_params['classification'] = None

        config_params['available_gpus'] = None
        config_params['lowest_gpu_memory'] = None

        config_params['report_features'] = False

        config_params['write_img_file'] = False
        config_params['write_det_img_file'] = True
        config_params['prev_ver'] = True

        config_params['labels_participating_in_training'] = []

        # If True, loads the PKL file of raw dataset information
        config_params['load_database_dataset'] = False

        # If True, loads the train and test dataset from former split.
        config_params['load_train_eval_datasets'] = False

        # If True, loads the training and test dataset from a ready tiles_dataset
        config_params['load_train_eval_datasets_from_tile_datasets'] = False
        config_params['pyramid_path'] = None
        config_params['scan_uuid_pyramids_path'] = None
        config_params['scan_id_pyramids_path'] = None
        config_params['res_dir'] = None
        config_params['labels_dir'] = None

        config_params['use_sparse_inference'] = False

        config_params['fine_tune_train'] = False

        config_params['test_data_id'] = None

        config_params['small_regions_split'] = False

        # V4 iterative task training option:
        config_params['iterative_training_mode'] = False

        config_params['train_iteration'] = 0

        # While training, disable/enable the evaluation.
        config_params['enable_evaluation'] = False

        config_params['train_with_hardness_rate_class'] = False

        config_params['use_ref_scan_split_info'] = False

        config_params['segmentation_mode'] = False

        config_params['semantic_segmentation_mode'] = False

        config_params['det_th'] = None

        config_params['net_ver'] = 1

        config_params['prev_ver'] = True

        config_params['x_100_model_res'] = False

        config_params['tile_center_inference'] = False

        config_params['score_compatibility'] = False

        config_params['ndc_seg_debug_post_process'] = False

        config_params['particles_dist_img_mask'] = None

        config_params['use_cloud_imageio'] = False

        config_params['use_cloud_imageio_runhttp'] = True

        config_params['predict_pyramid_to_model_resolution_factor'] = 1.

        config_params['model_resolution'] = 0.133071583818

        config_params['low_res_det'] = False

        config_params['new_data_tags'] = []

        config_params['merge_loaded_dataset'] = True

        config_params['use_sparse_single_component'] = False

    else:

        # Config params was received, and sanity check are required.
        config_params['species'] = config_params.get('species', 'unknown').lower()
        config_params['staining'] = config_params.get('staining', 'unknown').lower()
        config_params['enable_evaluation'] = config_params.get('enable_evaluation', False)

        config_params['run'] = config_params.get('run', None)
        config_params['data'] = config_params.get('data', None)
        config_params['detection'] = config_params.get('detection', None)
        config_params['segmentation'] = config_params.get('segmentation', None)
        config_params['classification'] = config_params.get('classification', None)

        config_params['labels_participating_in_training'] = config_params.get('labels_participating_in_training', [])

        config_params['particles_dist_img_mask'] = config_params.get('particles_dist_img_mask', None)

        config_params['new_data_tags'] = config_params.get('new_data_tags', [])

        config_params['merge_loaded_dataset'] = config_params.get('merge_loaded_dataset', True)

        config_params['available_gpus'] = config_params.get('available_gpus', None)
        config_params['lowest_gpu_memory'] = config_params.get('lowest_gpu_memory', None)

        if 'target' in config_params:
            config_params['target'] = config_params['target'].lower()
            if config_params['target'] == 'plt':
                config_params['training_type'] = 'single_task_training'
            elif config_params['target'] in ['4-wbc', '5-wbc', '6-wbc', '5-wbc-hard-neu', 'wbc-det', 'mk-det', 'mk-seg', 'mk-class'
                                             'left-shift-wbc', 'lym-wbc', 'lym-mon-blast-wbc',
                                             'lgl_al_abl_lym', 'blast_mon_lym', 'lym_nrbc', 'smudge',
                                             'nc-det', 'nc-multi-class-det', 'nc-seg', 'nc-wbc-rbc', 'nc-wbc', 'nc-rbc',
                                             'nc-neutrophil', 'nc-lymphocyte', 'nc-aberrant-lym', 'nc-plasma',
                                             'nc-blast', 'nc-eos', 'nc-bas', 'nc-macrophage', 'nc-neg',
                                             'ndc_region_select', 'bm_particle_regions']:
                config_params['training_type'] = 'single_task_training'
            elif config_params['target'] in ['2-wbc', '2-nc']:
                config_params['training_type'] = 'single_task_binary_training'

                if 'binary_positive_class' not in config_params:
                    raise ValueError('Trying to train Binary WBC Engine without specifying the positive label')
                config_params['binary_positive_class'] = str(config_params['binary_positive_class'])

        else:
            config_params['training_type'] = 'single_task_training'


        # Flags related to generating datasets
        # ------------------------------------
        # Loading Dataset dataset pickle file, if provided:
        config_params['load_database_dataset'] = config_params.get('load_database_dataset', False)
        # Loading Dataset training and testing datasets pickle files, if provided:
        config_params['load_train_eval_datasets'] = config_params.get('load_train_eval_datasets', False)
        # Where the PKL are located
        config_params['database_dataset_path'] = config_params.get('database_dataset_path', None)
        # Use setup for common dataset

        config_params['use_ref_scan_split_info'] = config_params.get('use_ref_scan_split_info', False)

        config_params['use_cloud_imageio'] = config_params.get('use_cloud_imageio', False)

        config_params['use_cloud_imageio_runhttp'] = config_params.get('use_cloud_imageio_runhttp', True)

        # or - deciding to open link to local DB
        config_params['generate_training_data_from_local_db'] = \
            config_params.get('generate_training_data_from_local_db', True)

        # Using sessions and scans dictionary from external file
        config_params['sessions_and_scans_dict_filename_path'] = config_params.get(
            'sessions_and_scans_dict_filename_path', None)
        # config_params['sessions_and_scans_dict_filename_path'] = '/tmp'  # tmp override
        if config_params['sessions_and_scans_dict_filename_path']:
            config_params['sessions_and_scans_dict_filename_path'] = str(
                config_params['sessions_and_scans_dict_filename_path'])

        config_params['fine_tune_train'] = config_params.get('fine_tune_train', False)

        # Flags to allow hard mining:
        # Attributes set by the user from where to load the data for current run: (path to /reference_trained_model/)
        config_params['reference_model_dir_list'] = config_params.get('reference_model_dir_list', None)
        if config_params['reference_model_dir_list'] is not None:
            param_type = type(config_params['reference_model_dir_list'])
            if param_type is not list:
                if param_type is str:
                    config_params['reference_model_dir_list'] = [config_params['reference_model_dir_list']]
                else:
                    raise ValueError('Illegal reference model parameter')

        # Passing the location of the pyramids, if accessing them directly.
        config_params['pyramid_path'] = config_params.get('pyramid_path', None)
        config_params['scan_uuid_pyramids_path'] = config_params.get('scan_uuid_pyramids_path', None)
        config_params['scan_id_pyramids_path'] = config_params.get('scan_id_pyramids_path', None)

        # TEMP assuming all pyramids are with the same resolution
        config_params['predict_pyramid_to_model_resolution_factor'] = config_params.get('predict_pyramid_to_model_resolution_factor',1.)

        # For debug... (Not used)
        config_params['res_dir'] = config_params.get('res_dir', None)

        config_params['labels_dir'] = config_params.get('labels_dir', None)

        config_params['use_sparse_inference'] = config_params.get('use_sparse_inference', False)

        config_params['report_features'] = config_params.get('report_features', False)

        config_params['write_img_file'] = config_params.get('write_img_file', False)
        config_params['write_det_img_file'] = config_params.get('write_det_img_file', True)

        config_params['test_data_id'] = config_params.get('test_data_id', None)

        # Sanity check on user input
        if config_params['load_database_dataset']:
            if config_params['database_dataset_path'] == None:
                raise ValueError('User set load_database_dataset==True from external file,' +
                                 ' and file path was not provided in config_params via database_dataset_path')
            dataset_pkl_file_name = os.path.join(config_params['database_dataset_path'],
                                                 core_path.DATABASE_DATASET_PICKLE_FILE_NAME)
            if not os.path.isfile(dataset_pkl_file_name):
                raise ValueError('Could not find a database dataset pickle file at {0}'.format(dataset_pkl_file_name))
            config_params['dataset_pkl_file_name'] = dataset_pkl_file_name


        # Iterative Task Training support. Setting required defaults.
        config_params['iterative_training_mode'] = config_params.get('iterative_training_mode', False)

        config_params['segmentation_mode'] = config_params.get('segmentation_mode', False)

        config_params['semantic_segmentation_mode'] = config_params.get('semantic_segmentation_mode', False)

        config_params['ndc_seg_debug_post_process'] = config_params.get('ndc_seg_debug_post_process', False)

        config_params['use_sparse_single_component'] = config_params.get('use_sparse_single_component', False)

        config_params['train_iteration'] = config_params.get('train_iteration', 0)

        config_params['low_res_det'] = config_params.get('low_res_det', False)

        config_params['det_th'] = config_params.get('det_th', None)

        config_params['net_ver'] = config_params.get('net_ver', 1)

        config_params['prev_ver'] = config_params.get('prev_ver', True)

        config_params['x_100_model_res'] = config_params.get('x_100_model_res', False)

        config_params['tile_center_inference'] = config_params.get('tile_center_inference', False)

        config_params['score_compatibility'] = config_params.get('score_compatibility', False)

        config_params['train_with_hardness_rate_class'] = config_params.get('train_with_hardness_rate_class', False)

        config_params['model_resolution'] = config_params.get('model_resolution', 0.000133071583818)

        if config_params['iterative_training_mode']:
            if config_params['reference_model_dir_list'] == None:
                raise ValueError('Running in iterative task training mode. No initial reference model was provided')

            # 17.17.2018: Currently supporting only 1 iteration in 'iterative training mode':
            config_params['generate_training_data_from_local_db'] = False
            config_params['load_database_dataset'] = False
            config_params[
                'load_train_eval_datasets'] = False  # Will be set to true in iterative mode, from second iteration
            config_params['database_dataset_path'] = None #link to training_data/ folder



            logger.info(
            '*** Note: you are running iterative training after conflicts resolution on provided reference model ***')

    if print_config:
        for key in config_params:
            logger.info('{0} : {1}'.format(key, config_params[key]))
    return config_params


# Utilities to read the training and evaluation datset info

def get_full_training_dataset(data_dir):
    """
    Returns full training dataset created in data_dir by the core_tfrecords functions

    """
    training_tiles_dataset_file = os.path.join(data_dir, core_path.TRAINING_FULL_DATA_SET_PATH)
    full_training_dataset = read_json_file(training_tiles_dataset_file)
    return full_training_dataset


def get_training_tfrecords_blueprint(data_dir):
    """
    Returns full training dataset created in data_dir by the core_tfrecords functions

    """
    tfrecords_blueprint_file = os.path.join(data_dir, core_path.TRAINING_TFRECORDS_BLUEPRINTS_PATH)
    tfrecords_blueprint = read_json_file(tfrecords_blueprint_file)
    return tfrecords_blueprint


def get_full_evaluation_dataset(data_dir):
    """
    Returns full training dataset created in data_dir by the core_tfrecords functions

    """
    evaluation_tiles_dataset_file = os.path.join(data_dir, core_path.EVALUATION_DATA_SET_PATH)
    full_evaluation_dataset = read_json_file(evaluation_tiles_dataset_file)
    return full_evaluation_dataset


def get_testing_conflicts_dataset(data_dir):
    """
    Returns full training dataset created in data_dir by the core_tfrecords functions

    """
    testing_conflicts_dataset_file = os.path.join(data_dir, core_path.TESTING_CONFLICTS_DATA_SET_PATH)
    testing_conflicts_dataset = read_json_file(testing_conflicts_dataset_file)
    return testing_conflicts_dataset


def get_evaluation_conflicts_dataset(data_dir):
    """
    Returns full training dataset created in data_dir by the core_tfrecords functions

    """
    evaluation_conflicts_dataset_file = os.path.join(data_dir, core_path.EVALUATION_CONFLICTS_DATA_SET_PATH)
    evaluation_conflicts_dataset = read_json_file(evaluation_conflicts_dataset_file)
    return evaluation_conflicts_dataset


def get_training_conflicts_dataset(data_dir):
    """
    Returns full training dataset created in data_dir by the core_tfrecords functions

    """
    training_conflicts_dataset_file = os.path.join(data_dir, core_path.TRAINING_CONFLICTS_DATA_SET_PATH)
    training_conflicts_dataset = read_json_file(training_conflicts_dataset_file)
    return training_conflicts_dataset


def read_json_file(path_to_json):
    if not os.path.isfile(path_to_json):
        raise ValueError('Missing .json in %s' % path_to_json)
    with open(path_to_json, 'r') as f:
        json_data = json.load(f)
    return json_data


def get_tfrecords_file_names_from_tiles_dataset(tiles_dataset, root_user_dir):
    # The path in the tiles_dataset is relative to the root_user_dir
    tfrecords_filenames = []
    print ('Reading TFRecords files from {0}'.format(root_user_dir))
    for scan in tiles_dataset:
        for roi in scan['regions_list']:
            for tile in roi['tile_list']:
                new_path = os.path.join(root_user_dir, tile['tfrecords_file_path'])
                tfrecords_filenames.append(new_path)
    return tfrecords_filenames


def get_tfrecords_metadata_from_tiles_dataset(tiles_dataset, root_user_dir):
    tfrecords_filenames_metadata = []
    print ('Reading TFRecords files from {0}'.format(root_user_dir))
    for scan in tiles_dataset:
        for roi in scan['regions_list']:
            for tile in roi['tile_list']:
                new_path = os.path.join(root_user_dir, tile['tfrecords_file_path'])
                tfrecords_filenames_metadata.append({'filename': new_path,
                                                     'metadata': tile['tile_metadata']})
    return tfrecords_filenames_metadata


def get_test_tile_list(tiles_dataset, evaluation_level, is_detection_model = False):
    # Used in model selection.
    # This utility receives a tiles_dataset and returns a list of smaller tiles_dataset to evaluate.
    # Break down is decided by the evaluation_level
    # PER_ROI - will return small tiles_dataset consisting of 1 ROI
    # PER_LABEL - will return 1 ROI with all labels

    test_tile_list = []

    if (evaluation_level == 'ROI') or (evaluation_level == 'ROI_per_SCAN') or  \
            ((evaluation_level == 'SCAN') and is_detection_model) or  \
            ((evaluation_level == 'ROI total labels ratio') and is_detection_model):
        for scan in tiles_dataset:
            scan_copy = copy.deepcopy(scan)
            scan_copy['regions_list'] = []
            for roi in scan['regions_list']:

                roi_copy = copy.deepcopy(roi)
                for tile in roi_copy['tile_list']:
                    tile['scan_id'] = scan['scan_id']
                    tile['ROI_id'] = roi_copy['ROI_id']
                    tile['session_id'] = roi_copy['ROI_session']

                new_tiles_dataset = copy.deepcopy(scan_copy)
                new_tiles_dataset['regions_list'] = [roi_copy]
                test_tile_list.append(new_tiles_dataset)
        return test_tile_list

    elif (evaluation_level == 'LABEL') or \
            (((evaluation_level == 'SCAN') or (evaluation_level == 'ROI total labels ratio'))  and (not is_detection_model)):

        # Creates a template of a returned test tile list
        new_tiles_dataset = copy.deepcopy(tiles_dataset[0])
        new_tiles_dataset['regions_list'] = [copy.deepcopy(tiles_dataset[0]['regions_list'][0])]
        new_tiles_dataset['regions_list'][0]['tile_list'] = []
        new_tiles_dataset['scan_db_resolution'] = None
        for scan in tiles_dataset:
            for roi in scan['regions_list']:
                roi_copy = copy.deepcopy(roi)
                for tile in roi_copy['tile_list']:
                    tile['scan_id'] = scan['scan_id']
                    tile['ROI_id'] = roi_copy['ROI_id']
                    tile['session_id'] = roi_copy['ROI_session']
                    tile['scan_resolution'] = scan['scan_db_resolution']
                    new_tiles_dataset['regions_list'][0]['tile_list'].append(tile)

        return [new_tiles_dataset]


def empty_tiles_dataset(tiles_dataset):
    # utility to check if a tiles_dataset is empty - i.e., with no labels
    tiles_dataset_empty = False
    if len(tiles_dataset) == 0:
        tiles_dataset_empty = True
    if not tiles_dataset[0]['regions_list'][0]['tile_list']:
        tiles_dataset_empty = True

    return tiles_dataset_empty


# TODO add distributed over the network support
# cluster_spec = {
#     "ps": ["ps0:2222", "ps1:2222"],
#     "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]}
# with tf.device(tf.train.replica_device_setter(cluster=cluster_spec)):
#   v = tf.get_variable("v", shape=[20, 20])  # this variable is placed
#                                             # in the parameter server
#                                             # by the replica_device_setter

# if __name__ == '__main__':
    # x = get_full_evaluation_dataset('/scopio/scratch5/shahar/benchmark_integration/data/tfrecords')
    # # y = get_full_training_dataset('/scopio/scratch5/shahar/benchmark_integration/data/tfrecords')
    # # get_test_tile_list(x, 'PER_ROI')
    #
    # # Playground
    # import itertools
    # import tensorflow as tf
    # import numpy as np
    #
    # from matplotlib import pyplot as plt
    #
    #
    # def get_balanced_filenames(self):
    #     # It's much more efficient to balance files names (still pointers) rather than full size images.
    #     balancing_criteria_list = [x[self.Flags.balancing_criteria] for x in self.tfrecords_filenames_metadata]
    #
    #     file_name_dataset = tf.data.Dataset.from_tensor_slices((self.tfrecords_filenames_list, balancing_criteria_list))
    #
    #     print (file_name_dataset)
    #
    #     balanced_dataset = file_name_dataset.apply(rejection_resample(class_func=self.balancing_class_mapping_function,
    #                                                                   target_dist=self.target_distribution)
    #                                                )
    #     balanced_dataset = balanced_dataset.map(lambda _, data: data)
    #     print(balanced_dataset)
    #     pass
    #
    #
    # def balancing_class_mapping_function(tfrecord_file_name, balancing_criteria):
    #     return balancing_criteria
    #
    #
    # # def gen():
    # #     for i in itertools.count(1):
    # #         yield (i, [1] * i)
    # #
    # #
    # # ds = tf.data.Dataset.from_generator(
    # #     gen, (tf.int64, tf.int64), (tf.TensorShape([]), tf.TensorShape([None])))
    #
    # # create a random vector of shape (100,2)
    # # x = np.random.sample((10, 2))
    # x = np.arange(10)
    # y = np.arange(0, 10)
    #
    # highly_unbalanced = False
    # if highly_unbalanced:
    #     y[y < 8] = 0
    #     y[y >= 8] = 1
    #     target_dist = [0.5, 0.5]
    # else:
    #     y = np.arange(0, 10)
    #     target_dist = [1. / 10] * 10
    #
    # print(x, y)
    # num_of_repeats = 200
    # len_x = num_of_repeats * len(x)
    # # make a dataset from a numpy array
    # dataset = tf.data.Dataset.from_tensor_slices((x, y))
    # print(dataset)
    # # dataset = dataset.shuffle(buffer_size=2)
    # dataset = dataset.repeat(num_of_repeats)
    #
    # dataset = dataset.apply(tf.contrib.data.rejection_resample(class_func=balancing_class_mapping_function,
    #                                                            target_dist=target_dist))
    #
    # dataset = dataset.map(lambda _, data: data)
    #
    # # balanced_dataset.repeat(200)
    # dataset = dataset.shuffle(buffer_size=10)
    #
    # value = dataset.make_one_shot_iterator().get_next()
    #
    # data = []
    # with tf.Session() as sess:
    #     zero_count = 0
    #     one_count = 0
    #     try:
    #         while True:
    #             x1 = sess.run(value)  # (1, array([1]))
    #
    #             if x1[1] == 0:
    #                 zero_count += 1
    #             else:
    #                 one_count += 1
    #
    #             print(x1)
    #             data.append(x1[0])
    #     except:
    #         print
    #         print(zero_count, one_count)
    #         print('rejected:', len_x - zero_count - one_count)
    #
    # # fixed bin size
    # bins = np.arange(0, 12)  # fixed bin size
    #
    # plt.xlim([min(data) - 1, max(data) + 3])
    #
    # plt.hist(data, bins=bins, alpha=0.5)
    # plt.title('Random Gaussian data (fixed bin size)')
    # plt.xlabel('variable X (bin size = 5)')
    # plt.ylabel('count')
    #
    # plt.show()
