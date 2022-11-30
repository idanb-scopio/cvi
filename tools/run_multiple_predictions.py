#!/usr/bin/env python3

import argparse
import json
import os
import logging
import time
import sys


"""
This scripts goes over all the json files in 'dataset-dir' directory and runs prediction with 'model-dir'
Output json files will be saved to 'output-dir', saving debug images is optional when 'save-debug'
"""

TIME_ITERATIONS = 5


def run_predict_cmd(predict_executable, params):
    cmd_params = ''
    for line in params:
        cmd_params += f'{line} '
    cmd = f'{predict_executable} {cmd_params}'
    logging.info(f'running: {cmd}')
    rc = os.system(cmd)
    if rc != 0:
        raise RuntimeError('command error')


def main():
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] -%(levelname).1s- : %(message)s',
                        datefmt='%d-%m-%Y %H:%M:%S')

    # Load parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, help='dir containing json files of datasets', required=True)
    parser.add_argument('--scans-dir', type=str, help='dir containing the scans pyramids',
                        default='/mnt/perfio/dataset/unified_symlinks')
    parser.add_argument('--model-dir', type=str, help='dir containing the model - for core, single model')
    parser.add_argument('--models-root', type=str, help='dir containing the models - for WBC')
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--save-debug', type=bool, help='save debug images, will be saved to output-dir', default=False)
    parser.add_argument('--predict-type', type=str, help='which prediction function to run (pbs/core/plt)', default='core')
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    scans_dir = args.scans_dir
    save_debug = args.save_debug
    model_dir = args.model_dir
    models_root = args.models_root
    predict_type = args.predict_type

    if not output_dir:
        output_dir = os.getcwd()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    out_text_file = os.path.join(output_dir, 'times_v2.txt')
    if os.path.exists(out_text_file):
        os.remove(out_text_file)

    times_info = {}
    for json_in in os.listdir(dataset_dir):
        if '.json' not in json_in:
            continue
        logging.info(f'Loading dataset file: {json_in}')
        json_path = os.path.join(dataset_dir, json_in)
        with open(json_path, 'r') as f:
            dataset = json.load(f)
        scan_id = dataset['scan_id']
        times_info[scan_id] = []
        resolution = dataset['pyramid_resolution']
        pyramid_path = os.path.join(scans_dir, scan_id)
        input_params = ['--input', pyramid_path,
                        '--input-res', str(resolution),
                        '--output-dir', output_dir]
        if save_debug:
            input_params.extend(['--debug-save-to', os.path.join(output_dir, scan_id)])
            logging.info(f"Running with saving debug. Timing measurements are not comparable to production!!")

        # Run prediction
        logging.info(f"Running prediction function: {predict_type}")
        if predict_type == 'core':
            if model_dir is None:
                sys.exit(f'Missing model_dir param')
            input_params.extend(['--model-dir', model_dir,
                                 '--rois-from-labels', json_path])
            predict_executable = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              '..', 'core', 'predict.py')
            for i in range(TIME_ITERATIONS):
                t_start = time.time()
                run_predict_cmd(predict_executable, input_params)
                t_end = time.time()
                times_info[scan_id].append(round(t_end - t_start, 2))
        elif predict_type == 'plt':
            if model_dir is None:
                sys.exit(f'Missing model_dir param')
            input_params.extend(['--model-dir', model_dir,
                                 '--rois-from-json', json_path])
            predict_executable = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              '..', 'core', 'predict.py')
            for i in range(TIME_ITERATIONS):
                t_start = time.time()
                run_predict_cmd(predict_executable, input_params)
                t_end = time.time()
                times_info[scan_id].append(round(t_end - t_start, 2))
        elif predict_type == 'pbs':
            if models_root is None:
                sys.exit(f'Missing models_root param')
            input_params.extend(['--models-root', models_root,
                                 '--rois-from-json', json_path])
            predict_executable = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              '..', 'apps', 'pbs', 'predict.py')
            for i in range(TIME_ITERATIONS):
                t_start = time.time()
                run_predict_cmd(predict_executable, input_params)
                t_end = time.time()
                times_info[scan_id].append(round(t_end - t_start, 2))
        else:
            sys.exit(f'Unknown input prediction function: {predict_type}')
        # Save times while running
        file_object = open(out_text_file, 'a')
        file_object.write(f'Scan id: {scan_id}\n')
        file_object.write(f'Ran prediction {TIME_ITERATIONS} times\n')
        file_object.write(f'Times: {times_info[scan_id]}\n\n')
        file_object.close()

    logging.info(f"Finished running prediction on directory: {dataset_dir}")
    logging.info(f"Results saved to: {output_dir}")

    # Get statistics of all the scans together
    json_file = f'{output_dir}/predictions_statistics.json'
    with open(json_file, 'w') as f:
        json.dump(times_info, f, indent=4)


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        raise err
