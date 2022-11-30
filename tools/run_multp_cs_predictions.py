#!/usr/bin/env python3

import argparse
import json
import os
import logging
import time


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
    t_start = time.time()
    rc = os.system(cmd)
    t_end = time.time()
    if rc != 0:
        raise RuntimeError('command error')
    run_time = t_end - t_start
    return run_time


def main():
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] -%(levelname).1s- : %(message)s',
                        datefmt='%d-%m-%Y %H:%M:%S')

    # Load parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir-wbc', type=str, help='dir containing wbc json files of datasets')
    parser.add_argument('--dataset-dir-plt', type=str, help='dir containing plt json files of datasets')
    parser.add_argument('--scans-dir', type=str, help='dir containing the scans pyramids',
                        default='/mnt/perfio/dataset/unified_symlinks')
    parser.add_argument('--models-root', type=str, help='dir containing the models')
    parser.add_argument('--output-dir', type=str)
    args = parser.parse_args()
    dataset_dir_wbc = args.dataset_dir_wbc
    dataset_dir_plt = args.dataset_dir_plt
    scans_dir = args.scans_dir
    models_root = args.models_root
    output_dir = args.output_dir

    if not output_dir:
        output_dir = os.getcwd()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    stats_dir = os.path.join(output_dir, 'statistics')
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    out_text_file = os.path.join(stats_dir, 'times_v2.txt')
    if os.path.exists(out_text_file):
        os.remove(out_text_file)

    times_info = {}
    for json_in in os.listdir(dataset_dir_wbc):
        if '.json' not in json_in:
            continue
        logging.info(f'Loading dataset file: {json_in}')
        json_path_wbc = os.path.join(dataset_dir_wbc, json_in)
        with open(json_path_wbc, 'r') as f:
            dataset_wbc = json.load(f)
        json_path_plt = os.path.join(dataset_dir_plt, json_in)
        if not os.path.exists(json_path_plt):
            logging.info(f'Dataset file: {json_in} does not exist for PLT, continuing to next file')
            continue

        scan_id = dataset_wbc['scan_id']
        resolution = dataset_wbc['pyramid_resolution']
        pyramid_path = os.path.join(scans_dir, scan_id)
        input_params = ['--input', pyramid_path,
                        '--input-res', str(resolution),
                        '--models-root', models_root,
                        '--output-dir', output_dir,
                        '--plt-rois-from-json', json_path_plt,
                        '--rois-from-json', json_path_wbc]
        times_info[scan_id] = []

        # Run prediction
        predict_executable = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          '..', 'apps', 'pbs', 'pbsapp_cs.py')
        for i in range(TIME_ITERATIONS):
            run_time = run_predict_cmd(predict_executable, input_params)
            times_info[scan_id].append(round(run_time, 2))
            logging.info(f'Waiting 1 seconds for killing servers.....')
            time.sleep(1)
            logging.info(f'Done waiting!')

        # Save times while running
        file_object = open(out_text_file, 'a')
        file_object.write(f'Scan id: {scan_id}\n')
        file_object.write(f'Ran prediction {TIME_ITERATIONS} times\n')
        file_object.write(f'Times: {times_info[scan_id]}\n\n')
        file_object.close()

    logging.info(f"Finished running prediction on directories: {dataset_dir_wbc}, {dataset_dir_plt}")
    logging.info(f"Results saved to: {stats_dir}")

    # Get statistics of all the scans together
    json_file = f'{stats_dir}/predictions_statistics.json'
    with open(json_file, 'w') as f:
        json.dump(times_info, f, indent=4)


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        raise err
