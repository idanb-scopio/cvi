#!/usr/bin/env python3

import argparse
import json
import os
import logging
from tools.convert_json_v1_to_v2 import main as run_convert_json_v1_to_v2
from tools.jsm.cvjoin_multiples import main as cvjoin_multiples

"""
Getting results in V1 format (subdirectories of scans and regions)
Saving all results_dict.json files to one directory in v2 format and renaming them by uuid 
--test_type : how the subdirectories are organized
  - cv_test: running with cv_test on a pkl dataset file 
  - cv_runner: running on SB in production mode
"""


def convert_cv_test_results(json_dataset, results_dicts_path_v1, output_dir):
    # running with cv_algo
    # Create mapping id to uuid
    with open(json_dataset, 'r') as f:
        json_dataset_dict = json.load(f)
    scan_id_to_uuid = {}
    for scan in json_dataset_dict:
        if 'scan_uuid' in scan:
            scan_id_to_uuid[str(scan['scan_id'])] = scan['scan_uuid']
        else:
            scan_id_to_uuid[str(scan['scan_id'])] = str(scan['scan_id'])

    # Get results dict for each scan
    for scan_output in os.listdir(results_dicts_path_v1):
        if 'scan' in scan_output:
            scan_id = scan_output.split('scan_')[1]
            scan_uuid = scan_id_to_uuid[scan_id]
            for region in os.listdir(os.path.join(results_dicts_path_v1, scan_output)):
                file_path = os.path.join(results_dicts_path_v1, scan_output, region, 'result_dict.json')
                if os.path.exists(file_path):
                    print(file_path)
                    continue
                for area in os.listdir(os.path.join(results_dicts_path_v1, scan_output, region)):
                    file_path = os.path.join(results_dicts_path_v1, scan_output, region, area, 'result_dict.json')
                    if os.path.exists(file_path):
                        print(file_path)
                        continue
                    for sub_area in os.listdir(os.path.join(results_dicts_path_v1, scan_output, region, area)):
                        file_path = os.path.join(results_dicts_path_v1, scan_output, region, area, sub_area,
                                                 'result_dict.json')
                        if os.path.exists(file_path):
                            print(file_path)
                            continue
            params = ['--json_in', file_path, '--out_path', output_dir, '--scan_id', scan_uuid]
            run_convert_json_v1_to_v2(params)


def convert_cv_runner_results(all_paths, output_dirs, save_prediction_history=None):
    # running SB production version
    for results_dict_path, out_dir in zip(all_paths, output_dirs):
        for scan_id in os.listdir(results_dict_path):
            results_path = os.path.join(results_dict_path, scan_id, 'results', 'result_dict_full.json')
            params = ['--json_in', results_path, '--out_path', out_dir, '--scan_id', scan_id]
            if save_prediction_history:
                params.append('--pred-history')
            run_convert_json_v1_to_v2(params)


def main():
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] -%(levelname).1s- : %(message)s',
                        datefmt='%d-%m-%Y %H:%M:%S')

    # Load previous version of "result_dict.json" and re-save it as version 2
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dataset", type=str, help='Required only for cv_test results')
    parser.add_argument("-in", "--v1_results_input_dir", type=str)
    parser.add_argument("-o", "--output_dir", type=str, default='')
    parser.add_argument("--test_type", type=str, default='cv_test', help='Which output format to load - cv_test / cv_runner')
    parser.add_argument('--pred-history', action='store_true', help='add prediction score history to json')
    parser.add_argument('--look-for-rois', action='store_true',
                        help='check if more dirs exists for dataset with multiple rois')

    args = parser.parse_args()
    json_dataset = args.json_dataset
    v1_results_input_dir = args.v1_results_input_dir
    test_type = args.test_type
    output_dir = args.output_dir
    look_for_rois = args.look_for_rois
    save_prediction_history = True if args.pred_history else False

    # Input dir
    if 'WBC_test' in os.listdir(v1_results_input_dir):
        results_dicts_path_v1 = os.path.join(v1_results_input_dir, 'WBC_test')
    else:
        results_dicts_path_v1 = v1_results_input_dir

    # Set default output dir
    if not output_dir:
        output_dir = os.path.join(v1_results_input_dir, 'results_v2_format')

    # Find dirs of other rois - available for 'cv_runner' only
    all_paths = [results_dicts_path_v1]
    output_dirs = [output_dir]
    if look_for_rois:
        results_dir_name = os.path.split(results_dicts_path_v1)[1]
        main_analyses_path = os.path.abspath(os.path.join(results_dicts_path_v1, os.pardir))
        for dir in os.listdir(main_analyses_path):
            if results_dir_name + '_roi' in dir:
                logging.info(f"Found ROI directory to convert: {dir}")
                all_paths.append(os.path.join(main_analyses_path, dir))
                output_dirs.append(f'{output_dir}{dir.split(results_dir_name)[1]}')

    # Create output dir
    for outdir in output_dirs:
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

    if test_type == 'cv_test':
        convert_cv_test_results(json_dataset, all_paths[0], output_dirs[0])
    elif test_type == 'cv_runner':
        convert_cv_runner_results(all_paths, output_dirs, save_prediction_history)

        # save combined json files
        if look_for_rois and len(all_paths) > 1:
            input_params = []
            for out_dir in output_dirs:
                input_params.extend([out_dir])
            input_params.extend(['--output', f'{output_dir}_combined_rois'])
            cvjoin_multiples(input_params)
            logging.info(f"Combined {len(all_paths)} rois from jsons and saving to {output_dir}_combined_rois")


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        raise err
