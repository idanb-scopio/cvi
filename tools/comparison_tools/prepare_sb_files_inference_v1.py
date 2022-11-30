#!/usr/bin/env python3

import argparse
import json
import os
import logging

"""
This scripts gets either json file (json-file-path) or directory of jsons (json-dir) of datasets with rois 
It will create a configuration files in the necessary directories order
That will able to run inference with v1 on a production machine
The command to run in the SB will be printed at the end
Note: in order to run the command it will expect in the SB to have: 
      1. sb-scans-path - contains pyramids, 2. sb-analyses-path - contains input files 

Script output (for multiple scans):
 - output_dir/commands_v1.txt
 - output_dir/analysis/#scan_id#/input.json (multiple files if json-dir)
 When running not on SB --->> move analysis dir to 'sb-analyses-path'
 - Run 'run_cmd_on_sb_v1.py' from SB for inference 
"""

MAX_WBC = 2000000  # Number of max wbc to run on


def main():
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] -%(levelname).1s- : %(message)s',
                        datefmt='%d-%m-%Y %H:%M:%S')

    # Load input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-file-path", type=str, help='Full path of a single json file')
    parser.add_argument("--json-dir", type=str, default=None, help='Directory that contains multiple json files')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--sb-analyses-path', type=str, default='/srv/webroot/analyses',
                        help='In which path the output will be saved in the SB/machine running on')
    parser.add_argument('--sb-scans-path', type=str, default='/srv/webroot/scans',
                        help='Pyramids path')
    parser.add_argument('--analysis-type', type=str, default='WBC', help='WBC / PLT')

    args = parser.parse_args()
    json_file_path = args.json_file_path
    json_dir = args.json_dir
    output_dir = args.output_dir
    sb_analyses_path = args.sb_analyses_path
    sb_scans_path = args.sb_scans_path
    analysis_type = args.analysis_type

    # Load dataset json
    if json_dir:
        json_files = []
        for json_in in os.listdir(json_dir):
            if '.json' not in json_in:
                continue
            json_files.append(os.path.join(json_dir, json_in))
    else:
        json_files = [json_file_path]

    # Create output and analysis dir
    if not os.path.exists(output_dir):
        logging.info(f"Creating directory {output_dir}")
        os.makedirs(output_dir)
    if not os.path.exists(sb_analyses_path):
        logging.info(f"Creating directory {sb_analyses_path}")
        os.makedirs(sb_analyses_path)

    # Create text file of commands
    out_text_file = os.path.join(output_dir, f'commands_v1_{analysis_type}.txt')
    if os.path.exists(out_text_file):
        os.remove(out_text_file)
    file_object = open(out_text_file, 'a')

    # Go over all dataset json files
    for json_file_path in json_files:
        with open(json_file_path, 'r') as f:
            dataset = json.load(f)
        scan_id = dataset['scan_id']
        logging.info(f"Scan id: {dataset['scan_id']}")
        # Create output directories
        # More than 1 roi will be saved in a new directory
        if len(dataset['ROIs']) > 1:
            logging.info(f"Scan id: {dataset['scan_id']} has more than 1 ROI")
        for count_roi, roi in enumerate(dataset['ROIs']):
            if count_roi > 0:
                current_sb_analyses_path = f'{sb_analyses_path}_roi{count_roi+1}'
                if not os.path.exists(current_sb_analyses_path):
                    logging.info(f"Creating directory {current_sb_analyses_path}")
                    os.makedirs(current_sb_analyses_path)
            else:
                current_sb_analyses_path = sb_analyses_path

            inference_input_file = {"species": "human",
                                    "um_per_pixel": dataset['pyramid_resolution']*1000,
                                    "pyramid": os.path.join(sb_scans_path, scan_id, 'pyramid', 'pyramid.dzi'),
                                    "analysis_type": analysis_type,
                                    "segmentation": False,
                                    "model_network": "human_blood",
                                    "max_wbc": MAX_WBC,
                                    "region": roi[:4],
                                    "region_name": roi[4]}

            # Save config file
            output_json_path = os.path.join(current_sb_analyses_path, scan_id, 'input')
            if not os.path.exists(output_json_path):
                logging.info(f"Creating directory {output_json_path}")
                os.makedirs(output_json_path)
            with open(os.path.join(output_json_path, 'input.json'), 'w') as f:
                json.dump(inference_input_file, f, indent=4)
            logging.info(f'Input json for inference saved in: {output_json_path}')

            # Print inference v1 command
            cmd_to_run = (f'V8_SCAN_ID={scan_id} '
                          # f'/srv/app_root/cv_algo/cv_runner/cv_run.sh '  # disabled when running with venv
                          f'/srv/app_root/cv_results_api/scripts/run_cv_inference_new.py '
                          f'--input {current_sb_analyses_path}/{scan_id}/input/input.json '
                          f'--logfile {current_sb_analyses_path}/{scan_id}/logs/cv_single_roi_inference.log '
                          f'--output {current_sb_analyses_path}/{scan_id}/results/result_dict.json '
                          f'--workdir /mnt/ramdisk/v1_inference_tests '
                          f'--endpoint http://localhost:8082/progress/{scan_id}\n')
            logging.info(f'Transfer to SB: scan to {sb_scans_path}, input dir to {current_sb_analyses_path}')
            logging.info(f'Command to run inference in SB:')
            logging.info(cmd_to_run)

            # Write commands to text file
            file_object.write(cmd_to_run)
    file_object.close()

    logging.info(f'Commands out file saved in: {out_text_file}')


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        raise err
