#!/usr/bin/env python3
import argparse
import glob
import logging
import os
import sys

from utils import execute
from models import cellsfactory

DEFAULT_PYRAMIDS_ROOT = os.environ.get('CVI_PYRAMIDS_ROOT', None)

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def create_run_file(filename, commands):
    run_file_content = """#!/bin/bash

set -xe

"""
    for cmd in commands:
        run_file_content += cmd + '\n'

    with open(filename, 'w') as f:
        f.write(run_file_content)

    os.chmod(filename, 0o775)


def main():
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] -%(levelname).1s- : %(message)s',
                        datefmt='%d-%m-%Y %H:%M:%S')
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-dir', type=str, help='dir containing reference cvi json files', required=True)
    parser.add_argument('--models-dir', type=str, help='dir containing keras models', required=True)
    parser.add_argument('--output-dir', type=str, help='dir to place output results inside', required=True)
    parser.add_argument('--pyramids-root', type=str, help='pyramids root dir', required=False)
    parser.add_argument('--run-files', action='store_true', help='create run files instead of actual execution',
                        required=False)
    args = vars(parser.parse_args())

    output_dir = args['output_dir']
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if args['pyramids_root']:
        pyramids_root = args['pyramids_root']
    else:
        pyramids_root = DEFAULT_PYRAMIDS_ROOT

    if not os.path.exists(pyramids_root):
        sys.exit(f'pyramids root not found: {pyramids_root}')

    eval_dir = args['eval_dir']
    if not os.path.exists(eval_dir):
        sys.exit(f'eval dir not found: {eval_dir}')

    # gather model files
    models_dir = args['models_dir']

    model_cfg_yaml = f'{models_dir}/model.yml'
    if not os.path.exists(model_cfg_yaml):
        sys.exit(f'model yaml not found: {model_cfg_yaml}')

    cells_model = cellsfactory.model_instance_from_yaml(model_cfg_yaml)
    model_type = cells_model.get_model_type()

    model_files = glob.glob(f'{models_dir}/*.h5')
    if not model_files:
        sys.exit(f'no model files found in: {models_dir}')

    logging.info(f'found {len(model_files)} model files')
    for model_file in model_files:
        run_file_commands = []

        basename = os.path.basename(model_file)
        # model output dir is the model file name without the .h5 extension
        model_output_dir = f'{output_dir}/{basename.split(".")[0]}'
        if not os.path.exists(model_output_dir):
            os.mkdir(model_output_dir)

        eval_files = set(os.path.basename(f) for f in glob.glob(f'{eval_dir}/*.json'))
        tst_files = set(os.path.basename(f) for f in glob.glob(f'{model_output_dir}/*.json'))

        if eval_files == tst_files:
            logging.info(f'all json output files exist for model: {basename}')
        else:
            # run core/predict
            if model_type == 'cells_detector':
                predict_cmd = f'{SCRIPT_DIR}/predict.py --rois-from-json {eval_dir} --model-dir {models_dir} '\
                              f'--input {pyramids_root} --model-filename {basename} --output-dir {model_output_dir} '\
                               '--limit-gpu-mem 7000'
            elif model_type == 'cells_classifier':
                predict_cmd = f'{SCRIPT_DIR}/predict.py --rois-from-labels {eval_dir} --model-dir {models_dir} ' \
                              f'--input {pyramids_root} --model-filename {basename} --output-dir {model_output_dir} ' \
                              '--limit-gpu-mem 7000 --adjust-labels --load-weights-only'
            else:
                sys.exit(f'unknown model type: {model_type}')

            logging.info(f'running: {predict_cmd}')

            if args['run_files']:
                run_file_commands.append(predict_cmd)
            else:
                execute(predict_cmd)

        compare_log_file = f'{model_output_dir}/compare.log'
        if os.path.exists(compare_log_file):
            logging.info(f'compare results file exists: {compare_log_file}')
        else:
            logging.info(f'comparing results for {basename}')

            if model_type == 'cells_detector':
                compare_cmd = f'{SCRIPT_DIR}/../tools/cvjcompare.py --neg-label neg '\
                              f'--model-dir {models_dir} {eval_dir} {model_output_dir} > {compare_log_file}'
            elif model_type == 'cells_classifier':
                compare_cmd = f'{SCRIPT_DIR}/../tools/cvjcompare.py --neg-label neg --wide-border 0 '\
                              f'--model-dir {models_dir} {eval_dir} ' \
                              f'{model_output_dir} > {compare_log_file}'

            if args['run_files']:
                run_file_commands.append(compare_cmd)
                run_filename = f'run_{basename.split(".")[0]}.sh'
                create_run_file(run_filename, run_file_commands)
            else:
                execute(compare_cmd)


if __name__ == '__main__':
    main()
