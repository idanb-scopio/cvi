#!/usr/bin/env python3
import os
import time
import json
import argparse
import logging

TIME_ITERATIONS = 1  # how many times to run each inference


"""
This script will execute commands file 'commands_v1.txt' that was created with 'prepare_sb_files_inference_v1.py'
It will run v1 inference in production mode and will measure the running time
Output:
    predictions_statistics.json, times_v1.txt files will be saved
    result_dict.json files will be saved for each scan in its directory
Run 'save_multiple_result_dict_to_v2.py' to save json outputs to v2
"""


def main():
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] -%(levelname).1s- : %(message)s',
                        datefmt='%d-%m-%Y %H:%M:%S')
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd_path', type=str, help='command file path')
    parser.add_argument('--output-dir', type=str, default='.')
    args = vars(parser.parse_args())
    cmd_path = args['cmd_path']
    output_dir = args['output_dir']

    with open(cmd_path) as f:
        lines = f.readlines()
    results = {}

    # Create output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create text file of commands
    out_text_file = os.path.join(output_dir, 'times_v1.txt')
    if os.path.exists(out_text_file):
        logging.info(f'Removing previous file of times')
        os.remove(out_text_file)

    for line in range(len(lines)):
        current_cmd = lines[line]
        # Get scan id between a known paths
        # TODO - fix to a more generic way
        start_phrase, end_phrase = 'analyses/', '/input'
        scan_id = current_cmd[current_cmd.find(start_phrase) + len(start_phrase):
                              current_cmd.rfind(end_phrase) - len(end_phrase)]
        logging.info(f'Running on scan: {scan_id}')
        results[scan_id] = []
        for i in range(TIME_ITERATIONS):
            t_start = time.time()
            os.system(current_cmd)
            t_end = time.time()
            results[scan_id].append(round(t_end - t_start, 2))
        file_object = open(out_text_file, 'a')
        file_object.write(f'Scan id: {scan_id}\n')
        file_object.write(f'Ran {TIME_ITERATIONS} times command:\n{current_cmd}')
        file_object.write(f'Times: {results[scan_id]}\n\n')
        file_object.close()

    print(results)

    # Get statistics of all the scans together
    json_file = f'{output_dir}/predictions_statistics.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        raise err
