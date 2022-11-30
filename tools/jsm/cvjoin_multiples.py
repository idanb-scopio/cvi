#!/usr/bin/env python3

# join cvi json files from multiple directories to one directory using "cvjoin"
# files that exist only in one directory will remain the same way as the original file

import argparse
import os
from tools.jsm.cvjoin import main as cvjoin


def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('json_dirs', type=str, nargs='+', help='json files to join')
    parser.add_argument('--output-dir', type=str, help='output dir')
    args = parser.parse_args(raw_args)

    json_dirs = args.json_dirs

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # get all files in all the directories given
    all_json_list = []
    for jd in json_dirs:
        all_json_list.extend(os.listdir(jd))

    # go over all the files and save them in output dir, join if more than 1 file exists
    all_json_list_no_repeats = set(all_json_list)
    for jf in all_json_list_no_repeats:
        if '.json' in jf:
            input_params = []
            for jd in json_dirs:
                if jf in os.listdir(jd):
                    input_params.extend([os.path.join(jd, jf)])
            input_params.extend(['--output', os.path.join(output_dir, jf)])
            if len(input_params) > 3:
                print(f"Combining json files of uuid: {jf.split('.json')[0]}")
            cvjoin(input_params)

    print('ok')


if __name__ == '__main__':
    main()
