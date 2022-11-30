#!/usr/bin/env python3

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser('join multiple json files')
    parser.add_argument('meta', type=str, help='.txt meta file to be converted to json files')
    parser.add_argument('--output-dir', type=str, required=True, help='output dir for json files')
    args = vars(parser.parse_args())

    meta_file = args['meta']
    if not os.path.exists(meta_file):
        sys.exit(f'file not found: {meta_file}')

    output_dir = args['output_dir']
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with open(meta_file, 'r') as f:
        samples = f.readlines()

    prev_scan_id = ''
    dataset = {}
    for sample in samples:
        fields = sample.strip().split()
        sub_fields = fields[1].split(',')

        scan_id = sub_fields[0]
        if scan_id != prev_scan_id:
            if dataset:
                with open(f'{output_dir}/{prev_scan_id}.json', 'w') as f:
                    json.dump(dataset, f, indent=4)

            pyr_res = fields[2].split('=')[1]
            dataset = {'scan_id': scan_id,
                       'pyramid_resolution': float(pyr_res),
                       'labels': [],
                       'ROIs': []}
            prev_scan_id = scan_id

        # take the first 4 integer coordinates and cast to int
        lbl = sub_fields[1:5]
        lbl = [int(e) for e in lbl]

        # append the label string
        lbl.append(sub_fields[5])
        dataset['labels'].append(lbl)


if __name__ == '__main__':
    main()
