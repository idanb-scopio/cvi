#!/usr/bin/env python3

import argparse
import json
import sys


def main(raw_args=None):
    parser = argparse.ArgumentParser('join multiple json files')
    parser.add_argument('json_files', type=str, nargs='+', help='json files to join')
    parser.add_argument('--output', type=str, required=True, help='output json file')
    args = parser.parse_args(raw_args)

    json_files = args.json_files
    output_file = args.output

    scan_id = None
    output_dict = None
    labels_set = set()
    rois_set = set()
    for jf in json_files:
        with open(jf, 'r') as f:
            js_dict = json.load(f)

        if not scan_id:
            scan_id = js_dict['scan_id']
            output_dict = js_dict.copy()
            del output_dict['labels']
            del output_dict['ROIs']

        else:
            if scan_id != js_dict['scan_id']:
                sys.exit(f'scan_id mismatch')

        print(f'adding {len(js_dict["labels"])} labels')
        for label in js_dict['labels']:
            labels_set.add(tuple(label))

        print(f'adding {len(js_dict["ROIs"])} ROIs')
        for roi in js_dict["ROIs"]:
            rois_set.add(tuple(roi))

    output_dict['labels'] = list(labels_set)
    output_dict['ROIs'] = list(rois_set)

    print(f'saving: {output_file}')
    print(f'labels: {len(labels_set)} ROIs: {len(rois_set)}')

    with open(output_file, 'w') as f:
        json.dump(output_dict, f, indent=4)


if __name__ == '__main__':
    main()
