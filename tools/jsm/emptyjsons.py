#!/usr/bin/env python3
# This utility creates headers-only CVI JSON files from a given JSON file
# with the following as its input
# { "key1": ["uuid1", "uuid2", ..., "uuidN"],
#   "key2": ["uuid" ... ],
#   ...
# }

import argparse
import json
import os
import sys


CVI_HEADER = {
    "pyramid_resolution": 0.0,
    "scan_name": "generated-empty-json",
    "labels": [],
    "ROIs": [],
}

def main():
    parser = argparse.ArgumentParser('create header only CVI json files')
    parser.add_argument('json_file', type=str, help='json file containing dict with uuids')
    parser.add_argument('--output-dir', type=str, required=False, help='output json file')
    args = vars(parser.parse_args())

    if not args['output_dir']:
        output_dir = os.getcwd()
    else:
        output_dir = args['output_dir']

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    scans_split_json = args['json_file']
    if not os.path.exists(scans_split_json):
        sys.exit(f'file not found: {scans_split_json}')

    with open(scans_split_json, 'r') as f:
        scans_split = json.load(f)

    for key in scans_split:
        jsons_output_dir = f'{output_dir}/{key}'
        if not os.path.exists(jsons_output_dir):
            os.mkdir(jsons_output_dir)

        scans_uuid_list = scans_split[key]
        for scan_id in scans_uuid_list:
            cvi_header = CVI_HEADER.copy()
            cvi_header["scan_id"] = scan_id

            with open(f'{jsons_output_dir}/{scan_id}.json', 'w') as f:
                json.dump(cvi_header, f, indent=4)
            print(f'{key}: {scan_id}')


if __name__ == '__main__':
    main()
