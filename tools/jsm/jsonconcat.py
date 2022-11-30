#!/usr/bin/env python3

import argparse
import json


def main():
    parser = argparse.ArgumentParser('join multiple json files')
    parser.add_argument('json_files', type=str, nargs='+', help='json files to join')
    parser.add_argument('--output', type=str, required=True, help='output json file')
    args = parser.parse_args()

    json_files = args.json_files
    output_file = args.output

    dataset = {}
    for jf in json_files:
        print(f'loading: {jf}')
        with open(jf, 'r') as f:
            js_dict = json.load(f)

        for entry in js_dict:
            scan_id = entry['scan_uuid']
            if scan_id in dataset:
                print(f'{scan_id} already exists in dataset: {jf}')
            dataset[scan_id] = entry

    print(f'saving: {output_file}')
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=4)


if __name__ == '__main__':
    main()
