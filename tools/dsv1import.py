#!/usr/bin/env python3

# transform dataset from GCP to CVI JSON format

import argparse
import json
import logging
import os
import sys


def main():
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] -%(levelname).1s- : %(message)s',
                        datefmt='%d-%m-%Y %H:%M:%S')

    parser = argparse.ArgumentParser()
    parser.add_argument('json_file', type=str, help='JSON dataset file')
    parser.add_argument('-o', '--output-dir', required=True, type=str, help='output dir')
    args = vars(parser.parse_args())

    json_file = args['json_file']
    with open(json_file, 'r') as f:
        l_dataset = json.load(f)

    output_dir = args['output_dir']
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not output_dir:
        output_dir = '.'

    logging.info(f'labels dataset is saved to: {os.path.abspath(output_dir)}')

    for scan_idx, scan in enumerate(l_dataset):
        scan_id = scan['scan_uuid']
        scan_name = scan['scan_name']
        pyramid_res = scan['scan_resolution_in_mm_per_pixel']
        roi_list = scan['roi_list']
        logging.info(f'{scan_id} {pyramid_res:.7f} {len(roi_list)}')

        cvi_scan_dict = {'scan_id': scan_id,
                         'pyramid_resolution': pyramid_res,
                         'scan_name': scan_name,
                         'labels': [],
                         'ROIs': []
                         }

        for roi_idx, roi in enumerate(roi_list):
            roi_details = [roi['x'], roi['y'], roi['w'], roi['h'], f'ROI idx {roi_idx}']

            v1_labels = roi['labels']
            v2_labels = []
            for v1_label in v1_labels:
                label = [
                    v1_label['x_gl'],
                    v1_label['y_gl'],
                    v1_label['w'],
                    v1_label['h'],
                    ''.join(v1_label['label_name']),
                    v1_label['database_id']
                         ]
                v2_labels.append(label)

            cvi_scan_dict['labels'] += v2_labels
            cvi_scan_dict['ROIs'].append(roi_details)
            logging.info(f'{scan_id}: adding {len(v2_labels)} labels for ROI {roi_idx}')

        output_filename = f'{output_dir}/{scan_id}.json'
        logging.info(f'writing: {output_filename}')
        with open(output_filename, 'w') as f:
            json.dump(cvi_scan_dict, f, indent=4)

    print('ok')


if __name__ == '__main__':
    main()
