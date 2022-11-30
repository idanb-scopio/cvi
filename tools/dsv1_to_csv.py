#!/usr/bin/env python3

# Create a csv file that contains each scan details from V1 (google cloud) JSON

import argparse
import json
import logging
import os
import pandas as pd
import csv


def main():
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] -%(levelname).1s- : %(message)s',
                        datefmt='%d-%m-%Y %H:%M:%S')

    parser = argparse.ArgumentParser()
    parser.add_argument('json_file', type=str, help='JSON dataset file')
    parser.add_argument('--output-dir', type=str, help='output dir')
    parser.add_argument('--csv-sub-scans', type=str, help='Currently not implemented! '
                                                          'csv file with specific scan names to select')
    parser.add_argument('--json-dir-sub-scans', type=str, help='dir with json files with scan names to select')
    args = vars(parser.parse_args())

    json_file = args['json_file']
    with open(json_file, 'r') as f:
        l_dataset = json.load(f)

    output_dir = args['output_dir']
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not output_dir:
        output_dir = '.'

    if args['csv_sub_scans']:
        # TODO - add this metric
        csv_sub_scans_path = args['csv_sub_scans']
        input_file = csv.DictReader(open(csv_sub_scans_path))

    if args['json_dir_sub_scans']:
        json_dir_sub_scans_path = args['json_dir_sub_scans']
        json_files = os.listdir(json_dir_sub_scans_path)
        uuid_list = []
        for json_file in json_files:
            uuid_list.append(json_file.split('.json')[0])
        logging.info(f'found {len(json_files)} json files.')
    else:
        uuid_list = None

    ds_dict = {'scan_uuid': [],
               'scan_id': [],
               'scan_name': [],
               'scan_resolution': [],
               'scanning_device': [],
               'roi_count': [],
               'roi_session': [],
               'roi_number_cells': [],
               'roi_name': []
               }

    # get total scans info
    num_scans = 0
    more_than_1roi = 0
    scan_skipped = 0

    for scan_idx, scan in enumerate(l_dataset):
        # scan info
        scan_uuid = scan['scan_uuid']
        scan_id = scan['scan_id']
        scan_name = scan['scan_name']
        pyramid_res = scan['scan_resolution_in_mm_per_pixel']
        roi_list = scan['roi_list']
        scanning_device = scan['scanning_device']

        if uuid_list and scan_uuid not in uuid_list:
            scan_skipped += 1
            continue
        num_scans += 1

        if len(roi_list) > 1:
            more_than_1roi += 1
            logging.info(f'Scan uuid {scan_uuid} has more than 1 ROI')

        for roi_idx, roi in enumerate(roi_list):
            ds_dict['scan_uuid'].append(scan_uuid)
            ds_dict['scan_id'].append(scan_id)
            ds_dict['scan_name'].append(scan_name)
            ds_dict['scan_resolution'].append(pyramid_res)
            ds_dict['scanning_device'].append(scanning_device)

            # ROI info
            ds_dict['roi_count'].append(f'{roi_idx+1}/{len(roi_list)}')
            ds_dict['roi_session'].append(roi['session'])
            ds_dict['roi_name'].append(roi['name'])
            ds_dict['roi_number_cells'].append(len(roi['labels']))

    ds_pd = pd.DataFrame.from_dict(ds_dict)
    ds_pd.to_csv(f'{output_dir}/ds_{num_scans}scans.csv', encoding='utf-8', index=False)
    logging.info(f'csv dataset is saved to: {os.path.abspath(output_dir)}')
    logging.info(f'Total scans count: {num_scans}')
    logging.info(f'Total scans skipped: {scan_skipped}')
    logging.info(f'Number of scans with more than 1 roi: {more_than_1roi}')


if __name__ == '__main__':
    main()
