#!/usr/bin/env python3
# Legacy DataSet Utils

import argparse
import glob
import json
import logging
import pickle
import os
import sys
import uuid
import yaml
import cv2
from lib import dzimage
from lib.dsutils import save_to_tiled_jpegs, make_histogram_sumstr


DEFAULT_BASE_DATASET = '/mnt/ssd/unified_wbc'
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
with open(f'{SCRIPT_DIR}/id2uuid.yml', 'r') as f:
    id2uuid = yaml.safe_load(f)

COLORS = [(255,   0,   0),
          (0,   255,   0),
          (0,     0, 255),
          ]

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] -%(levelname).1s- : %(message)s',
                    datefmt='%d-%m-%Y %H:%M:%S')

parser = argparse.ArgumentParser()
parser.add_argument('pickle_file', type=str, help='pickle dataset file')
parser.add_argument('--output-marks-dir', type=str, help='mark labels from pickle, save under output dir')
parser.add_argument('--limit-scans', type=int, help='limit action to first N scans')
parser.add_argument('--base-dataset', type=str, help='basedir where the dataset resides (for validating scans exist)')
parser.add_argument('--export-labels', type=str, help='output directory for saving labels')
args = vars(parser.parse_args())

pkl_file = args['pickle_file']
if not os.path.isfile(pkl_file):
    sys.exit(f'error: file not found: {pkl_file}')

# load pickled dataset file
logging.info(f'loading dataset: {pkl_file}')
with open(pkl_file, 'rb') as f:
    pds = pickle.load(f)
logging.info(f'done. pickle dataset contains {len(pds)} records.')
if len(pds) == 0:
    sys.exit('no records found.')

if args['export_labels']:
    labels_dir = args['export_labels']

    if not os.path.exists(labels_dir):
        os.mkdir(labels_dir)

    # labels histogram summary
    scans_summary = {}
    total_summary = {}
    scan_res = {}

    labels_file = f'{labels_dir}/labels.txt'
    logging.info(f'exporting labels to: {labels_file}')
    f = open(labels_file, 'w')

    for scan_idx, scan in enumerate(pds):
        try:
            scan_id = scan.scan_uuid
        except AttributeError:
            scan_id = id2uuid[scan.scan_id]

            # if scan.scan_id in id2uuid:
            #     scan_id = id2uuid[scan.scan_id]
            # else:
            #     # generate uuid
            #     scan_id = str(uuid.uuid4())
            #     print(f'ln -s /mnt/perfio/dataset/alpha_ichilov/ichilov_no_uuid/{scan.scan_id} {scan_id}')

        scans_summary[scan_id] = {}

        try:
            pyramid_res = scan.pyramid_resolution
        except AttributeError:
            pyramid_res = 0.000133

        scan_res[scan_id] = pyramid_res

        # for JSON marks
        j_labels = []
        j_ROIs = []
        j_summary = {'scan_id': scan_id, 'pyramid_resolution': pyramid_res, 'labels': j_labels, 'ROIs': j_ROIs}

        for roi_idx, roi in enumerate(scan.roi_list):
            j_ROIs.append([roi.x, roi.y, roi.w, roi.h, f'ROI idx {roi_idx}'])

            for labels_dict in roi.labels:
                x = labels_dict['x_gl']
                y = labels_dict['y_gl']
                w = labels_dict['w']
                h = labels_dict['h']

                if len(labels_dict['label_name']) > 1:
                    logging.error(f'multiple labels for {scan_id} {x} {y}: {",".join(labels_dict["labels_name"])}')
                    raise ValueError

                label = labels_dict['label_name'][0]
                j_labels.append([x, y, w, h, label])

                if label in scans_summary[scan_id]:
                    scans_summary[scan_id][label] += 1
                else:
                    scans_summary[scan_id][label] = 1

                if label in total_summary:
                    total_summary[label] += 1
                else:
                    total_summary[label] = 1

                f.write(f'{scan_id} {x} {y} {w} {h} {label}\n')

        # write scan label data in json format (supported by pdsview)
        json_file = f'{labels_dir}/{scan_id}.json'
        logging.info(f'[{scan_idx+1}/{len(pds)}] writing: {json_file}')
        with open(json_file, 'w') as jf:
            json.dump(j_summary, jf, indent=4)

    f.close()

    with open(f'{labels_dir}/summary.txt', 'w') as f:
        for scan in scans_summary:
            f.write(f'{scan} (resolution [mm/pixel]: {scan_res[scan]:.7f}):\n')

            for label in scans_summary[scan]:
                f.write(f'  {label}: {scans_summary[scan][label]}\n')
            f.write('\n')

        f.write(f'\ntotal labels in {len(scans_summary.keys())} scans:\n')
        for label in total_summary:
            f.write(f'  {label}: {total_summary[label]}\n')

if args['limit_scans']:
    num_scans = min(args['limit_scans'], len(pds))
    logging.warning(f'limit scans: operations are limited to the first {num_scans} scans')
else:
    num_scans = len(pds)

# stats dictionary, keyed by UUID
stats_dict = {}

# validate scans existence
if args['base_dataset']:
    base_dataset = args['base_dataset']
else:
    base_dataset = DEFAULT_BASE_DATASET
    logging.info(f'using default base dataset location: {base_dataset}')

pds_validated = []
missing = 0
print(f'{"old id   ":11s}{"scan id (uuid)":38s}{"scan name":40s}')
print(f'{"=========":11s}{"==============":38s}{"=========":40s}')
for idx in range(num_scans):
    entry = pds[idx]
    scan_id = entry.scan_uuid
    scan_path = '{0}/{1}'.format(base_dataset, scan_id)

    if not os.path.exists(scan_path):
        logging.warning('scan not found at: {0}'.format(scan_path))
        stats_dict[scan_id] = None
        missing += 1
        print(f'{"!"+str(entry.scan_id):11s}{"!"+entry.scan_uuid:38s}{"!"+entry.scan_name:40s}')
        continue

    print(f'{str(entry.scan_id):11s}{entry.scan_uuid:38s}{entry.scan_name:40s}')

    pds_validated.append(entry)
    stats_dict[scan_id] = scan_id


if missing == 0:
    logging.info(f'all {num_scans} scans found')
else:
    if missing == 1:
        logging.warning(f'{missing} scan is missing!')
    else:
        logging.warning(f'{missing} scans are missing!')

# *** Debug Bounding Box Markings ***
if args['output_marks_dir']:

    output_dir = args['output_marks_dir']
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # traverse all scans in the list, save ROI images with rectangle marks.
    # images may be tiled due to JPEG's limits (65K x 65K)
    for scan_idx, scan in enumerate(pds_validated):
        if scan_idx >= num_scans:
            logging.warning(f'reached scans limit of {num_scans}')
            break

        scan_id = scan.scan_uuid
        scan_path = f'{base_dataset}/{scan_id}'
        dzi_file = f'{scan_path}/pyramid/pyramid.dzi'

        if not os.path.exists(dzi_file):
            logging.error(f'missing pyramid file: {dzi_file}')
            stats_dict[scan_id] = None
            continue

        target_dir = f'{output_dir}/{scan_id}'
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)

        dzi = dzimage.DZImageFs.fromfile(dzi_file)

        # iterate over ROIs
        for roi_idx, roi in enumerate(scan.roi_list):

            marked_img_file = '{}/roi_{}_marked.jpg'.format(target_dir, roi_idx)
            existing_img_files = glob.glob(marked_img_file[:-4]+'*.jpg')

            roi_img = None
            if len(existing_img_files) > 0:
                logging.warning(f'skipping rectangle drawing for roi file(s): {",".join(existing_img_files)}')
            else:
                logging.info(f'[{scan_idx+1}/{len(pds_validated)}] Reading ROI: {roi.w}x{roi.h} at ({roi.x},{roi.y})'
                             f'for scan id {scan_id}')
                roi_img = dzi.read(x=roi.x, y=roi.y, width=roi.w, height=roi.h)

            labels_dict = roi.labels
            logging.info('Marking {0} labels'.format(len(labels_dict)))
            labels_hist = {}

            for entry in labels_dict:
                x = entry['x_gl'] - roi.x
                y = entry['y_gl'] - roi.y
                label_str = ",".join(entry['label_name'])
                if label_str not in labels_hist:
                    labels_hist[label_str] = 1
                else:
                    labels_hist[label_str] += 1

                # draw rectangle
                if roi_img is not None:
                    start_point = (x, y)
                    end_point = (x + entry['w']-1, y + entry['h']-1)
                    color = COLORS[0]

                    roi_img = cv2.rectangle(roi_img, start_point, end_point, color, 2)
                    cv2.putText(roi_img, label_str, (x, y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            with open(f'{target_dir}/labels.txt', 'a') as f:
                labels_output_str = f'{scan_id:38s} ROI:{roi_idx + 1}/{len(scan.roi_list)} total: {len(labels_dict)} -- {make_histogram_sumstr(labels_hist)}'
                f.write(labels_output_str + '\n')
                print(labels_output_str)

            if roi_img is not None:
                logging.info('Saving as: {}'.format(marked_img_file))
                save_to_tiled_jpegs(marked_img_file, roi_img)
