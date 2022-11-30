#!/usr/bin/env python3
# Tiles Dataset Utils - for tiles_dataset.json

import argparse
import json
import logging
import os
import sys
from lib.dsutils import validate_scans_existence, scan_path_unified, Rect, save_to_tiled_jpegs, \
    is_rect_contains_rect
from lib.debugutils import mark_rectangle
from lib import dzimage

DEFAULT_BASE_DATASET = '/mnt/ssd/unified_wbc'

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] -%(levelname).1s- : %(message)s',
                    datefmt='%d-%m-%Y %H:%M:%S')

parser = argparse.ArgumentParser()
parser.add_argument('json_file', type=str, help='json dataset file')
parser.add_argument('--output-marks-dir', type=str, help='mark tiles from json, save under output dir')
parser.add_argument('--output-json-dir', type=str, help='create summarized json file, save under output dir')
parser.add_argument('--limit-scans', type=int, help='limit action to first N scans')
parser.add_argument('--select-scans', type=str, help='enable action for selected scan ids')
parser.add_argument('--base-dataset', type=str, help='basedir where the dataset resides (for validating scans exist)')
parser.add_argument('--scale', type=float, help='scale factor for loaded coordinates')
args = vars(parser.parse_args())

json_file = args['json_file']
if not os.path.isfile(json_file):
    sys.exit(f'error: file not found: {json_file}')

# Load the JSON Dataset
logging.info(f'loading tiles dataset: {json_file}')
with open(json_file, 'r') as f:
    tiles_dataset = json.load(f)
logging.info(f'done. tiles dataset contains {len(tiles_dataset)} records.')

if len(tiles_dataset) == 0:
    sys.exit('no records found.')

if args['limit_scans']:
    num_scans = min(args['limit_scans'], len(tiles_dataset))
    logging.warning(f'limit scans: operations are limited to the first {num_scans} scans')
else:
    num_scans = len(tiles_dataset)

selected_scans = None
if args['select_scans']:
    selected_scans = args['select_scans'].split(',')
    logging.warning(f'filtering for selected scans: {",".join(selected_scans)}')

if args['base_dataset']:
    base_dataset = args['base_dataset']
else:
    base_dataset = DEFAULT_BASE_DATASET
    logging.info(f'using default base dataset location: {base_dataset}')

scale_factor = 1.0
if args['scale']:
    scale_factor = args['scale']
    logging.info(f'using scale factor: {scale_factor}')

# validate scans existence
validated_tiles_dataset = validate_scans_existence(tiles_dataset[:num_scans], base_dataset)

# *** Summarized JSON creation ***
if args['output_json_dir']:

    output_dir = args['output_json_dir']
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for scan_idx, scan in enumerate(validated_tiles_dataset):
        if scan_idx >= num_scans:
            logging.warning(f'reached scans limit of {num_scans}')
            break

        scan_id = scan['scan_uuid']
        if selected_scans is not None and scan_id not in selected_scans:
            logging.debug(f'skipping unselected scan: {scan_id}')
            continue

        labels = []
        ROIs = []
        summary = {'scan_id': scan_id, 'labels': labels, 'ROIs': ROIs}

        # iterate over ROIs
        regions_list = scan['regions_list']
        for roi_idx, roi in enumerate(regions_list):
            rect = Rect(*roi['rect'])

            if scale_factor is not 1.0:
                prev_rect = rect
                rect = Rect(*(int(round(scale_factor*e)) for e in rect))
                logging.info(f'ROI scale: {repr(prev_rect)} -> {repr(rect)}')

            ROIs.append([*rect, f'ROI idx {roi_idx}'])

            tile_list = roi['tile_list']
            logging.info(f'{scan_id}: creating summarized JSON for {len(tile_list)} tiles')
            for tile in tile_list:
                top_left = (rect.x + int(round(scale_factor * tile['tile_top_left_x'])),
                            rect.y + int(round(scale_factor * tile['tile_top_left_y'])))
                bot_right = (top_left[0] + int(round(scale_factor * tile['tile_size'])),
                             top_left[1] + int(round(scale_factor * tile['tile_size'])))
                tile_size = tile['tile_size']
                tile_rect = Rect(*top_left, tile_size, tile_size)

                # sanity check: ensure bounding box coordinates are inside the ROI rectangle
                if not is_rect_contains_rect(rect, tile_rect):
                    logging.error(f'OOB error: ROI: {",".join(rect)} does not contain tile rect: {",".join(tile_rect)}')
                    continue

                text_label = tile['tile_metadata']['train_rate_class']
                labels.append([*tile_rect, text_label])

        # save JSON file
        target_dir = f'{output_dir}/{scan_id}'
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)

        labels_json_file = f'{target_dir}/labels.json'
        logging.info(f'{scan_id}: writing labels file: {labels_json_file}')
        with open(labels_json_file, 'w') as f:
            json.dump(summary, f)


# *** Debug Bounding Box Markings ***
if args['output_marks_dir']:

    output_dir = args['output_marks_dir']
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # traverse all scans in the list, save ROI images with rectangle marks.
    # images may be tiled due to JPEG's limits (65K x 65K)
    for scan_idx, scan in enumerate(validated_tiles_dataset):
        if scan_idx >= num_scans:
            logging.warning(f'reached scans limit of {num_scans}')
            break

        scan_id = scan['scan_uuid']
        if selected_scans is not None and scan_id not in selected_scans:
            logging.debug(f'skipping unselected scan: {scan_id}')
            continue

        legacy_id = scan['scan_id']
        scan_path = scan_path_unified(base_dataset, scan_id, legacy_id)
        dzi_file = f'{scan_path}/pyramid/pyramid.dzi'

        if not os.path.exists(dzi_file):
            logging.error(f'missing pyramid file: {dzi_file}')
            continue

        target_dir = f'{output_dir}/{scan_id}'
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)

        dzi = dzimage.DZImageFs.fromfile(dzi_file)

        # iterate over ROIs
        regions_list = scan['regions_list']
        for roi_idx, roi in enumerate(regions_list):
            rect = Rect(*roi['rect'])

            # read the ROI
            logging.info(f'[{scan_idx + 1}/{len(validated_tiles_dataset)}] Reading ROI: {rect.w}x{rect.h}'
                         f' at ({rect.x},{rect.y}) for scan id {scan_id}')
            roi_img = dzi.read(x=rect.x, y=rect.y, width=rect.w, height=rect.h)

            tile_list = roi['tile_list']
            logging.info(f'marking {len(tile_list)} tiles')
            for tile in tile_list:
                top_left = (tile['tile_top_left_x'], tile['tile_top_left_y'])
                bot_right = (top_left[0] + tile['tile_size'], top_left[1] + tile['tile_size'])
                tile_rect = Rect(*top_left, tile['tile_size'], tile['tile_size'])

                tile_desc = tile['tile_metadata']['train_rate_class']
                mark_rectangle(roi_img, tile_rect, text=tile_desc)

            marked_img_file = '{}/tiles_roi_{}_marked.jpg'.format(target_dir, roi_idx)
            logging.info('Saving as: {}'.format(marked_img_file))
            save_to_tiled_jpegs(marked_img_file, roi_img)
