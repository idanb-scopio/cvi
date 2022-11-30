#!/usr/bin/env python3
# Tool for exporting conflicts, and related actions

import argparse
import json
import os
import sys
import yaml

from core.labels import is_point_inside_bbox, get_center_point, get_longform_label_mapping
from core.postprocess import euclidian_distance

COMMANDS = ['update-mapping', 'export-conflicts']


def get_output_dir(output_dir_arg):
    if output_dir_arg is None:
        return '.'

    if not os.path.exists(output_dir_arg):
        os.mkdir(output_dir_arg)

    return output_dir_arg


def export_conflicts(args):
    json_dataset = args['json_dataset']
    conflicts_meta = args['conflicts_meta']
    output_file = args['output_file']
    target_class = args['target_class']
    longform_map = get_longform_label_mapping()
    sample_type = args['sample_type']

    with open(json_dataset, 'r') as f:
        dataset = json.load(f)

    with open(conflicts_meta, 'r') as f:
        meta = f.readlines()

    exported_conflicts = {}
    for line in meta:

        # the second field (separated by whitespace) are the conflict details
        conflict_data = line.split()[1]

        conflict_arr = conflict_data.split(',')
        scan_id, x, y, w, h, predicted_class = conflict_arr[0:6]
        if predicted_class in longform_map:
            predicted_long_label = longform_map[predicted_class]
        elif predicted_class == 'neg':
            predicted_long_label = 'null'
        else:
            predicted_long_label = predicted_class

        scan_data = dataset[scan_id]
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        label_center_point = get_center_point(x, y, w, h)

        roi_list = scan_data['roi_list']
        selected_roi = None
        for roi in roi_list:
            roi_bbox = (roi['x'], roi['y'], roi['w'], roi['h'])
            if is_point_inside_bbox(label_center_point, roi_bbox):
                selected_roi = roi
                break

        roi = selected_roi
        if not roi:
            print(f'containing roi not found for: {scan_id} {x},{y},{w},{h},{predicted_class}')
            continue

        # locate the label
        db_id = None
        for label in roi['labels']:
            db_center_point = get_center_point(label['x_gl'], label['y_gl'], label['w'], label['h'])
            if euclidian_distance(label_center_point, db_center_point) <= 4.0:
                db_id = label['database_id']
                break

        if not db_id:
            print(f'not found in db: {scan_id} {x},{y},{w},{h},{predicted_class}')
            continue

        numeric_scan_id = scan_data['scan_id']
        session_id = roi['session']
        if session_id not in exported_conflicts:
            exported_conflicts[session_id] = {}

        if numeric_scan_id not in exported_conflicts[session_id]:
            exported_conflicts[session_id][numeric_scan_id] = []

        exported_entry = {
            'center_x': label_center_point[0],
            'center_y': label_center_point[1],
            'target_class': target_class,
            'predict_class': predicted_long_label,
            'database_id': db_id,
            'sample_type': sample_type,
        }

        exported_conflicts[session_id][numeric_scan_id].append(exported_entry)

    with open(output_file, 'w') as f:
        json.dump(exported_conflicts, f, indent=2)


def update_mapping(args):
    json_dataset_file = args['json_dataset']
    if not json_dataset_file:
        sys.exit(f'--json-dataset is mandatory for update-mapping command')

    if not os.path.exists(json_dataset_file):
        sys.exit(f'file not found: {json_dataset_file}')

    print(f'loading dataset: {json_dataset_file}')
    with open(json_dataset_file, 'r') as f:
        dataset = json.load(f)

    print(f'loaded {len(dataset)} scan entries.')

    output_dir = get_output_dir(args['output_dir'])
    output_map_file = f'{output_dir}/legacy_mappings.yml'
    if os.path.exists(output_map_file):
        print(f'loading existing mapping file: {output_map_file}')
        with open(output_map_file, 'r') as f:
            legacy_mappings = yaml.safe_load(f)
    else:
        legacy_mappings = {}

    for entry in dataset:
        scan_id = entry['scan_uuid']
        map_entry = {
            'legacy_id': entry['scan_id'],
            'ROIs': []
        }

        for roi in entry['roi_list']:
            roi_entry = {
                'x': roi['x'],
                'y': roi['y'],
                'w': roi['w'],
                'h': roi['h'],
                'session': roi['session']
            }
            map_entry['ROIs'].append(roi_entry)

        legacy_mappings[scan_id] = map_entry

    if os.path.exists(output_map_file):
        os.rename(output_map_file, f'{output_map_file}.prev')

    print(f'saving legacy mapping: {output_map_file}')
    with open(output_map_file, 'w') as f:
        yaml.dump(legacy_mappings, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', type=str, help=f'available commands: {",".join(COMMANDS)}')
    parser.add_argument('--json-dataset', type=str, help='v1 json dataset file')
    parser.add_argument('--map-file', type=str, help='.yml map file of: uuid, numeric id, session id, etc..')
    parser.add_argument('--output-dir', type=str, help='dir to store the output files')
    parser.add_argument('--conflicts-meta', type=str, help='conflicts .meta.txt file')
    parser.add_argument('--target-class', type=str, help='target class (ground truth)')
    parser.add_argument('--sample-type', type=str, help='FN, FP, Err')
    parser.add_argument('--output-file', type=str, help='JSON output file')
    args = vars(parser.parse_args())

    command = args['command']
    if command == 'update-mapping':
        update_mapping(args)
    elif command == 'export-conflicts':
        export_conflicts(args)
    else:
        sys.exit(f'unknown command: {command}')


if __name__ == '__main__':
    main()
