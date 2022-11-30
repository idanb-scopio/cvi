#!/usr/bin/env python3

import argparse
import json
import os
import logging
from tabulate import tabulate

MODEL_STAGE_MAP = {
    0: 'smudge',
    1: 'wbc_5_class',
    2: 'neu_leftshift',
    3: 'nrbc',
    4: 'blast_mon_lym',
    5: 'lgl_al',
    6: 'plasma'
}


def main(raw_args=None):
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] -%(levelname).1s- : %(message)s',
                        datefmt='%d-%m-%Y %H:%M:%S')

    # Load previous version of "result_dict.json" and re-save it as version 2
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--json_in", type=str)
    parser.add_argument("-o", "--out_path", type=str, default='.')
    parser.add_argument("-s", "--scan_id", type=str, default='')
    parser.add_argument('--pred-history', action='store_true', help='add prediction score history to json')

    # args = parser.parse_args()
    args = vars(parser.parse_args(raw_args))
    json_in = args['json_in']
    out_path = args['out_path']
    scan_id = args['scan_id']

    save_prediction_history = True if args['pred_history'] else False
    if save_prediction_history:
        debug_log_dir = os.path.join(out_path, 'debug')
        if not os.path.exists(debug_log_dir):
            logging.info(f"Creating directory {debug_log_dir}")
            os.makedirs(debug_log_dir)
        headers = ["uuid", "x", "y", "w", "h", "predicted label", "deciding model", "scores"]

    logging.info(f'Loading result_dict file of version 1: {json_in}')
    with open(json_in, 'r') as f:
        results_dict = json.load(f)

    um_per_pixel = results_dict['um_per_pixel']
    rois = results_dict['region']
    rois.append('ROI idx 0')
    output_dict = {'pyramid_resolution': um_per_pixel*1e-3,
                   'ROIs': [rois]}
    labels = []
    history_log = []
    for prediction in range(len(results_dict['labels'])):
        prediction_dict = results_dict['labels'][prediction]
        current_label = [prediction_dict['x_gl'], prediction_dict['y_gl'], prediction_dict['w'], prediction_dict['h'],
                         prediction_dict['label_name'][0]]
        labels.append(current_label)
        # optional - save to log file
        if save_prediction_history:
            history_log.append([scan_id, prediction_dict['x_gl'], prediction_dict['y_gl'],
                                prediction_dict['w'], prediction_dict['h'], prediction_dict['label_name'][0],
                                MODEL_STAGE_MAP[prediction_dict['classification_stage']], prediction_dict['score']])
    output_dict['labels'] = labels

    # Add scan_id to dict
    if scan_id:
        if 'scan_' in scan_id:
            scan_num = scan_id.replace('scan_', '')
            output_dict['scan_id'] = scan_num
        else:
            output_dict['scan_id'] = scan_id
    else:
        output_dict['scan_id'] = 'unknown_scan_id'

    # Save json file of v2
    json_file = f"{out_path}/{output_dict['scan_id']}.json"
    logging.info(f'Saving results file in version 2 format to: {json_file}')
    with open(json_file, 'w') as f:
        json.dump(output_dict, f, indent=4)

    # save prediction history log file
    if save_prediction_history:
        logging.info(f'Saving log score history to {debug_log_dir}/{scan_id}.log')
        with open(f'{debug_log_dir}/{scan_id}.log', 'w') as f:
            f.write(tabulate(history_log, headers=headers))


if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        raise err
