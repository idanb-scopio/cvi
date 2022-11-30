#!/usr/bin/env python3

"""
"Cut" a region of interest out of a pyramid and save as jpeg image.
"""

import argparse
import copy
import json
import logging
import os
import sys

import cv2

from lib import dzimage
from lib.debugutils import mark_rectangle
from lib.dsutils import Rect


def main():
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] -%(levelname).1s- : %(message)s',
                        datefmt='%d-%m-%Y %H:%M:%S')

    parser = argparse.ArgumentParser()
    parser.add_argument('--pyramid-dir', type=str, help='pyramid data dir', required=True)
    parser.add_argument('--json-dataset', type=str, help='json dataset file', required=True)
    parser.add_argument('--roi', type=str, help='region of interest in the following format: x,y,w,h', required=True)
    parser.add_argument('--output-dir', type=str, help='output dir', default='.', required=False)
    parser.add_argument('--bench-only', help='benchmark pyramid reads. no actual writes', action='store_true')
    args = vars(parser.parse_args())

    output_dir = args['output_dir']
    if not os.path.exists(output_dir):
        sys.exit(f'error: output dir does not exist: {output_dir}')

    roi_str = args['roi']
    try:
        roi = roi_str.split(',')
        roi = [int(e) for e in roi]
    except Exception as e:
        sys.exit(f'error: invalid roi format: {roi_str} ({str(e)})')

    pyramid_dir = args['pyramid_dir']
    dzi_file = f'{pyramid_dir}/pyramid/pyramid.dzi'
    if not os.path.exists(dzi_file):
        sys.exit(f'error: dzi file does not exist: {dzi_file}')
    dzi = dzimage.DZImageFs.fromfile(dzi_file)

    json_dataset_file = args['json_dataset']
    if not os.path.exists(json_dataset_file):
        sys.exit(f'error: json dataset file does not exist: {json_dataset_file}')

    with open(json_dataset_file, 'r') as f:
        json_dataset = json.load(f)

    # ROI boundaries as requested by the user
    tl_x, tl_y, br_x, br_y = roi[0], roi[1], roi[0]+roi[2], roi[1]+roi[3]

    # each label is marked by top left x,y and width, height. ROI's bottom right may be expanded to contain
    # overflowing label bounding boxes.
    ebr_x, ebr_y = br_x, br_y

    # get all points with centers contained in the selected ROI
    labels = json_dataset['labels']
    roi_labels = []
    for label in labels:
        x, y, w, h, lbl = label[0], label[1], label[2], label[3], label[4]

        if tl_x <= x <= br_x and tl_y <= y <= br_y:

            if x + w > ebr_x:
                ebr_x += w

            if y + h > ebr_y:
                ebr_y += h

            x_rel = x - tl_x
            y_rel = y - tl_y
            roi_labels.append((x_rel, y_rel, w, h, lbl))

    # adjust selected ROI to contain all points bounding boxes
    if br_x != ebr_x or br_y != ebr_y:
        print(f'ROI adjusted to contain all points bounding boxes: {[tl_x, tl_y, br_x, br_y]} -> '
              f'{[tl_x, tl_y, ebr_x, ebr_y]}')

    # load the image according to the effective ROI
    roi_w, roi_h = ebr_x-tl_x, ebr_y-tl_y
    logging.info(f'reading pyramid data, size: {roi_w}x{roi_h}')
    roi_img = dzi.read(tl_x, tl_y, roi_w, roi_h)
    logging.info(f'done')

    if args['bench_only']:
        sys.exit(0)

    # create a marked image
    roi_img_marked = roi_img.copy()
    for label in roi_labels:
        x, y, w, h, lbl = label[0], label[1], label[2], label[3], label[4]
        rect = Rect(x=x, y=y, w=w, h=h)
        mark_rectangle(image=roi_img_marked, rect=rect, text=lbl)

    out_dataset = copy.deepcopy(json_dataset)
    out_dataset['labels'] = roi_labels
    out_dataset['ROIs'] = [[tl_x, tl_y, ebr_x-tl_x, ebr_y-tl_y, 'Cropped_ROI']]
    scan_id = json_dataset['scan_id']

    out_json_file = f'{output_dir}/{scan_id}.json'
    if os.path.abspath(out_json_file) == os.path.abspath(json_dataset_file):
        sys.exit(f'error: output and input json dataset files are the same.')

    with open(out_json_file, 'w') as f:
        json.dump(out_dataset, f, indent=4)
    print(f'saved: {out_json_file}')

    unmarked_file = f'{output_dir}/{scan_id}.jpg'
    cv2.imwrite(unmarked_file, roi_img)
    print(f'saved: {unmarked_file}')

    marked_file = f'{output_dir}/marked-{scan_id}.jpg'
    cv2.imwrite(marked_file, roi_img_marked)
    print(f'saved: {marked_file}')


if __name__ == '__main__':
    main()
