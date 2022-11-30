#!/usr/bin/env python3

import argparse
import json
import os
import sys

import cv2
from lib.scanimsrc import ScanImageSource


# log_2_K for pyramid scale argument to level selection.
# only values that are in the keys are allowed for level scale pyramid reads.
PYRAMID_SCALE_TO_LEVEL = {
    1: 0,
    2: 1,
    4: 2,
    8: 3,
}


DEFAULT_JPEG_QUALITY = 95


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pyramid-dir', type=str, help='pyramid data dir', required=True)
    parser.add_argument('--roi', type=str, help='region of interest in the following format: x,y,w,h', required=True)
    parser.add_argument('-o', '--output-dir', type=str, help='output dir', default='.', required=False)
    parser.add_argument('--downsize', type=int, help='downsize by a factor of N. for example: 4 (wbc_det).')
    parser.add_argument('-q', '--quality', type=int, help=f'jpeg quality. default: {DEFAULT_JPEG_QUALITY}')
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
    image_source = ScanImageSource(pyramid_dir, 1.0)

    if args['quality']:
        quality = args['quality']
    else:
        quality = DEFAULT_JPEG_QUALITY

    level_scale_div = args['downsize']
    if level_scale_div is None:
        level_scale_div = 1

    if level_scale_div not in PYRAMID_SCALE_TO_LEVEL:
        raise ValueError(f'invalid level scale divider. allowed values: {PYRAMID_SCALE_TO_LEVEL.keys()}')

    level = image_source.base_level - PYRAMID_SCALE_TO_LEVEL[level_scale_div]

    orig_roi = roi.copy()
    if level_scale_div > 1:
        roi = [e // level_scale_div for e in roi]

    x, y, w, h = roi
    scan_id = image_source.infer_scan_id_from_src()

    # read ROI from pyramid
    print(f'reading data from pyramid: scan_id={scan_id} @ ({x},{y}) {w}x{h}')
    roi_img = image_source.read(x, y, w, h, level=level)

    output_file = f'{output_dir}/{scan_id}.jpg'
    cv2.imwrite(output_file, roi_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    print(f'saved: {output_file}')

    metadata = {'scan_id': scan_id,
                'roi': orig_roi,
                'level_scale_div': level_scale_div}

    metadata_file = f'{output_dir}/export-meta-{scan_id}.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f'saved: {metadata_file}')


if __name__ == '__main__':
    main()
