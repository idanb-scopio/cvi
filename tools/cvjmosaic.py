#!/usr/bin/env python3
# Tool for creating mosaics out of CVI json files
import argparse
import glob
import json
import os
import sys

from lib.pyramiddata import apply_blacklist_res_swap
from lib.scanimsrc import ScanImageSource, is_same_resolution
from lib.mosaicsaver import MosaicSaver
from lib.pyramiddata import read_rois_from_pyramid

# default cell size and target resolution
CELL_SIZE = 96
BASE_RESOLUTION = 0.0002016


def json_to_mosaic(json_file, pyramids_dir, output_dir, progress=None):

    with open(json_file, 'r') as f:
        scan_data = json.load(f)

    scan_id = scan_data['scan_id']
    pyr_res = scan_data['pyramid_resolution']
    pyr_res_wa = apply_blacklist_res_swap(pyr_res)
    if pyr_res != pyr_res_wa:
        print(f'warning: blacklisted resolution changed: {pyr_res} -> {pyr_res_wa}')
        pyr_res = pyr_res_wa

    if is_same_resolution(pyr_res, BASE_RESOLUTION):
        resize_factor = 1.0
    else:
        resize_factor = pyr_res / BASE_RESOLUTION

    pyramid_dir = f'{pyramids_dir}/{scan_id}'
    image_source = ScanImageSource(pyramid_dir, pyr_res)

    labels = scan_data['labels']

    # read label images from pyramid
    if progress:
        progress_str = f'[{progress[0]}/{progress[1]}] '
    else:
        progress_str = ''

    print(f'{progress_str}{scan_id}: reading {len(labels)} samples')
    rois_map = read_rois_from_pyramid(rois=labels,
                                      image_source=image_source,
                                      target_size=(CELL_SIZE, CELL_SIZE),
                                      resize_factor=resize_factor,
                                      thread_count=16)

    sub_image_shape = (CELL_SIZE, CELL_SIZE, 3)
    tag = os.path.basename(json_file).split('.')[0]
    mosaic_saver = MosaicSaver(sub_image_shape=sub_image_shape,
                               mosaic_w=20000, mosaic_h=20000,
                               output_dir=output_dir,
                               tag=tag,
                               save_meta=True)

    for label in rois_map:
        meta_str = f'{",".join([str(e) for e in label])}'
        image = rois_map[label]
        mosaic_saver.add_image(image, meta=meta_str)

    mosaic_saver.save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_input', type=str, help='reference cvi json file, or directory containing jsons')
    parser.add_argument('--save-to', type=str, help='output dir')
    args = vars(parser.parse_args())

    json_input = args['json_input']

    json_files = []
    if os.path.isfile(json_input):
        json_files.append(os.path.abspath(json_input))
    elif os.path.isdir(json_input):
        json_files = glob.glob(f'{json_input}/*.json')
    else:
        sys.exit(f'invalid input: {json_input}')

    if args['save_to']:
        output_dir = args['save_to']
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
    else:
        output_dir = os.path.abspath('.')

    print(f'using cell size: {CELL_SIZE}x{CELL_SIZE}, base resolution: {BASE_RESOLUTION:.7f} mm/pixel.')
    pyramids_dir = os.environ['CVI_PYRAMIDS_ROOT']

    total = len(json_files)
    for idx, jf in enumerate(json_files):
        json_to_mosaic(jf, pyramids_dir, output_dir, progress=(idx+1, total))


if __name__ == '__main__':
    main()
