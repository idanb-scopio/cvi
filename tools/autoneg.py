#!/usr/bin/env python3
# Dirt (negatives) example generator for detection ROIs

import argparse
import json
import os
import random
import sys

import numpy as np
import cv2

from core.labels import is_point_inside_bbox, get_center_point
from lib.scanimsrc import ScanImageSource, is_same_resolution
from lib.pyramiddata import read_roi_from_pyramid, read_rois_from_pyramid, \
    apply_blacklist_res_swap
from lib.mosaicsaver import MosaicSaver


# labels come in (x, y, w, h) format, in which bounding box size may not suite
# the model / content physical size, etc. Therefore, a new bounding box with
# the following bbox (square) size in pixels is "drawn" from the center points.
DEFAULT_CELLS_BBOX_SIZE = 100


# base resolution is what ties the 150 pixels to actual physical size (mm/pixel).
# given there are different scans with different resolutions, the diameter may
# need scaling. This scaling factor is calculated from the ratio of BASE_RESOLUTION
# and the scan resolution.
DEFAULT_BASE_RESOLUTION = 0.0002016

# default max number of samples (labels) to generate
DEFAULT_MAX_SAMPLES = 1000

# maximum number of placement attempts for a random bbox before quitting
MAX_RANDOM_ATTEMPTS = 100

# label string for the negative labels
NEG_LABEL_STR = 'dirt'


def generate_presence_mask(roi, labels, bbox_size):
    """
    Given an ROI, which is *fully* labeled - i.e. for WBC, there's no unlabeled WBC
    inside the ROI - automatically generate 'negative' / no_cell labels that fit
    non-labeled area.
    :param roi: fully labeled region of interest
    :param labels: list of labels within the ROI
    :param bbox_size: the size of the cells, in pixels
    :param neg_label: string label for the negative example
    :return: list of labels
    """
    roi_x, roi_y, roi_w, roi_h = roi

    # presence mask is used to mark presence of all (present and generated) labels
    # 0 - empty, 1 - occupied by a label
    # note: the size of the label is determined by bbox_size
    presence_mask = np.zeros(shape=(roi_h, roi_w), dtype='uint8')

    # populate existing labels into the presence mask
    for label in labels:

        label_center_point = get_center_point(*label[0:4])

        # skip labels that are not in the roi (there may be multiple rois)
        if not is_point_inside_bbox(label_center_point, roi):
            continue

        # transform label bbox to have the bbox_size size
        x_c, y_c = label_center_point
        x = x_c - bbox_size // 2
        y = y_c - bbox_size // 2
        adjusted_label = (x, y, bbox_size, bbox_size)

        # mask coordinates: bounding box x, y start and end offsets
        mc_xs = max(0, x - roi_x)
        mc_xe = min(mc_xs + bbox_size, roi_w)

        mc_ys = max(0, y - roi_y)
        mc_ye = min(mc_ys + bbox_size, roi_h)

        presence_mask[mc_ys:mc_ye, mc_xs:mc_xe] = 1

    return presence_mask


def find_empty_bbox(p_mask, bbox_size, max_attempts):
    """
    Given a presence mask array and bounding box size, find_empty_box returns
    a bounding box (x, y, w, h) such that it doesn't intersect any place which
    contains other labels.
    It is implemented using random attempts of bbox placement, until a free one
    is found.
    :param p_mask: 2d array of presence mask (1 - present, 0 - free), ROI sized.
    :param bbox_size: size of the bounding box (square), in pixels.
    :param max_attempts: if no available space was found after N attempts, return None.
    :return: (x, y, w, h) of the bounding box, or None.
    """
    roi_h, roi_w = p_mask.shape

    for i in range(max_attempts):
        # x,y are the top left coordinates of the bounding box, randomly chosen
        x = random.randint(0, roi_w - bbox_size)
        y = random.randint(0, roi_h - bbox_size)
        x_e = x + bbox_size
        y_e = y + bbox_size

        candidate = p_mask[y:y_e, x:x_e]
        if np.all(candidate == 0):
            return x, y, bbox_size, bbox_size

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_input', type=str, help='ground truth cvi json file with fully labeled ROIs')
    parser.add_argument('--output-dir', type=str, help='output dir for generated examples')
    parser.add_argument('--max-samples', type=int, help='maximum number of samples to generate. default: '
                        f'{DEFAULT_MAX_SAMPLES}')
    parser.add_argument('--cells-bbox-size', type=int, help=f'size of the (square) bounding box, in pixels. '
                                                            f'default: {DEFAULT_CELLS_BBOX_SIZE}')
    parser.add_argument('--base-resolution', type=float, help=f'resolution (mm/pixel) which ties cells-bbox-size to'
                                                              f'physical size. default: {DEFAULT_BASE_RESOLUTION}')
    parser.add_argument('--debug-pyramid-dir', type=str, help='pyramid dir of the scan, for debug image creation')
    args = vars(parser.parse_args())

    json_input = args['json_input']
    if not os.path.exists(json_input):
        sys.exit(f'file not found: {json_input}')

    try:
        with open(json_input, 'r') as f:
            label_dataset = json.load(f)
    except Exception as e:
        sys.exit(f'error reading input file: {str(e)}')

    if args['output_dir']:
        output_dir = args['output_dir']
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
    else:
        dir_name = os.path.dirname(json_input)
        if dir_name:
            output_dir = dir_name
        else:
            output_dir = '.'

    if args['max_samples']:
        max_samples = args['max_samples']
    else:
        max_samples = DEFAULT_MAX_SAMPLES

    if args['cells_bbox_size']:
        cells_bbox_size = args['cells_bbox_size']
    else:
        cells_bbox_size = DEFAULT_CELLS_BBOX_SIZE

    if args['base_resolution']:
        base_resolution = args['base_resolution']
    else:
        base_resolution = DEFAULT_BASE_RESOLUTION

    scan_id = label_dataset['scan_id']
    pyr_res = label_dataset['pyramid_resolution']
    pyr_res_wa = apply_blacklist_res_swap(pyr_res)
    if pyr_res != pyr_res_wa:
        print(f'blacklisted resolution for scan id: {scan_id}: {pyr_res} fixed to: {pyr_res_wa}')
        pyr_res = pyr_res_wa

    if is_same_resolution(pyr_res, base_resolution):
        resize_factor = 1.0
    else:
        resize_factor = pyr_res / base_resolution

    if resize_factor != 1.0:
        cells_bbox_size = int(cells_bbox_size / resize_factor)

    print(f'using cells bounding box size: {cells_bbox_size}')

    rois = label_dataset['ROIs']
    labels = label_dataset['labels']

    neg_labels = []
    debug_pyramid_dir = args['debug_pyramid_dir']

    for roi in rois:
        roi_x = roi[0]
        roi_y = roi[1]
        p_mask = generate_presence_mask(roi=roi[0:4],
                                        labels=labels,
                                        bbox_size=cells_bbox_size)

        # if debug_pyramid_dir:
        #     image_source = ScanImageSource(debug_pyramid_dir, pyr_res)
        #
        #     # read image from scan in the ROI
        #     scan_image = read_roi_from_pyramid(image_source, roi)
        #
        #     dbg_w = min(p_mask.shape[1], 20000)
        #     dbg_h = min(p_mask.shape[0], 20000)
        #     scan_im_dbg = scan_image[0:dbg_h, 0:dbg_w].astype('float32')
        #     p_mask_dbg = p_mask[0:dbg_h, 0:dbg_w]
        #
        #     p_mask_im = np.empty(shape=(dbg_h, dbg_w, 3), dtype='float32')
        #     p_mask_im[p_mask_dbg == 1] = (255, 255, 255)
        #     p_mask_im = 0.5 * p_mask_im + 0.5 * scan_im_dbg
        #     p_mask_im = p_mask_im.astype('uint8')
        #
        #     debug_image_file = f'{scan_id}_debug_mask.jpg'
        #     cv2.imwrite(debug_image_file, p_mask_im)

        print(f'generating samples for roi: {roi}.')

        for i in range(max_samples):
            bbox = find_empty_bbox(p_mask, cells_bbox_size, MAX_RANDOM_ATTEMPTS)

            # None is returned when no place was found for the bounding box after
            # max_attempts tries.
            if not bbox:
                break

            x, y, w, h = bbox

            # mark the bounding box in the placement mask
            p_mask[y:y+h, x:x+w] = 1

            # add the label to the labels list
            neg_labels.append((roi_x + x, roi_y + y, w, h, NEG_LABEL_STR))

    if not neg_labels:
        sys.exit('no labels were generated.')

    output_filename = f'{output_dir}/{scan_id}_autoneg.json'
    ds = label_dataset.copy()
    ds['labels'] = neg_labels
    with open(output_filename, 'w') as f:
        json.dump(ds, f, indent=4)

    if debug_pyramid_dir:
        mosaic_saver = MosaicSaver(sub_image_shape=(cells_bbox_size, cells_bbox_size, 3),
                                   mosaic_w=10000, mosaic_h=10000,
                                   output_dir='.',
                                   tag='autogen',
                                   save_meta=True)
        image_source = ScanImageSource(debug_pyramid_dir, pyr_res)
        rois_map = read_rois_from_pyramid(rois=neg_labels,
                                          image_source=image_source,
                                          target_size=(cells_bbox_size, cells_bbox_size),
                                          resize_factor=None)

        for lbl, img in rois_map.items():
            mosaic_saver.add_image(img, meta=str(lbl))
        mosaic_saver.save()


if __name__ == '__main__':
    main()

