import logging
import math
import os
import cv2
from collections import namedtuple
import lib.dzimage

Rect = namedtuple('Rect', ['x', 'y', 'w', 'h'])


def is_rect_contains_rect(big, small):
    """
    Checks if a rectangle (big) fully contains another rectangle.
    :param big: The containing rectangle (Rect type)
    :param small: The contained rectangle (Rect type)
    :return: True / False
    """
    return small.x >= big.x and \
        small.y >= big.y and \
        small.x + small.w <= big.x + big.w and \
        small.y + small.h <= big.y + big.h


def make_histogram_sumstr(hist_dict):
    """takes a histogram (dictionary) as input, and returns a string with the following format:
       k1:v1 k2:v2 k3:v3
       keys are sorted."""
    s = ''
    for k in sorted(hist_dict.keys()):
        s += f'{k}:{hist_dict[k]} '
    return s[:-1]


def save_to_tiled_jpegs(filename, data, max_size=65000):
    """ save large numpy image array to JPEG. Work around the 65000x65000 max resolution
        by splitting the data into tiles. """
    # calculate the number of tiles in each axis
    num_tiles_x = math.ceil(data.shape[1] / max_size)
    num_tiles_y = math.ceil(data.shape[0] / max_size)

    # non split image retains its original filename
    if num_tiles_x == 1 and num_tiles_y == 1:
        cv2.imwrite(filename, data)
        return

    for iy in range(num_tiles_y):
        for ix in range(num_tiles_x):
            tiled_filename = f'{filename[:-4]}_x{ix}_y{iy}.jpg'

            x_s = ix * max_size
            x_e = min(x_s + max_size, data.shape[1])
            y_s = iy * max_size
            y_e = min(y_s + max_size, data.shape[0])
            logging.info(f'[X: {ix + 1}/{num_tiles_x}, Y: {iy + 1}/{num_tiles_y}] writing {tiled_filename}')
            cv2.imwrite(tiled_filename, data[y_s:y_e, x_s:x_e])


def scan_path_unified(basedir, scan_id, legacy_id):
    """returns the scan path, taking into account scan id (uuid) and legacy scan id (numeric)"""
    dzi_suffix = '/pyramid/pyramid.dzi'
    scan_path = f'{basedir}/{scan_id}'
    legacy_path = f'{basedir}/{legacy_id}'

    # first priority - scan_id (UUID)
    dzi_path = scan_path + dzi_suffix
    if os.path.exists(dzi_path):
        return scan_path

    # check if legacy id is relevant:
    # valid legacy scan id values are in the low thousands
    legacy_dzi_path = legacy_path + dzi_suffix
    if legacy_id < 10000 and os.path.exists(legacy_dzi_path):
        return legacy_path

    return None


def validate_scans_existence(tiles_dataset, scans_basedir):
    """
    Validates entries from dataset correspond to real pyramid data.
    :param tiles_dataset: data structure - list of scan related records
    :param scans_basedir: path to basedir of the pyramid data.
    :return: list of scan related records - only those which the pyramid data exist for them.
    """
    validated = []
    for idx in range(len(tiles_dataset)):
        entry = tiles_dataset[idx]

        # scan id is the scan UUID. Old numbers are referred to by 'old id' / 'legacy id'
        scan_id = entry['scan_uuid']
        legacy_id = entry['scan_id']

        scan_path = scan_path_unified(scans_basedir, scan_id, legacy_id)
        if scan_path is None:
            logging.warning(f'missing scan. id: {scan_id} (legacy id: {legacy_id})')
            continue

        dzi_file = f'{scan_path}/pyramid/pyramid.dzi'
        dzi = lib.dzimage.DZImageFs.fromfile(dzi_file)
        logging.info(f'{dzi.width}x{dzi.height}: {scan_path}')

        validated.append(entry)

    return validated


def save_image(image, filename, is_bgr=False):
    logging.info(f'saving: {filename}')
    if is_bgr:
        cv2.imwrite(filename, image)
    else:
        cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def get_gaussian_image(size):
    # gaussian data saved in png. TODO: calculate this from sigma
    gaussian_file = f'{os.path.dirname(os.path.abspath(__file__))}/gaussian-{size}x{size}-240.png'

    gaussian_image = cv2.imread(gaussian_file)
    gaussian_image = gaussian_image[:, :, 0]

    gaussian_image = cv2.resize(gaussian_image, dsize=(2*gaussian_image.shape[0], 2*gaussian_image.shape[0]))
    return gaussian_image
