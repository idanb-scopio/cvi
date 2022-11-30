import math
import multiprocessing
from typing import Tuple

import cv2

from lib import dzimage
from lib.scanimsrc import ScanImageSource
from lib.epyramid import get_pyramid_reader


# mapping for known wrong resolution in the dataset, and the correct ones.
# floating point error here is expressed in the same units the pyramid
# fix table has.
FIX_TABLE_FP_RES_ERR = 1e-7
PYRAMID_RESOLUTION_FIX = {
    0.0001898: 0.000133071583818,
}


# log_2_K for pyramid scale argument to level selection.
# only values that are in the keys are allowed for level scale pyramid reads.
PYRAMID_SCALE_TO_LEVEL = {
    1: 0,
    2: 1,
    4: 2,
    8: 3,
}


def apply_blacklist_res_swap(res):
    for bad_res in PYRAMID_RESOLUTION_FIX:
        if abs(bad_res - res) < FIX_TABLE_FP_RES_ERR:
            return PYRAMID_RESOLUTION_FIX[bad_res]
    return res


def get_center_point(x, y, w, h):
    center_x = x + w // 2
    center_y = y + h // 2
    return center_x, center_y


def read_roi_from_pyramid(image_source, roi, level_scale_div=None, resize_factor=None):

    if not isinstance(image_source, ScanImageSource):
        image_source = get_pyramid_reader(image_source)

    if roi is None:
        roi = (0, 0, image_source.width, image_source.height)

    if level_scale_div is None:
        level_scale_div = 1

    if level_scale_div not in PYRAMID_SCALE_TO_LEVEL:
        raise ValueError(f'invalid level scale divider. allowed values: {PYRAMID_SCALE_TO_LEVEL.keys()}')

    level = image_source.base_level - PYRAMID_SCALE_TO_LEVEL[level_scale_div]

    x, y, w, h = roi[0], roi[1], roi[2], roi[3]
    roi_img = image_source.read(x=x, y=y, width=w, height=h, level=level)

    if resize_factor is not None and resize_factor != 1.0:
        w_ = int(w * resize_factor)
        h_ = int(h * resize_factor)
        roi_img = cv2.resize(roi_img, dsize=(w_, h_))

    return roi_img


def get_bounding_box(labels, margin, level_scale_div):
    """
    calculates a bounding box for a given list of labels.
    :param labels: list of labels [x, y, w, h, label_str]
    :param margin: margin to add around the calculated tight bounding box
    :param level_scale_div: divider for scaling
    :return: x, y, w, h of a bounding box
    """
    max_size = 100000000
    min_x, min_y = max_size, max_size
    max_x, max_y = 0, 0
    for label in labels:
        x, y, w, h, _ = label
        x = x // level_scale_div
        y = y // level_scale_div

        min_x = min(min_x, x - margin)
        min_y = min(min_y, y - margin)
        max_x = max(max_x, x + margin)
        max_y = max(max_y, y + margin)

    return min_x, min_y, max_x - min_x, max_y - min_y


def label_to_crop(label, target_size: Tuple[int, int],
                  level_scale_div: float = None,
                  resize_factor: float = None):
    if level_scale_div is None:
        level_scale_div = 1
    x, y, w, h = label[0], label[1], label[2], label[3]

    c_x, c_y = get_center_point(x, y, w, h)
    if level_scale_div > 1:
        c_x = c_x // level_scale_div
        c_y = c_y // level_scale_div

    # select roi size based on the target size
    t_w, t_h = target_size
    if resize_factor is not None:
        t_w = math.ceil(t_w / resize_factor)
        t_h = math.ceil(t_h / resize_factor)
    t_x = c_x - t_w // 2
    t_y = c_y - t_h // 2
    return t_x, t_y, t_w, t_h


def get_opencv_interpolation_type(interpolation_type):

    if interpolation_type == 'linear':
        opencv_interpolation_type = cv2.INTER_LINEAR
    elif interpolation_type == 'cubic':
        opencv_interpolation_type = cv2.INTER_CUBIC
    elif interpolation_type == 'lanczos':
        opencv_interpolation_type = cv2.INTER_LANCZOS4
    else:
        raise ValueError('Illegal interpolation %s'%interpolation_type)

    return opencv_interpolation_type


def read_rois_from_pyramid(rois, image_source, target_size, resize_factor=None, level_scale_div=None,
                           use_bulk_read=True, thread_count=None, interpolation_type='linear'):
    """
        Read list of labels (small ROIs) from a given pyramid.
        rois: list of ROIs: each roi is a tuple with the first 4 elements as (x, y, w, h, ...).
        dzi_path: path to pyramid.dzi
        target_size: tuple of (W, H) which specify the desired dimensions of the image.
                     note: effective ROI to read from pyramid is a bbox based on the center point. the input w,h are
                           used to calculate the center point.
        resize_factor: apply resize to the image, such that output dimensions is the target_size.
                       for instance, a resize_factor of 2 will get half the target size on each dimensions data, resized
                       to the target size.
        level_scale_div: read data from a lower level of the pyramid. levels are scaled down by a factor of 2 so only
                         values that are a power of 2 are allowed. the meaning of it is: read pyramid data scaled by
                         1 / level_scale_div.
        use_bulk_read: use optimized reads - by default True. For testing purposes (legacy read), set to False.

        Return:
            a dictionary mapping: tuple(label) -> numpy image
    """
    if not isinstance(image_source, ScanImageSource):
        image_source = get_pyramid_reader(image_source)

    if level_scale_div is None:
        level_scale_div = 1

    if level_scale_div not in PYRAMID_SCALE_TO_LEVEL:
        raise ValueError(f'invalid level scale divider. allowed values: {PYRAMID_SCALE_TO_LEVEL.keys()}')

    level = image_source.base_level - PYRAMID_SCALE_TO_LEVEL[level_scale_div]

    if use_bulk_read:
        crops = [dzimage.Rect(*label_to_crop(label, target_size,
                                             level_scale_div=level_scale_div,
                                             resize_factor=resize_factor))
                 for label in rois]
        cropped_images = image_source.bulk_read(crops, level=level, thread_count=thread_count)
    else:
        cropped_images = []
        for label in rois:
            t_x, t_y, t_w, t_h = label_to_crop(label, target_size,
                                               level_scale_div=level_scale_div,
                                               resize_factor=resize_factor)
            cropped_images.append(image_source.read(x=t_x, y=t_y, width=t_w, height=t_h, level=level))

    # Resize cropped images as necessary with resize factor
    cv_inter_type = get_opencv_interpolation_type(interpolation_type)
    if resize_factor is not None and resize_factor != 1.0:
        if thread_count is None:
            roi_images = []
            for roi_img in cropped_images:
                roi_img = cv2.resize(roi_img, (target_size[0], target_size[1]), interpolation=cv_inter_type)
                roi_images.append(roi_img)
        else:
            proc_count = thread_count
            resize_args = []
            for roi_img in cropped_images:
                resize_args.append((roi_img, (target_size[0], target_size[1]), cv_inter_type))
            with multiprocessing.Pool(proc_count) as p:
                roi_images = p.starmap(resize_image, resize_args)
    else:
        roi_images = cropped_images

    roi_map = {}
    for roi_img, pyramid_roi in zip(roi_images, rois):
        roi_map[tuple(pyramid_roi)] = roi_img

    return roi_map


def resize_image(in_img, target_size, open_cv_interpolation):

    resized_img = cv2.resize(in_img, (target_size[0], target_size[1]), interpolation=open_cv_interpolation)

    return resized_img