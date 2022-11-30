from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

def get_margin_rect(rect, margin):

    """
    :param rect: [min_x, min_y, max_x, max_y]
    :param margin: expand margin
    :return: margin rectangle
    """

    margin_rect = rect.copy()
    for idx in range(2):
        margin_rect[idx] -= margin
        margin_rect[idx + 2] += margin

    return margin_rect


def get_margin_roi_rect(roi_rect, margin):

    """
    :param roi_rect: [min_x, min_y, width, height]
    :param margin: expand margin
    :return: margin roi
    """

    margin_roi_rect = roi_rect.copy()
    for idx in range(2):
        margin_roi_rect[idx] -= margin
        margin_roi_rect[idx + 2] += (2 * margin)

    return margin_roi_rect


def region_to_rect(region):

    """
    :param region: [min_x, min_y, width, height]
    :return: rectangle [min_x, min_y, max_x, max_y]
    """
    rect = region.copy()
    rect[2] += rect[0]
    rect[3] += rect[1]

    return rect


def rect_to_region(rect):

    """
    :param rect: [min_x, min_y, max_x, max_y]
    :return: roi [min_x, min_y, width, height]
    """

    region = rect.copy()
    region[2] -= region[0]
    region[3] -= region[1]

    return region


def get_scaled_region(region, res_factor):

    """
    :param region: [min_x, min_y, width, height]
    :param res_factor: increase factor
    :return: scaled [min_x, min_y, width, height]
    """

    if math.fabs(1. - res_factor) < 1e-4:
        scaled_region = region
    else:
        x, y, w, h = tuple(region)
        scaled_roi_x = int(math.floor(float(x) * res_factor))
        scaled_roi_y = int(math.floor(float(y) * res_factor))
        scaled_roi_w = int(math.ceil(float(w) * res_factor))
        scaled_roi_h = int(math.ceil(float(h) * res_factor))
        scaled_region = [scaled_roi_x, scaled_roi_y, scaled_roi_w, scaled_roi_h]

    return scaled_region


def get_scaled_rect(rect, res_factor):

    """
    :param rectangle: [min_x, min_y, max_x, max_y]
    :param res_factor: increase factor
    :return: scaled [min_x, min_y, max_x, max_y]
    """

    region = region_to_rect(rect)
    scaled_region = get_scaled_region(region, res_factor)
    scaled_rect = rect_to_region(scaled_region)

    return scaled_rect

def point_in_rect(rect, point):
    """
    :param rect: [min_x, min_y, max_x, max_y]
    :param point: [x, y]
    :return: True for inclusion
    """
    in_rect = point[0] >= rect[0] and point[0] < rect[2] and point[1] >= rect[1] and point[1] < rect[3]
    return in_rect


