import logging
import math

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.morphology import closing, square
from skimage.measure import label, regionprops
from skimage.feature import peak_local_max
from scipy.spatial import distance

THRESHOLD_ABS = 10.0
PEAK_MIN_DISTANCE = 2


NEIGHBOUR_INDICES = [
    (-1, -1), (0, -1), (1, -1),
    (-1,  0),          (1,  0),
    (-1,  1), (0,  1), (1,  1)
]


def label_image_regions(density_img, binary_threshold=10.0):
    """
    This function takes a density image (WxHx1) containing 2d gaussians where objects
    are detected.
    """

    logging.debug('running gaussian filter')
    filtered_image = gaussian_filter(density_img, sigma=2.0/3.0)

    local_max_image = peak_local_max(filtered_image, min_distance=PEAK_MIN_DISTANCE, threshold_abs=THRESHOLD_ABS,
                                     exclude_border=False)

    return local_max_image


def np_to_list_of_tuples(detections: np.ndarray):
    return list(map(tuple, detections))


def euclidian_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def filter_duplicate_detections(detections_map, dist_threshold=25):
    if not detections_map:
        raise ValueError('empty detections map')

    # find the width and height of the tiles matrix
    # x for columns, y for rows
    max_ix = 0
    max_iy = 0
    for indices in detections_map:
        ix, iy = indices
        if ix > max_ix:
            max_ix = ix
        if iy > max_iy:
            max_iy = iy

    assert len(detections_map) == (max_ix + 1) * (max_iy + 1)

    filtered_detections_map = detections_map.copy()
    discarded_detections_map = {}
    for iy in range(max_iy + 1):
        for ix in range(max_ix + 1):

            for nbr in NEIGHBOUR_INDICES:
                nbr_ix = ix + nbr[0]
                nbr_iy = iy + nbr[1]

                # skip out of bound neighbours
                if not 0 <= nbr_ix <= max_ix or not 0 <= nbr_iy <= max_iy:
                    continue

                this_roi = filtered_detections_map[(ix, iy)]
                nbr_roi = filtered_detections_map[(nbr_ix, nbr_iy)]

                this_roi_filtered = []
                discarded_points = []
                for this_point in this_roi:
                    if any([euclidian_distance(this_point, nbr_point) <= dist_threshold for nbr_point in nbr_roi]):
                        discarded_points.append(this_point)
                    else:
                        this_roi_filtered.append(this_point)

                filtered_detections_map[(ix, iy)] = this_roi_filtered
                discarded_detections_map[(ix, iy)] = discarded_points

    return filtered_detections_map
