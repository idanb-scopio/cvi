# ** labeling related operations and tools **
#

import logging
import os
import yaml
import cv2
import numpy as np
from collections import Counter
from lib import pyramiddata


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def get_shortform_label_mapping():
    with open(f'{SCRIPT_DIR}/label_alias_map.yml', 'r') as f:
        cfg = yaml.safe_load(f.read())

    smap_orig = cfg['shortform_map']
    smap_lower = {k.lower(): v.lower() for k, v in smap_orig.items()}
    return smap_lower


def get_longform_label_mapping():
    with open(f'{SCRIPT_DIR}/label_alias_map.yml', 'r') as f:
        cfg = yaml.safe_load(f.read())

    shortform_map = cfg['shortform_map']
    longform_map = {v.lower(): k for k, v in shortform_map.items()}
    return longform_map


def get_containing_tiles(x, y, tile_size, overlap):
    """
    Given point x,y tile size and overlap values, return a list of tile indices (Tx, Ty)
    which contain the point.
    """

    skip = tile_size - overlap
    tx = set()
    if x < overlap:
        tx.add(0)
    else:
        tx.add((x - overlap) // skip)
    tx.add(x // skip)

    ty = set()
    if y < overlap:
        ty.add(0)
    else:
        ty.add((y - overlap) // skip)
    ty.add(y // skip)

    tiles_indices = [(_x, _y) for _x in tx for _y in ty]
    return tiles_indices


def split_to_tiles(centers, tile_size, overlap):
    """
    Given a list of points on a plane (centers) tile size and overlap, return a dictionary with
    tile indices (tile_x, tile_y) as keys and a list of centers contained within the tile.
    """

    tiles_map = {}

    # sort x, y points into containing tiles
    for center in centers:
        x, y = center[0], center[1]
        tiles_indices = get_containing_tiles(x=x, y=y, tile_size=tile_size, overlap=overlap)

        for ti in tiles_indices:
            if ti not in tiles_map:
                tiles_map[ti] = []
            tiles_map[ti].append((x, y))

    return tiles_map


def create_centers_map(labels, flat_label_mapping, level_scale_div, resize_factor):

    # resize_factor is None when pyramid and model resolution are the same
    if resize_factor is None:
        resize_factor = 1.0

    # scale down to the resolution that the images were loaded from
    scaled_centers_map = {}
    for label in labels:
        # convert x,y,w,h bounding box of the label, to its center point
        center = pyramiddata.get_center_point(label[0], label[1], label[2], label[3])

        # scale down according to level_scale_div (downsize of factor 2^k)
        # and by resize_factor (ratio between pyramid res and model res)
        scaled_center = (round(resize_factor * center[0] / level_scale_div),
                         round(resize_factor * center[1] / level_scale_div))

        # value for each center is the int value of the mapped label
        try:
            scaled_centers_map[scaled_center] = flat_label_mapping[label[4].lower()]
        except KeyError:
            # labels which are not relevant to the model - i.e. don't appear in
            # flat_label_mapping - are ignored.
            continue

    return scaled_centers_map


def create_label_images(rois_map, flat_label_mapping, level_scale_div, source_image_size,
                        scale_down_factor, gaussian_image, resize_factor, scaled_centers_map=None):
    # resize_factor is None when pyramid and model resolution are the same
    if resize_factor is None:
        resize_factor = 1.0

    labels = rois_map.keys()

    if not scaled_centers_map:
        scaled_centers_map = create_centers_map(labels, flat_label_mapping, level_scale_div, resize_factor)

    # label image size adds margin for rotation augmentation
    li_size = source_image_size
    li_scaled_size = li_size // scale_down_factor

    # gaussian image size (image is a square)
    g_size = gaussian_image.shape[0]

    # extended source image size: with added margin for gaussian drawings
    ext_size = li_size + 2 * g_size

    # distance from the center of a given point at which gaussians are drawn
    dd = ext_size // 2 - g_size // 2

    tile_size = 10 * ext_size
    overlap = ext_size
    tiles_map = split_to_tiles(centers=scaled_centers_map.keys(),
                               tile_size=tile_size,
                               overlap=overlap)

    # draw images
    rois_with_labels_map = {}
    for label in labels:

        # calculate the scaled center for this point
        center = pyramiddata.get_center_point(label[0], label[1], label[2], label[3])
        scaled_center = (round(resize_factor * center[0] / level_scale_div),
                         round(resize_factor * center[1] / level_scale_div))

        # get the tile(s) containing the current point
        cx, cy = scaled_center[0], scaled_center[1]
        label_int = scaled_centers_map[(cx, cy)]
        if label_int is None:
            logging.warning(f'ignored label point: invalid mapping for: {[*label]}')
            continue

        # list of sets, each set contains the tiles indices in which a given point is contained.
        # 4 corner points distanced "drawing_distance" from the center return which tile contains them
        # given that drawing_distance is smaller than the overlap, there will always be at least one tile
        # which contains all the 4 of them.
        containing_tiles = []
        containing_tiles.append(set(get_containing_tiles(x=cx-dd, y=cy-dd, tile_size=tile_size, overlap=overlap)))
        containing_tiles.append(set(get_containing_tiles(x=cx-dd, y=cy+dd, tile_size=tile_size, overlap=overlap)))
        containing_tiles.append(set(get_containing_tiles(x=cx+dd, y=cy-dd, tile_size=tile_size, overlap=overlap)))
        containing_tiles.append(set(get_containing_tiles(x=cx+dd, y=cy+dd, tile_size=tile_size, overlap=overlap)))
        tile_idx = set.intersection(*containing_tiles).pop()

        ext_label_image = np.zeros(shape=(ext_size, ext_size), dtype='uint8')

        # top left corner of the extended label image, calculated from the center.
        ext_im_tl = cx - ext_size // 2, cy - ext_size // 2

        # draw the current point, if it's a 'cell' label
        if label_int == 1:
            # gaussian top left corner, in extended image coordinates
            g_tl_rel = ext_size // 2 - g_size // 2, ext_size // 2 - g_size // 2
            ext_label_image[g_tl_rel[1]:g_tl_rel[1]+g_size, g_tl_rel[0]:g_tl_rel[0]+g_size] = gaussian_image

        # traverse all neighbour center points
        for cx_n, cy_n in tiles_map[tile_idx]:

            # skip self
            if cx_n == cx and cy_n == cy:
                continue

            # no_cell don't draw gaussians
            if scaled_centers_map[(cx_n, cy_n)] == 0:
                continue

            # for cells that should be drawn, do it if in drawing range
            if abs(cx - cx_n) <= dd and abs(cy - cy_n) <= dd:
                # gaussian top left corner, in extended image coordinates
                g_tl_rel = cx_n - ext_im_tl[0] - g_size // 2, cy_n - ext_im_tl[1] - g_size // 2
                ext_label_image[g_tl_rel[1]:g_tl_rel[1]+g_size, g_tl_rel[0]:g_tl_rel[0]+g_size] = np.maximum(
                    ext_label_image[g_tl_rel[1]:g_tl_rel[1]+g_size, g_tl_rel[0]:g_tl_rel[0]+g_size],
                    gaussian_image)

        label_image = ext_label_image[g_size:g_size+li_size, g_size:g_size+li_size]
        label_image_scaled = cv2.resize(label_image, dsize=(li_scaled_size, li_scaled_size))
        rois_with_labels_map[label] = (rois_map[label], label_image_scaled)

    return rois_with_labels_map


def apply_labels_remap(label_mapping, label_remap, str_labels=False):
    """ apply remap (group mapping) and then return the resulting label -> int map (dict)."""
    result_mapping = {}

    # label mapping, keys in lowercase
    label_mapping_lc = {k.lower(): v for k, v in label_mapping.items()}
    final_classes = label_mapping_lc.keys()

    for orig_lbl, mapped_lbl in label_remap.items():
        if mapped_lbl.lower() not in final_classes:
            raise ValueError(f'{mapped_lbl.lower()} not found in label_mapping')

        result_mapping[orig_lbl.lower()] = mapped_lbl.lower()

    # add the trivial class_str -> class_str mapping
    for class_lbl in final_classes:
        if class_lbl not in result_mapping:
            result_mapping[class_lbl] = class_lbl

    # convert value class_str to class_num
    if not str_labels:
        for orig_lbl, mapped_str_label in result_mapping.items():
            result_mapping[orig_lbl] = label_mapping_lc[mapped_str_label]

    return result_mapping


def create_class_debug_titles(label_mappings, truncate_text=5):
    """
    Create a short text describing the class number and class name. Names may be truncated.
    :param label_mappings: dictionary of class_num (int) -> class_desc (str)
    :param truncate_text: (int) max length of text.
    :return: dictionary of class_num (int) -> title (str) with values of '{class_num}:{class_str}'
    """
    debug_titles = {}
    for k, v in label_mappings.items():
        title = f'{v}:{k[:truncate_text]}'
        debug_titles[v] = title

    return debug_titles


def generate_label_stats(label_counters, label_mapping, label_remap):
    flat_total_counter = Counter()

    # summarize all worker counters
    for lc in label_counters:
        flat_total_counter += lc

    total_counter = {}
    for label in label_mapping:
        total_counter[label] = 0

    for unmapped_label in flat_total_counter:
        if unmapped_label in label_remap:
            mapped_label = label_remap[unmapped_label]
        else:
            mapped_label = unmapped_label

        total_counter[mapped_label] += flat_total_counter[unmapped_label]

    stats = {'labels_stats': total_counter,
             'flat_labels_stats': dict(flat_total_counter),
             'total': sum(flat_total_counter.values())}
    return stats


def is_point_inside_bbox(xy, bbox):
    """
    Returns if a point (x, y) is inside a bounding box specified by
    (x, y, w, h)
    :param xy: x, y coordinates
    :param bbox: x, y, w, h bounding box coordinates
    :return: True if inside, False otherwise
    """
    x, y = xy
    bx, by, bw, bh = bbox[0:4]
    if bx <= x < bx+bw and \
       by <= y < by+bh:
        return True
    return False


def get_center_point(x, y, w, h):
    center_x = x + w // 2
    center_y = y + h // 2
    return center_x, center_y

