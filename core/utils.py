import logging
import math
import subprocess
import sys
import tensorflow as tf
import numpy as np
from lib import pyramiddata


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    np_ser = tf.io.serialize_tensor(value).numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[np_ser]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def read_rois_from_pyramid(image_source, rois, cells_model, model_res,
                           use_bulk_read=True, thread_count=None, interpolation_type='linear'):
    """
    Read a list of ROIs/labels from an image source (pyramid), under configuration determined by
    a CellsModel instance. Image is rescaled to match the target model resolution parameter.
    :param image_source: ScanImageSource instance of scan data.
    :param rois: list of tuples: (x, y, w, h, label_str). note: the ROI defined by this tuple is used to calculate
                 the center point. the actual ROI which is read is configured by the CellsModel instance.
    :param cells_model: instance of CellsModel. Contains configuration such as target image size in pixels, etc.
    :param model_res: target resolution of the model, in mm/pixel. image source that are at another resolution are
                      resized to match the model res.
    :param use_bulk_read: use bulk read.
    :param thread_count: number of threads (or processes) for image read and resampling.
    :param interpolation_type: type of resize interpolation (cv2)
    :return: rois map
    """
    # level_scale_div is the the factor by which the image and labels are scaled (down) prior to taking it as
    # input. for example, cells detection work at quarter resolution (level_scale_div=4)
    level_scale_div = cells_model.get_level_scale_div()

    # input shape dimensions. tuple of (W, H, C)
    input_shape = cells_model.get_image_shape_in_dataset()

    # calculate resize factor, if input pyramid res is different than target model res
    if image_source.is_same_resolution(model_res):
        resize_factor = None
    else:
        image_res = image_source.get_resolution()
        resize_factor = image_res / model_res

        # suppress warnings in case of randomized reads using this function
        if len(rois) > 1:
            logging.warning(f'model res is different than image res ({image_res:.7f}). '
                            f'using resize factor: {resize_factor:.4f}')

    # read all labeled ROIs from input pyramid. resize if needed.
    rois_map = pyramiddata.read_rois_from_pyramid(rois=rois,
                                                  image_source=image_source,
                                                  target_size=(input_shape[0], input_shape[1]),
                                                  resize_factor=resize_factor,
                                                  level_scale_div=level_scale_div,
                                                  use_bulk_read=use_bulk_read,
                                                  thread_count=thread_count,
                                                  interpolation_type=interpolation_type)

    return rois_map


def get_lengths_array(tile_size, overlap, num_tiles, start_pos, end_pos):
    skip = tile_size - overlap
    lengths_array = [tile_size] * (num_tiles - 1)
    pos_start_of_last_tile = start_pos + (num_tiles - 1) * skip
    size_of_last_tile = end_pos - pos_start_of_last_tile

    if size_of_last_tile < overlap:
        lengths_array[-1] += size_of_last_tile
    else:
        lengths_array.append(size_of_last_tile)

    return lengths_array


def split_rois(large_roi, tile_size, overlap):
    """
    split large ROI into smaller overlapping ROIs.
    :param large_roi: tuple: (x, y, w, h) of the large ROI
    :param tile_size: int: size of the (square) roi
    :param overlap: int: overlap between rois
    :return: list of rois: [(x', y', roi_size, roi_size), ...] that cover the large_roi.
    """

    x0, y0, large_w, large_h = large_roi
    x_end = x0 + large_w
    y_end = y0 + large_h

    if large_w < overlap or large_h < overlap:
        raise ValueError(f'roi dimensions must be larger than the overlap size. ({large_w}x{large_h}), '
                         f'overlap={overlap}')

    # distance between adjacent overlapping ROIs
    skip = tile_size - overlap

    # number of horizontal and vertical smaller ROIs
    n_horizontal = math.ceil((large_w - overlap) / skip)
    n_vertical = math.ceil((large_h - overlap) / skip)

    roi_widths = get_lengths_array(tile_size, overlap, n_horizontal, x0, x_end)
    roi_heights = get_lengths_array(tile_size, overlap, n_vertical, y0, y_end)

    indexed_rois = {}
    for iy in range(len(roi_heights)):
        for ix in range(len(roi_widths)):
            x = x0 + ix * skip
            y = y0 + iy * skip
            roi_w = roi_widths[ix]
            roi_h = roi_heights[iy]
            indexed_rois[(ix, iy)] = (x, y, roi_w, roi_h)

    return indexed_rois


def expand_to_multiples_of(img, multiple, fill_value):
    """expands an rgb numpy image to dimensions that are multiples of a number.
       expanded space is filled with fill_value."""

    if img.shape[0] % multiple == 0 and img.shape[1] % multiple == 0:
        return img

    h, w = img.shape[0], img.shape[1]
    exp_h = int(math.ceil(h / multiple) * multiple)
    exp_w = int(math.ceil(w / multiple) * multiple)
    expanded_img = np.full(shape=(exp_h, exp_w, 3), fill_value=fill_value, dtype='uint8')
    expanded_img[0:h, 0:w, :] = img
    return expanded_img


def get_bbox_from_center_points(centers, bbox_size, label_str=""):
    """"
    Given center points (x,y), return label format as: (x_topleft, y_topleft, w, h, label)
    """
    bbox_w, bbox_h = bbox_size
    assert bbox_w % 2 == 0, "currently supporting only even sizes"
    assert bbox_h % 2 == 0, "currently supporting only even sizes"

    labels = []
    for center in centers:
        # center is a numpy int64, converted here to regular int so saving to JSON is easy
        x_c, y_c = int(center[0]), int(center[1])
        x = x_c - bbox_w // 2
        y = y_c - bbox_h // 2
        rect = [x, y, bbox_w, bbox_h, label_str]
        labels.append(rect)

    return labels


def execute(cmd, capture_output=False, ignore_failure=False, failure_exit_msg=None, verbose=False):
    if verbose:
        print("execute: {}".format(cmd))

    if capture_output:
        retcode, output = subprocess.getstatusoutput(cmd)
        if retcode != 0 and not ignore_failure:
            raise Exception("Command failed: %s" % cmd)
        return output

    else:   # output goes live on stdout
        error = subprocess.call(cmd, shell=True)
        if error != 0 and not ignore_failure:
            if failure_exit_msg:
                sys.exit(failure_exit_msg)
            else:
                raise Exception("Command failed: %s" % cmd)
