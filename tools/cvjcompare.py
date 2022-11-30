#!/usr/bin/env python3
import argparse
import glob
import json
import itertools
import math
import os
import sys
from collections import Counter
import pandas as pd

import numpy as np

from lib.pyramiddata import apply_blacklist_res_swap
from lib.scanimsrc import ScanImageSource, is_same_resolution
from lib.mosaicsaver import MosaicSaver
from models.cellsfactory import model_instance_from_yaml
from core.labels import is_point_inside_bbox
import lib.pyramiddata

from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

# shortform label map cache. loaded on first use.
flat_labels_map = {}
all_labels = []


# default threshold in pixels
DEFAULT_THRESHOLD = 65

# wide border is about throwing out cells that are very close to the ROI border
# and are partially out of bounds.
DEFAULT_WIDE_BORDER_THRESHOLD = 60   # in pixels


class ConfusionMosaics:

    def __init__(self, pyramids_root, cells_model, save_to):
        self.pyramids_root = pyramids_root
        self.save_to = save_to

        self.scan_id = None
        self.pyramid_res = None
        self.resize_factor = None

        self.sub_image_shape = cells_model.get_input_image_shape()
        self.level_scale_div = cells_model.get_level_scale_div()
        self.model_res = cells_model.get_model_res()
        self.mosaics = {}

        self.scan_image_source = None

    def set_current_pyramid(self, scan_id, pyramid_res):
        self.scan_id = scan_id
        self.pyramid_res = pyramid_res

        pyramid_dir = f'{self.pyramids_root}/{scan_id}'
        self.scan_image_source = ScanImageSource(image_source=pyramid_dir,
                                                 resolution=pyramid_res)

        if self.scan_image_source.is_same_resolution(self.model_res):
            self.resize_factor = 1.0
        else:
            self.resize_factor = self.scan_image_source.get_resolution() / self.model_res

    def add_label(self, true_pred_pair, label):
        if true_pred_pair in self.mosaics:
            mosaic = self.mosaics[true_pred_pair]
        else:
            tag = f'true_{true_pred_pair[0]}_pred_{true_pred_pair[1]}'
            mosaic = MosaicSaver(sub_image_shape=self.sub_image_shape,
                                 mosaic_w=5000, mosaic_h=5000,
                                 output_dir=self.save_to,
                                 tag=tag,
                                 save_meta=True)
            self.mosaics[true_pred_pair] = mosaic

        # read sample from pyramid
        rois_map = lib.pyramiddata.read_rois_from_pyramid(rois=[label],
                                                          image_source=self.scan_image_source,
                                                          target_size=self.sub_image_shape[0:2],
                                                          level_scale_div=self.level_scale_div,
                                                          resize_factor=self.resize_factor)
        image = rois_map[tuple(label)]
        meta_str = f'{self.scan_id},{",".join([str(e) for e in label[0:5]])} res={self.pyramid_res:.7f}'
        mosaic.add_image(image, meta=meta_str)

    def save_all(self):
        for mosaic in self.mosaics.values():
            mosaic.save()


def save_locations_match_list(filename, locations_match_list, scan_id):
    with open(filename, 'a') as f:
        for ref_entry, tst_entry in locations_match_list:
            log_str = f'{scan_id} '

            if ref_entry:
                log_str += f'R:{ref_entry[4]},{ref_entry[0:4]} '
            else:
                log_str += 'R:None '

            if tst_entry:
                log_str += f'T:{tst_entry[4]},{tst_entry[0:4]} '
            else:
                log_str += 'T:None '

            log_str += '\n'

            f.write(log_str)


def is_on_wider_border(point, border_rect, border_size):
    """
    Checks if a point in on a wider border, defined by the band between
    border_rect, with size (inward) of border_size pixels.
    :param point: (x,y) tuple
    :param border_rect: (x, y, w, h) tuple, where x and y are top left coordinates.
    :param border_size: border size in pixels
    :return: True if point lies on the border area, False otherwise.
    """
    x, y = point
    roi_xs, roi_ys, w, h = border_rect
    roi_ye = roi_ys + h
    roi_xe = roi_xs + w
    bs = border_size

    is_on_top_band = roi_ys <= y < roi_ys + bs
    is_on_bot_band = roi_ye - bs <= y < roi_ye
    is_on_lft_band = roi_xs <= x < roi_xs + bs
    is_on_rgt_band = roi_xe - bs <= x < roi_xe

    if is_on_top_band or is_on_bot_band or is_on_lft_band or is_on_rgt_band:
        return True

    return False


def print_precision_recall_line(c_matrix, labels_array, prefix):
    report_line = f'{prefix}'

    for idx, cls in enumerate(labels_array):
        tp_fp = np.sum(c_matrix, axis=0)[idx]
        if tp_fp > 0:
            precision = c_matrix[idx, idx] / tp_fp * 100
            p_str = f'p:{precision:6.2f}%'
        else:
            p_str = f'p: N/A    '

        tp_fn = np.sum(c_matrix, axis=1)[idx]
        if tp_fn > 0:
            recall = c_matrix[idx, idx] / tp_fn * 100
            r_str = f'r:{recall:6.2f}%'
        else:
            r_str = f'r: N/A    '

        cls_trunc = (cls[:8] + '..') if len(cls) > 10 else cls
        report_line += f' | {cls_trunc:10s} {p_str} {r_str}'

    print(report_line)


def plot_confusion_matrix(c_matrix, target_names, save_to_file, title='Confusion matrix', cmap=None, normalize=False):
    """
    c_matrix: confusion matrix from sklearn.metrics.confusion_matrix
    target_names:     given classification classes such as [0, 1, 2]
                      the class names, for example: ['high', 'medium', 'low']
    title:            the text to display at the top of the matrix
    cmap:             the gradient of the values displayed from matplotlib.pyplot.cm
    normalize:        If False, plot the raw numbers
                      If True, plot the proportions
    """
    plt.figure(figsize=(10, 10)), plt.title(title)

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        c_matrix = c_matrix.astype('float') / (1e-10 + c_matrix.sum(axis=1)[:, np.newaxis])
        c_matrix = np.around(c_matrix, 2)

    plt.imshow(c_matrix, interpolation='nearest', cmap=cmap)
    thresh = c_matrix.max() / 1.5 if normalize else c_matrix.max() / 2
    for i, j in itertools.product(range(c_matrix.shape[0]), range(c_matrix.shape[1])):
        plt.text(j, i, "{:,}".format(c_matrix[i, j]),
                 horizontalalignment="center",
                 color="white" if c_matrix[i, j] > thresh else "black",
                 fontsize='small' if len(target_names) > 15 else 'medium')

    plt.ylabel('True label'), plt.xlabel('Predicted label')
    plt.colorbar(fraction=0.046, pad=0.04), plt.tight_layout()

    print(f'saving figure to: {save_to_file}')
    plt.savefig(save_to_file)
    plt.close()


def save_confusion_xlsx(c_matrix, save_to, neg_label):

    eps = 1e-07
    # calculate cls statistics
    sum_predicted_tp_plus_fp = c_matrix.sum(axis=0).astype('float32')
    sum_predicted_tp_plus_fp[sum_predicted_tp_plus_fp == 0] = eps

    sum_target_tp_plus_fn = c_matrix.sum(axis=1).astype('float32')
    sum_target_tp_plus_fn[sum_target_tp_plus_fn == 0] = eps

    tp_cls = c_matrix.diagonal()

    precision = 100 * tp_cls / sum_predicted_tp_plus_fp
    recall = 100 * tp_cls / sum_target_tp_plus_fn

    # calc detection recall
    if neg_label is None:
        neg_label = 'neg'
    predicted_neg = c_matrix[:, all_labels.index(neg_label)]
    tp_det = sum_target_tp_plus_fn - predicted_neg
    recall_det = 100 * tp_det / sum_target_tp_plus_fn

    # calc precision without wbc, broken
    if 'broken' and 'wbc' in all_labels:
        target_broken = c_matrix[all_labels.index('broken'), :]
        target_wbc = c_matrix[all_labels.index('wbc'), :]
        sum_predicted_exclude_broken_wbc = sum_predicted_tp_plus_fp - target_broken - target_wbc
        precision_exclude_broken_wbc = 100 * tp_cls / sum_predicted_exclude_broken_wbc

    # save to excel
    df_pred = pd.DataFrame(data=c_matrix, columns=all_labels)
    df_pred.index = all_labels
    df_pred.loc['Total predicted'] = sum_predicted_tp_plus_fp.astype(int)
    df_pred.loc['Precision [%] (TP/TP+FP)'] = np.around(precision, 2)
    if 'broken' and 'wbc' in all_labels:
        df_pred.loc['Total (exclude wbc, broken)'] = sum_predicted_exclude_broken_wbc.astype(int)
        df_pred.loc['Precision [%] (exclude wbc, broken)'] = np.around(precision_exclude_broken_wbc, 2)

    df_pred['Total true'] = np.append(sum_target_tp_plus_fn, [0, 0, 0, 0]).astype(int)
    df_pred['TP det'] = np.append(np.around(tp_det, 0), [0, 0, 0, 0])
    df_pred['Recall det [%]'] = np.append(np.around(recall_det, 2), [0, 0, 0, 0])
    df_pred['TP'] = np.append(tp_cls, [0, 0, 0, 0])
    df_pred['Recall [%] (TP/TP+FN)'] = np.append(np.around(recall, 2), [0, 0, 0, 0])

    with pd.ExcelWriter(f'{save_to}/confusion_matrix.xlsx') as writer:
        df_pred.to_excel(writer, sheet_name='Number predictions')
    print(f'Saving excel report to: {save_to}/confusion_matrix.xlsx')


def get_center_point(x, y, w, h):
    center_x = x + w // 2
    center_y = y + h // 2
    return center_x, center_y


def process_label(label):

    # if long to short form label map is not loaded (first use), load it
    label_str = label[4].lower()
    if label_str in flat_labels_map:
        label_str = flat_labels_map[label_str]
    label_t = (*label[0:4], label_str)
    return label_t


def compare_locations(ref_set, tst_set, threshold):
    """
    return match list: [(R, T), (R, None), (None, T)]
    :param ref_set:
    :param tst_set:
    :param threshold:
    :return:
    """
    unmatched_ref_set = ref_set.copy()
    match_list = []

    for tst_label in tst_set:

        # for an empty unmatched ref set, rest of the tst set is consumed and
        # appends (None, tst_label) to the match list
        if not unmatched_ref_set:
            match_list.append((None, tst_label))
            continue

        # test center point
        tst_cx, tst_cy = get_center_point(*tst_label[0:4])

        # find minimum distance with respect to unmatched ref set
        min_distance = 1.0e9
        min_ref_label = None
        for ref_label in unmatched_ref_set:

            # test center point
            ref_cx, ref_cy = get_center_point(*ref_label[0:4])

            # calculate L2 distance
            dist = math.sqrt((ref_cx - tst_cx)**2 + (ref_cy - tst_cy)**2)

            if dist < min_distance:
                min_distance = dist
                min_ref_label = ref_label
            elif dist == min_distance and min_ref_label:
                m_cx, m_cy = get_center_point(*min_ref_label[0:4])
                if ref_cx < m_cx or (ref_cx == m_cx and ref_cy < m_cy):
                    min_ref_label = ref_label

        if min_distance <= threshold:
            # it's a match!
            match_list.append((min_ref_label, tst_label))

            # matched labels are removed from unmatched set
            unmatched_ref_set.remove(min_ref_label)
        else:
            # no match, this is a 'lone' tst cell
            match_list.append((None, tst_label))

    # after all tst cells we're iterated on, all remaining unmatched ref cells
    # are a 'lone' ref cells
    for ref_label in unmatched_ref_set:
        match_list.append((ref_label, None))

    return match_list


def calculate_detection_stats(locations_match_list, neg_label=None):
    """

    :param locations_match_list:
    :param neg_label: label to be treated as negative (must be lowercase), or None
    :return:
    """
    tp = 0
    fp = 0
    fn = 0

    total = 0
    for entry in locations_match_list:
        # entry is a tuple:
        # (ref, tst) for match     <=> TP
        # (ref, None) for lone ref <=> FN
        # (None, tst) for lone tst <=> FP
        ref_e, tst_e = entry

        if neg_label:
            if ref_e and ref_e[4].lower() == neg_label:
                ref_e = None
            if tst_e and tst_e[4].lower() == neg_label:
                tst_e = None

        if ref_e is None and tst_e is None:
            continue

        if ref_e is None:
            fp += 1
        elif tst_e is None:
            fn += 1
        else:
            tp += 1

        total += 1

    return tp, fp, fn, total


def gen_diffs_details(match_list, scan_header, save_to, neg_label=None, c_mosaics=None, labels_map=None):

    # true match list excludes all 'lone' points from ref/tst, remaining
    # only with matches on both sides.

    # confusion mosaics negative string
    if neg_label:
        cm_neg = neg_label
    else:
        cm_neg = 'neg'

    filtered_match_list = []
    labels_diff = []
    for entry in match_list:

        ref_e, tst_e = entry

        # only test entry - missing from ref
        if ref_e is None:
            label = list(tst_e)

            if c_mosaics:
                c_mosaics.add_label((cm_neg, label[4]), label)

            filtered_entry = ((*tst_e[0:4], cm_neg), tst_e)
            label[4] = f't: {label[4]}'
            label.append('orange')

        # only ref entry - missing from tst
        elif tst_e is None:
            label = list(ref_e)

            if c_mosaics:
                c_mosaics.add_label((label[4], cm_neg), label)

            filtered_entry = (ref_e, (*ref_e[0:4], cm_neg))
            label[4] = f'r: {label[4]}'
            label.append('red')

        # both ref and tst entries exist
        else:
            label = list(ref_e)

            if ref_e[4] == tst_e[4]:
                label.append('grey')
            else:
                if c_mosaics:
                    c_mosaics.add_label((ref_e[4], tst_e[4]), tst_e)

                label[4] = f'r: {ref_e[4]} t: {tst_e[4]}'
                label.append('pink')

            filtered_entry = entry

        filtered_match_list.append(filtered_entry)
        labels_diff.append(label)

    if save_to:
        det_save_to = f'{save_to}/labels_diff'
        if not os.path.exists(det_save_to):
            os.mkdir(det_save_to)

        cvi_ds = scan_header.copy()
        cvi_ds['labels'] = labels_diff
        scan_id = cvi_ds['scan_id']

        with open(f'{det_save_to}/{scan_id}.json', 'w') as f:
            json.dump(cvi_ds, f, indent=4)

    # without labels mapping taking from a model, return empty confusion map
    if not labels_map:
        return None

    y_true = []
    y_pred = []
    for entry in filtered_match_list:
        ref_e, tst_e = entry
        ref_lbl = ref_e[4]

        assert ref_lbl.lower() == ref_lbl, "not every label is lower case at this point"

        # some labels do not participate in this classification, and should be ignored also
        # in comparison. For example: there's no point in comparing 'dirt' labels for a 5-class
        # classifier.
        if ref_lbl not in labels_map:
            continue

        tst_lbl = tst_e[4]

        y_true.append(ref_lbl)
        y_pred.append(tst_lbl)

    if not y_true:
        return None

    c_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=all_labels)

    return c_matrix


def compare_json_pair(ref_json, tst_json, **kwargs):
    """
    Compares pair of json files.
    First, comparison is made on euclidean distances of the centers of
    the given labels (rois: x, y, w, h).
    Then for ref/tst labels with matching centers, a secondary comparison is made for
    their labels.
    :param ref_json: reference json file
    :param tst_json: test json file. must be the same res and scan id of the ref file.
    :param kwargs:
        threshold: (int) max euclidean distance in pixels between ref/tst centers, so that
                   they are considered the same cell.
        neg_label: (list of strings) list of labels that are considered negative examples,
                   so that in the statistics absence of this detection isn't considered a
                   misses cell.
    :return:
    """

    threshold = kwargs['threshold']
    neg_label = kwargs['neg_label']
    save_to = kwargs['save_to']
    c_mosaics = kwargs.get('c_mosaics', None)
    wide_border = kwargs.get('wide_border', None)
    ref_filter_mode = kwargs.get('ref_filter_mode', None)

    with open(ref_json, 'r') as f:
        ref_ds = json.load(f)

    with open(tst_json, 'r') as f:
        tst_ds = json.load(f)

    # black listed pyramid resolutions are swapped here for the correct ones.
    ref_res = apply_blacklist_res_swap(ref_ds['pyramid_resolution'])
    tst_res = apply_blacklist_res_swap(tst_ds['pyramid_resolution'])
    if not is_same_resolution(ref_res, tst_res):
        raise RuntimeError(f'resolution mismatch: {ref_json} vs. {tst_json}')

    if not ref_ds['scan_id'] == tst_ds['scan_id']:
        raise RuntimeError(f'scan_id mismatch: {ref_json} vs. {tst_json}')

    scan_id = ref_ds['scan_id']
    rois = ref_ds['ROIs']
    scan_header = ref_ds.copy()
    del scan_header['labels']

    if c_mosaics:
        c_mosaics.set_current_pyramid(scan_id, ref_res)

    ref_labels_set = set()
    for label in ref_ds['labels']:
        label_t = process_label(label)
        ref_labels_set.add(label_t)

    tst_labels_set = set()
    for label in tst_ds['labels']:
        label_t = process_label(label)
        tst_labels_set.add(label_t)

    locations_match_list = compare_locations(ref_set=ref_labels_set,
                                             tst_set=tst_labels_set,
                                             threshold=threshold)

    if ref_filter_mode:
        filtered_ds = ref_ds.copy()
        filtered_ds['labels'] = []

        for ref_e, tst_e in locations_match_list:
            if not ref_e or not tst_e:
                continue
            filtered_ds['labels'].append(tst_e)

        filtered_dir = f'{save_to}/filtered'
        if not os.path.exists(filtered_dir):
            os.mkdir(filtered_dir)

        with open(f'{filtered_dir}/{scan_id}.json', 'w') as f:
            json.dump(filtered_ds, f, indent=4)

        return {}, None

    if save_to:
        save_locations_match_list(f'{save_to}/locations_match_list.log', locations_match_list, scan_id)

    if wide_border:
        removed_points_log = ''

        # this list contains the original list minus borderline points
        updated_locations_match_list = []
        for entry in locations_match_list:
            ref_e, tst_e = entry
            # check for borderline points only when tst has a point that ref doesn't
            if not ref_e or not tst_e:
                if ref_e:
                    data_e = ref_e
                    origin = 'ref'
                else:
                    data_e = tst_e
                    origin = 'tst'
            else:
                data_e = ref_e
                origin = 'ref'

            x_c, y_c = get_center_point(*data_e[0:4])

            is_inside_rois = any(is_point_inside_bbox((x_c, y_c), roi[0:4]) for roi in rois)
            if not is_inside_rois:
                removed_points_log += f'oob {origin}: {scan_id},{",".join(str(e) for e in data_e[0:5])}\n'
                continue

            # there may be more than one ROI. if a label is on the wider border of one of
            # those roi, True is returned.
            is_on_border = any(is_on_wider_border((x_c, y_c), roi[0:4], wide_border) for roi in rois)

            if is_on_border:
                removed_points_log += f'borderline {origin}: {scan_id},{",".join(str(e) for e in data_e[0:5])}\n'
                continue

            updated_locations_match_list.append(entry)

        locations_match_list = updated_locations_match_list

        if save_to:
            with open(f'{save_to}/removed_points.log', 'a') as f:
                f.write(removed_points_log)

    if save_to:
        save_locations_match_list(f'{save_to}/locations_match_list_filtered.log', locations_match_list, scan_id)

    # save per-scan diffs and calculate confusion matrix, if applicable.
    # otherwise None is returned
    c_matrix = gen_diffs_details(locations_match_list, scan_header, save_to,
                                 neg_label=neg_label,
                                 c_mosaics=c_mosaics,
                                 labels_map=flat_labels_map)

    if c_matrix is None:
        # no valid comparison points were found
        n_cls = len(set(flat_labels_map.values()))
        return {'tp': 0, 'fp': 0, 'fn': 0, 'counted': 0, 'neg': 0}, np.zeros((n_cls, n_cls), dtype='int64')

    print_precision_recall_line(c_matrix, all_labels, scan_id)

    # detection statistics
    det_tp, det_fp, det_fn, counted = calculate_detection_stats(locations_match_list, neg_label=neg_label)
    assert det_tp + det_fp + det_fn == counted

    if counted == 0:
        return {
            'tp': 0,
            'fp': 0,
            'fn': 0,
            'counted': 0,
            'neg': 0,
        }, c_matrix

    neg = len(locations_match_list) - counted

    results = {
        'tp': det_tp,
        'fp': det_fp,
        'fn': det_fn,
        'counted': counted,
        'neg': neg,
    }

    return results, c_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ref', type=str, help='reference cvi json file, or directory containing jsons')
    parser.add_argument('tst', type=str, help='test cvi json file, or directory containing jsons')
    parser.add_argument('--model-dir', type=str, help='model dir to get label mapping for confusion matrix report',
                        required=True)
    parser.add_argument('--threshold', type=int, help=f'distance threshold. default: {DEFAULT_THRESHOLD}')
    parser.add_argument('--neg-label', type=str, help='negative label for location comparison')
    parser.add_argument('--save-to', type=str, help='save additional details to dir')
    parser.add_argument('--save-mosaics', action='store_true', help='save confusion matrix related images')
    parser.add_argument('--wide-border', type=int, help='remove borderline points. border width in pixels')
    parser.add_argument('--ref-filter-mode', action='store_true', help='run in filter mode')
    args = vars(parser.parse_args())

    ref_json = args['ref']
    tst_json = args['tst']
    if args['threshold']:
        threshold = args['threshold']
    else:
        threshold = DEFAULT_THRESHOLD

    neg_label = None
    if args['neg_label']:
        neg_label = args['neg_label'].lower()

    save_to = None
    if args['save_to']:
        save_to = args['save_to']
        if not os.path.exists(save_to):
            os.mkdir(save_to)

    # ref filter mode requires disabling the wide border mode
    if args['ref_filter_mode']:
        args['wide_border'] = 0

    if args['wide_border'] is not None:
        wide_border = args['wide_border']
        if wide_border == 0:
            wide_border = None
    else:
        wide_border = DEFAULT_WIDE_BORDER_THRESHOLD

    cells_model = None
    if args['model_dir']:
        model_dir = args['model_dir']
        model_cfg_yaml = f'{model_dir}/model.yml'
        if not os.path.exists(model_cfg_yaml):
            sys.exit(f'file not found: {model_cfg_yaml}')

        cells_model = model_instance_from_yaml(model_cfg_yaml)
        global flat_labels_map
        global all_labels
        flat_labels_map = cells_model.get_flat_label_mapping(str_labels=True)
        label_mapping = cells_model.get_label_mapping()
        all_labels = [None] * len(label_mapping)
        for k, v in label_mapping.items():
            all_labels[v] = k

    c_mosaics = None
    if args['save_mosaics']:
        if not cells_model:
            sys.exit('save_mosaics requires the model-dir argument')

        pyramids_root = os.environ['CVI_PYRAMIDS_ROOT']
        if not os.path.exists(pyramids_root):
            sys.exit(f'invalid pyramids root: {pyramids_root}')

        if not save_to:
            sys.exit('save_mosaics requires the save-to argument')

        save_to_mosaics = f'{save_to}/confusion-mosaics'
        if not os.path.exists(save_to_mosaics):
            os.mkdir(save_to_mosaics)

        c_mosaics = ConfusionMosaics(pyramids_root=pyramids_root, cells_model=cells_model,
                                     save_to=save_to_mosaics)

    ref_json_files = []
    tst_json_files = []
    if os.path.isfile(ref_json):

        # ensure symmetry of inputs
        if not os.path.isfile(tst_json):
            sys.exit(f'error: tst must also be a file: {tst_json}')

        ref_json_files.append(os.path.abspath(ref_json))
        tst_json_files.append(os.path.abspath(tst_json))

    elif os.path.isdir(ref_json):

        if not os.path.isdir(tst_json):
            sys.exit(f'error: tst must also be a directory: {tst_json}')

        ref_json_files = glob.glob(f'{ref_json}/*.json')
        print(f'ref: found {len(ref_json_files)} json files.')

        for ref_file in ref_json_files:
            basename = os.path.basename(ref_file)
            tst_file = f'{tst_json}/{basename}'

            if not os.path.isfile(tst_file):
                print(f'unable to find tst file: {tst_file}')
                continue

            tst_json_files.append(tst_file)

    total_results = Counter()
    total_c_matrix = None
    n = len(all_labels)
    if n > 0:
        total_c_matrix = np.zeros(shape=(n, n), dtype=int)

    for idx in range(len(ref_json_files)):
        ref_file = ref_json_files[idx]
        tst_file = tst_json_files[idx]

        assert os.path.basename(ref_file) == os.path.basename(tst_file), \
            'compared file must be the same <scan_id>.json for both files'

        results, c_matrix = compare_json_pair(ref_file, tst_file,
                                              threshold=threshold,
                                              neg_label=neg_label,
                                              save_to=save_to,
                                              c_mosaics=c_mosaics,
                                              wide_border=wide_border,
                                              ref_filter_mode=args['ref_filter_mode'])

        total_results += Counter(results)
        if c_matrix is not None:
            total_c_matrix += c_matrix

    if args['ref_filter_mode']:
        return

    if c_mosaics:
        c_mosaics.save_all()

    counted = total_results['counted']
    tp = total_results['tp']
    fp = total_results['fp']
    fn = total_results['fn']
    neg = total_results['neg']

    tp_per = tp / counted * 100.0
    fp_per = fp / counted * 100.0
    fn_per = fn / counted * 100.0

    print(f'** Total ** TP: {tp:<5} ({tp_per:.2f}%), FP: {fp} ({fp_per:.2f}%), FN: {fn} ({fn_per:.2f})% -- '
          f'total {counted} (neg: {neg})')

    if total_c_matrix is not None:
        print_precision_recall_line(total_c_matrix, all_labels, 'CM Total --')

        if save_to:
            plot_confusion_matrix(c_matrix=total_c_matrix,
                                  target_names=all_labels,
                                  save_to_file=f'{save_to}/confusion_matrix.png',
                                  title='global confusion matrix')

            # extended confusion matrix, for csv file
            cm_shape = total_c_matrix.shape
            ec_matrix = np.zeros(shape=(cm_shape[0]+1, cm_shape[1]+1), dtype=total_c_matrix.dtype)

            ec_matrix[:-1, :-1] = total_c_matrix

            # last row sums columns
            ec_matrix[-1, :-1] = np.sum(total_c_matrix, axis=0)

            # column sums rows
            ec_matrix[:-1, -1] = np.sum(total_c_matrix, axis=1)

            header_str = ','.join(all_labels) + ',ROW_SUM'
            np.savetxt(f'{save_to}/ec_matrix.csv', ec_matrix, fmt='%i', delimiter=',',
                       header=header_str, comments='')

            # Save to excel
            save_confusion_xlsx(c_matrix=total_c_matrix, save_to=save_to, neg_label=neg_label)


if __name__ == '__main__':
    main()
