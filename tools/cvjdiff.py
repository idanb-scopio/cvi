#!/usr/bin/env python3
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.pyramiddata import apply_blacklist_res_swap
from tools.cvjcompare import gen_diffs_details

# mapping of labels
UNIFIED_LABELS_MAP = {
    'neu': 'segmented neutrophil',
    'lym': 'lymphocyte',
    'gcd': 'smudge cell',
    'mon': 'monocyte',
    'band': 'band neutrophil',
    'eos': 'eosinophil',
    'lgl': 'large granular lymphocyte',
    'bas': 'basophil',
    'me': 'metamyelocyte',
    'myle': 'myelocyte',
    'rfl': 'nrbc',
    'al': 'atypical lymphocyte',
    'abl': 'aberrant lymphocyte',
    'wbc': 'unclassified wbc',
    'plasma': 'plasma cell',
}


FP_RES_ERR = 1e-6


def process_label(label):
    label_str = label[4].lower()
    if label_str in UNIFIED_LABELS_MAP:
        label_str = UNIFIED_LABELS_MAP[label_str]
    label_t = (*label[0:4], label_str)
    return label_t


def compare_single(ref_file, tst_file, save_to=None):
    with open(ref_file, 'r') as f:
        ref_ds = json.load(f)

    with open(tst_file, 'r') as f:
        tst_ds = json.load(f)

    ref_res = apply_blacklist_res_swap(ref_ds['pyramid_resolution'])
    tst_res = apply_blacklist_res_swap(tst_ds['pyramid_resolution'])

    if not ref_ds['scan_id'] == tst_ds['scan_id']:
        raise RuntimeError(f'scan_id mismatch: {ref_file} vs. {tst_file}')

    scan_id = ref_ds['scan_id']

    ref_labels_map = {}
    ref_labels_set = set()
    for label in ref_ds['labels']:
        label_t = process_label(label)
        coords_t = label_t[0:4]
        ref_labels_set.add(label_t)
        ref_labels_map[coords_t] = label_t[4]

    tst_labels_map = {}
    tst_labels_set = set()
    for label in tst_ds['labels']:
        label_t = process_label(label)
        coords_t = label_t[0:4]
        tst_labels_set.add(label_t)
        tst_labels_map[coords_t] = label_t[4]

    diff_ref_only = []
    diff_tst_only = []
    diff_mismatch = []
    mismatch_list = []
    for roi in ref_labels_map:
        r_lbl_str = ref_labels_map[roi]
        ref_label = (*roi, r_lbl_str)

        # missing in tst
        if roi not in tst_labels_map:
            diff_ref_only.append(ref_label)
            mismatch_list.append((ref_label, None))
            continue

        t_lbl_str = tst_labels_map[roi]
        tst_label = (*roi, t_lbl_str)

        # label has changed
        if r_lbl_str != t_lbl_str:
            diff_mismatch.append((ref_label, tst_label))
            mismatch_list.append((ref_label, tst_label))

            # remove matched roi from tst_labels_map. remaining keys are iterated
            # at the end of this loop.
            del tst_labels_map[roi]

    # add all remaining labels of tst to mismatch list
    tst_only_set = set(tst_labels_map.keys()) - set(ref_labels_map.keys())
    for roi in tst_only_set:
        t_lbl_str = tst_labels_map[roi]
        tst_label = (*roi, t_lbl_str)
        diff_tst_only.append(tst_label)
        mismatch_list.append((None, tst_label))

    for label in sorted(diff_ref_only):
        print(f'{scan_id}: removed: {label}')

    for label in sorted(diff_tst_only):
        print(f'{scan_id}: added: {label}')

    for r_label, t_label in sorted(diff_mismatch):
        roi = r_label[0:4]
        r_str = r_label[4]
        t_str = t_label[4]
        print(f'{scan_id}: changed: {roi}: {r_str} -> {t_str}')

    if save_to and len(mismatch_list) > 0:
        # copy and leave only header: labels are removed here to be replaced
        scan_header = ref_ds.copy()
        del scan_header['labels']

        gen_diffs_details(mismatch_list, scan_header, save_to)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ref', type=str, help='reference cvi json file, or directory containing jsons')
    parser.add_argument('tst', type=str, help='test cvi json file, or directory containing jsons')
    parser.add_argument('--verbose', '-v', action='store_true', help='display verbose information')
    parser.add_argument('--summary', '-s', action='store_true', help='print only summary')
    parser.add_argument('--save-to', type=str, help='save diffs cvi jsons in a given folder')
    args = vars(parser.parse_args())

    ref_json = args['ref']
    tst_json = args['tst']
    is_verbose = args.get('verbose', False)

    save_to = args['save_to']
    if save_to and not os.path.exists(save_to):
        os.mkdir(save_to)

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

        ref_json_files = os.listdir(ref_json)
        ref_json_files = list(filter(lambda f: f.endswith('.json'), ref_json_files))
        print(f'ref: found {len(ref_json_files)} json files in {ref_json}')

        tst_json_files = os.listdir(tst_json)
        tst_json_files = list(filter(lambda f: f.endswith('.json'), tst_json_files))
        print(f'tst: found {len(tst_json_files)} json files in {tst_json}')

        ref_only = list(set(ref_json_files) - set(tst_json_files))
        tst_only = list(set(tst_json_files) - set(ref_json_files))
        common_files = sorted(list(set(ref_json_files) & set(tst_json_files)))

        print(f'ref has {len(ref_only)} scans that are not in tst.')
        print(f'tst has {len(tst_only)} scans that are not in ref.')
        print(f'{len(common_files)} are common to both ref and tst.')

        if is_verbose:
            for f in ref_only:
                print(f'missing in tst: {f}')
            for f in tst_only:
                print(f'missing in ref: {f}')
            for f in common_files:
                print(f'common in both: {f}')

        # convert to abs path
        ref_json_files = [f'{ref_json}/{f}' for f in common_files]
        tst_json_files = [f'{tst_json}/{f}' for f in common_files]

    if args['summary']:
        sys.exit()

    assert len(ref_json_files) == len(tst_json_files)
    for idx in range(len(ref_json_files)):
        ref_file = ref_json_files[idx]
        tst_file = tst_json_files[idx]

        compare_single(ref_file, tst_file, save_to=save_to)


if __name__ == '__main__':
    main()
