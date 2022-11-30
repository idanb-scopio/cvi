#!/usr/bin/env python3
import argparse
import glob
import json
import os
import sys
from collections import Counter


unified_labels_map = {
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

parser = argparse.ArgumentParser()
parser.add_argument('json_input', type=str, help='cvi json file/containing dir')
args = vars(parser.parse_args())

json_input = args['json_input']
if not os.path.exists(json_input):
    sys.exit(f'not found: {json_input}')

json_files = []
if os.path.isfile(json_input):
    json_files.append(os.path.abspath(json_input))
elif os.path.isdir(json_input):
    json_files += glob.glob(f'{json_input}/*.json')
    print(f'found {len(json_files)} json files.')
else:
    sys.exit(f'invalid input: {json_files}')

label_stats = Counter()
res_lbl_stats = Counter()
res_scan_stats = Counter()

for jf in json_files:

    with open(jf, 'r') as f:
        dataset = json.load(f)

        label_rois = dataset['labels']

        # get all the label strings, lowercase
        labels_str = [label[4].lower() for label in label_rois]

        # apply label name mapping, if applicable
        labels_str = [unified_labels_map[label] if label in unified_labels_map else label for label in labels_str]

        label_stats += Counter(labels_str)

        pyr_res = dataset['pyramid_resolution']
        if pyr_res not in res_lbl_stats:
            res_scan_stats[pyr_res] = 1
            res_lbl_stats[pyr_res] = len(label_rois)
        else:
            res_scan_stats[pyr_res] += 1
            res_lbl_stats[pyr_res] += len(label_rois)

labels_total = sum(label_stats.values())
print('\nTotal labels distribution summary:')
print(f'count     share  label name')
print(f'========  =====  ==========')
label_items = sorted(label_stats.items(), key=lambda x: x[1], reverse=True)
for k, v in label_items:
    share = v / labels_total * 100
    share_str = f'{share:.2f}%'
    print(f'{v:8d}  {share_str:>7s}  {k}')
print(f'--------------------')
print(f'total: {labels_total}')
