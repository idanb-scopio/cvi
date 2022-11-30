#!/usr/bin/env python3
#
# Labels Dataset tools

import argparse
import glob
import json
import math
import os
import random
import sys
from uuid import UUID
import numpy as np


DEFAULT_SPLIT = (70, 15, 15)


def is_uuid(uuid_str):
    # check if scan_id is a uuid string
    try:
        uuid_obj = UUID(uuid_str)

        # valid uuid
        return True
    except ValueError:
        # not a valid uuid
        return False


def split_train_eval_test(list_of_scans, percents):
    if len(percents) != 3:
        raise ValueError('percents args must contain 3 numbers')

    if sum(percents) != 100:
        raise ValueError('sum of percents should be 100')

    ratios = [p / 100.0 for p in percents]
    n_train = math.ceil(ratios[0] * len(list_of_scans))
    n_eval = math.ceil(ratios[1] * len(list_of_scans))

    res = [list(_l) for _l in np.split(list_of_scans, [n_train, n_train + n_eval])]
    return res


def resolution_summary(json_files, split_percents=DEFAULT_SPLIT):
    # get consistent randomization
    random.seed(123)

    summary = {}

    for jf in json_files:

        # load json dataset from file
        with open(jf, 'r') as f:
            json_ds = json.load(f)

        scan_id = os.path.basename(jf)[:-5]     # remove .json suffix
        if not is_uuid(scan_id):
            raise RuntimeError('debug me: no uuid')

        res_str = f'{float(json_ds["pyramid_resolution"]):.7f}'
        if res_str not in summary:
            summary[res_str] = [f'{scan_id}.json']
        else:
            summary[res_str].append(f'{scan_id}.json')

    f_train = open('train_total.txt', 'w')
    f_eval = open('eval_total.txt', 'w')
    f_test = open('test_total.txt', 'w')

    for res_str in summary:
        scan_ids = summary[res_str]
        random.shuffle(scan_ids)

        train_scans, eval_scans, test_scans = split_train_eval_test(scan_ids, split_percents)

        print(f'res: {res_str} ({len(scan_ids)} scans)')

        print(f'train:')
        print('\n'.join(train_scans))
        f_train.write('\n'.join(train_scans) + '\n')

        print(f'eval:')
        print('\n'.join(eval_scans))
        f_eval.write('\n'.join(eval_scans) + '\n')

        print(f'test:')
        print('\n'.join(test_scans))
        f_test.write('\n'.join(test_scans) + '\n')

        print('\n')

    f_train.close()
    f_eval.close()
    f_test.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('labels_root', type=str, help='root dir containing pyramids')
    parser.add_argument('--resolution-summary', action='store_true', help='root dir containing pyramids')
    args = vars(parser.parse_args())

    labels_root = args['labels_root']
    json_files = glob.glob(f'{labels_root}/*.json')
    if len(json_files) == 0:
        sys.exit(f'no json files found at: {labels_root}')
    print(f'found {len(json_files)} json files.')

    if args['resolution_summary']:
        resolution_summary(json_files)


if __name__ == '__main__':
    main()