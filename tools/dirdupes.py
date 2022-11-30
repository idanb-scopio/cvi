#!/usr/bin/env python3

import argparse
import os
import sys


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('directories', nargs='+', type=str, help='directories to check for subdir duplicates inside')
    parser.add_argument('-j', '--json-labels-dir', type=str, help='directory with json files to be considered as dirs')
    parser.add_argument('-v', '--verbose', action='store_true', help='print extra details')
    parser.add_argument('-u', '--unique', action='store_true', help='show unique entries')
    args = vars(parser.parse_args())

    dirs = args['directories']

    if len(dirs) < 2 and not args['json_labels_dir']:
        sys.exit('error: duplicate checking requires at least 2 dir paths.')

    for i, d in enumerate(dirs):
        if not os.path.exists(d):
            sys.exit(f'no such directory: {d}')

        if d.endswith('/'):
            trimmed_name = d[:-1]
            dirs[i] = trimmed_name

    dir_dict = {}
    if args['json_labels_dir']:
        j_dir = args['json_labels_dir']
        if not os.path.exists(j_dir):
            sys.exit(f'json labels dir does not exist: {j_dir}')

        entries = os.listdir(j_dir)
        counter = 0
        for e in entries:
            if e.endswith('.json'):
                name = e[:-5]
                if name not in dir_dict:
                    dir_dict[name] = [j_dir]
                else:
                    dir_dict[name].append(j_dir)
                counter += 1

        if args['verbose']:
            print(f'found {counter} json entries in {j_dir}')

    for scanned_dir in dirs:
        entries = os.listdir(scanned_dir)

        if args['verbose']:
            print(f'found {len(entries)} entries in {scanned_dir}')

        for e in entries:
            e_full_path = os.path.join(scanned_dir, e)
            if os.path.isdir(e_full_path) or os.path.islink(e_full_path):
                if e not in dir_dict:
                    dir_dict[e] = [scanned_dir]
                else:
                    dir_dict[e].append(scanned_dir)

    if args['unique']:
        for d in dir_dict:
            if len(dir_dict[d]) == 1:
                print(f'unique: {d} in {dir_dict[d][0]}')
    else:
        for d in dir_dict:
            if len(dir_dict[d]) > 1:
                print(f'duplicate: {d} in {",".join(dir_dict[d])}')


if __name__ == '__main__':
    main()
