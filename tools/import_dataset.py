#!/usr/bin/env python3
import argparse
import os
import stat
import subprocess
import sys
from uuid import UUID

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

# v1 json dataset file converter to cvi json
DSV1IMPORT_SCRIPT = f'{SCRIPT_DIR}/dsv1import.py'


def is_uuid(uuid_str):
    # check if scan_id is a uuid string
    try:
        uuid_obj = UUID(uuid_str)

        # valid uuid
        return True
    except ValueError:
        # not a valid uuid
        return False


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_file', type=str, help='JSON dataset files')
    parser.add_argument('--labels-dir', type=str, help='destination labels dir', required=True)
    parser.add_argument('--download-dir', type=str, help='destination pyramid download dir dir', required=True)
    args = vars(parser.parse_args())

    dataset_file = args['dataset_file']
    if not os.path.exists(dataset_file):
        sys.exit(f'error: file not found: {dataset_file}')

    labels_dir = args['labels_dir']
    if not os.path.exists(labels_dir):
        os.mkdir(labels_dir)

    download_dir = args['download_dir']
    if not os.path.exists(download_dir):
        os.mkdir(download_dir)

    entries = os.listdir(labels_dir)
    if not entries:
        dsv1_cmd = f'{DSV1IMPORT_SCRIPT} --output-dir {labels_dir} {dataset_file}'
        execute(dsv1_cmd, verbose=True)

    entries = os.listdir(labels_dir)
    dataset_scans = []
    for e in entries:
        if e.endswith('.json'):
            scan_uuid = e[:-5]
            print(f'found scan id: {scan_uuid}')
            dataset_scans.append(scan_uuid)
    print(f'total: {len(dataset_scans)} files.')

    entries = os.listdir(download_dir)
    existing_scans = []
    for e in entries:
        if is_uuid(e):
            existing_scans.append(e)

    print(f'found: {len(existing_scans)} scans in download dir')
    new_scans = list(set(dataset_scans) - set(existing_scans))
    duplicates = list(set(dataset_scans) & set(existing_scans))
    for d in duplicates:
        print(f'duplicate: {d}')
    print(f'{len(new_scans)} new scans to be downloaded.')

    if len(new_scans) == 0:
        sys.exit(0)

    download_script = """
#!/bin/bash

set -xe

"""
    download_script += 'gsutil -m cp -r gs://scopio_serving_scans_prod_eur/webroot/union/{'
    for idx in range(len(new_scans) - 1):
        scan_id = new_scans[idx]
        print(f'will download: {scan_id}')
        download_script += scan_id + ','

    scan_id = new_scans[-1]
    print(f'will download: {scan_id}')
    download_script += scan_id + '} .\n'

    download_script_file = 'download_dataset.sh'
    with open(download_script_file, 'w') as f:
        f.write(download_script)

    st = os.stat(download_script_file)
    os.chmod(download_script_file, st.st_mode | stat.S_IEXEC)
    print(f'download script: {download_script_file}')


if __name__ == '__main__':
    main()

