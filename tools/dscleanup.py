#!/usr/bin/env python3
#
# Pyramid Dataset tools - manipulating and fixing directoy structure and names for pyramids exported from the app/cloud

import argparse
import os
import sys
from uuid import UUID


def is_uuid(uuid_str):
    # check if scan_id is a uuid string
    try:
        uuid_obj = UUID(uuid_str)

        # valid uuid
        return True
    except ValueError:
        # not a valid uuid
        return False


def move_lut3d(pyramids_root, lut3d_dir):
    if not os.path.exists(lut3d_dir):
        os.mkdir(lut3d_dir)

    dir_content = os.listdir(pyramids_root)
    for scan_id in dir_content:
        if not is_uuid(scan_id):
            print(f'not a uuid: {scan_id}')
            continue

        pyr_lut3d = f'{pyramids_root}/{scan_id}/pyramid/lut3d'
        if os.path.exists(pyr_lut3d):
            target_dir = f'{lut3d_dir}/{scan_id}'
            cmd = f'mkdir -p {target_dir} && mv {pyr_lut3d} {target_dir}'
            print(cmd)


def fix_double_dirs(pyramids_root):

    dir_content = os.listdir(pyramids_root)
    for scan_id in dir_content:
        if not is_uuid(scan_id):
            print(f'not a uuid: {scan_id}')
            continue

        pf_dir = f'{pyramids_root}/{scan_id}/pyramid/pyramid_files'
        deeper_dir = f'{pyramids_root}/{scan_id}/pyramid/pyramid_files/pyramid_files'
        deeper_renamed_dir = f'{pyramids_root}/{scan_id}/pyramid/_pyramid_files/pyramid_files'
        renamed_dir = f'{pyramids_root}/{scan_id}/pyramid/_pyramid_files'

        if os.path.exists(pf_dir) and os.path.exists(deeper_dir):
            print(f'mv {pf_dir} {renamed_dir} && mv {deeper_renamed_dir} {pf_dir}')


def rename_ycbcr(pyramids_root):

    dir_content = os.listdir(pyramids_root)
    for scan_id in dir_content:
        if not is_uuid(scan_id):
            print(f'not a uuid: {scan_id}')
            continue

        ycbcr_dir = f'{pyramids_root}/{scan_id}/pyramid/ycbcr/pyramid_files'
        pf_dir = f'{pyramids_root}/{scan_id}/pyramid/pyramid_files'

        # this state is a data duplicate (ycbcr is not a symlink)
        if os.path.exists(ycbcr_dir) and os.path.exists(pf_dir):
            if not os.path.islink(ycbcr_dir) and not os.path.islink(pf_dir):

                ycbcr_files = os.listdir(f'{ycbcr_dir}/pyramid_files')
                pf_file = os.listdir(f'{ycbcr_dir}/pyramid_files')

                if len(ycbcr_files) != len(pf_file):
                    print(f'double issue: {ycbcr_dir}')
                else:
                    print(f'double trouble: {ycbcr_dir} ({len(ycbcr_files)}), {pf_dir} ({len(pf_file)})')

                print(f'rm -rf {ycbcr_dir}')
                continue
            else:
                print(f'symlink {ycbcr_dir}')
                continue

        # this state should be investigated:
        if not os.path.exists(ycbcr_dir) and not os.path.exists(pf_dir):
            print(f'negative double trouble: {ycbcr_dir}, {pf_dir}')
            continue

        if os.path.exists(ycbcr_dir) and not os.path.exists(pf_dir):
            print(f'mv {ycbcr_dir} {pf_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pyramids_root', type=str, help='root dir containing pyramids')
    parser.add_argument('--move-lut3d', type=str, help='destination to move lut3d subdirs')
    parser.add_argument('--rename-ycbcr', action='store_true', help='rename ycbcr to pyramid_files')
    parser.add_argument('--fix-double-dirs', action='store_true', help='fix double pyramid_files/ issue')
    args = vars(parser.parse_args())

    pyramids_root = args['pyramids_root']
    if not os.path.exists(pyramids_root):
        sys.exit(f'pyramid root does not exist: {pyramids_root}')

    if args['move_lut3d']:
        move_lut3d(pyramids_root, args['move_lut3d'])

    if args['rename_ycbcr']:
        rename_ycbcr(pyramids_root)

    if args['fix_double_dirs']:
        fix_double_dirs(pyramids_root)


if __name__ == '__main__':
    main()