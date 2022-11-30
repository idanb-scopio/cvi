#!/usr/bin/env python3

import os
import sys

if len(sys.argv) != 2:
    sys.exit('please specify path as an argument')

path = sys.argv[1]
if not os.path.exists(path):
    sys.exit(f'path does not exist: {path}')

entries = os.listdir(path)

for e in entries:
    full_path = os.path.join(path, e)
    if os.path.islink(full_path):
        resolved_link = os.readlink(full_path)
        if resolved_link[0] == '/':
            abs_resolved_link = resolved_link
        else:
            abs_resolved_link = os.path.abspath(os.path.join(path, resolved_link))

        if not os.path.exists(abs_resolved_link):
            print(f'error: target link does not exist: {abs_resolved_link}')
            continue

        os.unlink(full_path)
        os.symlink(abs_resolved_link, full_path)
        print(f'{e} -> {os.path.abspath(abs_resolved_link)}')
