#!/usr/bin/env python3

import json
import os
import sys

if len(sys.argv) < 2:
    sys.exit(f'please supply json file')

json_dataset = sys.argv[1]
if not os.path.exists(json_dataset):
    sys.exit(f'file not found: {json_dataset}')

with open(sys.argv[1], 'r') as f:
    dataset = json.load(f)

output_file = os.path.join(os.path.dirname(json_dataset),
                           os.path.basename(json_dataset[:-5]) + '_indent' + '.json')

with open(output_file, 'w') as f:
    json.dump(dataset, f, indent=4)

