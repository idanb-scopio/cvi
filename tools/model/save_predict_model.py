#!/usr/bin/env python3

import os
import re
import shutil
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models.cellsfactory import model_instance_from_yaml

if len(sys.argv) < 2:
    sys.exit(f'h5 model file required')

tr_h5_file = sys.argv[1]
if not os.path.isfile(tr_h5_file):
    sys.exit(f'file not found: {tr_h5_file}')

basename = os.path.basename(tr_h5_file)
print(basename)
m = re.match('^model_tr-(\d+)\.h5', basename)
if not m:
    sys.exit(f'error parsing filename: {basename}')

model_dir = os.path.dirname(os.path.abspath(tr_h5_file))

output_model_file = f'{model_dir}/model-{m.group(1)}.h5'
tr_model_cfg_yaml = f'{model_dir}/model_tr.yml'
model_cfg_yaml = f'{model_dir}/model.yml'
if not os.path.exists(model_cfg_yaml):
    shutil.copy(tr_model_cfg_yaml, model_cfg_yaml)

cells_model = model_instance_from_yaml(model_cfg_yaml)

keras_model = cells_model.get_keras_model(training=False)
keras_model.load_weights(tr_h5_file, by_name=True)
keras_model.save(filepath=output_model_file, include_optimizer=False)
print(f'{tr_h5_file} -> {output_model_file}')