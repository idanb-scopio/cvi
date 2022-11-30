import os
import yaml

from models.cellscls import CellsClassifier
from models.cellsdet import CellsDetector
from models.cellscascade import CellsCascade

FACTORY_MAP = {
    'cells_classifier': CellsClassifier,
    'cells_detector': CellsDetector,
    'cells_cascade': CellsCascade,
}


def model_instance_from_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        model_cfg = yaml.safe_load(f)

    if 'model_type' not in model_cfg:
        raise ValueError(f'missing "model_type" field in model config: {yaml_file}')

    model_type = model_cfg['model_type']
    if model_type not in FACTORY_MAP:
        raise ValueError(f'invalid model type: {model_type}')

    model_dir = os.path.dirname(os.path.abspath(yaml_file))
    model_cfg['model_dir'] = model_dir

    instance = FACTORY_MAP[model_type](model_cfg=model_cfg)

    return instance
