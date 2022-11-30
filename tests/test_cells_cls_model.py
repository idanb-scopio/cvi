from models.cellsfactory import model_instance_from_yaml
import os


def test_init():
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    model_cfg_file = f'{tests_dir}/../models/classifiers/wbc_5class.yml'
    wbccls = model_instance_from_yaml(model_cfg_file)
    print(wbccls)
