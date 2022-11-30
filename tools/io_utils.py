from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import yaml
import pickle

# I/O utilities
def read_json_file(file_name):
    try:
        with open(file_name, 'r') as f:
            data = json.load(f)
    except:
        raise ValueError('Can not read file %s'%file_name)

    return data

def write_json_file(data, file_name, ident=4, sort=False):
    try:
        with open(file_name, 'w') as f:
            f.write(json.dumps(data, sort_keys=sort, indent=ident))
    except:
        raise ValueError('Can not write json file %s'%file_name)


def write_yaml_file(data, file_name):
    try:
        with open(file_name, 'w') as yaml_file:
            yaml_file.write(yaml.dump(data))
    except:
        raise ValueError('Can not write yaml file %s'%file_name)

def read_yaml_file(file_name):
    try:
        with open(file_name, 'r') as yaml_file:
            data = yaml.safe_load(yaml_file)
    except:
        raise ValueError('Can not read yaml file %s'%file_name)

    return data

def read_pickle_file(file_name):
    try:
        data = pickle.load(open(file_name, "rb"))
    except:
        raise ValueError('Can not read file %s' % file_name)

    return data

def write_pickle_file(data, file_name):
    try:
        with open(file_name, 'w') as f:
            pickle.dump(data, open(file_name, "wb"))
    except:
        raise ValueError('Can not write pickle file %s'%file_name)

