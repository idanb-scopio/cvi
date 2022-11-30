#!/usr/bin/env python3
import argparse
import json
import pathlib
import os
import sys
from flask import Flask, send_from_directory, render_template

DEFAULT_PORT = 4141

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
viewer_file = SCRIPT_DIR / 'templates/viewer.html'
assert viewer_file.exists()

parser = argparse.ArgumentParser()
parser.add_argument('scan_folder', type=str, help='scan folder containing pyramid data')
parser.add_argument('-l', '--labels-file', type=str, required=False, help='path to a labels/json format file')
parser.add_argument('--omit-label', type=str, help='do not draw label text')
parser.add_argument('-p', '--port', type=int, required=False, help='listen on a given port')
args = vars(parser.parse_args())

scan_folder = pathlib.Path(args['scan_folder'])

if args['port']:
    port = args['port']
else:
    port = DEFAULT_PORT

if not scan_folder.exists():
    sys.exit(f'path does not exist: {scan_folder}')

omit_label = args['omit_label']

dzi_file = scan_folder.absolute() / 'pyramid/pyramid.dzi'
if not dzi_file.exists():

    # attempt to infer scan folder from labels file uuid
    if args['labels_file']:
        scan_id = os.path.basename(args['labels_file'])[0:36]
        scan_folder = scan_folder / scan_id
        dzi_file = scan_folder.absolute() / 'pyramid/pyramid.dzi'

    if not dzi_file.exists():
        sys.exit(f'dzi file does not exist: {dzi_file}')

pyramid_prefix = f'{str(scan_folder.absolute())}/pyramid'

# handle labels
# if no labels json is found, an empty label dataset is served
default_labels_json = '{"labels": [], "ROIs": []}'
labels_json_str = None

pyramid_res = ''
if args['labels_file']:
    labels_file = f'{args["labels_file"]}'
    if not os.path.exists(labels_file):
        sys.exit(f'file not found: {labels_file}')

    try:
        with open(labels_file, 'r') as f:
            labels_json_str = f.read()
    except Exception as e:
        sys.exit(f'failed loading JSON data: {str(e)}')

    labels_dataset = json.loads(labels_json_str)
    if 'pyramid_resolution' not in labels_dataset:
        raise RuntimeError('missing field')

    pyramid_res = f' -- resolution: {float(labels_dataset["pyramid_resolution"]):.7f} mm/pixel'

    if omit_label:

        new_labels = []
        for label in labels_dataset['labels']:
            if label[4] == omit_label:
                new_label = label[0:4] + ['']
            else:
                new_label = label

            new_labels.append(new_label)

        labels_dataset['labels'] = new_labels
        labels_json_str = json.dumps(labels_dataset)

else:
    labels_json_str = default_labels_json

app = Flask(__name__)


@app.route("/")
def index():
    return render_template('viewer.html', scan_id=scan_folder.name, pyramid_res=pyramid_res)


@app.route("/static/<path:path>")
def static_dir(path):
    print(f"requested: {path}")
    return send_from_directory("static", path)


@app.route("/pyramid/<path:path>")
def pyramid_dir(path):
    return send_from_directory(pyramid_prefix, path)


@app.route("/labels")
def labels_data():
    return labels_json_str


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)
