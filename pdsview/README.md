# pdsview - Pyramid Dataset Viewer
## Intro
pdsview is a viewer for pyramid (scans) data, and can display labels saved in the CVI JSON format.

## Setup
pdsview is based uses `flask`. To get going open a new virtual environment with Python 3.6 and above and install packages from `requirements.txt`. 

**note:** this is a seperate `requirements.txt` than CVI projects `requirements.txt`.
```
$ python3 -m venv /path/to/pdsvenv
$ source /path/to/pdsvenv/bin/activate
(pdsvenv) $ pip install -r requirements.txt
```

## Usage
```
$ ./pdsview.py --help
usage: pdsview.py [-h] [--labels-file LABELS_FILE] [-p PORT] scan_folder

positional arguments:
  scan_folder           scan folder containing pyramid data

optional arguments:
  -h, --help            show this help message and exit
  --labels-file LABELS_FILE
                        path to a labels/json format file
  -p PORT, --port PORT  listen on a given port
```

### Example
Use pdsview to open a web server that hosts the pyramid viewer and files. Then open a web browser on the address that is given by pdsview.

```
(pdsvenv) $ ./pdsview.py /mnt/ssd/unified_wbc/269074f8-9c77-4828-ad9e-89ad4f71909a
 * Serving Flask app 'pdsview' (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on all addresses.
   WARNING: This is a development server. Do not use it in a production deployment.
 * Running on http://192.168.1.164:4141/ (Press CTRL+C to quit)

```

If more than one instance of pdsview is needed on the same machine (viewing more than one pyramid), use `--port` to specify a different TCP port:

## Displaying Labels
Use `--labels-file` to overlay labels data on top of the scan. The CVI JSON contains the following fields:
* `labels`: list of [x_topleft, y_topleft, width, height, label_str]
* `ROIs`: list of [x_topleft, y_topleft, width, height, roi_str]

Example:
```JSON
{
    "scan_id": "975a466a-be0d-47c8-a6dd-089ddd85b7a6",
    "pyramid_resolution": 0.0002016,
    "labels": [
        [
            10201,
            24052,
            79,
            79,
            "dirt"
        ],
        [
            11175,
            23879,
            79,
            79,
            "AL"
        ]
    ],
   "ROIs": [
        [
            10217,
            16947,
            29904,
            8694,
            "ROI idx 0"
        ]
    ]
}
```
