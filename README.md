# CVI - Computer Vision Infrastructure (v2)
## Introduction
CVI is the next generation of computer vision tools and infrastrucute, running with Keras API over the latest TensorFlow (v2).

**NOTE:** This is currently work in progress, taking the algorithms implemented from the production version (v1). 
In the first stage, PBS is reimplemented with v2 (White Blood Cells detection and classification).

## Structure
As of now, core capabilities are:
* Build
* DSPreview
* Train
* Predict

Those tools can be found under the `core` directory.

### Build
This stage takes pyramid scan data, and CVI JSON formatted files, a model configuration and output dir - and generates TFRecord files which are used as input for the training stage. Model configuration is saved inside the build dir.

### DSPreview
This stage takes a build dir (from the previous stage) and generates large JPEG mosaics (count can be limited with `--limit`). Those mosaics can be used to inspect the data which enters the training stage. 

### Train
This stage takes as input the build directory of a training set and evaluation set and trains a NN model. The keras model file (.h5) and configuration YAML are placed in the output dir.

### Predict
A utility for running prediction given a (single) model, input image source (pyramid or a JPEG file), input image resolution parameter and input ROI. Results are saved as CVI JSON format and can be viewed with pdsviewer. Debug data can also be saved per user request.

## Formats
### Model
The model parameters are represented in a YAML file, and can be configured. Model type is tied to a class instance which implements the specific model behaviour.

### Labels JSON
* Input images are read from pyramid scan data
* Labels are read from the CVI JSON format: a short version containing the following fields:
  - `scan_id`: uuid of the scan (pre-uuid scans with integers as id have uuid assigned to them)
  - `pyramid_resolution`: resolution in mm/pixel of the scanned image
  - `labels`: list of [x_topleft, y_topleft, width, height, label_string] describing a bounding box and a label.
  - `ROIs`: list of [x_topleft, y_topleft, width, height, ROI_string] describing a region of interest, usually the a region where all cells are labeled.

Example Format:
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
