"""
 Title         : Generating TFRecords from Scopio DB
 Project       : Training

 File          : convert_dataset.py
 Author        : Shahar Karny
 Created       : 02/10/2017
-----------------------------------------------------------------------------
 Description :   Reading labeled information and generating task-specific TFRecords
-----------------------------------------------------------------------------
 Copyright (c) This model is the confidential and
 proprietary property of ScopioLabs Ltd and the possession or use of this
 file requires a written license from ScopioLabs Ltd.
------------------------------------------------------------------------------
 Modification history :

"""

# Typical setup to include TensorFlow.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import shutil
import json
import pickle
import copy
import time
import yaml

from user_tfrecord import UserTfrecordBase as UserTfrecord

import tensorflow as tf
import numpy as np

from PIL import Image
import cv2
import random
import math

open_cv_version = float(cv2.__version__[0:3])
if open_cv_version < 3.4:
    raise ValueError(
        '\n\n\nUnable to run prediction on OpenCV version {0}. Please upgrade to version 3.4'.format(open_cv_version))

import user_flags
import core_flags
import core_path
import core_utils

import logging
logger = logging.getLogger(name='cv_tools')

# Import to support reading Pyramids directly
script_path = os.path.dirname(os.path.realpath(__file__))

cell_recognition_path = script_path + '/../cells_recognition/src'
sys.path.append(cell_recognition_path)

common_path = script_path + '/../bcc_web_app'
sys.path.append(common_path)

default_database_mappings = {
    "scopio3": '/mnt/ssd/chen',
    "scopio4": '/mnt/hd/cv_webroot',
    # "scopio5": '/mnt/ssd/computer_vision/webroot/',
    # "sclab5_3336": '/mnt/ssd/computer_vision/external_labeling_webroot_1/',

    'sclab5': {
        'pyramids': '/scratch/computer_vision/database/webroot/',
        'labels': '/mnt/ssd/computer_vision/webroot/'
    },
    'sclab5_3336': {
        'pyramids': '/scratch/computer_vision/database/external_labeling_webroot_1/',
        'labels': '/mnt/ssd/computer_vision/external_labeling_webroot_1/'
    },

    # Ichilov patch. Both labels pyramids path are pointing to the same folder as
    # Ichilov is using postgress DB and the pyramids path is hard coded anyways.
    # scopiobox8/srv/app/bcc_web_app/db_export/pg_db_export.py
    'scopiobox8': {
        'labels': os.environ.get('ROOT_DIR', '/mnt/storage/labeling_webroot'),
        'pyramids': os.environ.get('ROOT_DIR', '/mnt/hd1/labeling_webroot')
        # 'pyramids': '/mnt/storage/labeling_webroot'
    },
}

#segmentation_db_file = '/srv/segmentation/second_drop/segmentation_db.json'


def print_pair(key, value):
    logger.info("\033[91m%s \033[94m%s\033[0m" % (key, str(value)))
    sys.stdout.flush()


def get_pyramid_crop(PyramidAreaClass, tile_top_left_x,
                     tile_top_left_y,
                     width,
                     height,
                     pyramid_level,
                     pyramid_to_model_resolution_factor=None):
    if pyramid_to_model_resolution_factor is None or (math.fabs(pyramid_to_model_resolution_factor - 1.) < 1e-4):
        return PyramidAreaClass(tile_top_left_x, tile_top_left_y, width, height)

    # Get coordinates
    pyramid_tile_top_left_x_float = tile_top_left_x / pyramid_to_model_resolution_factor
    pyramid_tile_top_left_y_float = tile_top_left_y / pyramid_to_model_resolution_factor
    pyramid_width_float = width / pyramid_to_model_resolution_factor
    pyramid_height_float = height / pyramid_to_model_resolution_factor

    # Get integer coordinates for image that are bigger than actually required when working with float
    pyramid_tile_top_left_x = int(math.floor(pyramid_tile_top_left_x_float))
    pyramid_tile_top_left_y = int(math.floor(pyramid_tile_top_left_y_float))

    pyramid_tile_bottom_right_x_float = pyramid_tile_top_left_x_float + pyramid_width_float
    pyramid_tile_bottom_right_y_float = pyramid_tile_top_left_y_float + pyramid_height_float
    pyramid_tile_bottom_right_x = int(math.ceil(pyramid_tile_bottom_right_x_float))
    pyramid_tile_bottom_right_y = int(math.ceil(pyramid_tile_bottom_right_y_float))

    pyramid_width = pyramid_tile_bottom_right_x - pyramid_tile_top_left_x
    pyramid_height = pyramid_tile_bottom_right_y - pyramid_tile_top_left_y

    return PyramidAreaClass(pyramid_tile_top_left_x, pyramid_tile_top_left_y, pyramid_width, pyramid_height)


def normalize_pyramid_image(pyramid_tile_image, tile_top_left_x,
                       tile_top_left_y,
                       width,
                       height,
                       pyramid_to_model_resolution_factor=None,
                       interpolate_mode = None):

    # Get coordinates
    pyramid_tile_top_left_x_float = tile_top_left_x / pyramid_to_model_resolution_factor
    pyramid_tile_top_left_y_float = tile_top_left_y / pyramid_to_model_resolution_factor
    pyramid_width_float = width / pyramid_to_model_resolution_factor
    pyramid_height_float = height / pyramid_to_model_resolution_factor

    # Get integer coordinates for image that are bigger than actually required when working with float
    pyramid_tile_top_left_x = int(math.floor(pyramid_tile_top_left_x_float))
    pyramid_tile_top_left_y = int(math.floor(pyramid_tile_top_left_y_float))

    pyramid_tile_bottom_right_x_float = pyramid_tile_top_left_x_float + pyramid_width_float
    pyramid_tile_bottom_right_y_float = pyramid_tile_top_left_y_float + pyramid_height_float
    pyramid_tile_bottom_right_x = int(math.ceil(pyramid_tile_bottom_right_x_float))
    pyramid_tile_bottom_right_y = int(math.ceil(pyramid_tile_bottom_right_y_float))

    resize_options = {
        'bilinear_resample': dict(interpolation=cv2.INTER_LINEAR),
        'bicubic_resample': dict(interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT),
        'lanczos_resample': dict(interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
    }

    interpolate_mode = interpolate_mode if interpolate_mode is not None else 'bilinear_resample'
    if interpolate_mode not in resize_options:
        raise ValueError('Unsupported interploate mode %s' % interpolate_mode)
    options = resize_options[interpolate_mode]

    # this change was made in the performance eco by Yuval Gershon
    # the else part is old method for resize
    # branch : feature/performance/resize
    use_simple_resize = True
    if use_simple_resize:
        tile_image = cv2.resize(pyramid_tile_image, (width, height), interpolation=options['interpolation'])
    else:
        # Build the sampling grid
        x_sample_step = np.linspace((pyramid_tile_top_left_x_float - pyramid_tile_top_left_x),
                                    (pyramid_tile_bottom_right_x_float - pyramid_tile_top_left_x), width, endpoint=False)
        y_sample_step = np.linspace((pyramid_tile_top_left_y_float - pyramid_tile_top_left_y),
                                    (pyramid_tile_bottom_right_y_float - pyramid_tile_top_left_y), height,
                                    endpoint=False)

        resample_mask_x, resample_mask_y = np.meshgrid(x_sample_step, y_sample_step)
        resample_mask_x = resample_mask_x.astype('float32')
        resample_mask_y = resample_mask_y.astype('float32')

        tile_image = cv2.remap(src=pyramid_tile_image,
                               map1=resample_mask_x,
                               map2=resample_mask_y,
                               **options)

    return tile_image


class ScanData(object):
    def __init__(self,
                 scan_id,
                 scan_name,
                 scan_uuid=None,
                 roi_list=None,
                 species=None,
                 scanning_device=None,
                 scan_resolution=None,
                 pyramid_resolution=None, # Support scaled dataset
                 scan_tissue=None,
                 tags=None,
                 archive_path=None,
                 hardness=0.):
        self.scan_id = scan_id
        self.scan_uuid = '' if (scan_uuid is None) else str(scan_uuid)
        self.scan_name = str(scan_name) if (scan_name is not None) else None
        self.roi_list = roi_list or []
        self.species = str(species) if (species is not None) else None
        self.scan_tissue = str(scan_tissue) if (scan_tissue is not None) else None
        self.scanning_device = str(scanning_device) if (scanning_device is not None) else None
        self.tags = copy.deepcopy(tags) if (tags is not None) else None
        self.scan_resolution_in_mm_per_pixel = scan_resolution # Scan data resolution in mm per pixel
        self.pyramid_resolution = scan_resolution if (pyramid_resolution is None) else pyramid_resolution
        self.archive_path = archive_path
        self.hardness = hardness

    def get_copy_without_roi_list(self):
        scanning_device = self.__dict__.get('scanning_device', 'not_defined')
        species = self.__dict__.get('species', 'not_defined')
        scan_tissue = self.__dict__.get('scan_tissue', 'not_defined')
        scan_uuid = self.__dict__.get('scan_uuid', '')
        scan_tags = self.__dict__.get('tags', [])
        hardness = self.__dict__.get('hardness', 0.)
        pyramid_resolution = self.__dict__.get('pyramid_resolution', self.scan_resolution_in_mm_per_pixel)
        archive_path = self.__dict__.get('archive_path', None)
        copy_scan_dataset = ScanData(scan_id = self.scan_id,
                                     scan_name = self.scan_name,
                                     scan_uuid=scan_uuid,
                                     roi_list= [],
                                     species=species,
                                     scanning_device=scanning_device,
                                     scan_resolution=self.scan_resolution_in_mm_per_pixel,
                                     pyramid_resolution=pyramid_resolution,
                                     scan_tissue=scan_tissue,
                                     tags=scan_tags,
                                     archive_path=archive_path,
                                     hardness=hardness
                                     )
        return copy_scan_dataset

    def get_copy_without_labels(self):
        roi_list_aside = self.roi_list
        self.roi_list = []
        c = copy.deepcopy(self)
        for roi in roi_list_aside:
            c.roi_list.append(roi.get_copy_without_label_list())
        self.roi_list = roi_list_aside
        return c

    def __str__(self):
        # Overides how this class is printed.
        return ':::INFO:::\nscan name:' + str(self.scan_name) + \
               '\nscan #' + str(self.scan_id) + \
               '\nnumber of ROI within the scan: ' + str(len(self.roi_list))

    def split_scan(self, training_probability=None, split_level=None):
        # Splits Scan ROI list into 2 Scan with 2 ROI lists
        training_roi_list = []
        testing_roi_list = []

        if split_level in ['ROI', 'ROI_per_SCAN']:

            for roi in self.roi_list:
                # Divide the ROI between the 2 datasets
                # Get probability to be in Validation or Training file

                # for_training = np.random.choice(2, p=[1 - training_probability, training_probability])
                for chr in ['\n', '/', ':', ' ']:
                    if chr in roi.name:
                        roi.name = roi.name.replace(chr, '_')

                if roi.for_training:
                    training_roi_list.append(roi)
                else:
                    testing_roi_list.append(roi)

        elif split_level == 'LABEL':

            for roi in self.roi_list:

                training_roi_label_list = []
                testing_roi_label_list = []
                for label in roi.labels:
                    if 'for_training' not in label:
                        # This means that current label that was read from the complete dataset is not participating in current run
                        # This scenario occurs when we are read a PKL dataset that was build from another run and used in current run.
                        logger.info('[Build] Skipping label read from datset with class name {0}'.format(label['label_name']))
                        continue
                    elif label['for_training']:
                        training_roi_label_list.append(label)
                    else:
                        testing_roi_label_list.append(label)

                for chr in ['\n', '/', ':', ' ']:
                    if chr in roi.name:
                        roi.name = roi.name.replace(chr, '_')

                train_roi = copy.deepcopy(roi)
                train_roi.labels = training_roi_label_list
                training_roi_list.append(train_roi)

                testing_roi = copy.deepcopy(roi)
                testing_roi.labels = testing_roi_label_list
                testing_roi_list.append(testing_roi)

        else:
            raise ValueError('unxpected split level, for now')

        new_training_scan = ScanData(scan_id=self.scan_id,
                                     scan_name=self.scan_name,
                                     roi_list=training_roi_list,
                                     scan_resolution=self.scan_resolution_in_mm_per_pixel)

        new_testing_scan = ScanData(scan_id=self.scan_id,
                                    scan_name=self.scan_name,
                                    roi_list=testing_roi_list,
                                    scan_resolution=self.scan_resolution_in_mm_per_pixel)

        return new_training_scan, new_testing_scan

    def get_scan_scan_resolution_in_mm_per_pixel(self, pyramid_path):
        # This is a temp function to get scan_resolution_in_mm_per_pixel from a stitch map
        # When all scans are in the cloud, this should be different.

        stitch_map_path = os.path.join(pyramid_path, 'scans', str(self.scan_id), 'pyramid', 'stitch_map.yml')
        with open(stitch_map_path, 'r') as stream:
            data_loaded = yaml.safe_load(stream)
        self.scan_resolution_in_mm_per_pixel = data_loaded['mm_per_pixel_hr']
        self.pyramid_resolution = self.scan_resolution_in_mm_per_pixel


class RoiData(object):
    def __init__(self,
                 coordinates,
                 roi_id,
                 roi_name,
                 region_mask_contours = None,
                 cell_type=None,
                 image_file = None,
                 mask_image_file = None,
                 sparse_data=False,
                 scan=None,
                 scan_uuid=None,
                 session=None,
                 session_uuid=None
                 ):
        self.x = coordinates['x']
        self.y = coordinates['y']
        self.w = coordinates['width']
        self.h = coordinates['height']
        self.id = roi_id
        self.name = roi_name
        self.cell_type = str(cell_type) if (cell_type is not None) else None
        self.region_mask_contours = region_mask_contours
        self.image_file = image_file
        self.mask_image_file = mask_image_file
        self.sparse_data = sparse_data
        self.labels = []
        self.scan = scan
        self.scan_uuid = str(scan_uuid) if (scan_uuid is not None) else None
        self.session = session
        self.session_uuid = str(session_uuid) if (session_uuid is not None) else None
        self.for_training = True

    def get_copy_without_label_list(self):
        cell_type = self.__dict__.get('cell_type', '')
        scan_uuid = self.__dict__.get('scan_uuid', '')
        session_uuid = self.__dict__.get('session_uuid', '')
        sparse_data = self.__dict__.get('sparse_data', False)
        copy_roi_dataset = RoiData(coordinates = {'x': self.x,
                                                  'y': self.y,
                                                  'width': self.w,
                                                  'height': self.h
                                                  },
                                   roi_id = self.id,
                                   roi_name= self.name,
                                   region_mask_contours=None,
                                   cell_type=cell_type,
                                   image_file=None,
                                   mask_image_file=None,
                                   sparse_data=sparse_data,
                                   scan=self.scan,
                                   scan_uuid=scan_uuid,
                                   session=self.session,
                                   session_uuid=session_uuid)

        if self.__dict__.has_key('region_mask_contours'):
            copy_roi_dataset.region_mask_contours = copy.deepcopy(self.region_mask_contours)
        if self.__dict__.has_key('image_file'):
            copy_roi_dataset.image_file = self.image_file
        if self.__dict__.has_key('mask_image_file'):
            copy_roi_dataset.mask_image_file = self.mask_image_file

        return copy_roi_dataset

    def set_coordinates(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


class CoreTFrecord(object):
    def __init__(self,
                 user_tfrecord,
                 labels_db_host=None,
                 open_local_database_link=False,
                 do_not_declare_flags=False,
                 database_mapping=None,
                 config_params=None):

        # User running configuration settings:
        self.config_params = config_params

        # do_not_declare_flags - always False.
        self.open_local_database_link = open_local_database_link

        database_mapping = database_mapping or default_database_mappings

        if self.open_local_database_link:
            # This means we are working at training and creating a database directly from the DB

            # Import Scan DB (Scan) and labeling DB (DbExporter) interfaces
            from db_export import DbExporter
            from scan import Scan

            if labels_db_host not in database_mapping:
                raise ValueError("'%s' is not a valid database name, valid names are: \n-%s" % (
                    labels_db_host, "\n-".join(database_mapping.keys())))

            self.pyramid_path = database_mapping[labels_db_host].get('pyramids')
            self.labels_path = database_mapping[labels_db_host].get('labels')
            if self.pyramid_path is None or self.labels_path is None:
                raise ValueError(
                    "The data base mapping should hold pyramids and labels path per host. "
                    "{missing} is missing.".format(missing=[v for v in ('pyramids', 'labels')
                                                            if v not in database_mapping[labels_db_host]])
                )

            self.labels_db = DbExporter(self.labels_path)
            self.Scan = Scan

        self.user_tfrecord = user_tfrecord
        self.generate_tf_blueprints = True  # Signals to write a TF dictionary describing all features within a TF example.

        if 'database_segmentation_db_path' in config_params:
            segmentation_db_file = config_params['database_segmentation_db_path']
        else:
            segmentation_db_file = None

        if segmentation_db_file:
            self.segmentation_db = core_utils.read_json_file(segmentation_db_file)
        else:
            self.segmentation_db = []

        # Set default direcotry to generate TFRecords data

        # Declare mandatory flags to be used with convert_dataset to TFRecords.
        if not do_not_declare_flags:
            self.flags = core_flags.get_tf_flags()

            # Querying specific flags value
            # self.flags.DEFINE will only set value if and only if flag was not already declared.
            # We are using these values for sclab5:8000
            self.flags.DEFINE_integer("tfrecord_tile_size", 100,
                                      """How many pixels are used for tile size.""", setdefault=True)
            self.flags.DEFINE_integer("tf_channels", 3,
                                      """How many channels are therein the TF image.""", setdefault=True)
            self.flags.DEFINE_integer("roi_margin", 32,
                                      """Defines the membership distance from a label to an ROI(annotation)""",
                                      setdefault=True)
            self.flags.DEFINE_integer("training_image_width", 100,
                                      """Network expected input image width""", setdefault=True)
            self.flags.DEFINE_integer("training_image_height", 100,
                                      """Network expected input image height""", setdefault=True)
            self.flags.DEFINE_string("training_mode", 'train', """""", setdefault=True)
            self.flags.DEFINE_string("test_mode", 'test', """""", setdefault=True)
            self.flags.DEFINE_integer("training_dataset_ratio", 1, """""", setdefault=True)
            self.flags.DEFINE_string("scan_train_test_split_level", 'LABEL', """""", setdefault=True)
            self.flags.DEFINE_float("test_set_size", 0., """The test set size in numbers""", setdefault=True)
            self.flags.DEFINE_integer("inference_image_width", 100, """inference_image_width""", setdefault=True)
            self.flags.DEFINE_integer("inference_image_height", 100, """inference_image_height""", setdefault=True)
            self.tf_image_size = self.flags.tfrecord_tile_size
            self.tf_channels = self.flags.tf_channels
            self.roi_margin = self.flags.roi_margin
            self.minimum_roi_width = self.flags.inference_image_width
            self.minimum_roi_height = self.flags.inference_image_height
        else:
            # Currenlty this is used when calling Core TFRecords + Prediction in the same session
            self.roi_margin = 0
            self.minimum_roi_width = 1
            self.minimum_roi_height = 1

        # Iterator for post processing of existing TFRecord
        self.read_only_tfrecord_iterator = None

        # Unique ROI checker
        self.ROIs_seen_list = []

        # Debug variable for images
        self.debug = False
        self.training_scan_id_pyramids_directory = None
        self.training_scan_uuid_pyramids_directory = None

    ########################################
    # Fuctions required to create Datasets #
    ########################################

    # Help function to check unique ROI
    def _check_unique_roi(self, roi):
        """

        ROIs_seen_list structure:
        [{'scan': , 'name': , 'session': , 'id':}, {'scan': , 'name': , 'session': , 'id': }, {...}, ....]

        """
        new_roi_dict = {'scan': roi.scan, 'name': roi.name, 'session': roi.session, 'id': roi.id}
        for seen_roi_dict in self.ROIs_seen_list:
            if seen_roi_dict['id'] == new_roi_dict['id']:
                self.terminate_build_and_report_error(session=new_roi_dict['session'],
                                                      scan=new_roi_dict['scan'],
                                                      roi_name=new_roi_dict['id'],
                                                      error_message='Found the same ROI already participating in session {0}'.format(
                                                          seen_roi_dict['session']))

        self.ROIs_seen_list.append(new_roi_dict)
        return

    # Getting Scan data - required connection to DB
    def get_scan_dataset(self, session, scan, filter_list, species,
                         session_name=None,
                         scanning_device = None):
        '''
        get_scan_dataset is responsible to retrieve new labels from new sessions and scans stored in the database.

        Working methodology:
        - Run through the provided sessions list (Currently this is only 1 session)
        - Retrieve all associated scans
        - Get all labeling meta data {Specific per project/session/scan}
        - Build N label lists, a list per class.

        Args:
        `labels_db_path : A path to the local DB (Temp)
        `sessions_list : a list of sessions to access in the DB (Temp; Now we ony pass 1 session ID)
        `scan_list : scan ID.
        `training_patch_size : The patch size to retrieve from the DB
        `pyramid_level : The pyramid leel to scan annotations to. if 1 it means we are working with level 1 (highest resolution in the pyramid)

        Returns:
        `dataset, which is a list of ImageClass objects retrieved from specific sessions/scan.
         The object holds the class name and the metadata of the label.

        '''
        labels_db = self.labels_db
        roi_margin = self.roi_margin

        # Get Scan Info:
        scan_info = labels_db.get_scan_additional_info(scan)

        if session_name is not None:
            scan_name = session_name # Override DB scan name with session name
        else:
            scan_name = scan_info['scan_name']

        # Starts with a NULL dataset list
        # Scan dataset is a list of RoiData object, each holds a dictionary of labels.
        scan_dataset = ScanData(scan_id=scan_info['scan_id'],
                                scan_name=scan_name,
                                roi_list=[],
                                species=species,
                                scanning_device = scanning_device,
                                scan_resolution=self.flags.model_resolution)

        # Add scan_resolution_in_mm_per_pixel when generating dataset PKL
        scan_dataset.get_scan_scan_resolution_in_mm_per_pixel(self.pyramid_path)

        # Get scan annotations
        scan_annotations = labels_db.get_scan_annotations(scan)

        # Error #1:
        if not scan_annotations:
            self.terminate_build_and_report_error(session=session,
                                                  scan=scan,
                                                  roi_name=None,
                                                  error_message='No annotations were found')

        # Per specific session and scan, get all labels associated with the scan
        labels_dict = labels_db.get_scan_blobs(scan, session)

        # Mapping the dataset
        for label_data in labels_dict:
            # Mapping the labels received from the DB to labels that are expected by the system
            mapped_db_label_name = self.get_mapped_db_label(label_data['label_name'][0])
            label_data['label_name'] = [mapped_db_label_name]

        logger.info(('Scan #', scan, ' | Session #', session, ' | Labels found: ', len(labels_dict), ' | ROI(s) found: ',
              len(scan_annotations)))
        sys.stdout.flush()

        # iterate over the ROI and build the roi dataset object list
        for roi_member in scan_annotations:

            roi_candidate = RoiData(coordinates=roi_member['roi'],
                                    roi_id=roi_member['id'],
                                    roi_name=roi_member['name'],
                                    scan=scan,
                                    session=session)

            empty_roi = True  # At this point, we have an ROI, but we stil don't know if there will a label in it.

            # Perform 4 ROI checks
            # 1. If required, only use annotations names that are filtered by user preferences
            # 2. ROI ID is unique
            # 3. ROI region must at least have Height and Width as the network basic tile size
            # 4. There must be at least ONE label in a ROI

            # Check #1
            # --------
            if filter_list:
                filter_check_passed = False
                for filter_name in filter_list:
                    if filter_name in roi_candidate.name:
                        filter_check_passed = True

                if not filter_check_passed:
                    continue

            print_pair("Analyzing ROI ", roi_candidate.name)

            # Check #2
            # --------
            # This is a unique sanity check.
            # If ROI is not unique this break to program and the user is expected to fix the bug.
            # self._check_unique_roi(roi_candidate)

            # Check #3
            # --------
            minimum_roi_height = self.minimum_roi_height  # * pyramid_level
            minimum_roi_width = self.minimum_roi_width  # * pyramid_level
            if roi_candidate.h < minimum_roi_height or roi_candidate.w < minimum_roi_width:
                self.terminate_build_and_report_error(session=session,
                                                      scan=scan,
                                                      roi_name=roi_candidate.name,
                                                      error_message='ROI {0} is too small, resize to {1}'.format(
                                                          roi_candidate.name, minimum_roi_height))

            # For distribution reasons, between the different test sets, we try to break
            # the ROIs into smaller ones. currenlty assuming const size of 12,000 pixels
            small_roi_list = self.get_small_roi_list(roi_candidate)

            # iterate over the labels_dict and build the dataset object list
            # Append ROI only if it has associated labels

            for roi in small_roi_list:

                for label_data in labels_dict:


                    # Verify if label belongs to the current ROI
                    label_center_x = label_data['x_gl'] + label_data['w'] / 2.
                    label_center_y = label_data['y_gl'] + label_data['h'] / 2.

                    if (((roi.x - roi_margin) < label_center_x) and (
                            (roi.x + roi.w + roi_margin) > label_center_x) and (
                            (roi.y - roi_margin) < label_center_y) and ((roi.y + roi.h + roi_margin) > label_center_y)):
                        # Label is part of current ROI

                        # Verify this is a required label, that is participating in the training process
                        if label_data['label_name'][0] in self.flags.assert_classes_labels:
                            raise ValueError(
                                'Error. Found {0} label in session {1}, scan {2} that is in not allowed labels list'.format(
                                    label_data['label_name'][0], session, scan))

                        if len(self.flags.labels_participating_in_training) > 0:
                            if label_data['label_name'][0] not in self.flags.labels_participating_in_training:
                                logger.info(
                                    'Warning. Found {0} label in session {1}, scan {2} that is not part of the participating labels'.format(
                                        label_data['label_name'][0], session, scan))
                                continue

                        # OK, we can continue
                        label_data['database_id'] = label_data['id']
                        label_data['contour_line'] = None
                        label_data['nucleus_line'] = None

                        # Check if segmentation data is provided to current label, and add it.
                        segmentation_data = self.get_segmentation_data_by_database_id(label_data['id'])
                        if segmentation_data:
                            if (segmentation_data['center_x'] != int(label_center_x)) or (segmentation_data['center_y'] != int(label_center_y)):
                                raise ValueError('Segmentation top left x/y sanity failed for ID = {0}'.format(label_data['id']))
                            label_data['contour_line'] = segmentation_data['contour_line']
                            label_data['nucleus_line'] = segmentation_data['nucleus_line']

                        roi.labels.append(label_data)
                        empty_roi = False

                # # Check #4
                # if empty_roi:
                #     self.terminate_build_and_report_error(session=session,
                #                                           scan=scan,
                #                                           roi_name=roi.name,
                #                                           error_message='ROI has no labels. This may be a reason for some error, or, after automatic division there were no labels. try adding dirt labels in very empty areas')
                # Finally, we can add the ROI :-)
                scan_dataset.roi_list.append(copy.deepcopy(roi))

            # Verify that a big ROI (before it was split to small ROI), if its not empty - that it was unique
            if not empty_roi:
                # This is a unique sanity check.
                # If ROI is not unique this break to program and the user is expected to fix the bug.
                self._check_unique_roi(roi_candidate)

        # Check there are at least 2 ROI (after split) when generating data per ROI
        if self.flags.scan_train_test_split_level == 'ROI':
            if len(scan_dataset.roi_list) == 1:
                self.terminate_build_and_report_error(session=session,
                                                      scan=scan,
                                                      roi_name=None,
                                                      error_message='Scan does not have 2 valid ROIs')

        return scan_dataset, scan_info

    def get_segmentation_data_by_database_id(self, database_id):

        # Check if label id is in segmentation db:
        if str(database_id) in self.segmentation_db:
            return self.segmentation_db[str(database_id)]
        else:
            return None

    # Mapping of labeles from dataset
    def get_mapped_db_label(self, db_label):

        mapped_db_label = db_label

        for mapping_rule in self.flags.database_labels_mapping:
            if db_label in mapping_rule.keys():
                mapped_db_label = mapping_rule[db_label]
                break #To support loopable mappings

        return mapped_db_label

    # Filter scans
    def filter_dataset_scan_subset(self, in_dataset, scan_list):
        dataset = []
        for scan in in_dataset:
            if not scan.scan_id in scan_list:
                continue
            dataset.append(scan)
        return dataset

    # Maping and removing (filtering) labels
    def map_and_remove_non_participating_labels(self, dataset):
        for scan in dataset:
            for roi in scan.roi_list:
                filtered_labels = []
                for label in roi.labels:
                    label['orig_label_name'] = copy.deepcopy(label['label_name'])
                    class_key = copy.deepcopy(label['label_name'][0])

                    # Apply mapping
                    mapped_class_key = self.get_mapped_db_label(class_key)
                    if mapped_class_key is not class_key:
                        class_key = mapped_class_key
                        label['label_name'] = [class_key]

                    # Remove non-participating after mapping
                    if len(self.flags.labels_participating_in_training) > 0:
                        if class_key not in self.flags.labels_participating_in_training:
                            continue
                    filtered_labels.append(label)
                roi.labels = filtered_labels

    # Splitting big ROI - into smaller ROI regions.
    def get_small_roi_list(self, roi_candidate):

        # Splitting into small rois if core work at training per ROI.
        # If core works per label, no need to split the ROI
        roi_candidate_list = []

        if self.flags.scan_train_test_split_level == 'LABEL':
            roi_candidate_list.append(roi_candidate)

        else:
            # Training per ROI - thus we try to split ROI into smaller regions to enable distribution
            width_division = max(int(roi_candidate.w / float(self.flags.min_roi_size)), 1)
            height_division = max(int(roi_candidate.h / float(self.flags.min_roi_size)), 1)

            basic_width = int(roi_candidate.w / width_division)
            basic_height = int(roi_candidate.h / height_division)

            sub_roi_count = 0
            for x_step in range(0, width_division):
                for y_step in range(0, height_division):
                    coordinate_sub_roi = {'x': roi_candidate.x + x_step * basic_width,
                                          'y': roi_candidate.y + y_step * basic_height,
                                          'width': basic_width,
                                          'height': basic_height}

                    sub_roi_count += 1
                    sub_roi_candidate = RoiData(coordinates=coordinate_sub_roi,
                                                roi_id=roi_candidate.id,
                                                roi_name=roi_candidate.name + '_auto_divided' + str(sub_roi_count),
                                                scan=roi_candidate.scan,
                                                session=roi_candidate.session)

                    roi_candidate_list.append(sub_roi_candidate)

        return roi_candidate_list

    def terminate_build_and_report_error(self, session, scan, roi_name, error_message):

        full_error_msg = error_message + 'Session {0} | scan {1} | ROI name {2}'.format(session, scan, roi_name)

        logger.info('Detected error during build that requires your attention')
        logger.info(error_message)
        logger.info('Session {0} | scan {1} | ROI name {2}'.format(session, scan, roi_name))
        raise ValueError('Build terminated.' + full_error_msg)

    def get_train_eval_datasets(self):
        # Returns training and evaluation dataset
        # Read the PKL files and return

        # Set training and evaluation dataset file names
        train_dataset_pkl_file_name = os.path.join(self.config_params['database_dataset_path'], 'training_dataset.pkl')
        eval_dataset_pkl_file_name = os.path.join(self.config_params['database_dataset_path'], 'evaluation_dataset.pkl')

        if os.path.isfile(train_dataset_pkl_file_name):
            logger.info('Load %s ...'%train_dataset_pkl_file_name)
            training_dataset = pickle.load(open(train_dataset_pkl_file_name, "rb"))
        else:
            raise ValueError(
                'You are working at load_train_eval_datasets=True mode. Could not find training dataset in {0}'.format(
                    train_dataset_pkl_file_name))

        # train_datset_subset = self.filter_dataset_scan_subset(training_dataset, [1007, 1320, 825, 1334])
        # train_dataset_pkl_file_name = os.path.join(self.config_params['database_dataset_path'], 'training_dataset_sub.pkl')
        # pickle.dump(train_datset_subset, open(train_dataset_pkl_file_name, "wb"))
        # raise ValueError('END')

        if os.path.isfile(eval_dataset_pkl_file_name):
            logger.info('Load %s ...'%eval_dataset_pkl_file_name)
            evaluation_dataset = pickle.load(open(eval_dataset_pkl_file_name, "rb"))
        else:
            raise ValueError(
                'You are working at load_train_eval_datasets=True mode. Could not find test dataset in {0}'.format(
                    eval_dataset_pkl_file_name))

        return training_dataset, evaluation_dataset

    def get_test_datasets(self):
        # Returns training and evaluation dataset
        # Read the PKL files and return

        # Set testing dataset file names
        test_dataset_pkl_file_name = os.path.join(self.config_params['database_dataset_path'], 'testing_dataset.pkl')

        if os.path.isfile(test_dataset_pkl_file_name):
            testing_dataset = pickle.load(open(test_dataset_pkl_file_name, "rb"))
        else:
            raise ValueError(
                'Could not find testing dataset in {0}'.format(test_dataset_pkl_file_name))

        return testing_dataset

    def create_train_test_datasets(self,
                                   sessions_and_scans=None,
                                   output_dir=None):

        # There are 2 options:
        # 1. Create new dataset PKL and new training PKL and testing PKL datasets
        # 2. Load dataset PKL and create new training PKL and testing PKL dataset
        load_database_dataset = self.config_params['load_database_dataset']
        create_new_dataset = self.config_params['generate_training_data_from_local_db']

        merge_loaded_dataset = self.config_params.get('merge_loaded_dataset', True)  # Will merge scans that appear in several sessions
        logger.info('Creating new data folder at {0}'.format(output_dir))
        if not os.path.exists(output_dir):
            # raise ('{0} is already existing. Please provide non-existing dir'.format(output_dir))
            # shutil.rmtree(output_dir)
            os.makedirs(output_dir)

        # Set training and testing dataset file names
        train_dataset_pkl_file_name = os.path.join(output_dir, 'training_dataset.pkl')
        eval_dataset_pkl_file_name = os.path.join(output_dir, 'evaluation_dataset.pkl')
        test_dataset_pkl_file_name = os.path.join(output_dir, 'testing_dataset.pkl')

        if load_database_dataset:
            dataset = pickle.load(open(self.config_params['dataset_pkl_file_name'], "rb"))
            self.analyze_dataset(dataset=dataset,
                                 name='in_dataset',
                                 save_path=output_dir)
            if merge_loaded_dataset:
                filtered_dataset_by_session_and_scans = dataset

        # dataset_pkl_file_name = os.path.join(output_dir, 'in_dataset.pkl')
        # pickle.dump(loaded_dataset, open(dataset_pkl_file_name, "wb"))

        else:  # not loaded, building new dataset:
            merge_loaded_dataset = True
            filtered_dataset_by_session_and_scans = []
            for session_id, scan_list_dict in sessions_and_scans.iteritems():
                session = int(session_id)
                scan_list = scan_list_dict['scans']

                # ROI name filter list as provided in the sessions_and_scans dict
                filter_list = scan_list_dict['positive_annotations_filters']

                # Get species
                try:
                    species = scan_list_dict['species']
                except KeyError:
                    species = 'unknown_species'

                # Get session_name
                try:
                    session_name = scan_list_dict['session_name']
                except KeyError:
                    session_name = None

                # Get scanning_device
                try:
                    scanning_device = scan_list_dict['scanning_device']
                except KeyError:
                    scanning_device = 'alpha'

                # Get scanning_device
                try:
                    session_signed_off = scan_list_dict['signed_off']
                except KeyError:
                    session_signed_off = True # Assume session is valid

                # Only work on valid sessions!
                if not scan_list or scan_list == [] or not session_signed_off:
                    # If user has not provided a filter, ask the DB for all associated scans
                    # scan_list = self.labels_db.get_session_scans(session)

                    # New behavior - if sessions is provided with no scans, skip it.
                    print ('Session {0} is removed from dataset because -- session_signed_off: {1} | scan_list: {2}'.format(session, session_signed_off,scan_list))
                    continue


                for scan in scan_list:
                    # This is a newly created dataset
                    logger.info('Processing session %s and scan %d' % (session, scan))

                    scan_dataset, scan_info = self.get_scan_dataset(session = session,
                                                                    scan = scan,
                                                                    filter_list = filter_list,
                                                                    species = species,
                                                                    session_name = session_name,
                                                                    scanning_device = scanning_device)

                    if scan_dataset.roi_list != []:
                        filtered_dataset_by_session_and_scans.append(scan_dataset)

        if merge_loaded_dataset:
            dataset = []
            dataset_map = {}
            dataset_scans_list = []
            for scan_dataset in filtered_dataset_by_session_and_scans:
                if len(scan_dataset.roi_list) == 0:
                    continue
                scan_id = scan_dataset.scan_id
                if not dataset_map.has_key(scan_id):
                    dataset_scans_list.append(scan_id)
                    dataset_map[scan_id] = {}
                    dataset_map[scan_id]['roi_id_scan_map'] = {}
                    dataset_map[scan_id]['dataset'] = copy.deepcopy(scan_dataset)
                    dataset_map[scan_id]['dataset'].roi_list = []
                else:
                    logger.info('merge to one scan dataset - scan %d' % scan_id)
                for roi in scan_dataset.roi_list:
                    if len(roi.labels) > 0:
                        if not dataset_map[scan_id]['roi_id_scan_map'].has_key(roi.id):
                            dataset_map[scan_id]['dataset'].roi_list.append(copy.deepcopy(roi))
                            dataset_map[scan_id]['roi_id_scan_map'][roi.id] = scan_id
                        else:
                            prev_scan_id = dataset_map[scan_id]['roi_id_scan_map'][roi.id]
                            raise ValueError('ROI %d appears more than one time (scans %d and %d)' %
                                             (roi.id, prev_scan_id, scan_id))
            for scan_id in dataset_scans_list:
                dataset.append(dataset_map[scan_id]['dataset'])

            # Save a copy of the build query dataset (in both cases of load or newly read)
            dataset_pkl_file_name = os.path.join(output_dir, core_path.FILTERED_DATASET_PICKLE_FILE_NAME)
            pickle.dump(dataset, open(dataset_pkl_file_name, "wb"))

        # Saving statistics. At this point we have a complete dataset filtered by sessions and scans
        self.correct_roi_names(dataset)

        self.analyze_dataset(dataset=dataset,
                             name='filtered_sessions_scans',
                             save_path=output_dir)

        # Splitting the dataset into training and testing dataset
        training_and_eval_dataset, testing_dataset, train_test_in_data_info = \
            self.split_dataset_for_training_testing(dataset,
                                                    training_probability=1-self.flags.testing_dataset_ratio,
                                                    prev_in_data_info=None,
                                                    out_dir=output_dir, scan_split_str='train_test')

        eval_dataset_ratio = 1. - self.flags.training_dataset_ratio - self.flags.testing_dataset_ratio
        training_dataset_ratio =  self.flags.training_dataset_ratio / (self.flags.training_dataset_ratio + eval_dataset_ratio)
        training_dataset, evaluation_dataset, _ = \
            self.split_dataset_for_training_testing(training_and_eval_dataset,
                                                    training_probability=training_dataset_ratio,
                                                    prev_in_data_info=train_test_in_data_info,
                                                    out_dir=output_dir, scan_split_str='train_eval')

        pickle.dump(training_dataset, open(train_dataset_pkl_file_name, "wb"))
        pickle.dump(evaluation_dataset, open(eval_dataset_pkl_file_name, "wb"))
        pickle.dump(testing_dataset, open(test_dataset_pkl_file_name, "wb"))
        logger.info('Saved training dataset at {0}'.format(train_dataset_pkl_file_name))
        logger.info('Saved evaluation dataset at {0}'.format(eval_dataset_pkl_file_name))
        logger.info('Saved testing dataset at {0}'.format(test_dataset_pkl_file_name))

        # Analyze the newly splitted datasets:
        self.analyze_dataset(dataset=training_dataset,
                             name='training',
                             save_path=output_dir)

        self.analyze_dataset(dataset=evaluation_dataset,
                             name='evaluation',
                             save_path=output_dir)

        self.analyze_dataset(dataset=testing_dataset,
                             name='testing',
                             save_path=output_dir)

    def create_segmentation_dataset(self, dataset, out_data_path):
        seg_dataset_pkl_file_name = out_data_path + '/training_segmentation_dataset.pkl'
        calc_seg_info = True
        if calc_seg_info:
            self.correct_roi_names(dataset)
            db_info = {}
            db_info['dataset'] = dataset
            db_info['data_dir'] = out_data_path
            db_info['scan_id_pyramids_path'] = self.config_params['scan_id_pyramids_path']
            db_info['scan_uuid_pyramids_path'] = self.config_params['scan_uuid_pyramids_path']
            seg_dataset = self.user_tfrecord.create_dataset_with_seg_target_func_data(db_info)
            pickle.dump(seg_dataset, open(seg_dataset_pkl_file_name, "wb"))
        else:
            seg_dataset = pickle.load(open(seg_dataset_pkl_file_name, "rb"))

        return seg_dataset

    def create_tfrecords_dataset(self,
                                 training_dataset=None,
                                 evaluation_dataset=None,
                                 testing_dataset=None,
                                 reference_analysis_info=None,
                                 output_dir=None,
                                 project_name='',
                                 pyramid_level=1,
                                 mode=None,
                                 progress_manager=None):
        """
        A method to create TFRecords from provided scans.
        output_dir: the directory to output the scans TFRecords.
        sessions_and_scans : a dictionary of sessions and scans. See user_tfrecords for an example.
        image_tile_size : set this to any integer value to override image size extracted from the DB

        """
        if mode != self.flags.training_mode:
            raise ValueError('create_tfrecords_dataset is only used when building TFRecords for training phase')

        # logger.info('Creating new data folder at {0}'.format(output_dir))
        # if os.path.exists(output_dir):
        #     raise ('{0} is already existing. Please provide non-existing directory'.format(output_dir))
        #     # shutil.rmtree(output_dir)
        # os.makedirs(output_dir)
        if not os.path.exists(output_dir):
           os.makedirs(output_dir)

        # subset_scan_list = range(290) + [304, 305, 313, 314, 315, 336, 344, 346,
        #                                  354, 355, 356, 366, 367, 368, 395, 396, 397]
        # train_subset_scan_list = range(100,140)
        # eval_subset_scan_list = range(100,140)

        set_names = []
        pm_children_num = 0
        train_eval = (training_dataset is not None) and (evaluation_dataset is not None)
        test = (testing_dataset is not None)

        if train_eval:
            # full_dataset = training_dataset+evaluation_dataset
            # full_dataset = self.filter_dataset_scan_subset(full_dataset, subset_scan_list)
            # subset_scan_list = [718]
            # subset_scan_list = [249]
            # training_dataset = self.filter_dataset_scan_subset(training_dataset, subset_scan_list)
            # subset_scan_list = [720]
            # evaluation_dataset = self.filter_dataset_scan_subset(evaluation_dataset, subset_scan_list)

            # self.correct_roi_names(full_dataset)
            # self.map_and_remove_non_participating_labels(full_dataset)  # TBD check if you have enough labels
            # training_dataset = training_dataset[]
            # training_dataset = [training_dataset[26]]
            self.correct_roi_names(training_dataset)
            self.map_and_remove_non_participating_labels(training_dataset)  # TBD check if you have enough labels
            self.correct_roi_names(evaluation_dataset)
            self.map_and_remove_non_participating_labels(evaluation_dataset)
            # self.analyze_dataset(dataset=full_dataset, name='all', save_path=output_dir)

            if self.config_params['segmentation_mode']:
                out_data_path = os.path.join(output_dir, 'training')
                training_dataset = self.create_segmentation_dataset(training_dataset, out_data_path)

            self.analyze_dataset(dataset=training_dataset, name='train', save_path=output_dir)
            self.analyze_dataset(dataset=evaluation_dataset, name='eval', save_path=output_dir)
            set_names.extend(['training', 'evaluation_conflicts', 'training_conflicts'])
            pm_children_num += 3
            # set_names.extend(['training_conflicts'])
            #pm_children_num += 1
            #set_names.extend(['evaluation_conflicts'])
            #pm_children_num += 1
            #set_names.extend(['training', 'training_conflicts'])
            #pm_children_num += 2
            # set_names.extend(['training'])
            # pm_children_num += 1

        if test:
            self.correct_roi_names(testing_dataset)
            self.map_and_remove_non_participating_labels(testing_dataset)  # TBD check if you have enough labels

            # subset_scan_list = []
            # testing_dataset = self.filter_dataset_scan_subset(testing_dataset, subset_scan_list)

            self.analyze_dataset(dataset=testing_dataset, name='test', save_path=output_dir)
            set_names.extend(['testing_conflicts'])
            pm_children_num += 1

        # Create 4 progress manager children
        progress_manager.create_children(pm_children_num)
        # set_names = ['training_conflicts']
        for index, set in enumerate(set_names):
            set_progress_manager = progress_manager.children[index]

            tfrecord_path = os.path.join(output_dir, set)
            if not os.path.exists(tfrecord_path):
                os.makedirs(tfrecord_path)

            logger.info('Preparing {0} Tfrecords to be written to {1}'.format(set, tfrecord_path))

            # Set specific attributes for get tiles function:
            if set in ['training', 'evaluation']:
                get_tiles_mode = self.flags.training_mode
            else:
                get_tiles_mode = self.flags.test_mode

            if set in ['training', 'training_conflicts']:
                get_tiles_data_set = training_dataset
            elif set in ['evaluation', 'evaluation_conflicts']:
                get_tiles_data_set = evaluation_dataset
            else:
                get_tiles_data_set = testing_dataset

            # Show and save statistics
            self.analyze_dataset(get_tiles_data_set, set, save_path=tfrecord_path)

            # with open(set_stat_file, 'w') as f:
            #     f.write(json.dumps(set_stat, sort_keys=True, indent=4))

            if (set == 'training'):
                if self.flags.prepare_ref_analysis_info and (reference_analysis_info is not None):
                    ref_analysis_info = \
                        self.user_tfrecord.prepare_reference_analysis_info(training_dataset,
                                                                           reference_analysis_info,
                                                                           tfrecord_path,
                                                                           self.config_params['database_dataset_path'])
                else:
                    ref_analysis_info = reference_analysis_info
            else:
                ref_analysis_info = None

            tiles_list = self.user_tfrecord.get_tiles(data_set=get_tiles_data_set,
                                                      mode=get_tiles_mode,
                                                      reference_analysis_info=ref_analysis_info,
                                                      data_dir_path = tfrecord_path)
            logger.info('Gathering user tiles completed')

            # Generate a TFRecord file for every ROI in the training dataset
            logger.info('Gathering images for tiles')
            self.generate_tf_blueprints = True

            # Deciding from where to read the images:
            # For training there are 3 valid sources: DB, direct pyramid path or JPEG.

            # 1. From DB (a.k.a. DB_Export)
            from_db = self.open_local_database_link

            # 2. From pyramid (directly from the pyramid location)
            from_pyramid = not from_db
            if from_pyramid:
                self.training_scan_id_pyramids_directory = self.config_params['scan_id_pyramids_path']
                self.training_scan_uuid_pyramids_directory = self.config_params['scan_uuid_pyramids_path']

            # 3. From external JPEG
            from_jpeg = False  # Not enabled

            self.convert_to_tfrecord(mode=self.flags.training_mode,

                                     tiles_list=tiles_list,

                                     # Set image source

                                     from_db=from_db,  # if open_local_database_link, we are reading from the DB
                                     from_pyramid=from_pyramid,
                                     from_jpeg=from_jpeg,

                                     output_file=None,  # Build filename automatically
                                     output_dir=tfrecord_path,

                                     project_name=project_name,
                                     pyramid_level=pyramid_level,
                                     progress_manager=set_progress_manager)

        # cv_log(message = 'Completed building training dataset')

    def set_dataset_groups_info(self, dataset, dataset_stat):
        dataset_stat['scan_groups_lists'] = {}
        dataset_stat['scan_groups_lists']['unknown_group'] = []
        if self.flags.scan_groups_type == 'scan_name_group_info':
            dataset_stat['scan_groups_ratios'] = self.flags.scan_groups_ratio
            for name_signature in self.flags.scan_groups_name_signature:
                dataset_stat['scan_groups_lists'][name_signature] = []

        for scan in dataset:
            group_set = False
            if self.flags.scan_groups_type == 'scan_name_group_info':
                scan_name = scan.scan_name
                for name_signature in self.flags.scan_groups_name_signature:
                    if name_signature in scan_name.lower():
                        dataset_stat['scan_groups_lists'][name_signature].append(scan.scan_id)
                        group_set = True
                        break
            # Default - no groups were set assuming that distribution of the input dataset is similar to the
            # required target world test and evaluation distribution
            if not group_set:
                dataset_stat['scan_groups_lists']['unknown_group'].append(scan.scan_id)

    def analyze_dataset(self, dataset, name=None, save_path=None):

        # Caculates basic dataset statistics
        dataset_stat = {}

        dataset_stat['global_num_of_scans'] = 0
        dataset_stat['global_num_of_rois'] = 0
        dataset_stat['global_num_of_labels'] = 0
        dataset_stat['global_class_stat'] = {}
        dataset_stat['per_scan_stat'] = {}

        for scan in dataset:
            dataset_stat['global_num_of_scans'] += 1
            dataset_stat['per_scan_stat'][scan.scan_id] = {}
            dataset_stat['per_scan_stat'][scan.scan_id]['num_of_rois'] = 0
            dataset_stat['per_scan_stat'][scan.scan_id]['roi_pointers'] = []
            dataset_stat['per_scan_stat'][scan.scan_id]['classes'] = {}

            for roi in scan.roi_list:
                dataset_stat['global_num_of_rois'] += 1
                dataset_stat['per_scan_stat'][scan.scan_id]['num_of_rois'] += 1
                # Init all ROIs to be marked with 'For training'
                roi.for_training = True
                dataset_stat['per_scan_stat'][scan.scan_id]['roi_pointers'].append(roi)

                for label in roi.labels:

                    class_key = copy.deepcopy(label['label_name'][0])
                    # Filter non-required labels
                    if len(self.flags.labels_participating_in_training) > 0:
                        if class_key not in self.flags.labels_participating_in_training:
                            continue

                    # Init all labels to be marked with 'For training'
                    dataset_stat['global_num_of_labels'] += 1

                    # Update global statistics and per scan statistics
                    if class_key in dataset_stat['global_class_stat']:
                        dataset_stat['global_class_stat'][class_key] += 1
                    else:
                        dataset_stat['global_class_stat'][class_key] = 1

                    if class_key in dataset_stat['per_scan_stat'][scan.scan_id]['classes']:
                        dataset_stat['per_scan_stat'][scan.scan_id]['classes'][class_key]['count'] += 1
                        dataset_stat['per_scan_stat'][scan.scan_id]['classes'][class_key]['pointers'].append(label)
                    else:
                        dataset_stat['per_scan_stat'][scan.scan_id]['classes'][class_key] = {}
                        dataset_stat['per_scan_stat'][scan.scan_id]['classes'][class_key]['count'] = 1
                        dataset_stat['per_scan_stat'][scan.scan_id]['classes'][class_key]['pointers'] = [label]

        self.set_dataset_groups_info(dataset, dataset_stat)

        logger.info('Analyzing {0} dataset'.format(name))
        logger.info(('global_num_of_scans', dataset_stat['global_num_of_scans']))
        logger.info(('global_num_of_rois', dataset_stat['global_num_of_rois']))
        logger.info(('global_num_of_labels', dataset_stat['global_num_of_labels']))
        logger.info(('global_class_stat', dataset_stat['global_class_stat']))

        if save_path:
            # Prepare a copy of the dataset stat to be printed nicely
            dataset_stat_for_print = copy.deepcopy(dataset_stat)
            for scan_key in dataset_stat_for_print['per_scan_stat']:
                dataset_stat_for_print['per_scan_stat'][scan_key].pop('roi_pointers', None)
                for class_key in dataset_stat_for_print['per_scan_stat'][scan_key]['classes']:
                    dataset_stat_for_print['per_scan_stat'][scan_key]['classes'][class_key].pop('pointers', None)

            set_stat_file = os.path.join(save_path, name + '_statistics.json')
            with open(set_stat_file, 'w') as f:
                f.write(json.dumps(dataset_stat_for_print, sort_keys=True, indent=4))

        return dataset_stat

    def split_dataset_for_training_testing(self, dataset, training_probability=None, prev_in_data_info = None,
                                           progress_manager=None,
                                           out_dir=None, scan_split_str=None):

        # Splitting a dataset for training and test sets.
        in_data_info = None
        training_dataset = []
        testing_dataset = []

        # Initialize additional attributes for splitting
        for scan in dataset:
            for roi in scan.roi_list:
                for label in roi.labels:
                    label['for_training'] = True
                    label['ROI_id'] = roi.id

        if math.fabs(training_probability - 1.) < 1e-4:
            training_dataset = dataset
            return training_dataset, testing_dataset

        dataset_stat = self.analyze_dataset(dataset, 'complete set')

        # First calculate the sampling ratio
        test_ratio = 1. - training_probability
        test_ratio_limit = float(self.flags.test_set_size) / (dataset_stat['global_num_of_labels'])
        if test_ratio > test_ratio_limit:
            test_ratio = test_ratio_limit

        # Splitting the dataset per LABEL.
        # 1. Get total number of labels in complete dataset
        # 2. Get the total number of labels we would like to have in the test set (This is a number)
        # 3. from (1) and (2) calculate P which is test_size / dataset_size
        # 4. For each class calculate how manny representatives will be in the test set := P * #of total class members in dataset
        # 5. Sample per scan per class and add the label to the dataset.

        if self.flags.scan_train_test_split_level in ['ROI', 'ROI_per_SCAN']:
            # When working at per ROI, we are adding a mark per ROI if its part of the test or train set.
            # We take at least 1 ROI per scan. And continue add ROIs in a round-robin until test ratio is achieved.

            for scan in dataset:

                rois_in_scan = dataset_stat['per_scan_stat'][scan.scan_id]['num_of_rois']
                if (rois_in_scan == 2) or self.flags.scan_train_test_split_level == 'ROI_per_SCAN':
                    # select only 1 ROI in this scan to add to test
                    number_of_test_member = 1
                else:
                    number_of_test_member = int(math.ceil(max(test_ratio * rois_in_scan, 1)))

                test_members = random.sample(dataset_stat['per_scan_stat'][scan.scan_id]['roi_pointers'],
                                             number_of_test_member)
                for member in test_members:
                    member.for_training = False

        elif self.flags.scan_train_test_split_level == 'ROI total labels ratio':

            training_dataset, testing_dataset, in_data_info = \
                self.split_train_test_datasets_by_roi_total_labels(dataset, out_dir,
                                                                   training_probability,
                                                                   prev_in_data_info,
                                                                   scan_split_str)

        elif self.flags.scan_train_test_split_level == 'LABEL':

            # When working at per label, we are adding a mark per label if its part of the test or train set.

            for scan in dataset:
                for class_key in dataset_stat['per_scan_stat'][scan.scan_id]['classes']:

                    class_members_in_scan = dataset_stat['per_scan_stat'][scan.scan_id]['classes'][class_key]['count']
                    if class_members_in_scan == 1:
                        # This means that there is only 1 label per class in this scan - have it marked for training
                        dataset_stat['per_scan_stat'][scan.scan_id]['classes'][class_key]['pointers'][0][
                            'for_training'] = True
                    else:
                        number_of_test_member = int(math.ceil(max(test_ratio * class_members_in_scan, 1)))
                        test_members = random.sample(
                            dataset_stat['per_scan_stat'][scan.scan_id]['classes'][class_key]['pointers'],
                            number_of_test_member)
                        for member in test_members:
                            member['for_training'] = False

        elif self.flags.scan_train_test_split_level == 'SCAN':

                if self.flags.simple_random_scans_split:
                    training_dataset, testing_dataset = \
                        self.split_train_test_datasets_by_random_scans_subset(dataset, training_probability)
                else:
                    training_dataset, testing_dataset = \
                        self.split_train_test_datasets_by_scans(dataset, dataset_stat, out_dir,
                                                                training_probability, scan_split_str)

        # Splits a dataset into 2 datasets for training and testing
        if not self.flags.scan_train_test_split_level in ['SCAN','ROI total labels ratio']:
            for scan in dataset:
                new_training_scan, new_testing_scan = scan.split_scan(training_probability=training_probability,
                                                                      split_level=self.flags.scan_train_test_split_level)
                training_dataset.append(new_training_scan)
                testing_dataset.append(new_testing_scan)

        return training_dataset, testing_dataset, in_data_info

    def split_train_test_datasets_by_roi_total_labels(self, dataset, out_dir,
                                                      training_probability,
                                                      prev_in_data_info,
                                                      scan_split_str):

        # Gather roi and total stat
        info = {}
        info['total'] = {}
        info['roi'] = {}
        info['roi_num'] = 0
        info['train_prob'] = training_probability
        roi_id_list = []
        split_keys = self.flags.dataset_split_criteria.keys()
        roi_idx = 0
        for scan_dataset in dataset:
            for roi in scan_dataset.roi_list:
                info['roi_num'] += 1
                info['roi'][roi.id] = {}
                info['roi'][roi.id]['scan_id'] = scan_dataset.scan_id
                info['roi'][roi.id]['classes'] = {}
                info['roi'][roi.id]['idx'] = roi_idx
                roi_id_list.append(roi.id)
                roi_idx += 1
                for label in roi.labels:
                    label_name = str(label['label_name'][0])
                    if label_name in split_keys:
                        if not info['total'].has_key(label_name):
                            info['total'][label_name] = 0
                        info['total'][label_name] += 1
                        if not info['roi'][roi.id]['classes'].has_key(label_name):
                            info['roi'][roi.id]['classes'][label_name] = 0
                        info['roi'][roi.id]['classes'][label_name] += 1

        assert (info['roi_num'] >= 2), 'not enough ROIs - %d'%info['roi_num']

        # Find best split according to split criteria
        test_ratio = 1. - training_probability
        use_prev_test_ratio = prev_in_data_info is not None
        if use_prev_test_ratio:
            class_test_ratio =  test_ratio * prev_in_data_info['train_prob']
        else:
            class_test_ratio = test_ratio
        test_roi_num = [max(1, int(info['roi_num'] * (test_ratio - 0.03))),
                        max(1, int(info['roi_num'] * (test_ratio + 0.03))) + 1]
        max_iters = self.flags.scan_split_iterations
        split_results = []
        for iter_idx in xrange(max_iters):
            if iter_idx % 10000 == 9999:
                logger.info(iter_idx + 1)
            test_subset_size = np.random.randint(low=test_roi_num[0], high=test_roi_num[1], size=1)
            test_roi_subset = np.random.choice(a=info['roi_num'], size=test_subset_size,replace=False)
            scan_roi_info = {}
            for roi_idx in test_roi_subset:
                roi_id = roi_id_list[roi_idx]
                scan_id = info['roi'][roi_id]['scan_id']
                if not scan_roi_info.has_key(scan_id):
                    scan_roi_info[scan_id] = []
                scan_roi_info[scan_id].append(roi_id)
            total_score = 1.
            test_labels_summary = {}
            roi_train_map = np.ones((info['roi_num']), dtype='int32')
            for class_str in split_keys:
                test_scans_hist = {}
                test_labels_num = 0
                all_labels_num = prev_in_data_info['total'].get(class_str, 0) if use_prev_test_ratio \
                    else info['total'].get(class_str, 0)
                if all_labels_num == 0:
                    continue
                for roi_idx in test_roi_subset:
                    roi_train_map[roi_idx] = 0
                    roi_id = roi_id_list[roi_idx]
                    roi_info = info['roi'][roi_id]
                    test_labels_num += roi_info['classes'].get(class_str, 0)
                    if not test_scans_hist.has_key(roi_info['scan_id']):
                        test_scans_hist[roi_info['scan_id']] = 0
                    test_scans_hist[roi_info['scan_id']] += test_labels_num
                test_labels_ratio = float(test_labels_num) / all_labels_num
                criteria = self.flags.dataset_split_criteria[class_str]
                mean_val = criteria[0] * class_test_ratio
                min_val = criteria[1] * class_test_ratio
                max_val = criteria[2] * class_test_ratio
                sigma = criteria[3] * class_test_ratio
                scan_labels_values = np.array(test_scans_hist.values())
                if np.sum(scan_labels_values) > 0:
                    scan_labels_prob = scan_labels_values / np.sum(scan_labels_values)
                    scan_labels_prob = np.maximum(1e-15, scan_labels_prob)
                    scan_labels_prob = scan_labels_prob / np.sum(scan_labels_prob)
                    scan_labels_entropy = -np.sum(scan_labels_prob * np.log(scan_labels_prob))
                else:
                    scan_labels_entropy = 0
                ent_factor = criteria[4]
                class_score = 0
                if (test_labels_ratio > min_val) and (test_labels_ratio < max_val):
                    diff = test_labels_ratio - mean_val
                    class_score = (1. - ent_factor) * np.exp(-(diff * diff) / (2 * sigma * sigma)) + \
                                  ent_factor * scan_labels_entropy
                total_score = total_score * class_score
                test_labels_summary[class_str] = {'labels_num': all_labels_num,
                                                  'test_labels': test_labels_num,
                                                  'test_labels_ratio': test_labels_ratio,
                                                  'scans_label_entropy': scan_labels_entropy,
                                                  'class_score': class_score}
                pass

            roi_split_info = {}
            roi_split_info['test_labels_summary'] = test_labels_summary
            roi_split_info['roi_train_map'] = roi_train_map.tolist()
            roi_split_info['total_score'] = total_score
            roi_split_info['scan_roi_info'] = scan_roi_info
            split_results.append(roi_split_info)

        best_roi_train_map = None
        best_split_info = None
        best_score = -1e9
        for idx, split_info in enumerate(split_results):
            if split_info['total_score'] > best_score:
                best_score = split_info['total_score']
                best_roi_train_map = split_info['roi_train_map']
                best_split_info = split_info

        training_dataset = []
        testing_dataset = []
        for scan_dataset in dataset:
            train_scan_dataset = scan_dataset.get_copy_without_roi_list()
            test_scan_dataset = scan_dataset.get_copy_without_roi_list()
            for roi in scan_dataset.roi_list:
                roi_idx = info['roi'][roi.id]['idx']
                roi_cpy = copy.deepcopy(roi)
                if best_roi_train_map[roi_idx] != 0:
                    train_scan_dataset.roi_list.append(roi_cpy)
                else:
                    test_scan_dataset.roi_list.append(roi_cpy)
            if len(train_scan_dataset.roi_list) > 0:
                training_dataset.append(train_scan_dataset)
            if len(test_scan_dataset.roi_list) > 0:
                testing_dataset.append(test_scan_dataset)

        if (out_dir is not None):
            best_split_info_file_name = out_dir + '/best_%s_split_info.json'%scan_split_str
            with open(best_split_info_file_name, 'w') as f:
                f.write(json.dumps(best_split_info, sort_keys=True, indent=4))

        return training_dataset, testing_dataset, info

    def split_train_test_datasets_by_random_scans_subset(self, dataset, training_probability):

        training_dataset = []
        testing_dataset = []

        scans_num = len(dataset)
        scan_idx_list = np.random.permutation(scans_num)
        test_scans_num = int(round((1. - training_probability) * scans_num))
        test_scan_idx_subset = scan_idx_list[0:test_scans_num]

        for scan_idx, scan in enumerate(dataset):
            if scan_idx in test_scan_idx_subset:
                testing_dataset.append(scan)
            else:
                training_dataset.append(scan)

        return training_dataset, testing_dataset

    def split_train_test_datasets_by_scans(self, dataset, dataset_stat, out_dir, training_probability, scan_split_str):

        training_dataset = []
        testing_dataset = []

        scan_groups_name_signature = ['unknown_group']
        if self.flags.scan_groups_type == 'scan_name_group_info':
            scan_groups_name_signature = self.flags.scan_groups_name_signature

        # Optionally load reference scan split info
        use_ref_scan_split_info = self.config_params['use_ref_scan_split_info']
        if use_ref_scan_split_info:
            if out_dir is not None:
                raise ValueError('Must provide out_dir for ref scan split usage')
            prev_split_info_file_name = out_dir + '/prev_split_info.json'
            with open(prev_split_info_file_name, 'r') as f:
                prev_split_info = json.load(f)
            prev_test_groups_map = prev_split_info['test_groups_map']
            use_ref_scan_split_test_set = self.config_params['use_ref_scan_split_test_set']
            if use_ref_scan_split_test_set:
                test_scans_list = []
                for group_name in prev_test_groups_map.keys():
                    test_scans_list.extend(prev_test_groups_map[group_name])
                for scan in dataset:
                    if scan.scan_id in test_scans_list:
                        testing_dataset.append(scan)
                    else:
                        training_dataset.append(scan)
                self.correct_roi_names(training_dataset)
                self.correct_roi_names(testing_dataset)
                return training_dataset, testing_dataset
            else:
                prev_test_scans = 0
                for group_name in prev_test_groups_map.keys():
                    prev_test_scans += len(prev_test_groups_map[group_name])
                new_test_size = np.sum(np.array(self.flags.scan_groups_ratio),
                                       dtype='int32') * self.flags.scan_split_test_factor
                if prev_test_scans > new_test_size:
                    use_ref_scan_split_info = False

        if not use_ref_scan_split_info:
            prev_test_groups_map = {}
            for idx, group_name_signature in enumerate(scan_groups_name_signature):
                prev_test_groups_map[group_name_signature] = []

        # Choose random scan split according to groups distribution restriction while optimizing for
        # the required test / train ratio preferences for the participating cells - seek for
        # the test scans ratio with interval constraint
        logger.info('Scan %s splitting - iterations:'%scan_split_str)
        scans_class_num_map = {}
        for scan_str in dataset_stat['per_scan_stat'].keys():
            scans_class_num_map[int(scan_str)] = {}
            for class_str in dataset_stat['per_scan_stat'][scan_str]['classes'].keys():
                scans_class_num_map[int(scan_str)][class_str] = \
                    dataset_stat['per_scan_stat'][scan_str]['classes'][class_str]['count']

        max_iters = self.flags.scan_split_iterations
        scans_num = len(dataset)
        split_results = []
        for iter_idx in xrange(max_iters):
            if iter_idx % 10000 == 9999:
                logger.info(iter_idx + 1)
            group_lists = copy.deepcopy(dataset_stat['scan_groups_lists'])
            scan_train_map = {}
            train_groups_map = {}
            test_groups_map = {}
            test_labels_summary = {}
            for scan in dataset:
                scan_train_map[scan.scan_id] = True

            if self.flags.scan_groups_type == 'scan_name_group_info':
                test_scans = self.set_scans_split_group_maps(train_groups_map, test_groups_map, test_scans, group_lists,
                                                scan_train_map, prev_test_groups_map)
            else:
                test_scans = self.set_scans_split_default_maps(train_groups_map, test_groups_map,
                                                               group_lists, scan_train_map,
                                                               prev_test_groups_map, 1. -  training_probability)

            scan_ratio = float(test_scans) / scans_num
            total_score = 1.
            for class_str in self.flags.dataset_split_criteria.keys():
                test_labels_num = 0
                all_labels_num = dataset_stat['global_class_stat'].get(class_str, 0)
                if all_labels_num == 0:
                    continue
                for scan_id, is_train in scan_train_map.iteritems():
                    if not is_train:
                        if scans_class_num_map[scan_id].has_key(class_str):
                            test_labels_num += scans_class_num_map[scan_id][class_str]
                test_labels_ratio = float(test_labels_num) / all_labels_num
                criteria = self.flags.dataset_split_criteria[class_str]
                mean_val = criteria[0] * scan_ratio
                min_val = criteria[1] * scan_ratio
                max_val = criteria[2] * scan_ratio
                sigma = criteria[3] * scan_ratio
                class_score = 0
                if (test_labels_ratio > min_val) and (test_labels_ratio < max_val):
                    diff = test_labels_ratio - mean_val
                    class_score = np.exp(-(diff * diff) / (2 * sigma * sigma))
                total_score = total_score * class_score
                test_labels_summary[class_str] = {'labels_num': all_labels_num,
                                                  'test_labels': test_labels_num,
                                                  'test_labels_ratio': test_labels_ratio,
                                                  'class_score': class_score}

            scan_split_info = {}
            scan_split_info['group_scans_permutation'] = group_lists
            scan_split_info['train_scans_map'] = scan_train_map
            scan_split_info['train_groups_map'] = train_groups_map
            scan_split_info['test_groups_map'] = test_groups_map
            scan_split_info['test_labels_summary'] = test_labels_summary
            scan_split_info['total_score'] = total_score
            split_results.append(scan_split_info)

        best_scan_train_map = None
        best_split_info = None
        best_score = -1e9
        for idx, split_info in enumerate(split_results):
            if split_info['total_score'] > best_score:
                best_score = split_info['total_score']
                best_scan_train_map = split_info['train_scans_map']
                best_split_info = split_info

        for scan in dataset:
            if best_scan_train_map[scan.scan_id]:
                training_dataset.append(scan)
            else:
                testing_dataset.append(scan)

        if (out_dir is not None):
            best_split_info_file_name = out_dir + '/best_%s_split_info.json'%scan_split_str
            with open(best_split_info_file_name, 'w') as f:
                f.write(json.dumps(best_split_info, sort_keys=True, indent=4))
            # _ = self.analyze_dataset(training_dataset, name='training', save_path=out_dir)
            # _ = self.analyze_dataset(testing_dataset, name='testing', save_path=out_dir)

        return training_dataset, testing_dataset

    def set_scans_split_group_maps(self, train_groups_map, test_groups_map, test_scans, group_lists,
                                   scan_train_map, prev_test_groups_map):

        for idx, group_name_signature in enumerate(self.flags.scan_groups_name_signature):
            train_groups_map[group_name_signature] = []
            test_groups_map[group_name_signature] = []
            list_all = group_lists[group_name_signature]
            size = self.flags.scan_groups_ratio[idx] * self.flags.scan_split_test_factor
            if len(list_all) < size * 2:  # at most 50% test ratio per group, should be at most 20-25%
                # raise ValueError('insufficient group scans for test')
                if (len(list_all) / 2) >= 1:
                    size = 1
                else:
                    continue
            non_prev_list = list(set(list_all).difference(set(prev_test_groups_map[group_name_signature])))
            shuffled_non_prev_list = copy.deepcopy(non_prev_list)
            random.shuffle(shuffled_non_prev_list)
            shuffled_list = prev_test_groups_map[group_name_signature] + shuffled_non_prev_list
            test_scans += size
            for scan_id in shuffled_list[0:size]:
                scan_train_map[scan_id] = False
                test_groups_map[group_name_signature].append(scan_id)
            for scan_id in shuffled_list[size:]:
                train_groups_map[group_name_signature].append(scan_id)

    def set_scans_split_default_maps(self, train_groups_map, test_groups_map, group_lists, scan_train_map,
                                     prev_test_groups_map, test_ratio):

        list_all = group_lists['unknown_group']
        test_scans = int(len(list_all) * test_ratio)
        non_prev_list = list(set(list_all).difference(set(prev_test_groups_map['unknown_group'])))
        shuffled_non_prev_list = copy.deepcopy(non_prev_list)
        random.shuffle(shuffled_non_prev_list)
        shuffled_list = prev_test_groups_map['unknown_group'] + shuffled_non_prev_list
        test_groups_map['unknown_group'] = []
        train_groups_map['unknown_group'] = []
        for scan_id in shuffled_list[0:test_scans]:
            scan_train_map[scan_id] = False
            test_groups_map['unknown_group'].append(scan_id)
        for scan_id in shuffled_list[test_scans:]:
            train_groups_map['unknown_group'].append(scan_id)

        return test_scans

    def check_scans_pyramid_data(self, dataset):
        for scan in dataset:
            scan_folder = '/scopio/scratch5/computer_vision/ichilov/scans/%d' % scan.scan_id
            if not os.path.exists(scan_folder):
                logger.info('Missing scan folder - %s' % scan_folder)

    def correct_roi_names(self, dataset):
        for scan in dataset:
            for roi in scan.roi_list:
                for chr in ['\n', '/', ':', ' ']:
                    if chr in roi.name:
                        roi.name = roi.name.replace(chr, '_')

    def get_DB_dataset_from_tile_dataset(self, tile_dataset, for_train=True):

        dataset = []
        for scan_info in tile_dataset:
            scan_data = ScanData(scan_id=scan_info['scan_id'],
                                 scan_name=scan_info['scan_name'],
                                 roi_list=[],
                                 species=scan_info['species'],
                                 scan_resolution=self.flags.model_resolution)
            for reg_info in scan_info['regions_list']:
                rect = reg_info['rect']
                coordinates = {'x': rect[0],
                               'y': rect[1],
                               'width': rect[2],
                               'height': rect[3]
                               }
                ROI_data = RoiData(coordinates=coordinates,
                                   roi_id=reg_info['ROI_id'],
                                   roi_name=reg_info['region_name'],
                                   scan=reg_info['scan_id'],
                                   session=reg_info['ROI_session'])
                ROI_data.for_training = for_train
                ROI_data.labels.extend(reg_info['target_labels'])
                scan_data.roi_list.append(ROI_data)
            dataset.append(scan_data)

        return dataset

    # Records Feature types
    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _feature_casting(self, value):
        if type(value) is int:
            return self._int64_feature(value)
        elif type(value) is float:
            return self._float_feature(value)
        else:
            return self._bytes_feature(value)

    def get_segmentation_region_mask(self, roi_rect, roi):
        mask_img_scale = roi['target_mask_scale']
        scale = mask_img_scale * self.flags.detection_image_scale
        if roi['target_mask_file'] != None:
            mask_img_file = os.path.join(self.config_params['labels_dir'], roi['target_mask_file'])
            mask_img = cv2.imread(mask_img_file)
            if mask_img.ndim == 3:
                mask_img = np.max(mask_img, axis=-1, keepdims=True)
            set_binary = False
            if set_binary:
                binary_th = 128
                mask_img[mask_img >= binary_th] = 255
                mask_img[mask_img < binary_th] = 0
            else:
                max_val = np.max(mask_img)
                if max_val > 0.01:
                    mask_img = mask_img * (255. / max_val)
            mask_img = np.maximum(0., np.minimum(255., mask_img))
            mask_img = mask_img.astype('uint8')
        else:
            mask_sh = (int(roi_rect['h'] / scale) + 1, int(roi_rect['w'] / scale) + 1, 1)
            mask_img = np.zeros(mask_sh, dtype='uint8')

        mask_start_x = int(roi_rect['x'] / scale)
        mask_start_y = int(roi_rect['y'] / scale)
        return (mask_img, mask_img_scale, mask_start_x, mask_start_y)

    def get_tile_segmentation_image(self, tile, img, img_scale, start_x, start_y ):
        img_start_x = int(tile['tile_top_left_x'] / img_scale - start_x)
        img_start_y = int(tile['tile_top_left_y'] / img_scale - start_y)
        tile_size = int(tile['tile_size'])
        img_end_x = img_start_x + int(tile_size / img_scale)
        img_end_y = img_start_y + int(tile_size / img_scale)
        tile_image = img[img_start_y:img_end_y, img_start_x:img_end_x, :]
        # max_val = np.max(target_image)
        return tile_image

    def convert_to_tfrecord(self,
                            mode=None,
                            tiles_list=None,

                            from_db=False, pyramid_level=1,

                            from_jpeg=False, image_path=None, image=None,

                            from_pyramid=False, pyramid_reader=None, pyramid_area=None,

                            output_file=None, output_dir=None,
                            project_name='',
                            progress_manager=None,
                            tfrecords_for_test=False):
        """
        Receiving a data_set and converting it to a TFRecord file
        Args:
            data_set: a list of ImageClass objects all associated with specific scan
            output_dir: the output directory to save the TFRecord file
            name: Name of the saved file

        Returns:
            None. A new TFRecord will be saved at /output_dir/name.tfrecord

        """
        if not progress_manager:
            raise ValueError('progress_manager must be provided')

        # if from_jpeg or from_pyramid:
        #     if not output_file:
        #         raise ValueError('When creating a TFRecord from external JPEG image, output file name must be provided')

        if mode == self.flags.training_mode:
            scan_uuid_read_list = [1058, 1078, 1088, 1091, 1094, 1096, 1097, 1098, 1305, 1310, 1311, 1313, 1314, 1322,
                                   1361, 1363, 1365, 1366, 1369, 1442, 1597, 1598, 1643, 1644, 1645, 1653, 1654,
                                   1657, 1658, 1677, 1682, 1733, 1734, 1735, 1736, 1737, 1738, 1739, 1740, 1741,
                                   1742, 1743, 1744, 1745, 1746, 1747, 1748, 1750]
        else:
            scan_uuid_read_list = None

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        number_of_scans_in_tile_list = len(tiles_list)
        progress_manager.create_children(number_of_scans_in_tile_list)

        tiles_list_file = os.path.join(output_dir, 'tiles_dataset.json')

        scan_index = 0
        for scan in tiles_list:

            scan_progress_manager = progress_manager.children[scan_index]

            # Preparing file name to be processed by Facets visualization tool.

            # Extract scan ID from Scan info. Example: ETS_DOG_BLD_DIF_019
            scan_metadata = str(scan['scan_name'])

            try:
                lab_name, species, sample, staining, slide_id = scan_metadata.split('.')[:5]
            except ValueError:
                lab_name = 'NA'
                species = 'NA'
                sample = 'NA'
                staining = 'NA'
                slide_id = 0

            # Update the tiles_list with scan info
            scan['species'] = species
            scan['lab_name'] = lab_name
            scan['sample'] = sample
            scan['staining'] = staining

            scan_id = scan['scan_id']
            scan_uuid = str(scan.get('scan_uuid', ''))
            use_scan_id = (scan_uuid == '') or (scan_uuid == None)

            if scan_uuid_read_list is not None:
                use_scan_id = not (scan_id in scan_uuid_read_list)

            if scan_id >= 10000:
                use_scan_id = False

            # if use_scan_id:
            #     continue

            scan_id_str = str(scan_id)

            if (mode==self.flags.predict_mode):
                scan_resolution_factor = self.config_params['predict_pyramid_to_model_resolution_factor']
            else:
                scan_resolution_factor = scan['scan_resolution_factor']

            # Get the tile image - either from the DB or cropped from an image
            if from_db:
                # Build the Scan class that is used to retrieve image patch around a specific label
                # scan_api = self.Scan(scan_id=scan_id, webroot=self.pyramid_path, use_cache=False)
                scan_api = self.Scan(scan_id=scan_id, webroot=self.pyramid_path)

            scan_api = None
            if from_jpeg or from_pyramid:
                # tfrecords_count_file = os.path.join(output_dir, 'tfrecords_count_file.json')

                use_cloud_imageio = self.config_params.get('use_cloud_imageio', False)
                use_cloud_imageio_runhttp = self.config_params.get('use_cloud_imageio_runhttp', True)
                if from_pyramid and ((mode == self.flags.training_mode) or use_cloud_imageio):

                    # When training, scans/pyramids resolution should be defined. 
                    try:
                        from common import imageio
                    except ImportError:
                        pass
                        # raise ValueError(
                        #     'You must run the following command in your virtual env: pip install ../common')

                    if use_cloud_imageio:
                        if use_scan_id or (not use_cloud_imageio_runhttp):
                            import imageio_http
                            if use_scan_id:
                                pyramid_path = 'legacy/%s'%scan_id_str
                            else:
                                pyramid_path = 'union/%s' % scan_uuid
                            pyramid_reader = imageio_http.from_path(pyramid_path)
                        else:
                            import imageio_runhttp
                            pyramid_path = 'union/%s'%scan_uuid
                            pyramid_reader = imageio_runhttp.from_path(pyramid_path)
                    else:
                        for pyramid_dir, id_str in [(self.training_scan_id_pyramids_directory, scan_id_str),
                                                    (self.training_scan_uuid_pyramids_directory, scan_uuid)]:
                            pyramid_path = os.path.join(pyramid_dir, id_str, 'pyramid/pyramid.dzi')
                            path_exists = os.path.exists(pyramid_path)
                            # logger.info('Pyramid path - %s' % pyramid_path)
                            if path_exists:
                                break
                        assert path_exists, 'Pyramid files not available for scan %s'%scan_id
                        pyramid_reader = imageio.from_path(pyramid_path)
                    pyramid_area = imageio.Rect
                    logger.info('core-tfrecords: pyramid path - %s' % pyramid_path)
                    self.my_scan_id = scan_id  # For debug

            # Prepare to create a TFRecord and to count the number of examples within.

            # Running on all ROIs in a scan and creating a TFRECORD per ROI.
            # At a later stage these ROI files will be merged to training/evaluation/test/predict files.
            number_of_rois = len(scan['regions_list'])
            scan_progress_manager.create_children(number_of_rois)

            roi_index = 0
            for roi in scan['regions_list']:

                if len(roi['tile_list']) == 0:
                    continue

                # In training mode, update the ROI anchor top left and pyramid resolution
                # if (mode == self.flags.training_mode):
                #     print('YOU SHOULD DEFINE THE REOLUTION -- TBD')
                folder_session_id = str(0 if (mode == 'predict') else roi['ROI_session'])
                folder_scan_id = str(0 if (mode == 'predict') else roi['scan_id'])
                folder_roi_id = str(0 if (mode == 'predict') else roi['ROI_id'])

                new_folder = 'session_' + folder_session_id
                relative_path = new_folder
                save_folder_path = os.path.join(output_dir, new_folder)
                if not os.path.exists(save_folder_path):
                    os.makedirs(save_folder_path)

                # Create or use ../scan number
                new_folder = 'scan_' + folder_scan_id
                relative_path = os.path.join(relative_path, new_folder)
                save_folder_path = os.path.join(save_folder_path, new_folder)
                if not os.path.exists(save_folder_path):
                    os.makedirs(save_folder_path)

                # Create or use ../roi number
                new_folder = 'roi_' + folder_roi_id + '_' + roi['region_name']
                relative_path = os.path.join(relative_path, new_folder)
                save_folder_path = os.path.join(save_folder_path, new_folder)
                if not os.path.exists(save_folder_path):
                    os.makedirs(save_folder_path)

                # Count the number of labels in ROI, and prepare the progress manager
                example_counter = 0
                roi_tile_list = roi['tile_list']
                total_tiles_num = len(roi_tile_list)

                roi_image_file = roi.get('image_file', None)
                roi_image, roi_img_scale, scaled_roi_start_x, scaled_roi_start_y = (None, None, None, None)
                if roi_image_file is not None:
                    roi_img_scale = 1
                    scaled_roi_start_x = roi['tiles_rect'][0] / self.flags.detection_image_scale
                    scaled_roi_start_y = roi['tiles_rect'][1] / self.flags.detection_image_scale
                    roi_image = np.load(roi_image_file)
                    roi_image = roi_image.astype('float32')

                roi_progress_manager = scan_progress_manager.children[roi_index]
                roi_progress_manager.create_reporter(total_tiles_num)

                scan_info = (scan_id, scan_resolution_factor, lab_name, species, sample, staining, slide_id)
                folder_info = (output_dir, save_folder_path, relative_path)
                roi_seg_info = (roi_image_file, roi_image, roi_img_scale, scaled_roi_start_x, scaled_roi_start_y)
                image_source_info = (scan_api, from_db, from_jpeg, from_pyramid, image_path)

                subset_tile_indexes_data = roi.get('subset_tile_indexes_data', {})
                if len(subset_tile_indexes_data.keys()) == 0:
                    subset_tile_indexes_data = {0: range(total_tiles_num)}
                for subset_idx, subset_tile_indexes in subset_tile_indexes_data.iteritems():
                    if pyramid_reader is not None:
                        with pyramid_reader as p_reader:
                            example_counter += \
                                self.create_tile_list_tfrecords(roi_progress_manager,
                                                                roi,
                                                                subset_tile_indexes,
                                                                scan_info,
                                                                folder_info,
                                                                roi_seg_info,
                                                                image_source_info,
                                                                pyramid_area,
                                                                p_reader)
                    else:
                        example_counter += \
                            self.create_tile_list_tfrecords(roi_progress_manager,
                                                            roi,
                                                            subset_tile_indexes,
                                                            scan_info,
                                                            folder_info,
                                                            roi_seg_info,
                                                            image_source_info,
                                                            pyramid_area,
                                                            None)

                # -- End of ROI
                if tfrecords_for_test:
                    # Continue to create blueprints for each ROI
                    self.generate_tf_blueprints = True

                # -- End of TFRecord generation -- #

                # Update ROI progress manager index
                # print('Completed generating tfrecord for roi {0}/{1}'.format(roi_index + 1, len(scan['regions_list'])))
                roi_index += 1

            # print('Completed generating tfrecord for scan {0}/{1}'.format(scan_index + 1, number_of_scans_in_tile_list))
            scan_index += 1
            sys.stdout.flush()

        # print('Adding tile dataset JSON file ', tiles_list_file)
        write_tiles_list = not(self.config_params['semantic_segmentation_mode'] and (mode==self.flags.predict_mode))
        if write_tiles_list:
            with open(tiles_list_file, 'w') as f:
                f.write(json.dumps(tiles_list, indent=4))

        return tiles_list

    def create_tile_list_tfrecords(self,
                                   roi_progress_manager,
                                   roi,
                                   subset_tile_indexes,
                                   scan_info,
                                   folder_info,
                                   roi_seg_info,
                                   image_source_info,
                                   pyramid_area,
                                   p_reader):

        scan_id, scan_resolution_factor, lab_name, species, sample, staining, slide_id = scan_info
        output_dir, save_folder_path, relative_path = folder_info
        roi_image_file, roi_image, roi_img_scale, scaled_roi_start_x, scaled_roi_start_y = roi_seg_info
        scan_api, from_db, from_jpeg, from_pyramid, image_path = image_source_info
        tiles_num = len(subset_tile_indexes)

        example_counter = 0
        last_roi_idx = -1
        roi_tile_list = roi['tile_list']
        subset_tiles = [roi_tile_list[i] for i in subset_tile_indexes]
        tile_images = []  # TODO: add bulk_read
        create_tfrec_img = True
        if create_tfrec_img:
            if roi_image_file is not None:
                tile_images = [self.get_tile_segmentation_image(
                    tile,
                    roi_image,
                    roi_img_scale,
                    scaled_roi_start_x,
                    scaled_roi_start_y
                )
                    for tile in subset_tiles]
            else:
                tile_crops = [(int(x['tile_top_left_x']),
                               int(x['tile_top_left_y']),
                               int(x['tile_size']),
                               int(x['tile_size'])) for x in subset_tiles]
                pyramid_level = 1
                if p_reader is not None:
                    pyramid_level = self.flags.detection_image_scale
                tile_images = self.get_tiles(tile_crops,
                                             resolution_factor=scan_resolution_factor,
                                             from_file=from_jpeg,
                                             image_path=image_path,
                                             from_db=from_db,
                                             scan_api=scan_api,
                                             pyramid_level=pyramid_level,
                                             from_pyramid=from_pyramid,
                                             pyramid_reader=p_reader,
                                             pyramid_area=pyramid_area)
        else:
            ch = 4 if self.flags.classification_tf_input_segment_image else 3
            tile_images = [np.zeros((2, 2, ch), dtype='uint8') for _ in subset_tiles]

        for i, tile_index in enumerate(subset_tile_indexes):
            tile_image = tile_images[i]
            tile = roi_tile_list[tile_index]

            # Handle semantic segmentation case
            use_target_mask = False
            if self.config_params['semantic_segmentation_mode']:

                # Load region mask for segmentation
                roi_idx = int(tile['tile_metadata']['roi_idx'])
                use_target_mask = roi.has_key('target_mask_file')
                if use_target_mask:
                    if roi_idx != last_roi_idx:
                        mask_rect = {'x': roi['rect'][0],
                                     'y': roi['rect'][1],
                                     'w': roi['rect'][2],
                                     'h': roi['rect'][3]
                                     }
                        mask_img, mask_img_scale, mask_start_x, mask_start_y = \
                            self.get_segmentation_region_mask(roi_rect=mask_rect, roi=roi)

                if roi_idx != last_roi_idx:
                    last_roi_idx = roi_idx

            # Create unique TFRecord file name to hold tile data
            image_file_path = None
            write_img_file = self.config_params.get('write_img_file', False)
            create_tfrec_img = True
            tfrecords_file_path = os.path.join(save_folder_path, 'tile_index_' + str(tile_index) + '.tfrecords')
            if write_img_file:
                is_det = self.config_params.get('write_det_img_file', True)
                center_x = int(tile['tile_top_left_x'] + tile['tile_size'] / 2)
                center_y = int(tile['tile_top_left_y'] + tile['tile_size'] / 2)
                if is_det:
                    # tile_img_file_name = 'tile_index_%d_sc_%d_x_%d_y_%d.png' % \
                    #                      (tile_index, roi['scan_id'], center_x, center_y)
                    tile_img_file_name = 'tile_index_%d.jpeg' % (tile_index)
                    if self.flags.set_tf_target_image:
                        if tile['tile_metadata'].has_key('target_image_file') or use_target_mask:
                            target_tile_img_file_name = 'tile_index_%d_sc_%d_x_%d_y_%d_target.png' % \
                                                        (tile_index, roi['scan_id'], center_x, center_y)
                            target_image_file_path = os.path.join(save_folder_path, target_tile_img_file_name)
                        if tile['tile_metadata'].has_key('target_image_weights_file'):
                            target_weight_tile_img_file_name = 'tile_index_%d_sc_%d_x_%d_y_%d_target_weight.png' % \
                                                               (tile_index, roi['scan_id'], center_x, center_y)
                            target_weight_image_file_path = os.path.join(save_folder_path,
                                                                         target_weight_tile_img_file_name)
                else:
                    class_str = tile['tile_metadata'].get('class', 'NC')
                    tile_img_file_name = 'tile_index_%d_cls_%s_sc_%d_x_%d_y_%d.png' % \
                                         (tile_index, class_str, roi['scan_id'], center_x, center_y)
                    if self.flags.classification_tf_input_segment_image:
                        if tile['tile_metadata'].has_key('seg_image_file') or \
                                tile['tile_metadata'].has_key('classification_segment_contour'):
                            tile_seg_img_file_name = 'tile_index_%d_cls_%s_sc_%d_x_%d_y_%d_seg.png' % \
                                                     (tile_index, class_str, roi['scan_id'], center_x, center_y)
                            tile_seg_img_file_path = os.path.join(save_folder_path, tile_seg_img_file_name)

                image_file_path = os.path.join(save_folder_path, tile_img_file_name)
            file_relative_path = os.path.join(relative_path, 'tile_index_' + str(tile_index) + '.tfrecords')
            tile['tfrecords_file_path'] = file_relative_path

            with tf.python_io.TFRecordWriter(tfrecords_file_path) as record_writer:
                # Converts a dataset tile to tfrecords file.

                roi_progress_manager.report_progress()

                if from_jpeg and image_path == None:
                    image_path = roi['image_path']

                if roi_image_file is None and write_img_file:
                    cv2.imwrite(image_file_path, tile_image)

                if create_tfrec_img:
                    image_bytes = tile_image.tobytes()
                else:
                    null_tile_image = np.zeros((2,2,3), dtype='uint8')
                    image_bytes = null_tile_image.tobytes()

                # Build the TFRecord feature list
                # Static features:
                feature_dict = {'image_raw': self._feature_casting(image_bytes),
                                'session': self._feature_casting(roi['ROI_session']),
                                'scan_id': self._feature_casting(scan_id),
                                'lab_name': self._feature_casting(lab_name),
                                'species': self._feature_casting(species),
                                'sample': self._feature_casting(sample),
                                'staining': self._feature_casting(staining),
                                'slide_id': self._feature_casting(slide_id),
                                'top_left_x': self._feature_casting(tile['tile_top_left_x']),
                                'top_left_y': self._feature_casting(tile['tile_top_left_y']),
                                'database_id': self._feature_casting(tile['tile_database_id'])
                                }

                # User specific features:
                key_ignore_list = ['area', 'contour_line', 'is_null_seg', 'hardness_type', 'pos_tile', 'sample_uuid', 'prediction_probability']
                if self.config_params['semantic_segmentation_mode']:
                    key_ignore_list.extend(
                        ['class', 'class_number', 'roi_idx', 'hardness_level', 'scan_id', 'train_rate_class', 'x_idx',
                         'y_idx'])
                use_sparse_inference = self.config_params.get('use_sparse_inference', False)
                if use_sparse_inference:
                    key_ignore_list.extend(['x_idx', 'y_idx'])
                if self.flags.set_tf_target_image:
                    target_image = None
                    if use_target_mask and self.config_params['semantic_segmentation_mode']:
                        target_image = self.get_tile_segmentation_image(tile,
                                                                        mask_img,
                                                                        mask_img_scale,
                                                                        mask_start_x,
                                                                        mask_start_y
                                                                        )
                        if write_img_file:
                            cv2.imwrite(target_image_file_path, target_image)
                    if tile['tile_metadata'].has_key('target_image_file'):
                        key_ignore_list.append('target_image_file')
                        file_name = output_dir + tile['tile_metadata']['target_image_file']
                        target_image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
                        if write_img_file:
                            cv2.imwrite(target_image_file_path, target_image)
                    if tile['tile_metadata'].has_key('target_image_weights_file'):
                        key_ignore_list.append('target_image_weights_file')
                        file_name = output_dir + tile['tile_metadata']['target_image_weights_file']
                        weight_image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
                        if write_img_file:
                            cv2.imwrite(target_weight_image_file_path, weight_image)
                        target_image = np.stack((target_image, weight_image), axis=-1)
                    if target_image is None:
                        target_image_bytes = np.zeros((1), dtype='uint8').tobytes()
                    else:
                        target_image_bytes = target_image.tobytes()
                    feature_dict['target_image'] = self._feature_casting(target_image_bytes)

                if self.flags.classification_tf_input_segment_image:
                    seg_img = None
                    if tile['tile_metadata'].has_key('seg_image_file'):
                        key_ignore_list.append('seg_image_file')
                        file_name = output_dir + tile['tile_metadata']['seg_image_file']
                        seg_img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
                    if tile['tile_metadata'].has_key('classification_segment_contour'):
                        seg_img = self.user_tfrecord.get_tile_segment_image(tile)
                        key_ignore_list.extend(
                            ['classification_segment_contour', 'margin_rect', 'margin', 'is_synth_seg_mask'])
                    if seg_img is None:
                        seg_img = np.ones_like(tile_image[:, :, 0])
                    else:
                        if write_img_file:
                            cv2.imwrite(tile_seg_img_file_path, seg_img)
                    if seg_img.ndim == 2:
                        seg_img = np.expand_dims(seg_img, axis=-1)
                    in_img = tile_image
                    if create_tfrec_img:
                        in_img = np.concatenate([tile_image, seg_img], axis=-1)
                    image_bytes = in_img.tobytes()
                    feature_dict['image_raw'] = self._feature_casting(image_bytes)

                for key, value in tile['tile_metadata'].iteritems():
                    if key in key_ignore_list:
                        continue
                    feature_dict[key] = self._feature_casting(value)

                example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
                example_counter += 1
                record_writer.write(example.SerializeToString())

            # Here all tiles from a specific TFRecord were written to the file

            # Writing a one-time TF blueprint.
            # In here, you can decide what will be processed by TF training infrastructure
            if self.generate_tf_blueprints:

                blueprint_dict = {'image_raw': 'str'}
                # 'session': 'int',
                # 'scan_id': 'int',
                # 'lab_name': 'str',
                # 'species': 'str',
                # 'sample': 'str',
                # 'staining': 'str',
                # 'slide_id': 'int',
                # 'top_left_x': 'float',
                # 'top_left_y': 'float'

                for key, value in tile['tile_metadata'].iteritems():
                    if key in key_ignore_list:
                        continue
                    blueprint_dict[key] = type(value).__name__

                if self.flags.set_tf_target_image:
                    blueprint_dict['target_image'] = 'str'

                # Create a JSON file
                blueprints_file = os.path.join(output_dir, 'tf_blueprints.json')

                with open(blueprints_file, 'w') as f:
                    f.write(json.dumps(blueprint_dict, sort_keys=True, indent=4))
                # print_pair("Created blueprint at ", blueprints_file)
                self.generate_tf_blueprints = False

        return example_counter

    def _init_tfrecord_iterator(self, path_to_tfrecord_file):
        """
        Inits a TFRecord iterator is required.

        """
        if not self.read_only_tfrecord_iterator:
            self.read_only_tfrecord_iterator = tf.python_io.tf_record_iterator(path_to_tfrecord_file)


    def _read_tile_from_file(self, tile_top_left_x, tile_top_left_y, width, height, image_path=None):
        logger.info('Reading test image')
        if image_path.endswith('.npy'):
            full_prediction_image = np.load(image_path)
        elif image_path.endswith('.jpg') or image_path.endswith('.png'):
            full_prediction_image = cv2.imread(image_path)
        else:
            raise ValueError('Unsupported file format')

        # Check if the image is 2D or 3D
        if len(full_prediction_image.shape) == 2:
            return full_prediction_image[tile_top_left_y:tile_top_left_y + height,
                         tile_top_left_x:tile_top_left_x + width]

        return full_prediction_image[tile_top_left_y:tile_top_left_y + height,
                         tile_top_left_x:tile_top_left_x + width, :]

    def get_tiles(self, tile_crops, resolution_factor, pyramid_level=1, pyramid_reader=None, from_pyramid=False, pyramid_area=None, **kwargs):
        # call bulk_read instead
        if from_pyramid:
            crops = []
            for tile_top_left_x, tile_top_left_y, width, height in tile_crops:
                crop = get_pyramid_crop(pyramid_area, tile_top_left_x, tile_top_left_y, width, height, pyramid_level,
                                        pyramid_to_model_resolution_factor=resolution_factor)
                crops.append(crop)
            tiles = pyramid_reader.bulk_read(crops, resize=1. / pyramid_level, padding=True)
            tiles = [normalize_pyramid_image(tile, *tile_crops[i],
                                             pyramid_to_model_resolution_factor=resolution_factor,
                                             interpolate_mode=self.flags.resample_interpolate_mode)
                     for i, tile in enumerate(tiles)]
        else:
            tiles = []
            for tile_top_left_x, tile_top_left_y, width, height in tile_crops:
                tiles.append(self._get_tile(tile_top_left_x, tile_top_left_y, width, height, resolution_factor,
                                            pyramid_area=pyramid_area, pyramid_level=pyramid_level,
                                            pyramid_reader=pyramid_reader,
                                            **kwargs))

        return tiles


    # Getting a patch (a tile) from a big image
    def _get_tile(self,
                  tile_top_left_x, tile_top_left_y, width, height,
                  resolution_factor,
                  from_file=False, image_path=None, pyramid_level=1,
                  from_db=False, scan_api=None,
                  from_pyramid=False, pyramid_reader=None, pyramid_area=None):


        ##################
        # Get image data #
        ##################
        if from_db:
            # Used in training
            if pyramid_level > 1:
                tile_image = scan_api.read(tile_top_left_x, tile_top_left_y, width, height, resize=1. / pyramid_level)
            else:
                tile_image = scan_api.read(tile_top_left_x, tile_top_left_y, width, height)

        elif from_file:
            self._read_tile_from_file(tile_top_left_x, tile_top_left_y, width, height, image_path=image_path)

        else:
            raise ValueError('CV error: Trying to read image from unknown source')

        if self.debug:
            filename = '/scopio/scratch5/shahar/scaling/tile_{0}_x{1}_y{2}.png'.format(
                0, tile_top_left_x, tile_top_left_y)
            cv2.imwrite(filename, tile_image)


        return tile_image

    def get_tfrecord_for_prediction(self,
                                    image=None,
                                    pyramid_reader=None,
                                    pyramid_area=None,
                                    working_dir=None,
                                    top_left_x=None, top_left_y=None, width=None, height=None,
                                    tiles_grid_info = None,
                                    cascaded_prediction=False, cascaded_dataset=None,
                                    progress_manager=None):
        """
        Convinience function to generate a TFRecord for a prediction area.
        """

        if not cascaded_prediction:
            session_id = 0
            scan_id = 0
            roi_id = 0
            scan_uuid = None
            test_data_id = self.config_params.get('test_data_id', None)
            if test_data_id is not None:
                session_id = test_data_id.get('session_id', 0)
                scan_id = test_data_id.get('scan_id', 0)
                roi_id = test_data_id.get('roi_id', 0)
                scan_uuid = test_data_id.get('scan_uuid', '')

            # This means user is generating a dataset on all input image
            # Build the test scan dataset
            scan_dataset = ScanData(scan_id=scan_id,
                                    scan_name='Predict',
                                    scan_uuid=scan_uuid,
                                    scan_resolution=self.flags.model_resolution)

            prediction_area_dict = {
                'x': top_left_x,
                'y': top_left_y,
                'width': width,
                'height': height
            }
            roi = RoiData(coordinates=prediction_area_dict,
                          roi_id=roi_id,
                          roi_name='Predict',
                          scan=scan_id,
                          session=session_id)
            scan_dataset.roi_list.append(roi)

            # Build the test dataset with just one scan test datset
            dataset = [scan_dataset]
        else:
            # In cascaded pipeline, the TFRecord for the next network is received from previous engine, thus
            # dataset is provided.
            dataset = cascaded_dataset

        logger.info('Building user tiles for prediction')
        tiles_list = self.user_tfrecord.get_tiles(data_set=dataset,
                                                  mode=self.flags.predict_mode,
                                                  tiles_grid_info=tiles_grid_info,
                                                  data_dir_path=working_dir)
        logger.info('Gathering user tiles completed')

        logger.info('Gathering images for tiles')

        if image is not None:
            self.convert_to_tfrecord(mode=self.flags.predict_mode,
                                     tiles_list=tiles_list,
                                     from_jpeg=True,
                                     image=image,
                                     output_dir=working_dir,
                                     progress_manager=progress_manager)
        elif pyramid_reader is not None:
            self.pyramid_read_time = 0
            self.convert_to_tfrecord(mode=self.flags.predict_mode,
                                     tiles_list=tiles_list,
                                     from_pyramid=True,
                                     pyramid_reader=pyramid_reader,
                                     pyramid_area=pyramid_area,
                                     output_dir=working_dir,
                                     progress_manager=progress_manager)
        else:
            raise ValueError('CV error - Attempting to extract images for tiles from unknown source')

        logger.info('Gathering images for tiles completed')

        # Also, for post process needs, return tiles_list that was used for inference
        return tiles_list

    # Generic function to create TFRecords for training and validation
    def create_generic_tfrecords_dataset(self,
                                         output_dir=None,
                                         dataset=None,
                                         progress_manager=None):

        # Generic function to create TFRecords from user created images and meta-data
        # Output_dir - The directory where the TFRecords DB will be created
        # dataset - a list of dictionaries.
        #           In each dict, there are 2 keys: 'image_path' [STRING] and 'meta_data': [DICT]
        # Image_tile_size

        if dataset is None:
            logger.info('image_class_list is none. Expecting a list of [image_path, class_name, class_number]')
            raise ValueError('Missing image_class_list')

        output_dir = os.path.expanduser(output_dir)
        print_pair('output_dir is ', output_dir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        tfrecords_scan_dir = output_dir

        logger.info('Gathering user tiles')
        tiles_list = []

        scan_tile = {}
        scan_tile['scan_id'] = 0
        scan_tile['scan_name'] = 'Generic_Scan'
        scan_tile['regions_list'] = []

        roi_index = 0
        for data_element in dataset:
            # To match expected infrastructure tiles list format,
            # we assume that each image is a different scan (This make the code simpler)
            # In each scan there will be one tile.
            region_data = {}
            region_data['ROI_id'] = roi_index
            region_data['ROI_session'] = 0
            region_data['scan_id'] = 0
            region_data['tile_list'] = []
            region_data['image_path'] = data_element['image_path']

            tile_metadata = {}
            tile_metadata['meta_data'] = pickle.dumps(data_element['meta_data'])
            tile_center_x = int(self.flags.tfrecord_tile_size / 2)
            tile_center_y = int(self.flags.tfrecord_tile_size / 2)
            tile_top_left_x = 0
            tile_top_left_y = 0
            tile_database_id = ''

            # unpacked_samples_data = pickle.loads(tile_metadata['meta_data'])

            region_data['tile_list'].append(UserTfrecord.TileClass(tile_database_id=tile_database_id,
                                                                   tile_center_x=tile_center_x,
                                                                   tile_center_y=tile_center_y,
                                                                   tile_size=self.flags.tfrecord_tile_size,
                                                                   tile_top_left_x=tile_top_left_x,
                                                                   tile_top_left_y=tile_top_left_y,
                                                                   tile_metadata=tile_metadata).__dict__)
            scan_tile['regions_list'].append(region_data.copy())
            roi_index += 1
        tiles_list.append(scan_tile.copy())

        logger.info('Gathering user tiles completed')

        i = 0
        # for scan_tiles in tiles_list:
        #     file_name = 'image_{0}.tfrecords'.format(i)
        #     i += 1
        #     output_file = os.path.join(tfrecords_scan_dir, file_name)
        self.convert_to_tfrecord(tiles_list=tiles_list,
                                 from_jpeg=True,
                                 # image_path=scan_tiles['image_path'],
                                 # output_file=output_file,
                                 output_dir=tfrecords_scan_dir,
                                 project_name='',
                                 progress_manager=progress_manager)

        # self.convert_to_tfrecord(tiles_list, True, False, self.labels_db_path, None, session, tfrecords_scan_dir, project_name)
        logger.info('Gathering images for tiles completed')

        sys.stdout.flush()

    def get_next_image_from_tfrecord(self, path_to_tfrecord_file=None):
        """
        Returns a parsed example from a TFRecprd file
        Args:
            path_to_tfrecord_file ():

        Returns:

        """
        # Generate a TFRecord iterator if required
        self._init_tfrecord_iterator(path_to_tfrecord_file)

        try:
            next_serialized_example = self.read_only_tfrecord_iterator.next()
        except StopIteration:
            # TFRecord iterator is done
            return None

        example = tf.train.Example()
        example.ParseFromString(next_serialized_example)

        # example = tf.parse_single_example(
        #     next_serialized_example,
        #     features={'image_raw': tf.FixedLenFeature([], tf.string)})

        # get bytes as 1d image matrix
        image_nparr = np.fromstring(example.features.feature['image_raw'].bytes_list.value[0], dtype=np.uint8)

        # reshape 1d image to image_size x image_size x channels
        image = image_nparr.reshape((self.tf_image_size, self.tf_image_size, self.tf_channels))

        return image

#   ________                           _____.___.                .___
#  /  _____/___________ ___  __ ____   \__  |   |____ _______  __| _/
# /   \  __\_  __ \__  \\  \/ // __ \   /   |   \__  \\_  __ \/ __ | 
# \    \_\  \  | \// __ \\   /\  ___/   \____   |/ __ \|  | \/ /_/ | 
#  \______  /__|  (____  /\_/  \___  >  / ______(____  /__|  \____ | 
#         \/           \/          \/   \/           \/           \/ 
# 
# print('Preparing Evaluation tiles.')
# evaluation_roi_tfrecord_path = os.path.join(output_dir, 'evaluation_roi')
# if not os.path.exists(evaluation_roi_tfrecord_path):
#     os.makedirs(evaluation_roi_tfrecord_path)
# print('Evaluation TFRECORDS per ROI are written to: ', evaluation_roi_tfrecord_path)
# 
# evaluation_tiles_list = self.user_tfrecord.get_tiles(data_set=testing_dataset, mode=self.flags.training_mode)
# print('Gathering user tiles completed')
# 
# # Generate a TFRecord file for every ROI in the training dataset
# print('Gathering images for tiles')
# self.generate_tf_blueprints = True  # Make one blueprints for the evaluation dataset
# self.convert_to_tfrecord(tiles_list=evaluation_tiles_list,
#                          from_db=True,  # We are reading from the DB
#                          from_jpeg=False,
#                          output_file=None,  # Build filename automatically
#                          output_dir=evaluation_roi_tfrecord_path,
#                          project_name=project_name,
#                          pyramid_level=pyramid_level,
#                          progress_manager=evaluation_dataset_progress_manager)
# 
# # Third Stage: Model Selection Set
# # =========================
# print('Preparing test tiles.')
# evaluation_conflicts_tfrecord_path = os.path.join(output_dir, 'evaluation_conflicts_set')
# if not os.path.exists(evaluation_conflicts_tfrecord_path):
#     os.makedirs(evaluation_conflicts_tfrecord_path)
# print('Model selection TFRECORDS are written to: ', evaluation_conflicts_tfrecord_path)
# 
# evaluation_conflicts_tiles_list = self.user_tfrecord.get_tiles(data_set=testing_dataset, mode=self.flags.test_mode)
# print('Gathering user tiles completed')
# 
# # Generate a TFRecord file for every ROI in the training dataset
# print('Gathering images for tiles')
# self.generate_tf_blueprints = True  # Make one blueprints for the evaluation dataset
# self.convert_to_tfrecord(tiles_list=evaluation_conflicts_tiles_list,
#                          from_db=True,  # We are reading from the DB
#                          from_jpeg=False,
#                          output_file=None,  # Build filename automatically
#                          output_dir=evaluation_conflicts_tfrecord_path,
#                          project_name=project_name,
#                          pyramid_level=pyramid_level,
#                          progress_manager=evaluation_dataset_progress_manager)
# 
# # Forth Stage: Model Selection Set
# # =========================
# print('Preparing conflicts tiles.')
# conflicts_tfrecord_path = os.path.join(output_dir, 'conflicts_set')
# if not os.path.exists(conflicts_tfrecord_path):
#     os.makedirs(conflicts_tfrecord_path)
# print('Model selection TFRECORDS are written to: ', conflicts_tfrecord_path)
# 
# conflicts_tiles_list = self.user_tfrecord.get_tiles(data_set=training_dataset, mode=self.flags.test_mode)
# print('Gathering user tiles completed')
# 
# # Generate a TFRecord file for every ROI in the training dataset
# print('Gathering images for tiles')
# self.generate_tf_blueprints = True  # Make one blueprints for the evaluation dataset
# self.convert_to_tfrecord(tiles_list=conflicts_tiles_list,
#                          from_db=True,  # We are reading from the DB
#                          from_jpeg=False,
#                          output_file=None,  # Build filename automatically
#                          output_dir=conflicts_tfrecord_path,
#                          project_name=project_name,
#                          pyramid_level=pyramid_level,
#                          progress_manager=evaluation_dataset_progress_manager)
# 
# 
# # For the testing TFRecords we create for each test ROI 2 files:
# # 1 TFRecord and 1 JSON file to hold the tiles list dictionary.
# 
# self.generate_tf_blueprints = True  # Signals to write a TF dictionary describing all features within a TF example.        print('Preparing testing tiles.')
# 
# testing_roi_tfrecord_path = os.path.join(output_dir, 'testing_roi')
# if not os.path.exists(testing_roi_tfrecord_path):
#     os.makedirs(testing_roi_tfrecord_path)
# print('Test TFRECORDS (and their dictionaries) per ROI are written to: ', testing_roi_tfrecord_path)
# 
# testing_dataset_progress_manager.create_children(len(testing_dataset))
# test_scan_index = 0
# for test_scan in testing_dataset:
#     roi_test_dataset_list = test_scan.generate_roi_test_dataset_list()
# 
#     roi_test_pm = testing_dataset_progress_manager.children[test_scan_index].create_children(
#         len(roi_test_dataset_list))
#     roi_test_dataset_index = 0
# 
#     for roi_test_dataset in roi_test_dataset_list:
# 
#         # Get Test tiles for the test ROI
#         test_tiles_list = self.user_tfrecord.get_tiles(data_set=roi_test_dataset, mode=self.flags.test_mode)
#         if len(test_tiles_list) == 0:
#             continue
#         regions_list = test_tiles_list[0]['regions_list']
#         if len(regions_list) == 0:
#             continue
#         if len(regions_list[0]['tile_list']) == 0:
#             continue
# 
#         # Create a specific folder per Test ROI
#         roi_id_name = 'test_roi_id_' + str(roi_test_dataset[0].roi_list[0].id)
#         roi_test_folder_path = os.path.join(testing_roi_tfrecord_path, roi_id_name)
# 
#         if not os.path.exists(roi_test_folder_path):
#             os.makedirs(roi_test_folder_path)
# 
#         # Creating a TFRecord for a test ROI
#         self.convert_to_tfrecord(tiles_list=test_tiles_list,
#                                  from_db=True,  # We are reading from the DB
#                                  from_jpeg=False,
#                                  output_file=None,  # Build filename automatically
#                                  output_dir=roi_test_folder_path,
#                                  project_name=project_name,
#                                  pyramid_level=pyramid_level,
#                                  progress_manager=roi_test_pm[roi_test_dataset_index],
#                                  tfrecords_for_test=True)
#         roi_test_dataset_index += 1
#     test_scan_index += 1
# 
# # Forth Stage: Training data as Test dataset for metrics calculation and conflicts detection
# # ==========================================================================================
# # For the testing TFRecords we create for each test ROI 2 files:
# # 1 TFRecord and 1 JSON file to hold the tiles list dictionary.
# 
# self.generate_tf_blueprints = True  # Signals to write a TF dictionary describing all features within a TF example.        print('Preparing testing tiles.')
# 
# metrics_roi_tfrecord_path = os.path.join(output_dir, 'metrics_roi')
# if not os.path.exists(metrics_roi_tfrecord_path):
#     os.makedirs(metrics_roi_tfrecord_path)
# print('Metrics TFRECORDS (and their dictionaries) per ROI are written to: ', metrics_roi_tfrecord_path)
# 
# metrics_dataset_progress_manager.create_children(len(testing_dataset))
# test_scan_index = 0
# for test_scan in training_dataset:
#     roi_test_dataset_list = test_scan.generate_roi_test_dataset_list()
# 
#     roi_test_pm = metrics_dataset_progress_manager.children[test_scan_index].create_children(
#         len(roi_test_dataset_list))
#     roi_test_dataset_index = 0
# 
#     for roi_test_dataset in roi_test_dataset_list:
# 
#         # Get Test tiles for the test ROI
#         test_tiles_list = self.user_tfrecord.get_tiles(data_set=roi_test_dataset, mode=self.flags.test_mode)
#         if len(test_tiles_list) == 0:
#             continue
#         regions_list = test_tiles_list[0]['regions_list']
#         if len(regions_list) == 0:
#             continue
#         if len(regions_list[0]['tile_list']) == 0:
#             continue
# 
#         # Create a specific folder per Test ROI
#         roi_id_name = 'test_roi_id_' + str(roi_test_dataset[0].roi_list[0].id)
#         roi_test_folder_path = os.path.join(metrics_roi_tfrecord_path, roi_id_name)
# 
#         if not os.path.exists(roi_test_folder_path):
#             os.makedirs(roi_test_folder_path)
# 
#         # Creating a TFRecord for a test ROI
#         self.convert_to_tfrecord(tiles_list=test_tiles_list,
#                                  from_db=True,  # We are reading from the DB
#                                  from_jpeg=False,
#                                  output_file=None,  # Build filename automatically
#                                  output_dir=roi_test_folder_path,
#                                  project_name=project_name,
#                                  pyramid_level=pyramid_level,
#                                  progress_manager=roi_test_pm[roi_test_dataset_index],
#                                  tfrecords_for_test=True)
#         roi_test_dataset_index += 1
#     test_scan_index += 1
