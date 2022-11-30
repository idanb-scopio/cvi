"""
 Title         : Core TFRecord dictionary
 Project       : Infrastructure

 File          : user_tfrecord.py
 Author        : Shahar Karny
 Created       : 02/11/2017
-----------------------------------------------------------------------------
 Description :   Defining the feature structure of the TFRecords along woth metadata
-----------------------------------------------------------------------------
 Copyright (c) This model is the confidential and
 proprietary property of ScopioLabs Ltd and the possession or use of this
 file requires a written license from ScopioLabs Ltd.
------------------------------------------------------------------------------
 Modification history :

"""

"""
    User is expected to implement the following methods
    (1) extract_patch_size_from_db that sets the patch size around each label to read form the DB
    (2) class_label_to_id that maps between class label and class numerical IDs
    (3) dataset_changes that manipulates ImageClass objects as required
    (4) Sessions and Scan to convert to TFRecords.
"""

import tensorflow as tf
from core_flags import get_tf_flags

import logging
logger = logging.getLogger(name='cv_tools')

class UserTfrecordBase(object):
    def __init__(self):
        self.flags = get_tf_flags()
        self.res_dir = None
        self.scan_id_pyramids_path = None
        self.scan_uuid_pyramids_path = None

    def get_tiles(self,
                  data_set,
                  mode,
                  tiles_grid_info=None,
                  reference_analysis_info=None,
                  data_dir_path=None):
        """
        Method for the user to define training or inference tiles.
        A tile is an object with:
        1. image coordinates (x,y,w,h); and
        2. metadata

        Metadata examples -
        When we train the detector, meta data is a list of {x,y} detection within the tile image space.
        For classification tasks, metadata is a class number representing the tile image

        Inputs:
        mode
        data_set - a list that consists of the following:
            scan_list - a list of scan_data class
            scan_data class holds scan_id and a list of roi_data class
                roi_data class holds roi_id and a list of label_data class
        tiles_grid_info - Optional tiles grid info
        reference_analysis_info - reference analysis information of the data using reference model

        Returns a list named tiles_list.
        There is a list member per scan. In each member there is tile list of TileClass objects
        """
        tiles_list = []

        # Iterate over all scans in the received dataset
        for scan in data_set:
            # scan is a ScanData class object that holds
            # scan_id, scan_name and roi_list
            logger.info(('scan', scan.scan_id))
            scan_tiles = {}
            scan_tiles['scan_id'] = scan.scan_id
            scan_tiles['scan_name'] = scan.scan_name
            scan_tiles['regions_list'] = []
            for roi in scan.roi_list:
                region_data = {}
                region_data['tile_list'] = []
                region_data['ROI_session'] = roi.session
                region_data['scan_id'] = scan.scan_id
                region_data['ROI_id'] = roi.id

                for label in roi.labels:
                    # Create a tile metadata dictionary
                    tile_metadata = {}
                    tile_metadata['class_string'] = label['label_name'][0]
                    tile_metadata['class_number'] = 1  # just a fake number

                    # Labels from DB may arraive as float number. Make sure we only use integers.
                    tile_center_x = int(label['x_gl'] + label['w'] / 2)
                    tile_center_y = int(label['y_gl'] + label['h'] / 2)

                    tile_top_left_x = int(tile_center_x - self.flags.tfrecord_tile_size / 2)
                    tile_top_left_y = int(tile_center_y - self.flags.tfrecord_tile_size / 2)


                    tile_database_id = label['id']
                    region_data['tile_list'].append(self.TileClass(tile_database_id=tile_database_id,
                                                                  tile_center_x=tile_center_x,
                                                                  tile_center_y=tile_center_y,
                                                                  tile_size=self.flags.tfrecord_tile_size,
                                                                  tile_top_left_x=tile_top_left_x,
                                                                  tile_top_left_y=tile_top_left_y,
                                                                  tile_metadata=tile_metadata).__dict__)
                scan_tiles['regions_list'].append(region_data)
            tiles_list.append(scan_tiles)

        return tiles_list

    def get_sessions_and_scan(self):
        """
        Returns a dictionary of sessions and associated scans to build tiles from
        If Session list is empty, the system will try to retrieve all associated scans.
        """

        sessions_scans_dict = {
            # '109': [174 , 176 , 177 , 178 , 179 , 180, 182, 183]
            #'109': [],
            #'112': []
            # '87':[]
            '92': {'scans': [], 'positive_annotations_filters':[]}, # debug session
            # '87': [59 , 60 , 61 , 62 , 63 , 64 , 67 , 68 , 69 , 70 , 71 , 72 , 73 , 74 , 75 , 76 , 77 , 78 , 79 , 80 , 81 , 82 , 86 , 93 , 95 , 99 , 100 , 104 , 106 , 107 , 108 , 110 , 112 , 113 , 115 , 126 , 129 , 130 , 131 , 133 , 134 , 135 , 137 , 138 , 139 , 140 , 141 , 142 , 143 , 144 , 145 , 149 , 150]
            # '86': [103],
            # '96': [44, 56] # Some example...
            # Sclab5:3336
            #2 : []
        }

        return sessions_scans_dict

    def get_annotations_positive_filter_list(self):
        """

        Returns: A list of string that are POSITIVE filters for annotations.
        Example1: If you would like to take annotations that has WBC in their name, pass through the list 'WBC'
        Example2: If you would like to take all annotations, just pass an empty list

        """
        positive_filter_list = []
        #positive_filter_list = ['WBC']

        return positive_filter_list

    class TileClass:
        """
        The TileClass is used to construct a basic TFRecord example
        """

        def __init__(self,
                     tile_database_id,
                     tile_center_x=None,
                     tile_center_y=None,
                     tile_top_left_x=None,
                     tile_top_left_y=None,
                     tile_size=None,
                     tile_metadata={}):
            """
            Constructs TileClass instance.

            Args:
                tile_top_left_x : The top left x coordinate of the input image
                tile_top_left_y : The top left y coordinate of the input image
                tile_size: the square tile size to extract from the top left corner
                tile_metadata: a dictionary. It's keys names will be added to the TFRecords feature name
            """

            self.tile_database_id = tile_database_id
            self.tile_center_x = tile_center_x
            self.tile_center_y = tile_center_y
            self.tile_top_left_x = tile_top_left_x
            self.tile_top_left_y = tile_top_left_y
            self.tile_size = tile_size
            self.tile_metadata = tile_metadata
