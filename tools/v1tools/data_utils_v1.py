from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from tools.io_utils import read_json_file, write_pickle_file

# Dataset Scan and ROI classes from training.core_tfrecord.py dataset format
class ScanData(object):
  def __init__(self,
               scan_id,
               scan_name,
               scan_uuid=None,
               roi_list=None,
               species=None,
               scanning_device=None,
               scan_resolution=None,
               pyramid_resolution=None,  # Support scaled dataset
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
    self.scan_resolution_in_mm_per_pixel = scan_resolution  # Scban data resolution in mm per pixel
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
    copy_scan_dataset = ScanData(scan_id=self.scan_id,
                                 scan_name=self.scan_name,
                                 scan_uuid=scan_uuid,
                                 roi_list=[],
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

class RoiData(object):
  def __init__(self,
               coordinates,
               roi_id,
               roi_name,
               region_mask_contours=None,
               cell_type=None,
               image_file=None,
               mask_image_file=None,
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
    copy_roi_dataset = RoiData(coordinates={'x': self.x,
                                            'y': self.y,
                                            'width': self.w,
                                            'height': self.h
                                            },
                               roi_id=self.id,
                               roi_name=self.name,
                               region_mask_contours=None,
                               cell_type=cell_type,
                               image_file=None,
                               mask_image_file=None,
                               sparse_data=sparse_data,
                               scan=self.scan,
                               scan_uuid=scan_uuid,
                               session=self.session,
                               session_uuid=session_uuid)

    if 'region_mask_contours' in self.__dict__.keys():
      copy_roi_dataset.region_mask_contours = copy.deepcopy(self.region_mask_contours)
    if 'image_file' in self.__dict__.keys():
      copy_roi_dataset.image_file = self.image_file
    if 'mask_image_file' in self.__dict__.keys():
      copy_roi_dataset.mask_image_file = self.mask_image_file

    return copy_roi_dataset

  def set_coordinates(self, x, y, w, h):
    self.x = x
    self.y = y
    self.w = w
    self.h = h

def convert_dataset_json(in_dataset_file = None,
                         out_dataset_file = None):

    """
    :param in_dataset_file: input dataset.json from labeling tool export
    :param out_dataset_file: dataset pickle file
    """

    if in_dataset_file is None:
        in_dataset_file = 'PATH_TO_DATASET.json'

    if out_dataset_file is None:
        out_dataset_file = 'PATH_TO_OUT_DATASET.pkl'

    print ('Load %s ...'%in_dataset_file)
    in_dataset = read_json_file(in_dataset_file)

    dataset = []
    for scan_info in in_dataset:

        scan_dataset = ScanData(
            scan_id = scan_info['scan_id'],
            scan_name = scan_info['scan_name'],
            scan_uuid=scan_info['scan_uuid'],
            roi_list=None,
            species=scan_info['species'],
            scanning_device=scan_info['scanning_device'],
            scan_resolution=scan_info['scan_resolution_in_mm_per_pixel'],
            pyramid_resolution=scan_info['scan_resolution_in_mm_per_pixel'],  # Support scaled dataset
            scan_tissue=None,
            tags=[]
        )
        for roi_info in scan_info['roi_list']:
            coord = {}
            coord['x'], coord['y'], coord['width'], coord['height'] = \
                tuple([roi_info['x'],  roi_info['y'],  roi_info['w'],  roi_info['h']])
            roi_dataset = RoiData(
                coordinates=coord,
                roi_id=roi_info['id'],
                roi_name=roi_info['name'],
                region_mask_contours=None,
                cell_type=None,
                image_file=None,
                mask_image_file=None,
                sparse_data=False,
                scan=roi_info['scan'],
                scan_uuid=scan_info['scan_uuid'],
                session=roi_info['session'],
                session_uuid=None
            )
            roi_dataset.labels = roi_info['labels']
            scan_dataset.roi_list.append(roi_dataset)
        dataset.append(scan_dataset)

    print ('Save %s...'%out_dataset_file)
    write_pickle_file(dataset, out_dataset_file)
    pass


if __name__ == '__main__':

    convert_dataset_json()