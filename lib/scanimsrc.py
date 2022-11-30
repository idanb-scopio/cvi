# Scan Image Source represent an image with resolution metadata: mm per pixel
# Image source may be DZI pyramid (scan data) or .JPG file.

import os
from uuid import UUID
from lib import epyramid

# floating point resolution error tolerance: differences larger than this values means that the resolution
# is different
FP_RES_ERR = 1e-6


def is_same_resolution(lhs, rhs, fp_res_err=FP_RES_ERR):
    if abs(lhs - rhs) <= fp_res_err:
        return True
    return False


class ScanImageSource:

    def __init__(self, image_source, resolution):
        """
        Initialize the ScanImageSource object with an image source and resolution data (mm per pixel).
        :param image_source: path to DZI file or a dir which contains pyramid/pyramid.dzi file,
                             or a .JPG file (for debugging small scale data).
        :param resolution: floating point representing the pixel size in millimeters.
        """

        self.resolution = resolution
        self.image_source = image_source
        self.pyramid_reader = epyramid.get_pyramid_reader(image_source)
        self.width = self.pyramid_reader.width
        self.height = self.pyramid_reader.height
        self.base_level = self.pyramid_reader.base_level

    def read(self, *args, **kwargs):
        return self.pyramid_reader.read(*args, **kwargs)

    def bulk_read(self, *args, **kwargs):
        return self.pyramid_reader.bulk_read(*args, **kwargs)

    def get_resolution(self):
        return self.resolution

    def is_same_resolution(self, other_res, error_tolerance=FP_RES_ERR):
        return is_same_resolution(self.resolution, other_res, fp_res_err=error_tolerance)

    def infer_scan_id_from_src(self):
        # try to infer scan id from the input source
        if os.path.exists(f'{self.image_source}/pyramid/pyramid.dzi'):
            scan_id = os.path.basename(self.image_source)

            # check if scan_id is a uuid string
            try:
                uuid_obj = UUID(scan_id)

                # valid uuid
                return scan_id
            except ValueError:
                # not a valid uuid
                return None

        # standard path of a uuid scan_id does not exist. can't infer a uuid
        return None
