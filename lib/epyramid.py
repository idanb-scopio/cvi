"""
Extended Pyramid Reader:
    DZI format returns dzimage instance
    image format returns impyr instance

    Both use the same API with regard to CVI's operations.
"""

import os
from lib import dzimage, impyr


def get_pyramid_reader(image_source):
    if image_source[-4:] == '.jpg':
        return impyr.ImagePyramid(image_source)

    if image_source[-4:] == '.dzi':
        return dzimage.DZImageFs.fromfile(image_source)

    # attempt to find a dzi file
    dzi_file = f'{image_source}/pyramid/pyramid.dzi'
    if os.path.isdir(image_source) and os.path.exists(dzi_file):
        return dzimage.DZImageFs.fromfile(dzi_file)

    raise ValueError(f'invalid image source: {image_source}')
