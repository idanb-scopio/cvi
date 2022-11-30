#!/usr/bin/env python
import abc
import math
import shutil
from collections import namedtuple
from functools import partial
from typing import Optional

Rect = namedtuple('Rect', ['x', 'y', 'width', 'height'])

try:
    # python 3
    from io import BytesIO
except ImportError:
    # python 2
    try:
        from cStringIO import StringIO as BytesIO
    except ImportError:
        from StringIO import StringIO as BytesIO

import os
import re

import time
import urllib
import xml.dom.minidom
import logging
import traceback

# use xrange in python2
try:
    range = xrange
except NameError:
    pass

# opencv and numpy for image manipulation
import cv2
import numpy as np

from multiprocessing.pool import ThreadPool as MPThreadPool
import atexit

# part of DZI file
NS_DEEPZOOM = 'http://schemas.microsoft.com/deepzoom/2008'

# defaul color of blank pyramid
FILL_COLOR = [255, 255, 255]

DEFAULT_PYRAMID_NAME = 'pyramid'

DEFAULT_TILE_SIZE = 512
DEFAULT_TILE_OVERLAP = 1
DEFAULT_TILE_FORMAT = 'jpg'

_read_tiles_pool: Optional[MPThreadPool] = None


def get_read_thread_pool(thread_count: int = 10):
    global _read_tiles_pool
    if _read_tiles_pool is None:
        _read_tiles_pool = MPThreadPool(thread_count)
    return _read_tiles_pool


def cleanup_global_pools():
    if _read_tiles_pool is not None:
        _read_tiles_pool.close()


atexit.register(cleanup_global_pools)


def _read_tile(args):
    """this is a patch for multiprocessing on bulk_read just for DZImageFs"""
    pyramid_root, tile_format, x, y, level = args
    image_path = '%s/%d/%d_%d.%s' % (pyramid_root, level, x, y,
                                     tile_format)
    tile_im = cv2.imread(image_path)
    if tile_im is None:
        raise ValueError('warning: missing tile image file: %s' % image_path)
    return tile_im


def floor_to_even(n):
    if n % 2 == 0:
        return n
    else:
        return n - 1


def ceil_to_odd(n):
    if n % 2 == 0:
        return n + 1
    else:
        return n


class CropBuilder(object):
    def __init__(self, rect, level, dzi_image, dzi_descriptor):
        self.rect = rect
        self.level = level
        # self.dzi_image = dzi_image
        self.descriptor = dzi_descriptor

        # self._x_tile_offset, self._y_tile_offset = dzi_image.get_offset_inside_tile(rect.x, rect.y)
        self.image = np.empty((rect.height, rect.width, 3), dtype='uint8')

    def update_tile(self, tile_x, tile_y, tile_image):
        x_s, y_s, x_e, y_e = self.descriptor.get_tile_bounds(self.level, tile_x, tile_y)
        tile_width, tile_height = x_e - x_s, y_e - y_s

        tile_x_offset = max(0, self.rect.x - x_s)
        tile_y_offset = max(0, self.rect.y - y_s)

        rect_x_offset = max(0, x_s - self.rect.x)
        rect_y_offset = max(0, y_s - self.rect.y)

        take_width = min(self.rect.width - rect_x_offset, tile_width - tile_x_offset)
        take_height = min(self.rect.height - rect_y_offset, tile_height - tile_y_offset)

        crop_tile = tile_image[tile_y_offset:tile_y_offset + take_height,
                    tile_x_offset:tile_x_offset + take_width, :]
        self.image[
        rect_y_offset:rect_y_offset + take_height,
        rect_x_offset:rect_x_offset + take_width, :] = crop_tile

    def get_image(self):
        return self.image


class TileRanges(object):
    """TileRanges - Store tile ranges rectangles while merging overlapping

    TileRanges class stores tile ranges of the highest resolution level of the pyramid
    which are marked as changed should update its corresponding lower resolution tiles.
    It aims to merge overlapping rectangles in order to optimize the updating of lower
    resolution level tiles.

    Tile ranges are stored in the form of (x_min_idx, x_max_idx, y_min_idx, y_max_idx)
    where the indexes are referring to the tile array index and the range includes
    both lower bound and upper bound.

    adding add() rectangle to this class, merges it with other overlapping rectangles
    creating a list of larger rectangles.

    all() iterates over the rectangles after merging but restricting their sizes to
    max_tiles_in_axis (default: 50) tiles per axis of any rectangle
    """

    def __init__(self, max_tiles_in_axis=50):
        self._rects = []
        self._max_tiles_in_rect_axis = max_tiles_in_axis

    def _is_overlap_axis(self, rect1, rect2, axis):
        """return True if rect1 and rect2 are overlapping in axis (x=0, y=1)"""

        def _is_overlap_axis_inner(r1, r2):
            is_overlap = False
            for i in range(axis * 2, axis * 2 + 1):
                is_overlap |= r1[i] >= r2[axis * 2] and \
                              r1[i] <= r2[axis * 2 + 1]

            return is_overlap

        return _is_overlap_axis_inner(rect1, rect2) or \
               _is_overlap_axis_inner(rect2, rect1)

    def _is_adjacent_axis(self, rect1, rect2, axis):
        """return True if rect1 and rect2 are adjacent in axis (x=0, y=1)"""

        def _is_adjacent_axis_inner(r1, r2):
            return r2[axis * 2] == r1[axis * 2 + 1] + 1

        return _is_adjacent_axis_inner(rect1, rect2) or \
               _is_adjacent_axis_inner(rect2, rect1)

    def _should_merge(self, rect1, rect2):
        should_merge = False
        # overlapping both x and y
        should_merge |= self._is_overlap_axis(rect1, rect2, 0) and \
                        self._is_overlap_axis(rect1, rect2, 1)
        # overlapping x and adjacent y
        should_merge |= self._is_overlap_axis(rect1, rect2, 0) and \
                        self._is_adjacent_axis(rect1, rect2, 1)
        # overlapping y and adjacent x
        should_merge |= self._is_overlap_axis(rect1, rect2, 1) and \
                        self._is_adjacent_axis(rect1, rect2, 0)
        return should_merge

    def _merge(self, rect1, rect2):
        """Merges rect1 and rect2 and return the merged rectangle"""
        return (
            min(rect1[0], rect2[0]),
            max(rect1[1], rect2[1]),
            min(rect1[2], rect2[2]),
            max(rect1[3], rect2[3]),
        )

    def add(self, rect):
        """Add rect tile range to the store, merging it overlapping/adjacent rectangles"""
        to_merge = []
        rects = []
        for other_rect in self._rects:
            if self._should_merge(rect, other_rect):
                to_merge.append(other_rect)
            else:
                rects.append(other_rect)

        for other_rect in to_merge:
            rect = self._merge(other_rect, rect)

        rects.append(rect)
        self._rects = rects

    def all(self):
        """Iterates over the merged tile ranges
        Return tile ranges which are no larger than 50 tiles per axis"""
        m = self._max_tiles_in_rect_axis
        for rect in self._rects:
            w = rect[1] - rect[0]
            h = rect[3] - rect[2]
            for step_x in range(w // m + 1):
                for step_y in range(h // m + 1):
                    yield (
                        rect[0] + (step_x * m),
                        min(rect[1], rect[0] + ((step_x + 1) * m) - 1),
                        rect[2] + (step_y * m),
                        min(rect[3], rect[2] + ((step_y + 1) * m) - 1),
                    )


class DZImage(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, descriptor, **kwargs):

        self._logger = kwargs.get('logger', logging)

        # caches blank tile image files. A level of a pyramid may contain huge
        # amount of blank tiles with only a few different sizes (hence files).
        # The cache is used to avoid generating new tile images, and instead
        # creating a hard link to existing ones.
        # The cache is a map of: tile dimensions: (width, height) --> filename
        self.blank_cache = {}
        self.updated_tiles_dict = {}
        self.updated_tiles_read_dict = {}

        self.descriptor = descriptor
        self.blank_im_size = 0
        self.ref_tile_modified = False
        self.updated_tiles = []
        # parallel save related
        self.is_parallel = False
        self.context = None
        self.workers = []

        # update lowres tiles optimization
        # update lowres tiles only once every 2 calls
        self.tiles_range_cache = None

        if 'parallel' in kwargs and kwargs['parallel'] >= 1:
            self.is_parallel = True
            self.init_parallel(kwargs['parallel'])

    @property
    def width(self):
        return self.descriptor.width

    @property
    def height(self):
        return self.descriptor.height

    @property
    def base_level(self):
        return self.descriptor.num_levels - 1

    def tiles(self, level):
        """Iterator for all tiles in the given level. Returns (column, row) of a tile."""
        columns, rows = self.descriptor.get_num_tiles(level)
        for row in range(rows):
            for column in range(columns):
                yield (column, row)

    def get_tiles_range(self, x, y, w, h, level=None):
        '''Returns the tiles range [Xs,Xe] [Ys,Ye] covered by an image of
           dimensions: (w,h), at global pyramid coordinates (x,y).'''
        if x + w > self.width or y + h > self.height or x < 0 or y < 0:
            raise ValueError('rectangle borders out of image bounds: x %d, y: %d' % (x, y))

        tile_tl = self._get_containing_tile(x, y, level)
        tile_br = self._get_containing_tile(x + w - 1, y + h - 1, level)

        x_tile_s = tile_tl[0]
        x_tile_e = tile_br[0]
        y_tile_s = tile_tl[1]
        y_tile_e = tile_br[1]

        return (x_tile_s, x_tile_e, y_tile_s, y_tile_e)

    def is_inside_bounds(self, bounds, x=None, y=None):
        assert x is not None or y is not None
        # NOTE: x_e and y_e are past-the-end offsets
        x_s, y_s, x_e, y_e = bounds

        # adjust for overlap
        if x_s > 0:
            x_s += self.descriptor.tile_overlap
        if x_e < self.width:
            x_e -= self.descriptor.tile_overlap
        if y_s > 0:
            y_s += self.descriptor.tile_overlap
        if y_e < self.height:
            y_e -= self.descriptor.tile_overlap

        result = True
        if x is not None:
            result &= x_s <= x and x < x_e
        if y is not None:
            result &= y_s <= y and y < y_e

        return result

    def _get_containing_tile(self, x, y, level):
        row, col = x // self.descriptor.tile_size, y // self.descriptor.tile_size
        return row, col

    def _get_offset_inside_tile(self, x, y, tile_x, tile_y):
        '''Returns the offset from the top left tile of a given x,y coordinates.
           The top left tile is the tile containing the x,y coordinates.'''

        loc_x = x - tile_x * self.descriptor.tile_size
        loc_y = y - tile_y * self.descriptor.tile_size

        return (loc_x, loc_y)

    def get_offset_inside_tile(self, x, y, get_overlap=False, level=None):
        '''Returns a (offset_x, offset_y) location inside the tile for a given
           global x,y coordinates'''

        tile_x, tile_y = self._get_containing_tile(x, y, level)
        offset_x, offset_y = self._get_offset_inside_tile(x, y, tile_x, tile_y)

        if get_overlap:
            if tile_x > 0:
                offset_x += self.descriptor.tile_overlap
            if tile_y > 0:
                offset_y += self.descriptor.tile_overlap

        return (offset_x, offset_y)

    def read(self, x, y, width, height, level=None, **kwargs):
        '''read image data at location (x,y) with width (width,height)'''
        if level is None:
            level = self.descriptor.num_levels - 1
        tiles_range = self.get_tiles_range(x, y, width, height, level=level)

        # read tiles data
        image = self.read_tiles(level, *tiles_range, **kwargs)

        # get offset of the x,y coordinates inside the bounding tiles box
        ox, oy = self.get_offset_inside_tile(x, y)

        # crop to requested dimensions
        image = image[oy:oy + height, ox:ox + width, :]

        return image

    def crop_read(self, x, y, width, height, level=None, **kwargs):
        '''read image data at location (x,y) with width (width,height)'''
        if level is None:
            level = self.descriptor.num_levels - 1

        crop = CropBuilder(Rect(x, y, width, height), level, self, self.descriptor)
        tiles_range = self.get_tiles_range(x, y, width, height, level=level)
        for index, tile in self._enumerate_tiles_in_range(level, *tiles_range, **kwargs):
            crop.update_tile(index[0], index[1], tile)

        return crop.get_image()

    def bulk_read(self, rects, level=None, thread_count=None, **kwargs):
        """
        This method will read each required underlying tile only once.
        It also supports parallelism, governed by the @thread_count param. ( defaults no parallelism )

        @thread_count: will use this amount of threads or len of tiles if smaller.

        @return
        a 4D np.ndarray, where the first dimension is len(@rects)
        """

        if level is None:
            level = self.base_level
        rects = list(map(lambda x: Rect(*x), rects))

        tiles_to_read = dict()  # {@tile_pos: [@rect,..]}
        crop_builders = dict()  # {@rect: CropBuilder}

        for r in rects:
            tile_range = self.get_tiles_range(*r)
            for pos in self._index_tiles_in_range(*tile_range):
                rects_inside = tiles_to_read.get(pos, list())
                rects_inside.append(r)
                tiles_to_read[pos] = rects_inside

        # py2 backwards compatibility,  py2 is not ordered
        tiles_to_read_sorted = sorted(tiles_to_read.items())
        if thread_count \
                and isinstance(self, DZImageFs):
            t_count = min(thread_count, len(tiles_to_read_sorted))
            pool = get_read_thread_pool(t_count)
            data_for_multi = [(self.pyramid_root, self.descriptor.tile_format) + tile_pos + (level,) for
                              tile_pos, _
                              in tiles_to_read_sorted]
            results = pool.map(_read_tile, data_for_multi)
        else:
            results = [self._read_tile(*tile_pos, level=level) for tile_pos, recs_in_tile in
                       tiles_to_read_sorted]

        for (tile_pos, recs_in_tile), tile_im in zip(tiles_to_read_sorted, results):
            for r in recs_in_tile:
                if r in crop_builders:
                    current_crop = crop_builders[r]
                else:
                    current_crop = CropBuilder(r, level, self, self.descriptor)
                    crop_builders[r] = current_crop
                current_crop.update_tile(tile_pos[0], tile_pos[1], tile_im)

        results = np.array([crop_builders[rec].get_image() for rec in rects])
        return results

    def _read_tile(self, x, y, level, **kwargs):
        image_path = '%s/%d/%d_%d.%s' % (self.pyramid_root, level, x, y,
                                         self.descriptor.tile_format)

        return self.read_and_store(image_path, **kwargs)

    def _read_tile_wrapped(self, x, y, level, **kwargs):
        return x, y, self._read_tile(x, y, level, **kwargs)

    def _index_tiles_in_range(self, x_tile_s, x_tile_e, y_tile_s, y_tile_e):
        for row in range(y_tile_s, y_tile_e + 1):
            for col in range(x_tile_s, x_tile_e + 1):
                yield (col, row)

    def _enumerate_tiles_in_range(self, level, x_tile_s, x_tile_e, y_tile_s, y_tile_e, get_overlap=False, **kwargs):
        for pos in self._index_tiles_in_range(x_tile_s, x_tile_e, y_tile_s, y_tile_e):
            yield pos, self._read_tile(*pos, level=level, **kwargs)

    def read_tiles(self, level, x_tile_s, x_tile_e, y_tile_s, y_tile_e, get_overlap=False, **kwargs):
        '''read a rectangle of tiles, marked by tile corrdinates X,Y: [start, end]'''

        cols, rows = self.descriptor.get_num_tiles(level)
        # if x_tile_e >= cols or y_tile_e >= rows:
        #  raise ValueError('tile coordinates out of bounds')

        if x_tile_e >= cols:
            self._logger.warning('warning: out of bounds tile: level = %d, x_tile_e = %d, cols = %d' % \
                                 (level, x_tile_e, cols))
            x_tile_e = cols - 1

        if y_tile_e >= rows:
            self._logger.warning('warning: out of bounds tile: level = %d, y_tile_e = %d, rows = %d' % \
                                 (level, y_tile_e, rows))
            y_tile_e = rows - 1

        topleft_bounds = self.descriptor.get_tile_bounds(level, x_tile_s, y_tile_s)
        x_start = topleft_bounds[0]
        y_start = topleft_bounds[1]
        botright_bounds = self.descriptor.get_tile_bounds(level, x_tile_e, y_tile_e)
        x_end = botright_bounds[2]
        y_end = botright_bounds[3]

        # new tile data starts after the overlap margin
        if x_tile_s > 0 and not get_overlap:
            x_start += self.descriptor.tile_overlap
        if y_tile_s > 0 and not get_overlap:
            y_start += self.descriptor.tile_overlap
        if x_tile_e < cols - 1 and not get_overlap:
            x_end -= self.descriptor.tile_overlap
        if y_tile_e < rows - 1 and not get_overlap:
            y_end -= self.descriptor.tile_overlap

        im_w = x_end - x_start
        im_h = y_end - y_start
        #    print "read tiles dimensions: %dx%d" % (im_w, im_h)
        image = np.empty((im_h, im_w, 3), dtype='uint8')

        result = get_read_thread_pool().starmap(self._read_tile_wrapped,
                                                [(col, row, level)
                                                 for row in range(y_tile_s, y_tile_e + 1)
                                                 for col in range(x_tile_s, x_tile_e + 1)])

        for col, row, tile_im in result:
            # coordinates inside the bigger image
            x_s, y_s, x_e, y_e = self.descriptor.get_tile_bounds(level, col, row)

            # coordinates inside a tile image
            tx_s, ty_s = 0, 0
            ty_e, tx_e = tile_im.shape[0:2]  # numpy dimension (2D) are (height, width)

            # change to relative coordinates
            x_s -= x_start
            x_e -= x_start
            y_s -= y_start
            y_e -= y_start

            # adjust coordinates to take overlap into account
            if col > 0 and not get_overlap:
                x_s += self.descriptor.tile_overlap
                tx_s += self.descriptor.tile_overlap
            if col < cols - 1 and not get_overlap:
                x_e -= self.descriptor.tile_overlap
                tx_e -= self.descriptor.tile_overlap
            if row > 0 and not get_overlap:
                y_s += self.descriptor.tile_overlap
                ty_s += self.descriptor.tile_overlap
            if row < rows - 1 and not get_overlap:
                y_e -= self.descriptor.tile_overlap
                ty_e -= self.descriptor.tile_overlap

            # print '%s  X: [%d,%d) Y: [%d,%d) --- Tile: X: [%d,%d) Y: [%d,%d)' % \
            #      (image_path, x_s, x_e, y_s, y_e, tx_s, tx_e, ty_s, ty_e)

            # assign tile data to the bigger image at the matching coordinates
            image[y_s:y_e, x_s:x_e, :] = tile_im[ty_s:ty_e, tx_s:tx_e, :]
        return image

    @abc.abstractmethod
    def read_and_store(self, image_path, **kwargs):
        """ implemented by subclass"""


class DZImageFs(DZImage):
    def __init__(self, descriptor, dzi_filename, **kwargs):
        self.dzi_filename = dzi_filename
        # pyramid root is the folder containing the level subdirectories and image files.
        self.pyramid_root = os.path.dirname(dzi_filename) + '/' + \
                            os.path.splitext(os.path.basename(dzi_filename))[0] + '_files'

        super(DZImageFs, self).__init__(descriptor, **kwargs)

    @classmethod
    def fromscratch(cls, width, height, dzi_filename, **kwargs):
        tile_size = kwargs.get('tile_size', DEFAULT_TILE_SIZE)
        tile_overlap = kwargs.get('tile_overlap', DEFAULT_TILE_OVERLAP)
        tile_format = kwargs.get('tile_format', DEFAULT_TILE_FORMAT)

        # DZI descriptor contains the details about the pyramid levels, tiles, etc.
        instance = cls(DeepZoomImageDescriptor(width, height, tile_size, tile_overlap, tile_format),
                       dzi_filename, **kwargs)

        if not os.path.isdir(instance.pyramid_root):
            os.mkdir(instance.pyramid_root)

        if 'dynamic' in kwargs and kwargs['dynamic'] == True:
            instance.descriptor.save(dzi_filename, dynamic=kwargs['dynamic'])
        else:
            instance.descriptor.save(dzi_filename)
        return instance

    @classmethod
    def fromfile(cls, dzi_filename, **kwargs):
        descriptor = DeepZoomImageDescriptor()
        descriptor.open_dzi_file(dzi_filename)
        return cls(descriptor, dzi_filename, **kwargs)

    def init_blank(self, fill_color=FILL_COLOR, overwrite=False, debug=False):
        """Initialize blank pyramid"""

        # mkdir pyramid directory root
        if os.path.isdir(self.pyramid_root):
            if overwrite:
                # NOTE: rmtree on the wrong string can be harmful, assetions safeguard should be kept updated.
                assert re.match('^.*_files$', self.pyramid_root) is not None
                shutil.rmtree(self.pyramid_root)
            else:
                raise ValueError('pyramid root already exist: %s' % self.pyramid_root)

        os.mkdir(self.pyramid_root)

        for level in range(self.descriptor.num_levels):
            level_dir = '%s/%d' % (self.pyramid_root, level)
            os.mkdir(level_dir)

            # loop on all tile bounds
            for (column, row) in self.tiles(level):
                x_s, y_s, x_e, y_e = self.descriptor.get_tile_bounds(level, column, row)
                tile_w = x_e - x_s
                tile_h = y_e - y_s

                #        print "level: %d tile: (%d,%d): dims: %dx%d" % (level, column, row, tile_w, tile_h)
                image_file = '%s/%d_%d.%s' % (level_dir, column, row, self.descriptor.tile_format)

                # create a blank image
                self.create_blank_image(image_file, tile_w, tile_h, fill_color, debug=debug)

    def create_blank_image(self, filename, width, height, fill_color, **kwargs):
        if 'disable_cache' in kwargs and kwargs['disable_cache'] == True:
            disable_cache = True
        else:
            disable_cache = False

        if os.path.exists(filename):
            raise ValueError('image file already exist')

        dims = (width, height)
        if dims in self.blank_cache and not disable_cache:
            src_file = self.blank_cache[dims]
            os.link(src_file, filename)
        else:
            # create blank image
            image = np.zeros((height, width, 3), dtype='uint8')
            image[:] = fill_color
            cv2.imwrite(filename, image)
            self.blank_cache[(width, height)] = filename

            # if 'debug' in kwargs and kwargs['debug'] == True:
            # base_level = self.descriptor.num_levels - 1
            # reference_tile = '%s/%s/1_1.%s' % (self.pyramid_root, base_level, self.descriptor.tile_format)
            # if (filename == reference_tile):
            #   self.blank_im_size = os.stat(filename).st_size
            #   print "the size of blank image %s is %d" %(reference_tile, self.blank_im_size)

    def write(self, image_update, x, y, do_update=False, debug=False, **kwargs):
        '''write image data to the pyramid at a given (x,y) coordinate'''

        updated_tiles = None

        base_level = self.descriptor.num_levels - 1
        height, width = image_update.shape[0:2]

        # get bounding tiles box
        tiles_range = self.get_tiles_range(x, y, width, height)

        # self._logger.debug("start read_tiles")
        # read tiles data
        image = self.read_tiles(base_level, *tiles_range, get_overlap=True)
        # self._logger.debug("finish read_tiles")

        # get offset of the x,y coordinates inside the bounding tiles box
        ox, oy = self.get_offset_inside_tile(x, y, get_overlap=True)

        # overwrite with updated image data
        image[oy:oy + height, ox:ox + width, :] = image_update[:, :, :]

        x_tile_s, x_tile_e, y_tile_s, y_tile_e = tiles_range

        x_start, y_start = self.descriptor.get_tile_bounds(base_level, x_tile_s, y_tile_s)[0:2]
        #    print 'TopLeft point: (%d,%d)' % (x_start, y_start)
        self._logger.debug("going over the base level")
        for row in range(y_tile_s, y_tile_e + 1):
            for col in range(x_tile_s, x_tile_e + 1):
                image_file = '%s/%d/%d_%d.%s' % (self.pyramid_root, base_level, col, row,
                                                 self.descriptor.tile_format)

                # coordinates inside the high-res plane
                x_s, y_s, x_e, y_e = self.descriptor.get_tile_bounds(base_level, col, row)

                # adjust to coordinates inside the bounding tiles box
                x_s -= x_start
                x_e -= x_start
                y_s -= y_start
                y_e -= y_start

                tile_data = image[y_s:y_e, x_s:x_e, :]
                #        print 'write data in X: [%d,%d)   Y: [%d,%d)' % (x_s, x_e, y_s, y_e)

                # erase existing tile image file
                if not os.path.isfile(image_file):
                    for _ in range(100):
                        time.sleep(.1)
                        if os.path.isfile(image_file):
                            break
                    if not os.path.isfile(image_file):
                        raise ValueError('warning: missing tile image file: %s' % image_file)

                write_tiles_cb = self._create_write_tiles_cb(kwargs, base_level)

                self.write_image(image_file, tile_data, write_tiles_cb)

                self.updated_tiles.append([base_level, col, row])

        self._logger.debug("filled the cache with the base level")

        if not self.is_parallel:
            updated_tiles = self.update_lowres_levels(tiles_range, **kwargs)
        else:
            self._lowres_pool.do(tiles_range, **kwargs)

        self._logger.debug("finished write")
        return updated_tiles

    def write_image(self, image_file, tile_data, write_tiles_cb=None):
        '''Helper method to write image on disk.'''
        if self.is_parallel:
            # assign tile data with to dictionary keyed by tile filename
            with self.updated_tiles_dict_lock:
                self.updated_tiles_dict[image_file] = tile_data
                self.updated_tiles_read_dict[image_file] = tile_data

            # send the tile filename to the workers for writing
            self._write_pool.do(image_file, write_tiles_cb)
        else:
            cv2.imwrite(image_file, tile_data)
            if write_tiles_cb is not None:
                write_tiles_cb()

    def read_and_store(self, image_file, **kwargs):
        if self.is_parallel:
            if image_file in self.updated_tiles_read_dict:
                with self.updated_tiles_dict_lock:
                    if image_file in self.updated_tiles_read_dict:
                        return self.updated_tiles_read_dict[image_file]

        tile_im = cv2.imread(image_file)
        tries, max_tries = 0, 100
        while tile_im is None and tries < max_tries:
            time.sleep(.1)
            tile_im = cv2.imread(image_file)
            tries += 1

        if tile_im is None:
            raise ValueError('warning: missing tile image file: %s' % image_file)

        return tile_im

    def _create_write_tiles_cb(self, params, level):
        write_tiles_cb = None
        if 'write_tiles_cb' in params and params['write_tiles_cb'] is not None:
            write_tiles_cb = partial(params['write_tiles_cb'], self.dzi_filename, level, params['display_idx'] - 1)
        return write_tiles_cb

    def update_lowres_levels(self, hr_boundaries=None, **kwargs):
        '''Update lower pyramid levels (lower resolutions) of changes in the bottom
           pyramid level (full resolution).
           If higher levels images exists, they are deleted (unlinked) first.
           New files with updated content are then written.
        '''

        base_level = self.descriptor.num_levels - 1
        prev_level_bounds = hr_boundaries
        self._logger.debug("start updating the cache with lowres levels ")
        # iterating from high resolution to low resolution order, excluding the
        # highest resolution.
        # every iteration returns the tile xy bounds to be used in the next iteration
        for level in reversed(range(base_level)):
            #      print 'level %d: prev bounds: %s' % (level, prev_level_bounds)
            write_tiles_cb = self._create_write_tiles_cb(kwargs, level)
            prev_level_bounds = self._update_lowres_level(level, prev_level_bounds, write_tiles_cb)

        updated_tiles = self.updated_tiles
        self._logger.debug("finished updating the cache with lowres levels ")
        updated_tiles_msg = '[%s] ' % kwargs['display_idx'] if 'display_idx' in kwargs else ''
        updated_tiles_msg += "updated the following tiles: %s" % self.updated_tiles
        if 'updated_tiles_cb' in kwargs and kwargs['updated_tiles_cb'] is not None:
            kwargs['updated_tiles_cb'](kwargs.get('display_idx'), updated_tiles)
        self._logger.debug(updated_tiles_msg)
        self.updated_tiles = []
        return updated_tiles

    def _update_lowres_level(self, level, hr_boundaries=None, write_tiles_cb=None):
        '''Update a lower resolution level of pyramid from 1 high res level above it.
           If the lower resolution levels images exists, they are deleted (unlinked)
           first. New files with updated content are then written.

           Update can be limited to a subgroup of tiles using the boundaries
           parameter: (tile_x_min, tile_x_max, tile_y_min, tile_y_max).
           These boundaries refer to higher resolution level pyramid tiles.
        '''

        base_level = self.descriptor.num_levels - 1
        assert level >= 0 and level < base_level

        hcols, hrows = self.descriptor.get_num_tiles(level + 1)

        # calculate higher res level tiles range in x,y
        if hr_boundaries is not None:
            htx_min = floor_to_even(hr_boundaries[0])
            htx_max = ceil_to_odd(hr_boundaries[1])
            hty_min = floor_to_even(hr_boundaries[2])
            hty_max = ceil_to_odd(hr_boundaries[3])
        else:
            htx_min = 0
            htx_max = hcols - 1
            hty_min = 0
            hty_max = hrows - 1

        # read all updated tiles data into a single image.
        # NOTE: assumes update image size is small enough to fit in memory
        # self._logger.debug("start read_tiles")
        higher_res_image = self.read_tiles(level + 1, htx_min, htx_max, hty_min, hty_max)
        # self._logger.debug("finish read_tiles")

        # lower resolution TileXY min/max coordinates
        ltx_min, ltx_max, lty_min, lty_max = htx_min // 2, htx_max // 2, hty_min // 2, hty_max // 2

        tl_bounds = self.descriptor.get_tile_bounds(level, ltx_min, lty_min)
        br_bounds = self.descriptor.get_tile_bounds(level, ltx_max, lty_max)

        x_start = tl_bounds[0]
        x_end = br_bounds[2]
        y_start = tl_bounds[1]
        y_end = br_bounds[3]

        width = x_end - x_start
        height = y_end - y_start

        # width, height = self.descriptor.get_dimensions(level)

        # resize the image
        downsized_image = cv2.resize(higher_res_image, (width, height))
        # lower res tiles range in numbers is half the higher res range
        for row in range(lty_min, lty_max + 1):
            for col in range(ltx_min, ltx_max + 1):
                bounds = self.descriptor.get_tile_bounds(level, col, row)
                xs, ys, xe, ye = bounds

                xs -= x_start
                xe -= x_start
                ys -= y_start
                ye -= y_start
                # print 'updating level %d --- TileXY: [%d,%d] tile bounds:  X: [%d,%d) Y: [%d,%d)' % \
                # (level, col, row, xs, xe, ys, ye)

                image_file = '%s/%d/%d_%d.%s' % (self.pyramid_root, level, col, \
                                                 row, self.descriptor.tile_format)

                if not os.path.isfile(image_file):
                    self._logger.warning('warning: missing tile image file: %s' % image_file)

                tile_data = downsized_image[ys:ye, xs:xe, :]

                self.write_image(image_file, tile_data, write_tiles_cb)

                self.updated_tiles.append([level, col, row])

        # return boundaries for the next (lower res) level
        if hr_boundaries is None:
            return None
        else:
            if ltx_min == ltx_max and lty_min == lty_max:
                return None
            else:
                return (ltx_min, ltx_max, lty_min, lty_max)

    def update_image(self, image, x, y):
        '''updates pyramid with a given image at coordinates (x,y)'''

        im_h, im_w = image.shape[0:2]
        if x + im_w > self.width or y + im_h > self.height:
            raise ValueError('out of bounds update')

        tiles_range = self.get_tiles_range(x, y, im_w, im_h)

    def write_tiles(self, img_file, write_tiles_cb=None):
        try:
            tile = None
            with self.updated_tiles_dict_lock:
                if img_file in self.updated_tiles_dict:
                    tile = self.updated_tiles_dict[img_file]
                    del self.updated_tiles_dict[img_file]

            if tile is not None:
                parts = os.path.splitext(img_file)
                img_file_tmp = parts[0] + '.tmp' + parts[1]
                cv2.imwrite(img_file_tmp, tile)
                os.rename(img_file_tmp, img_file)
                if write_tiles_cb is not None:
                    write_tiles_cb()

                with self.updated_tiles_dict_lock:
                    # check that the tile wasn't changed in the meantime
                    if img_file not in self.updated_tiles_dict:
                        del self.updated_tiles_read_dict[img_file]

        except:
            self._logger.error(traceback.format_exc())

    def update_lowres_tiles(self, display_idx=None, updated_tiles_cb=None):
        if self.tiles_range_cache is None:
            return
        kwargs = dict(updated_tiles_cb=updated_tiles_cb)
        if display_idx is not None:
            kwargs['display_idx'] = display_idx

        self._lowres_pool.do(
            None,
            **kwargs
        )

    def update_lowres(self, tiles_range, **kwargs):
        try:
            defer_update_tiles = kwargs.get('defer_update_tiles', False)

            if self.tiles_range_cache is None:
                # first call: save the tiles_range for the next call
                self.tiles_range_cache = TileRanges()

            if tiles_range is not None:
                self._logger.info('dzimage: adding tiles range: {}'.format(tiles_range))
                self.tiles_range_cache.add(tiles_range)

            # update lowres levels every second call
            if not defer_update_tiles:
                for tr in self.tiles_range_cache.all():
                    self._logger.info('dzimage: updating tiles range {}'.format(tr))
                    self.update_lowres_levels(tr, **kwargs)
                self.tiles_range_cache = None
        except:
            self._logger.error(traceback.format_exc())

    def init_parallel(self, num_workers):
        self._logger.debug('init_parallel: start workers')
        raise ValueError('not supported in this fork of dzimage')
        # self._write_pool = ThreadPool(self.write_tiles, nworkers=3)
        # self._lowres_pool = ThreadPool(self.update_lowres)
        # self.updated_tiles_dict_lock = threading.Lock()

    def sync(self):
        '''blocks while there are files to be written '''
        if not self.is_parallel:
            return

        self._lowres_pool.wait()
        self._write_pool.wait()

        # make sure no lowres tiles need to be updated
        self.update_lowres_tiles()
        self._lowres_pool.wait()
        self._write_pool.wait()

    def close(self):
        if self.is_parallel:
            self.sync()
            self._write_pool.close()
            self._lowres_pool.close()


class DeepZoomImageDescriptor(object):
    def __init__(self, width=None, height=None,
                 tile_size=254, tile_overlap=1, tile_format='jpg'):
        self.width = width
        self.height = height
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.tile_format = tile_format
        self._num_levels = None

    def open_dzi_file(self, source):
        """Intialize descriptor from an existing descriptor file."""
        doc = xml.dom.minidom.parse(source)
        self.load_descriptor(doc)

    def open_dzi_string(self, source):
        """Intialize descriptor from an existing descriptor file."""
        doc = xml.dom.minidom.parseString(source)
        self.load_descriptor(doc)

    def load_descriptor(self, doc):
        image = doc.getElementsByTagName('Image')[0]
        size = doc.getElementsByTagName('Size')[0]
        self.width = int(size.getAttribute('Width'))
        self.height = int(size.getAttribute('Height'))
        self.tile_size = int(image.getAttribute('TileSize'))
        self.tile_overlap = int(image.getAttribute('Overlap'))
        self.tile_format = image.getAttribute('Format')

    def save(self, destination, **kwargs):
        """Save descriptor file."""
        if 'dynamic' in kwargs and kwargs['dynamic'] == True:
            ns_deepzoom = NS_DEEPZOOM + "/scopio"
        else:
            ns_deepzoom = NS_DEEPZOOM
        file = open(destination, 'wb')
        doc = xml.dom.minidom.Document()
        image = doc.createElementNS(NS_DEEPZOOM, 'Image')
        image.setAttribute('xmlns', ns_deepzoom)
        image.setAttribute('TileSize', str(self.tile_size))
        image.setAttribute('Overlap', str(self.tile_overlap))
        image.setAttribute('Format', str(self.tile_format))
        size = doc.createElementNS(NS_DEEPZOOM, 'Size')
        size.setAttribute('Width', str(self.width))
        size.setAttribute('Height', str(self.height))
        image.appendChild(size)
        doc.appendChild(image)
        descriptor = doc.toxml(encoding='UTF-8')
        file.write(descriptor)
        file.close()

    @property
    def num_levels(self):
        """Number of levels in the pyramid."""
        if self._num_levels is None:
            max_dimension = max(self.width, self.height)
            self._num_levels = int(math.ceil(math.log(max_dimension, 2))) + 1
        return self._num_levels

    def get_scale(self, level):
        """Scale of a pyramid level."""
        assert 0 <= level and level < self.num_levels, 'Invalid pyramid level'
        max_level = self.num_levels - 1
        return math.pow(0.5, max_level - level)

    def get_dimensions(self, level):
        """Dimensions of level (width, height)"""
        assert 0 <= level and level < self.num_levels, 'Invalid pyramid level'
        scale = self.get_scale(level)
        width = int(math.ceil(self.width * scale))
        height = int(math.ceil(self.height * scale))
        return (width, height)

    def get_num_tiles(self, level):
        """Number of tiles (columns, rows)"""
        assert 0 <= level and level < self.num_levels, 'Invalid pyramid level'
        w, h = self.get_dimensions(level)
        return (int(math.ceil(float(w) / self.tile_size)),
                int(math.ceil(float(h) / self.tile_size)))

    def get_tile_bounds(self, level, column, row):
        """Bounding box of the tile (x1, y1, x2, y2)"""
        assert 0 <= level and level < self.num_levels, 'Invalid pyramid level'
        offset_x = 0 if column == 0 else self.tile_overlap
        offset_y = 0 if row == 0 else self.tile_overlap
        x = (column * self.tile_size) - offset_x
        y = (row * self.tile_size) - offset_y
        level_width, level_height = self.get_dimensions(level)
        w = self.tile_size + (1 if column == 0 else 2) * self.tile_overlap
        h = self.tile_size + (1 if row == 0 else 2) * self.tile_overlap
        w = min(w, level_width - x)
        h = min(h, level_height - y)
        return (x, y, x + w, y + h)


def retry(attempts, backoff=2):
    """Retries a function or method until it returns or
    the number of attempts has been reached."""

    if backoff <= 1:
        raise ValueError('backoff must be greater than 1')

    attempts = int(math.floor(attempts))
    if attempts < 0:
        raise ValueError('attempts must be 0 or greater')

    def deco_retry(f):
        def f_retry(*args, **kwargs):
            last_exception = None
            for _ in range(attempts):
                try:
                    return f(*args, **kwargs)
                except Exception as exception:
                    last_exception = exception
                    time.sleep(backoff ** (attempts + 1))
            raise last_exception

        return f_retry

    return deco_retry


@retry(0)
def safe_open(path):
    return BytesIO(urllib.urlopen(path).read())


def _get_files_path(path):
    return os.path.splitext(path)[0] + '_files'


def _remove(path):
    os.remove(path)
    tiles_path = _get_files_path(path)
    shutil.rmtree(tiles_path)
