# Mosaic Saver saves a sequence of images in a mosaic like large image
import os
import logging
import re
import cv2
import numpy as np

from lib.debugutils import mark_rectangle


def factors(n):
    """Inefficient but simple way to find all factors of n."""
    if n > 100000:
        raise ValueError('not meant for big numbers')

    all_factors = []
    for i in range(1, n+1):
        if n % i == 0:
            all_factors.append(i)

    return all_factors


def find_nice_aspect_ratio(n):
    """finds nice (visually) Rows x Cols such that Rows x Cols <= n"""
    if n <= 0:
        raise ValueError('invalid value')

    if n == 1:
        return 1, 1

    min_ratio = 1.0
    max_ratio = 1.5

    while True:
        fac = factors(n)

        # perfect square
        n_fac = len(fac)
        if n_fac % 2 == 1:
            sz = fac[n_fac // 2]
            return sz, sz

        b_idx = n_fac // 2
        rows = fac[b_idx-1]
        cols = fac[b_idx]
        aspect_ratio = cols / rows
        if min_ratio <= aspect_ratio <= max_ratio:
            return rows, cols

        n = n + 1


def create_mosaic_image(mosaic_shape, images_array, cols_first=True):
    rows, cols = mosaic_shape
    if rows * cols < len(images_array):
        raise ValueError('matrix too small for the given images array')

    if len(images_array) == 0:
        raise ValueError('empty images array')

    sub_image_shape = images_array[0].shape

    full_image_w = mosaic_shape[1] * sub_image_shape[1]
    full_image_h = mosaic_shape[0] * sub_image_shape[0]

    mosaic_image = np.zeros(shape=(full_image_h, full_image_w, 3), dtype='uint8')
    im_h, im_w = sub_image_shape[0], sub_image_shape[1]

    for sub_image_idx in range(len(images_array)):
        if cols_first:
            ix = sub_image_idx // rows
            iy = sub_image_idx % rows
        else:
            ix = sub_image_idx % cols
            iy = sub_image_idx // cols

        # x,y start coordinates for assignment
        xs = ix * im_w
        ys = iy * im_h

        mosaic_image[ys:ys+im_h, xs:xs+im_w] = images_array[sub_image_idx]

    return mosaic_image


class MosaicSaver:

    def __init__(self, sub_image_shape, mosaic_w, mosaic_h, output_dir, tag='',
                 cols_first=True, save_meta=False, image_ext='jpg'):
        if not os.path.exists(output_dir):
            raise RuntimeError(f'output dir does not exist: {output_dir}')
        self.output_dir = output_dir

        if tag:
            if not re.match(r'^[a-zA-Z0-9_-]+$', tag):
                raise ValueError(f'tag formatting error: [tag={tag}]')
        self.tag = tag

        if not sub_image_shape[2] == 3:
            raise ValueError('supporting only RGB images')

        self.image_ext = image_ext

        sub_h, sub_w = sub_image_shape[0], sub_image_shape[1]
        rows = mosaic_h // sub_h
        cols = mosaic_w // sub_w
        self.mosaic_shape_max = (rows, cols)
        self.mosaic_max_sub_images = rows * cols
        self.sub_image_shape = sub_image_shape

        # temporary store for incoming images until saved or until mosaic_max_sub_images reached.
        self.mosaic_images = []

        self.full_images_counter = 1
        self.cols_first = cols_first

        # absolute counter across all mosaics
        self.save_meta = save_meta
        self.meta = []

    def get_max_sub_images(self):
        return self.mosaic_max_sub_images

    def save(self):
        if len(self.mosaic_images) == 0:
            return

        output_filename = f'{self.output_dir}/mosaic'
        if self.tag:
            output_filename += f'-{self.tag}'
        output_filename += f'-{self.full_images_counter}.{self.image_ext}'

        if len(self.mosaic_images) == self.mosaic_max_sub_images:
            mosaic_shape = self.mosaic_shape_max
        else:
            mosaic_shape = find_nice_aspect_ratio(len(self.mosaic_images))

        # create mosaic image
        mosaic_image = create_mosaic_image(mosaic_shape, self.mosaic_images, self.cols_first)

        logging.info(f'mosaic saver: writing {output_filename}')
        cv2.imwrite(output_filename, mosaic_image)

        if self.save_meta:
            output_filename = f'{self.output_dir}/mosaic'
            if self.tag:
                output_filename += f'-{self.tag}'
            output_filename += f'-{self.full_images_counter}.meta.txt'

            logging.info(f'mosaic saver: writing {output_filename}')
            with open(output_filename, 'w') as f:
                for line in self.meta:
                    f.write(f'{line}\n')

            self.meta = []

        self.mosaic_images = []
        self.full_images_counter += 1

    def add_image(self, image, with_index_caption=True, meta=None):
        """
        Adds an image into the larger mosaic of images. Image is placed col-stack (default) / row-stack in the order
        of arrival. When the mosaic is full it is saved to disk. Note: if the mosaic is not full a call to the .save()
        method must be made at the end to save the last mosaic image to disk.
        :param image: RGB numpy image
        :param with_index_caption: if True, the index and border are drawn around every added image.
        :param meta: metadata to save for each image (string)
        """
        if tuple(image.shape) != tuple(self.sub_image_shape):
            raise ValueError(f'shape error: expecting: {self.sub_image_shape}, received: {image.shape}')

        if image.dtype != 'uint8':
            raise ValueError(f'supporting only uint8 images')

        # actual writing of the image is delayed. create a copy of the numpy array
        image = image.copy()
        sub_image_idx = len(self.mosaic_images)

        if with_index_caption:
            mark_rectangle(image=image,
                           rect=(0, 0, self.sub_image_shape[1], self.sub_image_shape[0]),
                           text=f'{sub_image_idx}',
                           color=(0, 255, 0),
                           line_width=1,
                           draw_border=True)

        if meta:
            self.meta.append(f'{sub_image_idx}: {meta}')

        self.mosaic_images.append(image)

        if len(self.mosaic_images) == self.mosaic_max_sub_images:
            self.save()
