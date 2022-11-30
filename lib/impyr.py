"""
Image Pyramid reader: interface to access .jpg images with the same API as DZImage (pyramid).
"""

import math
import cv2


class ImagePyramid:

    def __init__(self, image_file):
        self.image = cv2.imread(image_file)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.width = self.image.shape[1]
        self.height = self.image.shape[0]
        largest_dim = max(self.image.shape)
        self.base_level = math.ceil(math.log(largest_dim, 2))

    def read(self, x, y, width, height, level=None):

        if level is None:
            level = self.base_level

        resize_factor = 2 ** (self.base_level - level)
        if resize_factor == 1:
            image = self.image
        else:
            resized_h = self.image.shape[0] // resize_factor
            resized_w = self.image.shape[1] // resize_factor
            image = cv2.resize(self.image, dsize=(resized_w, resized_h))

        return image[y:y+height, x:x+width, :]
