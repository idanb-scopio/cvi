import numpy as np
import cv2
from lib.impyr import ImagePyramid


def test_basic_operations():
    image_file = '/tmp/test_impyr_image.png'

    # generate patterned image
    dim_w = 256
    dim_h = 200

    row = np.array(range(0, dim_w), dtype='uint8')

    image = np.empty(shape=(dim_h, dim_w, 3), dtype='uint8')
    for i in range(dim_h):
        for c in range(3):
            image[i, :, c] = np.roll(row, i)

    cv2.imwrite(image_file, image)

    impyr = ImagePyramid(image_file)

    cropped = impyr.read(11, 10, 2, 4)
    exp1 = np.array(np.array([[1,   2],
                              [0,   1],
                              [255, 0],
                              [254, 255]]))
    expected_im = np.empty(shape=(*exp1.shape, 3), dtype='uint8')
    for i in range(3):
        expected_im[:, :, i] = exp1

    assert np.array_equal(cropped, expected_im)
