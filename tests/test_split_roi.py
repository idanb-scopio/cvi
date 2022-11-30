import random
from core.utils import split_rois


def test_split_roi():
    test_rois = [
        [3221, 12107, 8113, 31417],
        [3221, 12107, 1645, 1849]
    ]
    tile_size = 1024
    tile_overlap = 100

    for roi in test_rois:
        rois = split_rois(large_roi=roi, tile_size=tile_size, overlap=tile_overlap)
        max_x = 0
        max_y = 0
        for r in rois.values():
            x, y, w, h = r
            x_e = x + w
            if x_e > max_x:
                max_x = x_e
            y_e = y + h
            if y_e > max_y:
                max_y = y_e

        assert max_x == roi[0] + roi[2]
        assert max_y == roi[1] + roi[3]


