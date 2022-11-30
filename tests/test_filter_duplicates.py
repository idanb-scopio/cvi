import pickle
import os
from core.postprocess import filter_duplicate_detections

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def test_filter_duplicate_detections():
    pkl_file = f'{SCRIPT_DIR}/assets/global_detections_map.pickle'
    with open(pkl_file, 'rb') as f:
        global_detections_map = pickle.load(f)

    filtered_detections = filter_duplicate_detections(global_detections_map)
