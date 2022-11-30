import glob
import math
import os
import yaml
import tensorflow as tf


def parse_tfrec_subdirs(dataset_dir, sample_classes=None):
    """
    Traverse the dataset directory and subdirectory and return list of .tfrecs.
    If sample_classes (list of class strings) are given, a dictionary of lists
    is returned instead of flat list, keyed by the class strings.
    """
    dataset_dir_abs = os.path.abspath(dataset_dir)
    all_tfrecs = glob.glob(f'{dataset_dir_abs}/**/*.tfrec')
    if not sample_classes:
        return all_tfrecs

    cls_tfrecs = {cls: [] for cls in sample_classes}
    for tfrec in all_tfrecs:
        # verify directory structure
        tfrec_path_array = tfrec.split('/')
        basedir = '/'.join(tfrec_path_array[:-2])
        if not basedir == dataset_dir_abs:
            raise RuntimeError('invalid directory structure')

        class_str = tfrec_path_array[-2]
        cls_tfrecs[class_str].append(tfrec)

    return cls_tfrecs


class TFRDataset:

    def __init__(self, dataset_dir, cells_model=None, sampled=False, repeat=True):
        if sampled and not cells_model:
            raise ValueError('sampled TFRDataset requires cells_model instance.')

        if sampled:
            self.sample_classes = list(cells_model.get_label_mapping().keys())
        else:
            self.sample_classes = None

        self.dataset = None

        tfrecs = parse_tfrec_subdirs(dataset_dir, self.sample_classes)
        if len(tfrecs) == 0:
            return

        # flat_tfrecs is a list of all tfrecors under dataset_dir
        self.flat_tfrecs = None

        # hier_tfrecs is hierarchy based (dict, k: class str, v: list of tfrecs).
        self.hier_tfrecs = None

        if type(tfrecs) == list:
            self.flat_tfrecs = tfrecs
        else:
            self.hier_tfrecs = tfrecs
            self.flat_tfrecs = sum(tfrecs.values(), [])

        if not sampled:
            self.dataset = tf.data.TFRecordDataset(self.flat_tfrecs)
        else:
            datasets = []
            for cls in self.sample_classes:
                # create a dataset (per class) from its tfrecord file(s)
                ds = tf.data.TFRecordDataset(self.hier_tfrecs[cls])
                datasets.append(ds)

            self.dataset = tf.data.experimental.sample_from_datasets(datasets,
                                                                     stop_on_empty_dataset=True)
        if repeat:
            self.dataset = self.dataset.repeat()

        with open(f'{dataset_dir}/dist.yml', 'r') as f:
            self.dist = yaml.safe_load(f)

    def get_dataset(self):
        return self.dataset

    def get_steps_per_epoch(self, batch_size):
        total_samples = sum(self.dist.values())
        if not self.sample_classes:
            return math.ceil(total_samples / batch_size)
        else:
            # the number of samples to cover is such that the number of batches
            # required to see each example once
            samples_to_cover = max(self.dist.values()) * len(self.dist.values())
            return math.ceil(samples_to_cover / batch_size)

    def get_dist(self):
        return self.dist
