# Dataset Manifest
import json
import logging
import os
import random
import yaml
from collections import Counter

from core.labels import get_shortform_label_mapping
from lib.pyramiddata import apply_blacklist_res_swap
from lib.scanimsrc import FP_RES_ERR

from tabulate import tabulate


def manifest_to_str(manifest):
    """
    :returns a string version of a given manifest, for log purposes.
    :param manifest: array of (scan_id, x, y, w, h, label_str)
    :return: string listing all manifest entries, one for each line.
    """
    manifest_str = ''
    for entry in manifest:
        scan_id, x, y, w, h, label_str = entry
        manifest_str += f'{scan_id} {x},{y},{w},{h} "{label_str}"\n'

    return manifest_str


def apply_label_mapping(manifests, label_mapping):
    """
    Apply in-place label mapping for a given per-class manifest.
    :param manifests: dictionary of {class_str: manifest}
    :param label_mapping: dictionary of {unmapped label str: class str}
    """
    for cls_str in manifests:
        # per class manifest. list of tuples: (scan_id, x, y, w, h, label_str)
        cls_manifest = manifests[cls_str]

        # transform the label_str to a mapped version
        mapped_manifest = [(*e[0:5], label_mapping[e[5]]) for e in cls_manifest]

        manifests[cls_str] = mapped_manifest


def check_manifest_exclusivity(train_manifest, val_manifest):

    # validation set may be empty
    if not val_manifest:
        return

    # check all fields excluding label string
    train_set = set()
    for entry in train_manifest:
        train_set.add(entry[0:5])

    val_set = set()
    for entry in val_manifest:
        val_set.add(entry[0:5])

    common_set = train_set.intersection(val_set)
    if common_set:
        raise RuntimeError('train/val intersection')


class LabelsManifest:
    """
    A label manifest is defined as a tuple of (scan_id, x, y, w, h, label_str),
    identifying specific samples in the dataset.
    label manifest is used as a middle stage which allows processing labels at
    the global (whole dataset) scope.
    """
    def __init__(self, label_mapping=None, **kwargs):
        """
        :param shortform_map: translate long form labels (aberrant lymphocyte) to short form (abl)
        """

        self.class_manifests = {}

        # label stats, for each different input resolution
        self.res_stats = {}

        # label stats, per class
        self.cls_stats = {'train': {}, 'val': {}}

        # custom fp res error in mm can be set as a parameter to treat close
        # enough resolutions as the same and avoid resizing.
        self.fp_res_err = kwargs.get('fp_res_err', FP_RES_ERR)

        if label_mapping:
            self.label_mapping = label_mapping
        else:
            # default mapping is to the shortform (al, lgl, neu, etc..)
            self.label_mapping = get_shortform_label_mapping()

        self.random = random.Random(1337)

        self.train_manifests = {}
        self.val_manifests = {}

        # scan id resolution map: uuid -> mm/pixel for all scans in the manifest
        self.scan_id_res_map = {}

    def get_scan_id_res_map(self):
        return self.scan_id_res_map

    def create_from_files(self, json_files, val_split=0.0, shuffle=True, oversampling_map=None,
                          discarded_dir=None):
        """
        Add labels from given json files content to the manifest.
        :param json_files: dataset cvi json files.
        :param val_split: ratio of samples to allocate for validation set. 0.0 means
                          empty validation set (default)
        :param shuffle: shuffle output labels
        :param oversampling_map: input oversampling dictionary. key is a global
                                 label identifier, defined as its first 5 elements:
                                 (scan_id, x, y, w, h). label_str is not used in
                                 the identifier.
        :param discarded_dir: (optional) a directory to save all discarded cells as CVI json
                              files.
        """
        if val_split < 0.0 or val_split > 0.5:
            raise ValueError('invalid validation split value')

        if discarded_dir:
            logging.info(f'discarded cells will be saved to: {discarded_dir}')
            if not os.path.exists(discarded_dir):
                os.mkdir(discarded_dir)

        dup_log_str = ''

        logging.info(f'creating manifest from {len(json_files)} json files')
        logging.info(f'gathering raw label data')
        blacklisted_resolutions = 0
        for jf in json_files:
            with open(jf, 'r') as f:
                labeling_data = json.load(f)

            scan_id = labeling_data['scan_id']
            pyramid_resolution = labeling_data['pyramid_resolution']
            pyr_res_workaround = apply_blacklist_res_swap(pyramid_resolution)

            if pyr_res_workaround != pyramid_resolution:
                blacklisted_resolutions += 1
                logging.warning(f'blacklisted resolution for scan {scan_id} corrected: {pyramid_resolution} '
                                f'-> {pyr_res_workaround}')
                pyramid_resolution = pyr_res_workaround

            self.scan_id_res_map[scan_id] = pyramid_resolution

            # res_in_stats_dict is used as a floating point key inside a dictionary.
            # since 0.000133 and 0.00013302 may mean the same resolution but are
            # different floating point values, an existing resolution is searched
            # (up to fp_res_err: floating point error for mm/pixel units). If a
            # close enough resolution is found that key is used, otherwise the
            # current value is added as a new key.
            res_in_stats_dict = None
            for res in self.res_stats:
                if abs(res - pyramid_resolution) < self.fp_res_err:
                    res_in_stats_dict = res
                    break

            if not res_in_stats_dict:
                res_in_stats_dict = pyramid_resolution
                self.res_stats[res_in_stats_dict] = Counter()

            discarded_labels = []
            labels = labeling_data['labels']

            # for duplicate entries checks
            dup_check_dict = {}

            for label in labels:
                x, y, w, h = label[0:4]
                label_str = label[4].lower()

                if label_str not in self.label_mapping:
                    discarded_labels.append(label)
                    continue

                mapped_str = self.label_mapping[label_str]

                manifest_entry = (scan_id, x, y, w, h, label_str)
                dup_entry = (scan_id, x, y, w, h)
                if dup_entry in dup_check_dict:
                    dup_log_str += f'duplicate: {dup_entry[0]} {",".join(str(e) for e in dup_entry[1:5])} '\
                                   f'"{dup_check_dict[dup_entry]}" "{label_str}"\n'
                    continue
                else:
                    dup_check_dict[dup_entry] = label_str

                if mapped_str in self.class_manifests:
                    self.class_manifests[mapped_str].append(manifest_entry)
                else:
                    self.class_manifests[mapped_str] = [manifest_entry]

                self.res_stats[res_in_stats_dict][label_str] += 1

            if discarded_dir and discarded_labels:
                discarded_ds = labeling_data.copy()
                discarded_ds['labels'] = discarded_labels
                with open(f'{discarded_dir}/{scan_id}.json', 'w') as f:
                    json.dump(discarded_ds, f, indent=4)

        if blacklisted_resolutions > 0:
            logging.warning(f'worked around {blacklisted_resolutions} black listed resolutions.')

        if discarded_dir and dup_log_str:
            with open(f'{discarded_dir}/duplicates.log', 'w') as f:
                f.write(dup_log_str)

        logging.info(f'creating class manifests')
        self._create_manifests(val_split=val_split, shuffle=shuffle, oversampling_map=oversampling_map)
        logging.info(f'labels manifests creation done.')

    def _check_train_val_exclusivity(self):
        for cls_str in self.train_manifests:
            train_mf = self.train_manifests[cls_str]
            val_mf = self.val_manifests[cls_str]
            check_manifest_exclusivity(train_mf, val_mf)

    def _create_manifests(self, val_split=0.0, shuffle=True, oversampling_map=None):

        for label_str in self.class_manifests:
            # create a copy of the labels manifest list so it can be modified
            # by different user requests (shuffle, oversampling)
            manifest = self.class_manifests[label_str].copy()

            # oversampling buffer holds those entries which appear in the oversampling_map
            # while they are temporarily taken out of the manifest. Those entries then
            # return as N duplicate entries after the train/val split to the training
            # manifest. This is done to ensure all the samples that appear in the
            # oversampling_map end up only in the training set.
            oversampling_buffer = []

            if oversampling_map:
                modified_manifest = []
                for entry in manifest:
                    # manifest entry are (scan_id, x, y, w, h, label_str)
                    # for the purpose of identifying a label, only the (x, y, w, h)
                    # are used: oversampling_map is a dict: (scan_id, x, y, w, h) -> multiple (int)
                    global_label_id = entry[0:5]
                    if global_label_id in oversampling_map:
                        boost_factor = oversampling_map[global_label_id]
                        oversampling_buffer.append(boost_factor * [entry])
                    else:
                        modified_manifest.append(entry)

                # the modified manifest, which omits oversampled labels continues to the next
                # manifest processing steps
                manifest = modified_manifest

            if shuffle:
                self.random.shuffle(manifest)

            if val_split > 0.0:
                # train/val split
                train_split = 1.0 - val_split
                train_elements = int(len(manifest) * train_split)

                train_manifest = manifest[0:train_elements]
                val_manifest = manifest[train_elements:]
            else:
                train_manifest = manifest
                val_manifest = {}

            if oversampling_buffer:
                train_manifest.append(oversampling_buffer)
                if shuffle:
                    self.random.shuffle(train_manifest)
                else:
                    logging.warning('oversampling with no shuffling.')

            self.cls_stats['train'][label_str] = len(train_manifest)
            self.cls_stats['val'][label_str] = len(val_manifest)

            logging.info(f'created label manifest for {label_str}, train: {len(train_manifest)}, '
                         f'val: {len(val_manifest)}')

            self.train_manifests[label_str] = train_manifest
            self.val_manifests[label_str] = val_manifest

        # check there are no overlaps. this raises a RuntimeError in case there is.
        self._check_train_val_exclusivity()

    def get_manifests(self, apply_mapping=True, limit=None):
        """
        :return:
        tuple of two: train, val class manifests dictionaries. each dictionary is of
        label_str -> per class (label) manifest
        label_str -> [(scan_id, x, y ,w, h, label_str), ...]
        """
        if not self.train_manifests:
            raise RuntimeError('empty manifest')

        train_manifests = self.train_manifests.copy()
        val_manifests = self.val_manifests.copy()

        if apply_mapping:
            apply_label_mapping(train_manifests, self.label_mapping)
            apply_label_mapping(val_manifests, self.label_mapping)

        if limit:
            logging.warning(f'manifest limit is set to {limit} samples.')
            train_manifests = {cls: mfst[:limit] for cls, mfst in train_manifests.items()}
            val_manifests = {cls: mfst[:limit] for cls, mfst in val_manifests.items()}

        return train_manifests, val_manifests

    def get_summary(self):
        summary_str = 'labels manifest summary:\n\n'

        res_totals = {}
        cells_total = Counter()
        for res in self.res_stats:
            res_totals[res] = sum(self.res_stats[res].values())
            cells_total += self.res_stats[res]

            per_res_list = [('label', 'count')] + sorted(list(self.res_stats[res].items()),
                                                         key=lambda x: x[1], reverse=True)
            summary_str += f'statistics for resolution of {res:.7f} mm/pixel:\n'
            summary_str += tabulate(per_res_list, headers='firstrow')
            summary_str += '\n\n\n'

        all_res_list = [('resolution', 'total cells')] + sorted(list(res_totals.items()),
                                                                key=lambda x: x[1], reverse=True)
        summary_str += f'resolution based summary:\n'
        summary_str += tabulate(all_res_list, headers='firstrow')
        summary_str += '\n\n\n'

        all_cells_list = [('label', 'count')] + sorted(list(cells_total.items()),
                                                       key=lambda x: x[1], reverse=True)
        summary_str += f'cells based summary (all resolutions):\n'
        summary_str += tabulate(all_cells_list, headers='firstrow')
        summary_str += '\n\n\n'

        return summary_str

    def save_manifest_metadata(self, output_dir):
        logging.info(f'saving manifest logs in: {output_dir}')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        for label_str in self.train_manifests:
            mf_str = manifest_to_str(self.train_manifests[label_str])
            mf_file = f'{output_dir}/manifest-train-class_{label_str}.log'
            with open(mf_file, 'w') as f:
                f.write(mf_str)

        for label_str in self.val_manifests:
            if not self.val_manifests[label_str]:
                continue

            mf_str = manifest_to_str(self.val_manifests[label_str])
            mf_file = f'{output_dir}/manifest-val-class_{label_str}.log'
            with open(mf_file, 'w') as f:
                f.write(mf_str)

        summary_str = self.get_summary()
        with open(f'{output_dir}/summary.txt', 'w') as f:
            f.write(summary_str)

        with open(f'{output_dir}/dist.yml', 'w') as f:
            f.write(yaml.safe_dump(self.cls_stats))
