from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import random
import math
from tools.io_utils import read_json_file, write_json_file
from tools.geometry_utils import get_scaled_region, rect_to_region

class ImportTilesDataset(object):

    """
    Import TF v1 training tiles dataset and prepare CVI scan datasets, examples including balancing
    information
    """
    def __init__(self, config):

        """
        Init scans datasets creation based on TF v1 training configuration including balancing keys,
        target distribution, class name mapping, and balancing parameters
        """

        self.config = copy.deepcopy(config)
        self.is_training = self.config['training']
        self.tiles_dataset_file = self.config['tile_dataset_file']
        self.tiles_dataset = read_json_file(self.tiles_dataset_file)
        self.save_dbg_info = True
        if not os.path.exists(self.config['out_dataset_path']):
            os.makedirs(self.config['out_dataset_path'])
        if self.is_training:
            self.target_distribution = self.config['target_distribution']
            self.balancing_classes_keys = self.config['balancing_classes_keys']
            self.balancing_classes_keys_map = {}
            for idx, key in enumerate(self.balancing_classes_keys):
                self.balancing_classes_keys_map[key] = idx
            if not os.path.exists(self.config['out_dataset_info_path']):
                os.makedirs(self.config['out_dataset_info_path'])

        self.class_map = {
            'NEU': 'Segmented Neutrophil',
            'LYM': 'Lymphocyte',
            'MON': 'Monocyte',
            'BAS': 'Basophil',

            'Mast': 'Mast Cell',
            'EOS': 'Eosinophil',
            'ProEosinophil': 'Pro Eosinophil',
            'BAND': 'Band Neutrophil',
            'ME': 'Metamyelocyte',
            'MYLE': 'Myelocyte',
            'Promyelocyte': 'Promyelocyte',
            'BLAST': 'Blast',
            'Blast': 'Blast',
            'PLASMA': 'Plasma Cell',
            'Plasma': 'Plasma Cell',
            'LGL': 'Large Granular Lymphocyte',
            'AL': 'Atypical Lymphocyte',
            'ABL': 'Aberrant Lymphocyte',
            'Macrophage': 'Macrophage',
            'RFL': 'Normoblast',
            'NRBC': 'Normoblast',
            'Normoblast': 'Normoblast',
            'Basophilic Normoblast': 'Basophilic Normoblast',
            'Polychromatophilic Normoblast': 'Polychromatophilic Normoblast',
            'GCD': 'Smudge Cell',
            'clutter': 'Nucleated',
            'Smudge': 'Smudge Cell',
            'Broken': 'Smudge Cell',
            'FCD': 'Erythroblast',
            'Erythroblast': 'Erythroblast',
            'Myeloid': 'Unclassified WBC',
            'Erythroid': 'Unclassified RBC',
            'RBC': 'Unclassified RBC',
            'UC_NEU': 'Unclassified Neutrophil',
            'Unclassified Neutrophil': 'Unclassified Neutrophil',
            'UC_LYM': 'Lymphocyte',
            'Unclassified Lymphocyte': 'Lymphocyte',
            'Unclassified': 'Unclassified',
            'NC': 'Unclassified',
            'NOT_NEU': 'Not Neutrophil',
            'NOT_LYM': 'Not Lymphocyte',
            'NOT_MON': 'Not Monocyte',
            'NOT_BAS': 'Not Basophil',
            'NOT_EOS': 'Not Eosinophil',
            'NOT_PLASMA': 'Not Plasma Cell',
            'NOT_GCD': 'Not Smudge Cell',
            'NOT_RFL': 'Not Normoblast',
            'NOT_ProEosinophil': 'Eosinophil',
            'NOT_Mast': 'Basophil',
            'NOT_Macrophage': 'Unclassified WBC',
            'WBC': 'Unclassified WBC',
            'unknown': 'Consulation',
            'Consulation': 'Consultation',
            'Consultation': 'Consultation',
            'PLT': 'Megakaryocyte',
            'Megakaryocyte': 'Megakaryocyte',
            'Unclassified Megakaryocyte': 'Dirt',
            'maybe_PLT': 'Suspected Megakaryocyte',
            'dirt': 'Dirt',
            'Dirt': 'Dirt',
            'NEG': 'Dirt'
        }

    def get_mapped_class_name(self, class_value):

        """
        :param class_value: balancing class key
        :return: mapped class name
        """

        class_value = str(class_value)
        if '_' in class_value:
            str_vals = class_value.split('_')
            class_name = str_vals[-1]
        else:
            class_name = class_value

        if not class_name in self.class_map:
            print(class_name)
            assert False

        mapped_class = str(self.class_map[class_name])

        return mapped_class

    def class_factor_mapping(self, hardness_level):

        """
        :param hardness_level:
        :return: hardness level factor
        """

        is_high_factor = (hardness_level > 1.5)
        if is_high_factor:
            mapped_class_factor = self.config['high_hardness_factor']
        else:
            mapped_class_factor = self.config['low_hardness_factor']

        return mapped_class_factor, is_high_factor

    def write_scans_datasets(self):

        if self.is_training:
            self.write_training_scans_datasets()
        else:
            self.write_validation_scans_datasets()

    def write_training_scans_datasets(self):

        """
         Write CVI scans datasets with additional labels information on expansion factor and
         balancing distribution rate index
        """

        examples_per_class_high = {}
        examples_per_class_low = {}

        # Gather tiles balancing information
        high_factor_num = 0
        low_factor_num = 0
        high_factor = 1.
        low_factor = 1.
        for scan_idx, scan_info in enumerate(self.tiles_dataset):
            for region_idx, region_info in enumerate(scan_info['regions_list']):
                for tile_idx, tile_info in enumerate(region_info['tile_list']):
                    tile_metadata = tile_info['tile_metadata']
                    tile_key = (scan_idx, region_idx, tile_idx)
                    class_value = tile_metadata[self.config['balancing_criteria']]
                    class_factor, is_high_factor = self.class_factor_mapping(tile_metadata['hardness_level'])
                    if is_high_factor:
                        high_factor = class_factor
                    else:
                        low_factor = class_factor
                    class_value_mapped = self.balancing_classes_keys_map[class_value]


                    if is_high_factor:
                        high_factor_num += 1
                        examples_per_class_high.setdefault(class_value, []).append((tile_key, class_value_mapped, class_value))
                    else:
                        low_factor_num += 1
                        examples_per_class_low.setdefault(class_value, []).append((tile_key, class_value_mapped, class_value))

        # Gather classes num stat
        class_num = {}
        class_num_sum = 0
        max_class_num = 0
        low_keys_list = [str(key) for key in examples_per_class_low.keys()]
        high_keys_list = [str(key) for key in examples_per_class_high.keys()]
        all_file_names_keys =  list(set(low_keys_list + high_keys_list))
        for key in all_file_names_keys:
            class_num[key] = 0
            if key in examples_per_class_low:
                class_num[key] += len(examples_per_class_low[key]) * low_factor
            if key in examples_per_class_high:
                class_num[key] += len(examples_per_class_high[key]) * high_factor
            class_num_sum += class_num[key]
            if max_class_num < class_num[key]:
                max_class_num = class_num[key]

        # Init target dist
        target_distribution = {}
        sum_prob = 0.
        for idx, prob in enumerate(self.target_distribution):
            key = self.balancing_classes_keys[idx]
            target_distribution[key] = 0.
            if key in all_file_names_keys:
                target_distribution[key] = prob
                sum_prob += prob

        for idx in range(len(self.target_distribution)):
            key = self.balancing_classes_keys[idx]
            target_distribution[key] /= sum_prob
            self.target_distribution[idx] = target_distribution[key]

        # Gather factors info
        min_class_factor = 1e10
        min_class_factor_key = None
        for key in all_file_names_keys:
            class_num_ratio = float(class_num[key]) / class_num_sum
            factor = target_distribution[key] / class_num_ratio
            if (min_class_factor > factor) and (factor > 2e-3):
                min_class_factor = factor
                min_class_factor_key = key

        max_reject_ratio = min(1., self.config['max_reject_ratio'])
        if (max_reject_ratio > 0.) and (min_class_factor < max_reject_ratio):
            expand_factor = max_reject_ratio / min_class_factor
        else:
            expand_factor = 1.

        # Set tile lists examples factors
        update_distribution = False
        class_factor = {}
        list_class_factor_low = {}
        list_class_factor_high = {}
        list_class_exact_factor_low = {}
        list_class_exact_factor_high = {}
        sum_prob = 0.
        for idx in range(len(self.target_distribution)):
            key = self.balancing_classes_keys[idx]
            if not key in all_file_names_keys:
                target_distribution[key] = 0
                update_distribution = True
                continue
            class_num_ratio = float(class_num[key]) / class_num_sum
            factor = expand_factor * target_distribution[key] / class_num_ratio
            if factor > self.config['max_class_factor']:
                factor = self.config['max_class_factor']
                target_distribution[key] = factor * class_num_ratio / expand_factor
                update_distribution = True
            sum_prob += target_distribution[key]
            class_factor[key] = factor
            for examples_lists, list_factor, list_class_factor, list_class_exact_factor in \
                    [(examples_per_class_low, low_factor, list_class_factor_low, list_class_exact_factor_low),
                     (examples_per_class_high, high_factor, list_class_factor_high, list_class_exact_factor_high)]:
                if key in examples_lists:
                    list_class_exact_factor[key] = class_factor[key]*list_factor
                    list_class_factor[key] = max(1, int(round(list_class_exact_factor[key])))
                    # list_class_factor[key] = max(1,int(math.ceil(list_class_exact_factor[key])))

        # Update target distribution due to balancing constraints
        if update_distribution:
            for idx in range(len(self.target_distribution)):
                key = self.balancing_classes_keys[idx]
                target_distribution[key] /= sum_prob
                self.target_distribution[idx] = target_distribution[key]

        # Set expanded data distribution
        semi_balanced_distribution = {}
        semi_balanced_distribution_ratio = {}
        examples_num = {}
        examples_num_all = 0
        for key in all_file_names_keys:
            examples_num[key] = 0
            if key in examples_per_class_low:
                examples_num[key] += len(examples_per_class_low[key]) * list_class_factor_low[key]
            if key in examples_per_class_high:
                examples_num[key] += len(examples_per_class_high[key]) * list_class_factor_high[key]
            examples_num_all += examples_num[key]
        for key in all_file_names_keys:
            semi_balanced_distribution[key] = examples_num[key] / float(examples_num_all)
            semi_balanced_distribution_ratio[key] = semi_balanced_distribution[key] / max(1e-6, target_distribution[key])

        # Print summary
        group_names = self.config['groups_names']
        src_path = os.path.dirname(os.path.realpath(__file__))
        save_to_file = ('/scopio' in src_path) and self.save_dbg_info
        f1 = None
        f2 = None
        if save_to_file:
            base_path = self.config.get('base_path', '/nfs/cvdataset/amir/tests/WBC_class/test_11')
            target = self.config.get('target_name', 'smudge')
            dbg_path = base_path + '/chk_data_input_%s' % target
            if not os.path.exists(dbg_path):
                os.makedirs(dbg_path)
            dbg_file_1 = '%s/%s_input_1.txt'%(dbg_path, target)
            dbg_file_2 = '%s/%s_input_2.txt'%(dbg_path, target)
            f1 = open(dbg_file_1, 'w')
            f2 = open(dbg_file_2, 'w')

        group_keys_num = {}

        for group_str in group_names:
            group_keys_num[group_str] = {}
        all_key_num = {}
        t_num_sum = 0
        for key in self.balancing_classes_keys:
            if key in all_file_names_keys:
                low_factor_files_num = len(examples_per_class_low.get(key, []))
                high_factor_files_num = len(examples_per_class_high.get(key, []))
                balance_factor = target_distribution[key] / semi_balanced_distribution[key]
                t_num = balance_factor * examples_num[key]
                print_str = '%s : td %5.3f, sbd %5.3f, f %3.1f, num %d, t_num %5.3f (l %d, h %d), dist_ratio %5.3f'% \
                            (str(key), target_distribution[key], semi_balanced_distribution[key], class_factor[key],
                             examples_num[key], t_num, low_factor_files_num, high_factor_files_num,
                             semi_balanced_distribution_ratio[key])
                print(print_str)
                t_num_sum += t_num
                if save_to_file:
                    if print_str.startswith('hard'):
                        f2.write('%s\n' % print_str)
                    else:
                        f1.write('%s\n' % print_str)
                key_ = str(key)
                if 'hard_' in key_:
                    key_ = key_[5:]
                for group_str in group_names:
                    if key_.startswith(group_str):
                        class_str = key_[(len(group_str) + 1):]
                        if class_str not in group_keys_num[group_str]:
                            group_keys_num[group_str][class_str] = 0
                        group_keys_num[group_str][class_str] += examples_num[key]
                        if class_str not in all_key_num:
                            all_key_num[class_str] = 0
                        all_key_num[class_str] += examples_num[key]

        for class_str in sorted(all_key_num.keys()):
            print_str = '%s: '%class_str
            for group_str in group_names:
                num = 0
                if class_str in group_keys_num[group_str].keys():
                    num = group_keys_num[group_str][class_str]
                print_str = print_str +  '%s %d, '%(group_str, num)
            print_str = print_str + 'all %d'%all_key_num[class_str]
            print(print_str)
            if save_to_file:
                f2.write('%s\n'%print_str)
                f1.write('%s\n'%print_str)
        print_str = 'all: %d\nmin factor class: %s\nt_num all: %d'%(examples_num_all, min_class_factor_key, t_num_sum)
        print(print_str)
        if save_to_file:
            f2.write('%s\n' % print_str)
            f2.close()
            f1.write('%s\n' % print_str)
            f1.close()

        # Gather scan datasets meta data
        scan_datasets = {}
        scan_dataset_map = {}
        for scan_idx, scan_info in enumerate(self.tiles_dataset):
            scan_data = {
                'scan_id': scan_info['scan_uuid'],
                'pyramid_resolution': scan_info['scan_db_resolution'],
                'labels': [],
                'ROIs': []
            }
            scan_res_factor = 1. / scan_info['scan_resolution_factor']
            for region_idx, region_info in enumerate(scan_info['regions_list']):
                roi_model_rect = region_info['region_rect']
                roi_model_region = rect_to_region(roi_model_rect)
                roi_region = get_scaled_region(roi_model_region, scan_res_factor) # Pyramid resolution
                roi_data = roi_region + ['ROI idx %d'%region_info['ROI_id']]
                scan_data['ROIs'].append(roi_data)
                for tile_idx, tile_info in enumerate(region_info['tile_list']):
                    tile_key = (scan_idx, region_idx, tile_idx)
                    scan_dataset_map[tile_key] = scan_info['scan_uuid']
            scan_datasets[scan_info['scan_uuid']] = scan_data

        # Set scan datasets labels data
        est_target_examples_num = {}
        for examples_lists, list_class_factor, list_class_exact_factor in \
                [(examples_per_class_low, list_class_factor_low, list_class_exact_factor_low),
                 (examples_per_class_high, list_class_factor_high, list_class_exact_factor_high)]:
            for key in examples_lists.keys():
                example_exact_factor = list_class_exact_factor[key]
                example_expand_factor = list_class_factor[key]
                for tile_key, class_value_mapped, class_value in examples_lists[key]:
                    assert tile_key in scan_dataset_map
                    scan_uuid = scan_dataset_map[tile_key]
                    scan_idx, region_idx, tile_idx = tile_key
                    scan_info = self.tiles_dataset[scan_idx]
                    tile_info = scan_info['regions_list'][region_idx]['tile_list'][tile_idx]
                    scan_res_factor = 1./ scan_info['scan_resolution_factor']
                    tile_metadata = tile_info['tile_metadata']
                    class_name = str(tile_metadata['class'])
                    if class_name not in est_target_examples_num:
                        est_target_examples_num[class_name] = 0.
                    est_target_examples_num[class_name] += example_exact_factor
                    tile_model_region = [tile_info['tile_top_left_x'], tile_info['tile_top_left_y'],
                                         tile_info['tile_size'], tile_info['tile_size']]
                    tile_region = get_scaled_region(tile_model_region, scan_res_factor) # Tile ROI in pyramid resolution
                    label_data = tile_region + [ class_name,
                                                 class_value_mapped,            # Balancing rate class index
                                                 example_expand_factor          # Balancing expand factor
                                               ]
                    scan_datasets[scan_uuid]['labels'].append(label_data)

        est_total_examples_num = 0
        for class_name in est_target_examples_num.keys():
            est_total_examples_num += int(math.ceil(est_target_examples_num[class_name]))

        # Shuffle scan datasets labels data and save
        for scan_uuid in scan_datasets.keys():
            random.shuffle(scan_datasets[scan_uuid]['labels'])
            scan_dataset_file_name = self.config['out_dataset_path'] + '/%s.json'%scan_uuid
            write_json_file(scan_datasets[scan_uuid], scan_dataset_file_name)

        # Save balancing info
        balancing_info_file = self.config['out_dataset_info_path'] + '/balancing_info.json'
        balancing_info = {
            'rate_class_keys': self.balancing_classes_keys,
            'datasets_distribution': [],
            'target_distribution': [],
            'est_target_samples': est_target_examples_num,
            'est_target_samples_num': est_total_examples_num
        }
        for key in self.balancing_classes_keys:
            semi_balanced_dist_prob = semi_balanced_distribution.get(key, 0.)
            balancing_info['datasets_distribution'].append(semi_balanced_dist_prob)
            target_dist_prob = target_distribution.get(key, 0.)
            balancing_info['target_distribution'].append(target_dist_prob)
        write_json_file(balancing_info, balancing_info_file)

    def write_validation_scans_datasets(self):

        # Gather scan datasets meta data
        scan_datasets = {}
        for scan_idx, scan_info in enumerate(self.tiles_dataset):
            scan_data = {
                'scan_id': scan_info['scan_uuid'],
                'pyramid_resolution': scan_info['scan_db_resolution'],
                'labels': [],
                'ROIs': []
            }
            scan_res_factor = 1. / scan_info['scan_resolution_factor']
            for region_idx, region_info in enumerate(scan_info['regions_list']):
                roi_model_rect = region_info['scaled_rect']
                roi_model_region = rect_to_region(roi_model_rect)
                roi_region = get_scaled_region(roi_model_region, scan_res_factor) # Pyramid resolution
                roi_data = roi_region + ['ROI idx %d'%region_info['ROI_id']]
                scan_data['ROIs'].append(roi_data)
                for tile_idx, tile_info in enumerate(region_info['tile_list']):

                    tile_metadata = tile_info['tile_metadata']
                    class_name = str(tile_metadata['class'])
                    tile_model_region = [tile_info['tile_top_left_x'], tile_info['tile_top_left_y'],
                                         tile_info['tile_size'], tile_info['tile_size']]
                    tile_region = get_scaled_region(tile_model_region, scan_res_factor) # Tile ROI in pyramid resolution
                    label_data = tile_region + [ class_name,
                                                 0,            # Balancing rate class index - not used
                                                 1             # Balancing expand factor - not used
                                               ]
                    scan_data['labels'].append(label_data)
            scan_datasets[scan_info['scan_uuid']] = scan_data

        # Write scan datasets
        for scan_uuid in scan_datasets.keys():
            scan_dataset_file_name = self.config['out_dataset_path'] + '/%s.json'%scan_uuid
            write_json_file(scan_datasets[scan_uuid], scan_dataset_file_name)

if __name__ == '__main__':

    training = True

    # Train
    if training:
        tile_dataset_file = '/nfs/cvdataset/amir/tests/WBC_class/test_9/data_2_smudge_chk_B/training/tiles_dataset.json'
        balancing_info_file = '/nfs/cvdataset/amir/tests/WBC_class/test_9/data_2_smudge_chk_B/training/balancing_info.json'
        balancing_info =  read_json_file(balancing_info_file)
        config = {
            'tile_dataset_file': tile_dataset_file,
            'training': True,
            'out_dataset_path': '/nfs/cvdataset/amir/tests/WBC_class/test_11/in_data_2_chk/train',
            'out_dataset_info_path': '/nfs/cvdataset/amir/tests/WBC_class/test_11/in_data_2_chk/train/balancing_info',
            'base_path': '/nfs/cvdataset/amir/tests/WBC_class/test_11/data_2_in_chk',
            'target_name': 'smudge'
        }
        config.update(balancing_info)

    # Validation
    else:
        tile_dataset_file = '/nfs/cvdataset/amir/tests/WBC_class/test_9/data_2_smudge_chk_B/evaluation_conflicts/tiles_dataset.json'
        config = {
            'tile_dataset_file': tile_dataset_file,
            'training': False,
            'out_dataset_path': '/nfs/cvdataset/amir/tests/WBC_class/test_11/in_data_2_chk/valid',
            'base_path': '/nfs/cvdataset/amir/tests/WBC_class/test_11/data_2_in_chk',
            'target_name': 'smudge'
        }

    tiles_data_import = ImportTilesDataset(config)
    tiles_data_import.write_scans_datasets()








