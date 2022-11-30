# Copyright 2017 ScopioLabs Ltd. All Rights Reserved.
#
#
# Title         : Core Flags
# Project       : Training Core
#
# File          : core_flags.py
# Author        : Shahar Karny
# Created       : 20/08/2017
# =============================================================================
# Description :   Training core definitions and flags
# =============================================================================
# Copyright (c) This model is the confidential and
# proprietary property of ScopioLabs Ltd and the possession or use of this
# file requires a written license from ScopioLabs Ltd.
# =============================================================================
# Modification history :


# ==============================================================================
import copy


class _FLAGS(object):
    class comments(object):
        pass

    @classmethod
    def _define(cls, key, value, comment=None, setdefault=False):
        if not (hasattr(cls, key) and setdefault):
            setattr(cls, key, value)
            setattr(cls.comments, key, comment)

    @classmethod
    def DEFINE_integer(cls, key, value, *args, **kwargs):
        assert isinstance(value, int)
        cls._define(key, value, *args, **kwargs)

    @classmethod
    def DEFINE_boolean(cls, key, value, *args, **kwargs):
        assert isinstance(value, bool)
        cls._define(key, value, *args, **kwargs)

    @classmethod
    def DEFINE_float(cls, key, value, *args, **kwargs):
        assert isinstance(value, float)
        cls._define(key, value, *args, **kwargs)

    @classmethod
    def DEFINE_string(cls, key, value, *args, **kwargs):
        assert isinstance(value, basestring)
        cls._define(key, value, *args, **kwargs)

    @classmethod
    def DEFINE_list(cls, key, value, *args, **kwargs):
        assert isinstance(value, list)
        cls._define(key, value, *args, **kwargs)


flags = _FLAGS()


def get_tf_flags():
    return flags


def set_core_flags():
    # Multi GPU attributes
    # Model flags from multi_gpu_training
    # -----------------------------------
    flags.DEFINE_integer("max_num_gpus", 1,
                         """limits how many GPUs to use, even if more are available.""")

    flags.DEFINE_integer("num_gpus", 1,
                         """How many GPUs to use. We be set by the infrastructure according to availability""")

    flags.DEFINE_string("tower_name", "tower",
                        """Generic name for tower running on a GPU""")

    flags.DEFINE_string("variable_strategy", "CPU",
                        """Where to locate variable operations. Only CPU/GPU are valid""")

    flags.DEFINE_float("GPU_memory_fraction", 0.95,
                       """The percent (0 - 1.0) of GPU memory to allocate for Tensorflow sessions""")

    flags.DEFINE_boolean("GPU_allow_growth", True,
                         """If True TF does not use all memory allocated for it in GPU_memory_fraction""")

    flags.DEFINE_boolean("use_augmentation", True,
                         """Decide if doing image augmentation""")
    flags.DEFINE_boolean("use_color_augmentation", True,
                         """Decide if doing image color augmentation""")

    flags.DEFINE_boolean("sync", False,
                         """If present when running in a distributed environment will run on sync mode.""")

    flags.DEFINE_integer("num_intra_threads", 0,
                         """Number of threads to use for intra-op parallelism. When training on CPU set to 0 to have the system pick the appropriate number or alternatively set it to the number of physical CPU cores""")

    flags.DEFINE_integer("num_inter_threads", 0,
                         """Number of threads to use for inter-op parallelism. If set to 0, the system will pick an appropriate number.""")

    flags.DEFINE_boolean("log_device_placement", False,
                         """Whether to log device placement.""")

    flags.DEFINE_boolean("use_fp16", False,
                         """Train the model using fp16.""")

    flags.DEFINE_integer("num_examples_per_epoch_for_train", 100,
                         """Number of examples processed per epoch while training.""")

    flags.DEFINE_integer("num_examples_per_epoch_for_eval", 100,
                         """Number of examples processed per epoch while evaluating.""")

    # Test flags are decided by the infrastructure
    flags.DEFINE_integer("num_examples_for_test", 100,
                         """Number of examples processed for inference run.""")

    flags.DEFINE_integer("max_test_batch_size", 10,
                         """Max number of images to process in a batch. This batch will be divided into the number of GPUs""")

    flags.DEFINE_integer("test_batch_size", 10,
                         """The number of images to process in a batch at specific run. Will be calculated by the core""")

    flags.DEFINE_integer("number_of_cpu_cores", 6,
                         """Number of CPU cores to enable parallelism""")

    flags.DEFINE_integer("evaluation_start_delay_secs", 3600,
                         """While training, defines how many seconds to wait between each evaluation """)

    flags.DEFINE_string("training_mode", 'train',
                        """This is the string that should be comapred to the mode variable set by the infrastructure to query about the running mode""")
    flags.DEFINE_string("evaluation_mode", 'eval',
                        """This is the string that should be comapred to the mode variable set by the infrastructure to query about the running mode""")
    flags.DEFINE_string("test_mode", 'test',
                        """This is the string that should be comapred to the mode variable set by the infrastructure to query about the running mode""")
    flags.DEFINE_string("predict_mode", 'predict',
                        """This is the string that should be comapred to the mode variable set by the infrastructure to query about the running mode""")

    # Config on how to divide the dataset into training, evaluation and test dataset (evaluation = 1 - train - test)
    flags.DEFINE_float("training_dataset_ratio", 0.7,
                       """The percentage of the dataset to be allocated to the training dataset""")

    flags.DEFINE_float("testing_dataset_ratio", 0.15,
                       """The percentage of the dataset to be allocated to the testing dataset""")

    flags.DEFINE_string("scan_train_test_split_level", 'LABEL',
                        """How to split train.test dataset. Valid options: SCAN, ROI, LABEL""")

    flags.DEFINE_boolean("simple_random_scans_split", False,
                          """simple random scans split""")

    flags.DEFINE_float("test_set_size", 10000.,
                       """The test set size in numbers""")

    flags.DEFINE_integer("scan_split_test_factor", 1,
                       """scan split test factor""")

    flags.DEFINE_integer("scan_split_iterations", 50000,
                       """scan split itertions""")

    flags.DEFINE_integer("min_roi_size", 12000,
                         """The minimum size we want per ROI""")

    flags.DEFINE_boolean("use_pyramid_buffer", False,
                         """ use pyramyid buffer for pyramid data read in case of tiles grid""")

    flags.DEFINE_string("scan_groups_type", 'scan_name_group_info',
                        """Type of scans distribution groups""")

    flags.DEFINE_list("scan_groups_name_signature",
                      ['ichilov-a', 'ichilov-b', 'ichilov-c', 'ichilov-d',
                       'ichilov-e', 'ichilov-f', 'ichilov-g', 'ichilov-h', 'ichilov-normal'],
                      """ scan groups name signatures """)

    flags.DEFINE_list("scan_groups_ratio",
                      [2, 1, 1 ,2, 1, 1 ,1, 1, 10],
                      """ scan groups ratios """)

    flags.DEFINE_list("database_labels_mapping",
                      [{'Consultation': 'unknown'},
                       {'WBC' : 'unknown'},
                       {'LB2': 'PLT'}

                       ], """Per use case mapping of db labels""")
    # else:
    #     # TBD - pass it per project::::
    #     flags.DEFINE_list("database_labels_mapping",
    #                       [{'BAND': 'NEU'},  # Default mapping of band Neu to Neu
    #                        {'MYLE': 'NEU'},  # Myelocytes to unknown
    #                        {'Promyelocyte': 'NEU'},
    #                        {'ME': 'NEU'},  # Metamyelocytes to unknown
    #                        {'BLAST': 'MON'},
    #                        {'AL': 'LYM'},  # atypical lymphocyte to LYM
    #                        {'LGL': 'LYM'},  # large granular lymphocyte to LYM
    #                        {'PLASMA': 'LYM'},
    #                        {'RFL': 'LYM'}, #NRBC
    #                        {'ABL': 'LYM'},  # Aberrant LYM to LYM
    #                        {'GCD' : 'dirt'}, # Smudge
    #                        {'Consultation': 'unknown'},
    #                        {'LB2': 'PLT'} # LB2 are PLT in clumps. Currently we take it as PLT.
    #
    #                        ], """Per use case mapping of db labels""")

    # flags.DEFINE_list("database_labels_mapping",
    #                   [{'BAND': 'BAND'},  # Default mapping of band Neu to Neu
    #                    {'MYLE': 'unknown'},  # Myelocytes to unknown
    #                    {'Promyelocyte': 'unknown'},
    #                    {'ME': 'unknown'},  # Metamyelocytes to unknown
    #                    {'BLAST': 'unknown'},
    #                    {'AL': 'LYM'},  # atypical lymphocyte to LYM
    #                    {'LGL': 'LYM'},  # large granular lymphocyte to LYM
    #                    {'PLASMA': 'unknown'},
    #                    {'ABL': 'LYM'},  # Aberrant LYM to LYM
    #                    {'Consultation': 'unknown'},
    #                    {'LB2': 'PLT'} # LB2 are PLT in clumps. Currently we take it as PLT.
    #
    #                     # RFL - NRBC
    #                     # GCD - Smudge Cells
    #
    #                    ], """Per use case mapping of db labels""")

def set_predict_flags():
    tf_flags = get_tf_flags()

    tf_flags.max_num_gpus = 1

    tf_flags.num_gpus = 1

    tf_flags.max_test_batch_size = tf_flags.eval_batch_size

    tf_flags.test_batch_size = tf_flags.max_test_batch_size

    tf_flags.GPU_memory_fraction = 0.95

    tf_flags.GPU_allow_growth = True
