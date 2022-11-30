# Copyright 2017 ScopioLabs Ltd. All Rights Reserved.
#
#
# Title         : User Flags
# Project       : Training Core
#
# File          : User_flags.py
# Author        :
# Created       :
# =============================================================================
# Description :   Training core definitions and flags
# =============================================================================
# Copyright (c) This model is the confidential and
# proprietary property of ScopioLabs Ltd and the possession or use of this
# file requires a written license from ScopioLabs Ltd.
# =============================================================================
# Modification history :


# ==============================================================================

from core_flags import flags

def set_user_flags():
    # User Model Flags
    flags.DEFINE_integer("roi_margin", 32,
                                """Defines the membership distance from a label to an ROI(annotation)""")

    # To create a TFRecord tile image which is square - set a tile size and number of channels
    flags.DEFINE_integer("tfrecord_tile_size", 100,
                                """How many pixels are used for tile size.""")
    flags.DEFINE_integer("tf_channels", 3,
                                """How many channels are therein the TF image""")

    flags.DEFINE_string("tf_data_format", "channels_last",
                               """How is the TFRecord holding the images. Allowed values: channels_first (NCHW) channels_last (NHWC).""")

    # Define the image size that will enter the network after all pre-processing.
    flags.DEFINE_integer("training_image_height", 100,
                                """How many pixels are used for patch height.""")
    flags.DEFINE_integer("training_image_width", 100,
                                """How many pixels are used for patch width.""")
    flags.DEFINE_integer("training_image_depth", 3,
                                """How many pixels are used for patch depth.""")

    flags.DEFINE_string("data_format", "channels_last",
                               """If not set, the data format best for the training device is used. Allowed values: channels_first (NCHW) channels_last (NHWC).""")

    # Model flags from multi_gpu_training
    # -----------------------------------
    # Moved to core flags
    # flags.DEFINE_integer("num_gpus", 2,
    #                             """How many GPUs to use.""")

    # Flags related to run time
    flags.DEFINE_integer("number_of_epochs", 50,
                                """Number of Epochs to train""")
    flags.DEFINE_integer("train_batch_size", 10,
                                """Number of images to process in a batch. This batch will be divided into the number of GPUs""")
    flags.DEFINE_integer("eval_batch_size", 10,
                                """Number of images to process in a batch. This batch will be divided into the number of GPUs""")

    # Network model parameters
    # ------------------------
    flags.DEFINE_string("network_input_data_format", "channels_last",
                               """How is the network expecting to recieve images. Allowed values: channels_first (NCHW) channels_last (NHWC).""")

    flags.DEFINE_integer("num_classes", 10,
                                """Number of sub-class labels.""")

    # TODO -- interesting flags
    # flags.DEFINE_float("moving_average_decay", 0.9999,
    #                           """The decay to use for the moving average.""")
    #
    # flags.DEFINE_integer("num_epochs_per_decay", 350,
    #                             """Number of epochs after which learning rate decays.""")

    flags.DEFINE_float("weight_decay", 2e-4,
                              """Weight decay for regularization.""")

    # Optimizer Learning attributes
    # -----------------------------
    flags.DEFINE_string("optimizer_kind", "Momentum",
                               """Optimizer kind to use in training. May be any optimizer supported by TF""")

    flags.DEFINE_float("momentum_value", 0.0001,
                              """momentum_value.""")

    flags.DEFINE_float("initial_learning_rate", 0.0001,
                              """Initial learning rate.""")

    flags.DEFINE_float("learning_rate_decay_factor", 0.1,
                              """Learning rate decay factor.""")
