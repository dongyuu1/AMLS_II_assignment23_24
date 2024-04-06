# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

"""Configs."""
from fvcore.common.config import CfgNode

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

_C.LR_TRAIN_PATH = "./data/DIV2K_train_LR_bicubic/X4"

_C.LR_VAL_PATH = "./data/DIV2K_valid_LR_bicubic/X4"

_C.HR_TRAIN_PATH = "./data/DIV2K_train_HR"

_C.HR_VAL_PATH = "./data/DIV2K_valid_HR"

_C.TEST_PATH = "./data/benchmark"

_C.SAVE_DIR = "./"

_C.DEVICE = "cuda:0"

_C.SCALE = 4

_C.PATCH_SIZE = 256

_C.LOG_INTERVAL = 5

_C.RAND_SEED = 0

# -----------------------------------------------------------------------------
# Configs for training model
# -----------------------------------------------------------------------------
_C.TRAIN = CfgNode()

_C.TRAIN.BATCH = 32

_C.TRAIN.MAX_ITER = 250000

_C.TRAIN.T_MAX = 250000

_C.TRAIN.LR = 2e-3

_C.TRAIN.ETA_MIN = 1e-7

# -----------------------------------------------------------------------------
# Configs for testing model
# -----------------------------------------------------------------------------

_C.TEST = CfgNode()

_C.TEST.BATCH = 32

# -----------------------------------------------------------------------------
# Configs for model architecture
# -----------------------------------------------------------------------------

_C.MODEL = CfgNode()

_C.MODEL.CIN = 3

_C.MODEL.CMID = 48

_C.MODEL.CUP = 24

_C.MODEL.COUT = 3

_C.MODEL.N_BLOCK = 16


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()
