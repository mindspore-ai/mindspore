# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Hyper-parameters."""

import numpy as np
from easydict import EasyDict
from utils.serialization import yaml_load

cfg = EasyDict()

# COMMON CONFIGS
# source domain
cfg.SOURCE = 'GTA'

# target domain
cfg.TARGET = 'Cityscapes'

# Number of workers for dataloading
cfg.NUM_WORKERS = 4

# List of training images
cfg.DATA_LIST_SOURCE = str('dataset/gta5_list/{}.txt')
cfg.DATA_LIST_TARGET = str('dataset/cityscapes_list/{}.txt')
cfg.PSEUDO_LIST = str('dataset/cityscapes_list/{}.txt')

# Directories
cfg.DATA_DIRECTORY_SOURCE = str('/cache/datasets/domain_adaptation/GTAv')
cfg.DATA_DIRECTORY_TARGET = str('/cache/datasets/domain_adaptation/cityscapes')
cfg.DATA_DIRECTORY_PSEUDO = str('/cache/datasets/domain_adaptation/cityscapes')
cfg.DATA_REMOTE_DIRECTORY_SOURCE = str('chensj/datasets/domain_adaptation/GTAv')
cfg.DATA_REMOTE_DIRECTORY_TARGET = str('chensj/datasets/domain_adaptation/cityscapes')

# Number of object classes
cfg.NUM_CLASSES = 19

# Exp dirs
#cfg.EXP_NAME = ''
cfg.EXP_ROOT = './experiments/test'
cfg.EXP_REMOTE_ROOT = 'chensj/experiments/domain_adaptation/advent/experimet_8p'

#cfg.EXP_ROOT_SNAPSHOT = osp.join(cfg.EXP_ROOT, 'snapshots')
#cfg.EXP_ROOT_LOGS = osp.join(cfg.EXP_ROOT, 'logs')
# CUDA
cfg.GPU_ID = '0,1,2,3,4,5,6,7'

# TRAIN CONFIGS
cfg.TRAIN = EasyDict()
cfg.TRAIN.SET_SOURCE = 'all'
cfg.TRAIN.SET_TARGET = 'train'
cfg.TRAIN.BATCH_SIZE_SOURCE = 1
cfg.TRAIN.BATCH_SIZE_TARGET = 1
cfg.TRAIN.IGNORE_LABEL = 255
cfg.TRAIN.INPUT_SIZE_SOURCE = (1280, 720)
cfg.TRAIN.INPUT_SIZE_TARGET = (1024, 512)

# Class info
cfg.TRAIN.INFO_SOURCE = ''
cfg.TRAIN.INFO_TARGET = str('dataset/cityscapes_list/info.json')

# Segmentation network params
cfg.TRAIN.MODEL = 'DeepLabv2'
cfg.TRAIN.MULTI_LEVEL = True
cfg.TRAIN.RESTORE_FROM = ''
cfg.TRAIN.REMOTE_RESTORE_FROM = ''
cfg.TRAIN.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
cfg.TRAIN.LEARNING_RATE = 2.5e-4
cfg.TRAIN.MOMENTUM = 0.9
cfg.TRAIN.WEIGHT_DECAY = 0.0005
cfg.TRAIN.POWER = 0.9
cfg.TRAIN.LAMBDA_SEG_MAIN = 1.0
cfg.TRAIN.LAMBDA_SEG_AUX = 0.1  # weight of conv4 prediction. Used in multi-level setting.

# BN settings
cfg.TRAIN.FREEZE_BN = False
cfg.TRAIN.FREEZE_BN_AFFINE = False

# Domain adaptation
cfg.TRAIN.DA_METHOD = 'AdvEnt'

# Adversarial training params
cfg.TRAIN.GAN_MODE = 'vanilla'
cfg.TRAIN.LEARNING_RATE_D = 1e-4
cfg.TRAIN.LAMBDA_ADV_MAIN = 0.001
cfg.TRAIN.LAMBDA_ADV_AUX = 0.0002

# MinEnt params
cfg.TRAIN.LAMBDA_ENT_MAIN = 0.001
cfg.TRAIN.LAMBDA_ENT_AUX = 0.0002

# Semi supervised learning params
cfg.TRAIN.USE_SEMI = False
cfg.TRAIN.NUM_SEMI = 100
cfg.TRAIN.DEL_XL = False

# loss weight of self KL Loss
cfg.TRAIN.self_KL = False
cfg.TRAIN.LAMBDA_SELF_KL = 1
cfg.TRAIN.Tau = 0.01

# setting for KD
cfg.TRAIN.KD = False
cfg.TRAIN.OnLine_KD = False
cfg.TRAIN.REMOTE_KD_RESTORE_FROM_2 = ''
cfg.TRAIN.KD_RESTORE_FROM_2 = ''
cfg.TRAIN.REMOTE_KD_RESTORE_FROM = ''
cfg.TRAIN.KD_RESTORE_FROM = ''
cfg.TRAIN.LAMBDA_KL = 0.5
cfg.TRAIN.KL_T = 10

# Other params
cfg.TRAIN.PRINT_FREQ = 100
cfg.TRAIN.MAX_ITERS = 250000
cfg.TRAIN.EARLY_STOP = 120000
cfg.TRAIN.SAVE_PRED_EVERY = 2000
cfg.TRAIN.SOURCE_TRANS = False
cfg.TRAIN.SNAPSHOT_DIR = ''
cfg.TRAIN.RANDOM_SEED = 1234
cfg.TRAIN.TENSORBOARD_LOGDIR = ''
cfg.TRAIN.TENSORBOARD_VIZRATE = 100

# TEST CONFIGS
cfg.TEST = EasyDict()
cfg.TEST.DATA = 'Cityscapes'
cfg.TEST.MODE = 'single'  # {'single', 'best'}

# model
cfg.TEST.MODEL = ('DeepLabv2',)
cfg.TEST.MODEL_WEIGHT = (1.0,)
cfg.TEST.MULTI_LEVEL = (True,)
cfg.TEST.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
cfg.TEST.RESTORE_FROM = ('',)
cfg.TEST.SNAPSHOT_DIR = ('',)  # used in 'best' mode
cfg.TEST.SNAPSHOT_STEP = 2000  # used in 'best' mode
cfg.TEST.SNAPSHOT_MAXITER = 120000  # used in 'best' mode

# Test sets
cfg.TEST.SET = 'val'
cfg.TEST.BATCH_SIZE = 1
cfg.TEST.INPUT_SIZE = (1024, 512)
cfg.TEST.OUTPUT_SIZE = (2048, 1024)
cfg.TEST.DATA_DIRECTORY = str('/home/wangcong/hujingsong/deeplabv2/dataset/data/cityscapes')
cfg.TEST.DATA_LIST = str('dataset/cityscapes_list/{}.txt')
cfg.TEST.INFO = str('dataset/cityscapes_list/info.json')
cfg.TEST.WAIT_MODEL = True


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    #if type(a) is not EasyDict:
    if not isinstance(a, EasyDict):
        return

    for k, v in a.items():
        # a must specify keys that are in b
        # if not b.has_key(k):
        if k not in b:
            raise KeyError(f'{k} is not a valid config key')

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(f'Type mismatch ({type(b[k])} vs. {type(v)}) '
                                 f'for config key: {k}')

        # recursively merge dicts
        if isinstance(v, EasyDict):
            try:
                _merge_a_into_b(a[k], b[k])
            except Exception:
                print(f'Error under config key: {k}')
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options.
    """
    yaml_cfg = EasyDict(yaml_load(filename))
    _merge_a_into_b(yaml_cfg, cfg)
