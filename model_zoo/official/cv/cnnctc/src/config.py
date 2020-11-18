# Copyright 2020 Huawei Technologies Co., Ltd
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
"""network config setting, will be used in train.py and eval.py"""

class Config_CNNCTC():
    # model config
    CHARACTER = '0123456789abcdefghijklmnopqrstuvwxyz'
    NUM_CLASS = len(CHARACTER) + 1
    HIDDEN_SIZE = 512
    FINAL_FEATURE_WIDTH = 26

    # dataset config
    IMG_H = 32
    IMG_W = 100
    TRAIN_DATASET_PATH = 'CNNCTC_Data/ST_MJ/'
    TRAIN_DATASET_INDEX_PATH = 'CNNCTC_Data/st_mj_fixed_length_index_list.pkl'
    TRAIN_BATCH_SIZE = 192
    TEST_DATASET_PATH = 'CNNCTC_Data/IIIT5k_3000'
    TEST_BATCH_SIZE = 256
    TRAIN_EPOCHS = 3

    # training config
    CKPT_PATH = ''
    SAVE_PATH = './'
    LR = 1e-4
    LR_PARA = 5e-4
    MOMENTUM = 0.8
    LOSS_SCALE = 8096
    SAVE_CKPT_PER_N_STEP = 2000
    KEEP_CKPT_MAX_NUM = 5
