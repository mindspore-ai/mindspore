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
"""
network config setting, will be used in main.py
"""
from easydict import EasyDict as edict


config_gpu = edict({
    'random_seed': 1,
    'rank': 0,
    'group_size': 1,
    'work_nums': 8,
    'decay_method': 'cosine',
    "loss_scale": 1,
    'batch_size': 128,
    'epoch_size': 250,
    'num_classes': 1000,
    'smooth_factor': 0.1,
    'aux_factor': 0.2,
    'lr_init': 0.00004,
    'lr_max': 0.4,
    'lr_end': 0.000004,
    'warmup_epochs': 1,
    'weight_decay': 0.00004,
    'momentum': 0.9,
    'opt_eps': 1.0,
    'keep_checkpoint_max': 100,
    'ckpt_path': './checkpoint/',
    'is_save_on_master': 0
})
