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
    'work_nums': 8,
    'decay_method': 'cosine',
    "loss_scale": 1,
    'batch_size': 128,
    'epoch_size': 250,
    'num_classes': 1000,
    'ds_type': 'imagenet',
    'ds_sink_mode': True,
    'smooth_factor': 0.1,
    'aux_factor': 0.2,
    'lr_init': 0.00004,
    'lr_max': 0.4,
    'lr_end': 0.000004,
    'warmup_epochs': 1,
    'weight_decay': 0.00004,
    'momentum': 0.9,
    'opt_eps': 1.0,
    'keep_checkpoint_max': 10,
    'ckpt_path': './',
    'is_save_on_master': 0,
    'dropout_keep_prob': 0.5,
    'has_bias': True,
    'amp_level': 'O0'
})

config_ascend = edict({
    'random_seed': 1,
    'work_nums': 8,
    'decay_method': 'cosine',
    "loss_scale": 1024,
    'batch_size': 128,
    'epoch_size': 250,
    'num_classes': 1000,
    'ds_type': 'imagenet',
    'ds_sink_mode': True,
    'smooth_factor': 0.1,
    'aux_factor': 0.2,
    'lr_init': 0.00004,
    'lr_max': 0.4,
    'lr_end': 0.000004,
    'warmup_epochs': 1,
    'weight_decay': 0.00004,
    'momentum': 0.9,
    'opt_eps': 1.0,
    'keep_checkpoint_max': 10,
    'ckpt_path': './',
    'is_save_on_master': 0,
    'dropout_keep_prob': 0.8,
    'has_bias': False,
    'amp_level': 'O3'
})

config_cpu = edict({
    'random_seed': 1,
    'work_nums': 8,
    'decay_method': 'cosine',
    "loss_scale": 1024,
    'batch_size': 128,
    'epoch_size': 120,
    'num_classes': 10,
    'ds_type': 'cifar10',
    'ds_sink_mode': False,
    'smooth_factor': 0.1,
    'aux_factor': 0.2,
    'lr_init': 0.00004,
    'lr_max': 0.1,
    'lr_end': 0.000004,
    'warmup_epochs': 1,
    'weight_decay': 0.00004,
    'momentum': 0.9,
    'opt_eps': 1.0,
    'keep_checkpoint_max': 10,
    'ckpt_path': './',
    'is_save_on_master': 0,
    'dropout_keep_prob': 0.8,
    'has_bias': False,
    'amp_level': 'O0',
})
