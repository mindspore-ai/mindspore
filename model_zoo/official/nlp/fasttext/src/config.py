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
#" :===========================================================================
"""
network config setting, will be used in train.py and eval.py
"""
from easydict import EasyDict as ed

config_yelpp = ed({
    'vocab_size': 6414979,
    'buckets': [64, 128, 256, 512, 2955],
    'test_buckets': [64, 128, 256, 512, 2955],
    'batch_size': 2048,
    'embedding_dims': 16,
    'num_class': 2,
    'epoch': 5,
    'lr': 0.30,
    'min_lr': 1e-6,
    'decay_steps': 549,
    'warmup_steps': 400000,
    'poly_lr_scheduler_power': 0.5,
    'epoch_count': 1,
    'pretrain_ckpt_dir': None,
    'save_ckpt_steps': 549,
    'keep_ckpt_max': 10,
})

config_db = ed({
    'vocab_size': 6596536,
    'buckets': [64, 128, 256, 512, 3013],
    'test_buckets': [64, 128, 256, 512, 1120],
    'batch_size': 4096,
    'embedding_dims': 16,
    'num_class': 14,
    'epoch': 5,
    'lr': 0.8,
    'min_lr': 1e-6,
    'decay_steps': 549,
    'warmup_steps': 400000,
    'poly_lr_scheduler_power': 0.5,
    'epoch_count': 1,
    'pretrain_ckpt_dir': None,
    'save_ckpt_steps': 548,
    'keep_ckpt_max': 10,
})

config_ag = ed({
    'vocab_size': 1383812,
    'buckets': [64, 128, 467],
    'test_buckets': [467],
    'batch_size': 512,
    'embedding_dims': 16,
    'num_class': 4,
    'epoch': 5,
    'lr': 0.2,
    'min_lr': 1e-6,
    'decay_steps': 115,
    'warmup_steps': 400000,
    'poly_lr_scheduler_power': 0.001,
    'epoch_count': 1,
    'pretrain_ckpt_dir': None,
    'save_ckpt_steps': 116,
    'keep_ckpt_max': 10,
})
