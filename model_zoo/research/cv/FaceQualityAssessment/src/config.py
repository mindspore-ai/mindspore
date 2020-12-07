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
"""Network config setting, will be used in train.py and eval.py"""
from easydict import EasyDict as edict

faceqa_1p_cfg = edict({
    'task': 'face_qa',

    # dataset related
    'per_batch_size': 256,

    # network structure related
    'steps_per_epoch': 0,
    'loss_scale': 1024,

    # optimizer related
    'lr': 0.02,
    'lr_scale': 1,
    'lr_epochs': '10, 20, 30',
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'max_epoch': 40,
    'warmup_epochs': 0,
    'pretrained': '',

    'local_rank': 0,
    'world_size': 1,

    # logging related
    'log_interval': 10,
    'ckpt_path': '../../output',
    'ckpt_interval': 500,

    'device_id': 0,
})

faceqa_8p_cfg = edict({
    'task': 'face_qa',

    # dataset related
    'per_batch_size': 32,

    # network structure related
    'steps_per_epoch': 0,
    'loss_scale': 1024,

    # optimizer related
    'lr': 0.02,
    'lr_scale': 1,
    'lr_epochs': '10, 20, 30',
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'max_epoch': 40,
    'warmup_epochs': 0,
    'pretrained': '',

    'local_rank': 0,
    'world_size': 8,

    # logging related
    'log_interval': 10,  # 10
    'ckpt_path': '../../output',
    'ckpt_interval': 500,
})
