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

reid_1p_cfg = edict({
    'task': 'REID_1p',

    # dataset related
    'per_batch_size': 128,

    # network structure related
    'fp16': 1,
    'loss_scale': 2048.0,
    'input_size': (96, 64),
    'net_depth': 12,
    'embedding_size': 128,

    # optimizer related
    'lr': 0.1,
    'lr_scale': 1,
    'lr_gamma': 1,
    'lr_epochs': '30,60,120,150',
    'epoch_size': 30,
    'warmup_epochs': 0,
    'steps_per_epoch': 0,
    'max_epoch': 180,
    'weight_decay': 0.0005,
    'momentum': 0.9,

    # distributed parameter
    'is_distributed': 0,
    'local_rank': 0,
    'world_size': 1,

    # logging related
    'log_interval': 10,
    'ckpt_path': '../../output',
    'ckpt_interval': 200,
})


reid_8p_cfg = edict({
    'task': 'REID_8p',

    # dataset related
    'per_batch_size': 16,

    # network structure related
    'fp16': 1,
    'loss_scale': 2048.0,
    'input_size': (96, 64),
    'net_depth': 12,
    'embedding_size': 128,

    # optimizer related
    'lr': 0.8,  # 0.8
    'lr_scale': 1,
    'lr_gamma': 0.5,
    'lr_epochs': '30,60,120,150',
    'epoch_size': 30,
    'warmup_epochs': 0,
    'steps_per_epoch': 0,
    'max_epoch': 180,
    'weight_decay': 0.0005,
    'momentum': 0.9,

    # distributed parameter
    'is_distributed': 1,
    'local_rank': 0,
    'world_size': 8,

    # logging related
    'log_interval': 10,
    'ckpt_path': '../../output',
    'ckpt_interval': 200,
})
