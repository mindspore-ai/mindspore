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
"""
network config setting, will be used in train.py and eval.py
"""
from easydict import EasyDict as ed
# config optimizer for resnet50, imagenet2012. Momentum is default, Thor is optional.
cfg = ed({
    'optimizer': 'Momentum',
    })

config1 = ed({
    "class_num": 1000,
    "batch_size": 256,
    "loss_scale": 1024,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "epoch_size": 150,
    "pretrain_epoch_size": 0,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 5,
    "keep_checkpoint_max": 5,
    "decay_mode": "linear",
    "save_checkpoint_path": "./checkpoints",
    "hold_epochs": 0,
    "use_label_smooth": True,
    "label_smooth_factor": 0.1,
    "lr_init": 0.8,
    "lr_end": 0.0
})

config2 = ed({
    "class_num": 1000,
    "batch_size": 256,
    "loss_scale": 1024,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "epoch_size": 150,
    "pretrain_epoch_size": 0,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 5,
    "keep_checkpoint_max": 5,
    "decay_mode": "linear",
    "save_checkpoint_path": "./checkpoints",
    "hold_epochs": 0,
    "use_label_smooth": True,
    "label_smooth_factor": 0.1,
    "lr_init": 0.8,
    "lr_end": 0.0
})

config3 = ed({
    "class_num": 1000,
    "batch_size": 256,
    "loss_scale": 1024,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "epoch_size": 220,
    "pretrain_epoch_size": 0,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 5,
    "keep_checkpoint_max": 5,
    "decay_mode": "cosine",
    "save_checkpoint_path": "./checkpoints",
    "hold_epochs": 0,
    "use_label_smooth": True,
    "label_smooth_factor": 0.1,
    "lr_init": 0.8,
    "lr_end": 0.0
})
