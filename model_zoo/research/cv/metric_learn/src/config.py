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
config.
"""
from easydict import EasyDict as ed

#sop softmax
config0 = ed({
    "class_num": 5184,
    "batch_size": 80,
    "loss_scale": 1024,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "epoch_size": 30,
    "pretrain_epoch_size": 0,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 10,
    "keep_checkpoint_max": 5,
    "save_checkpoint_path": "./softmax/",
    "warmup_epochs": 0,
    "lr_decay_mode": "steps",
    "lr_end": 0.01,
    "lr_init": 0.0001,
    "lr_max": 0.3
})

#sop trpletloss
config1 = ed({
    "class_num": 5184,
    "batch_size": 60,
    "loss_scale": 1024,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "epoch_size": 30,
    "pretrain_epoch_size": 0,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 10,
    "keep_checkpoint_max": 1,
    "save_checkpoint_path": "./triplet/",
    "warmup_epochs": 0,
    "lr_decay_mode": "const",
    "lr_end": 0.01,
    "lr_init": 0.0001,
    "lr_max": 0.0001
})

#sop quadrupletloss
config2 = ed({
    "class_num": 5184,
    "batch_size": 60,
    "loss_scale": 1024,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "epoch_size": 30,
    "pretrain_epoch_size": 0,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 10,
    "keep_checkpoint_max": 1,
    "save_checkpoint_path": "./quadruplet/",
    "warmup_epochs": 0,
    "lr_decay_mode": "const",
    "lr_end": 0.01,
    "lr_init": 0.0001,
    "lr_max": 0.0001
})
