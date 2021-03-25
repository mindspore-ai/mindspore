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
"""config for train or evaluation"""
from easydict import EasyDict

config = EasyDict({
    "class_num": 1000,
    "batch_size": 128,
    "loss_scale": 1024,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "epoch_size": 120,
    "pretrained": False,
    "pretrain_epoch_size": 0,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 5,
    "keep_checkpoint_max": 5,
    "save_checkpoint_path": "./",
    "warmup_epochs": 5,
    "lr_decay_mode": "poly",
    "use_label_smooth": True,
    "use_autoaugment": True,
    "label_smooth_factor": 0.1,
    "weight_init": "xavier_uniform",
    "lr_init": 0,
    "lr_max": 0.6,
    "lr_end": 0.0
})
