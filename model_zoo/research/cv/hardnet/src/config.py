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
network config setting, will be used in main.py
"""
from easydict import EasyDict

config = EasyDict({
    "class_num": 1000,
    "batch_size": 256,
    "loss_scale": 1024,
    "momentum": 0.9,
    "weight_decay": 6e-5,
    "epoch_size": 150,
    "pretrain_epoch_size": 0,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 10,
    "image_height": 224,
    "image_width": 224,
    "keep_checkpoint_max": 10,
    "save_checkpoint_path": '/home/hardnet/result/',
    "warmup_epochs": 5,
    "lr_decay_mode": "cosine",
    "lr_init": 0.05,
    "lr_end": 0.00001,
    "lr_max": 0.1
})
