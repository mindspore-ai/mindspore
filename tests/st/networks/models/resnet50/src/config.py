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
network config setting, will be used in train.py and eval.py
"""
from easydict import EasyDict as ed

config = ed({
    "class_num": 1001,
    "batch_size": 32,
    "eval_interval": 1,
    "eval_batch_size": 50,
    "loss_scale": 1024,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "use_nesterov": True,
    "epoch_size": 90,
    "pretrained_epoch_size": 1,
    "buffer_size": 1000,
    "image_height": 224,
    "image_width": 224,
    "save_checkpoint": False,
    "save_checkpoint_epochs": 5,
    "keep_checkpoint_max": 10,
    "save_checkpoint_path": "./",
    "warmup_epochs": 0,
    "lr_decay_mode": "cosine",
    "use_label_smooth": True,
    "label_smooth_factor": 0.1,
    "lr_init": 0,
    "lr_max": 0.1,
    "use_lars": True,
    "lars_epsilon": 1e-8,
    "lars_coefficient": 0.001
})
