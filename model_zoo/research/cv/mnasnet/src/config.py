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

# config for mnas, imagenet2012.
config = ed({
    "class_num": 1000,
    "batch_size": 256,
    "loss_scale": 1024,
    "momentum": 0.9,
    "weight_decay": 1e-5,
    "epoch_size": 250,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 1,
    "keep_checkpoint_max": 5,
    "save_checkpoint_path": "./checkpoint",
    "opt": 'rmsprop',
    "opt_eps": 0.001,
    "warmup_epochs": 5,
    "lr_decay_mode": "other",
    "use_label_smooth": True,
    "label_smooth_factor": 0.1,
    "lr_init": 0.0001,
    "lr_max": 0.2,
    "lr_end": 0.00001
})
