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

config = ed({
    "num_classes": 1000,
    "batch_size": 128,
    "epoch_size": 500,
    "warmup_epochs": 20,
    "lr_init": 0.1,
    "lr_max": 0.4,
    'lr_end': 1e-6,
    'lr_decay_mode': 'cosine',
    "momentum": 0.9,
    "weight_decay": 4e-5,
    "label_smooth": 0.1,
    "loss_scale": 128,
    "use_label_smooth": True,
    "label_smooth_factor": 0.1,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 20,
    "keep_checkpoint_max": 10,
    "save_checkpoint_path": "./",
})
