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
    "class_num": 10572,
    "batch_size": 128,
    "learning_rate": 0.01,
    "lr_decay_epochs": [40, 80, 100],
    "lr_decay_factor": 0.1,
    "lr_warmup_epochs": 20,
    "p": 16,
    "k": 8,
    "loss_scale": 1024,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "epoch_size": 120,
    "buffer_size": 10000,
    "image_height": 128,
    "image_width": 128,
    "save_checkpoint": True,
    "save_checkpoint_steps": 195,
    "keep_checkpoint_max": 2,
    "save_checkpoint_path": "./"
})
