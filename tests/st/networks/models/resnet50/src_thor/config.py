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
    "loss_scale": 128,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "epoch_size": 45,
    "buffer_size": 1000,
    "image_height": 224,
    "image_width": 224,
    "save_checkpoint": True,
    "save_checkpoint_steps": 5004,
    "keep_checkpoint_max": 20,
    "save_checkpoint_path": "./",
    "label_smooth": 1,
    "label_smooth_factor": 0.1,
    "frequency": 834,
    "eval_interval": 1,
    "eval_batch_size": 32
})
