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

config_gpu = ed({
    "num_classes": 1000,
    "image_height": 224,
    "image_width": 224,
    "batch_size": 150,
    "epoch_size": 370,
    "warmup_epochs": 4,
    "lr": 1.54,
    "momentum": 0.9,
    "weight_decay": 4e-5,
    "label_smooth": 0.1,
    "loss_scale": 1024,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 1,
    "keep_checkpoint_max": 500,
    "save_checkpoint_path": "./checkpoint",
    "export_format": "MINDIR",
    "export_file": "mobilenetv3"
})

config_cpu = ed({
    "num_classes": 10,
    "image_height": 224,
    "image_width": 224,
    "batch_size": 32,
    "epoch_size": 120,
    "warmup_epochs": 5,
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "label_smooth": 0.1,
    "loss_scale": 1024,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 1,
    "keep_checkpoint_max": 500,
    "save_checkpoint_path": "./checkpoint",
    "export_format": "MINDIR",
    "export_file": "mobilenetv3"
})
