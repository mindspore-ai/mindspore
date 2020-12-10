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
config_ascend_quant = ed({
    "num_classes": 1000,
    "image_height": 224,
    "image_width": 224,
    "batch_size": 192,
    "data_load_mode": "mindata",
    "epoch_size": 60,
    "start_epoch": 200,
    "warmup_epochs": 0,
    "lr": 0.3,
    "momentum": 0.9,
    "weight_decay": 4e-5,
    "label_smooth": 0.1,
    "loss_scale": 1024,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 1,
    "keep_checkpoint_max": 300,
    "save_checkpoint_path": "./checkpoint",
})

config_gpu_quant = ed({
    "num_classes": 1000,
    "image_height": 224,
    "image_width": 224,
    "batch_size": 300,
    "epoch_size": 60,
    "start_epoch": 200,
    "warmup_epochs": 0,
    "lr": 0.05,
    "momentum": 0.997,
    "weight_decay": 4e-5,
    "label_smooth": 0.1,
    "loss_scale": 1024,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 1,
    "keep_checkpoint_max": 300,
    "save_checkpoint_path": "./checkpoint",
})

def config_quant(device_target):
    if device_target not in ["Ascend", "GPU"]:
        raise ValueError("Unsupported device target: {}.".format(device_target))
    configs = ed({
        "Ascend": config_ascend_quant,
        "GPU": config_gpu_quant
    })
    config = configs.Ascend if device_target == "Ascend" else configs.GPU
    config["device_target"] = device_target
    return config
