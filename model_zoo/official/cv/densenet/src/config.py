# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""config"""
from easydict import EasyDict as ed

#config for densenet100 and cifar10
config_100 = ed({
    "image_size": '32, 32',
    "num_classes": 10,

    "lr": 0.1,
    "lr_scheduler": 'exponential',
    "lr_epochs": '150, 225, 300',
    "lr_gamma": 0.1,
    "eta_min": 0,
    "T_max": 120,
    "max_epoch": 300,
    "per_batch_size": 64,
    "warmup_epochs": 0,

    "weight_decay": 0.0001,
    "momentum": 0.9,
    "is_dynamic_loss_scale": 0,
    "loss_scale": 1024,
    "label_smooth": 0,
    "label_smooth_factor": 0.1,

    "log_interval": 100,
    "ckpt_interval": 3124,
    "ckpt_path": 'outputs_cifar10/',
    "is_save_on_master": 1,

    "rank": 0,
    "group_size": 1
})

# config for densenet121 and imagenet
config_121 = ed({
    "image_size": '224,224',
    "num_classes": 1000,

    "lr": 0.1,
    "lr_scheduler": 'cosine_annealing',
    "lr_epochs": '30,60,90,120',
    "lr_gamma": 0.1,
    "eta_min": 0,
    "T_max": 120,
    "max_epoch": 120,
    "per_batch_size": 32,
    "warmup_epochs": 0,

    "weight_decay": 0.0001,
    "momentum": 0.9,
    "is_dynamic_loss_scale": 0,
    "loss_scale": 1024,
    "label_smooth": 0,
    "label_smooth_factor": 0.1,

    "log_interval": 100,
    "ckpt_interval": 50000,
    "ckpt_path": 'outputs_imagenet/',
    "is_save_on_master": 1,

    "rank": 0,
    "group_size": 1
})
