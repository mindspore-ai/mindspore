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
from easydict import EasyDict as edict

# config for vgg16, cifar10
cifar_cfg = edict({
    "num_classes": 10,
    "lr": 0.01,
    "lr_init": 0.01,
    "lr_max": 0.1,
    "lr_epochs": '30,60,90,120',
    "lr_scheduler": "step",
    "warmup_epochs": 5,
    "batch_size": 64,
    "max_epoch": 70,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "loss_scale": 1.0,
    "label_smooth": 0,
    "label_smooth_factor": 0,
    "buffer_size": 10,
    "image_size": '224,224',
    "pad_mode": 'same',
    "padding": 0,
    "has_bias": False,
    "batch_norm": True,
    "keep_checkpoint_max": 10,
    "initialize_mode": "XavierUniform",
    "has_dropout": False
})

# config for vgg16, imagenet2012
imagenet_cfg = edict({
    "num_classes": 1000,
    "lr": 0.01,
    "lr_init": 0.01,
    "lr_max": 0.1,
    "lr_epochs": '30,60,90,120',
    "lr_scheduler": 'cosine_annealing',
    "warmup_epochs": 0,
    "batch_size": 32,
    "max_epoch": 150,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "loss_scale": 1024,
    "label_smooth": 1,
    "label_smooth_factor": 0.1,
    "buffer_size": 10,
    "image_size": '224,224',
    "pad_mode": 'pad',
    "padding": 1,
    "has_bias": True,
    "batch_norm": False,
    "keep_checkpoint_max": 10,
    "initialize_mode": "KaimingNormal",
    "has_dropout": True
})
