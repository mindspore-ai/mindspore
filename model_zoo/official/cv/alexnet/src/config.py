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
network config setting, will be used in train.py
"""

from easydict import EasyDict as edict

alexnet_cifar10_cfg = edict({
    'num_classes': 10,
    'learning_rate': 0.002,
    'momentum': 0.9,
    'epoch_size': 30,
    'batch_size': 32,
    'buffer_size': 1000,
    'image_height': 227,
    'image_width': 227,
    'save_checkpoint_steps': 1562,
    'keep_checkpoint_max': 10,
    'air_name': "alexnet.air",
})

alexnet_imagenet_cfg = edict({
    'num_classes': 1000,
    'learning_rate': 0.13,
    'momentum': 0.9,
    'epoch_size': 150,
    'batch_size': 256,
    'buffer_size': None, # invalid parameter
    'image_height': 224,
    'image_width': 224,
    'save_checkpoint_steps': 625,
    'keep_checkpoint_max': 10,
    'air_name': "alexnet.air",

    # opt
    'weight_decay': 0.0001,
    'loss_scale': 1024,

    # lr
    'is_dynamic_loss_scale': 0,
})
