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
network config setting, will be used in main.py
"""
from easydict import EasyDict as edict

cifar_cfg = edict({
    'num_classes': 10,
    'lr_init': 0.05,
    'batch_size': 64,
    'epoch_size': 70,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'buffer_size': 10,
    'image_height': 224,
    'image_width': 224,
    'keep_checkpoint_max': 10
})
