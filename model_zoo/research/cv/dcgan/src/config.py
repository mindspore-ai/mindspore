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
network config setting, will be used in train.py
"""

from easydict import EasyDict as edict

dcgan_imagenet_cfg = edict({
    'num_classes': 1000,
    'epoch_size': 20,
    'batch_size': 128,
    'latent_size': 100,
    'feature_size': 64,
    'channel_size': 3,
    'image_height': 32,
    'image_width': 32,
    'learning_rate': 0.0002,
    'beta1': 0.5
})
