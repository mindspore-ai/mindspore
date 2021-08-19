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

crowd_cfg = edict({
    'lr': 0.000028,# 0.00001 if device_num == 1； 0.00003 device_num=8
    'momentum': 0.0,
    'epoch_size': 800,
    'batch_size': 1,
    'buffer_size': 1000,
    'save_checkpoint_steps': 1,
    'keep_checkpoint_max': 10,
    'air_name': "mcnn",
})
