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

awa_cfg = edict({
    'lr_att': 1e-5,
    'wd_att': 1e-2,
    'clip_att': 0.2,

    'lr_word': 1e-4,
    'wd_word': 1e-3,
    'clip_word': 0.5,

    'lr_fusion': 1e-4,
    'wd_fusion': 1e-2,
    'clip_fusion': 0.5,

    'batch_size': 64,
})

cub_cfg = edict({
    'lr_att': 1e-5,
    'wd_att': 1e-2,
    'clip_att': 0.5,

    'batch_size': 100,
})
