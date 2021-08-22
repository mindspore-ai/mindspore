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
network config setting, will be used in train.py and eval.py
"""

from easydict import EasyDict as edict

config_dqn = edict({
    'gamma': 0.8,
    'epsi_high': 0.9,
    'epsi_low': 0.05,
    'decay': 200,
    'lr': 0.001,
    'capacity': 100000,
    'batch_size': 512,
    'state_space_dim': 4,
    'action_space_dim': 2
})
