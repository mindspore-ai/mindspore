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
"""Network parameters."""

from easydict import EasyDict as edict

srcnn_cfg = edict({
    'lr': 1e-4,
    'patch_size': 33,
    'stride': 99,
    'scale': 2,
    'epoch_size': 20,
    'batch_size': 16,
    'save_checkpoint': True,
    'keep_checkpoint_max': 10,
    'save_checkpoint_path': 'outputs/'
})
