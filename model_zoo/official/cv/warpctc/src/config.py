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
"""Network parameters."""
from easydict import EasyDict

config = EasyDict({
    "max_captcha_digits": 4,
    "captcha_width": 160,
    "captcha_height": 64,
    "batch_size": 64,
    "epoch_size": 30,
    "hidden_size": 512,
    "learning_rate": 0.01,
    "momentum": 0.9,
    "save_checkpoint": True,
    "save_checkpoint_steps": 97,
    "keep_checkpoint_max": 30,
    "save_checkpoint_path": "./checkpoint",
})
