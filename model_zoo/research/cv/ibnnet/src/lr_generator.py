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
python lr_generator.py
"""
import numpy as np
from mindspore import Tensor
from src.config import cfg


def lr_generator(lr_init, total_epochs, steps_per_epoch):
    lr_each_step = []
    for i in range(total_epochs):
        if i in cfg.schedule:
            lr_init *= cfg.gamma
        for _ in range(steps_per_epoch):
            lr_each_step.append(lr_init)
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    return Tensor(lr_each_step)
