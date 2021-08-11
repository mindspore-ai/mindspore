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
"""generate learning rate"""
import numpy as np


def get_lr(total_epochs,
           steps_per_epoch,
           lr_init
           ):
    """
    generate learning rate
    """
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    for i in range(int(total_steps)):
        if i <= int(60 * steps_per_epoch):
            lr = lr_init
        elif i <= int(120 * steps_per_epoch):
            lr = lr_init * 0.1 + 0.01
        elif i <= int(160 * steps_per_epoch):
            lr = lr_init * 0.1 * 0.1 + 0.003
        elif i <= int(200 * steps_per_epoch):
            lr = 0.001
        elif i <= int(240 * steps_per_epoch):
            lr = 0.0008
        elif i <= int(260 * steps_per_epoch):
            lr = 0.0006
        lr_each_step.append(lr)

    lr_each_step = np.array(lr_each_step).astype(np.float32)

    return lr_each_step
