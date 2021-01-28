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

"""learning rate generator"""
import numpy as np


def get_lr(lr_init, total_epochs, steps_per_epoch):
    """
    generate learning rate array

    Args:
       lr_init(float): init learning rate
       total_epochs(int): total epoch of training
       steps_per_epoch(int): steps of one epoch

    Returns:
       np.array, learning rate array
    """
    lr_each_step = []
    half_epoch = total_epochs // 2
    for i in range(total_epochs):
        for _ in range(steps_per_epoch):
            if i < half_epoch:
                lr_each_step.append(lr_init)
            else:
                lr_each_step.append(lr_init / (1.1 ** (i - half_epoch)))
    learning_rate = np.array(lr_each_step).astype(np.float32)
    return learning_rate
