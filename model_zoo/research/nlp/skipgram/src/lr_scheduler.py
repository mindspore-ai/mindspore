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
"""Learning rate utilities."""

import numpy as np


def exp_decay_lr(learning_rate, decay_rate, decay_step, total_step, is_stair=False):
    """lr[i] = learning_rate∗pow(decay_rate, i / decay_step)
    """
    if is_stair:
        lrs = learning_rate * np.power(decay_rate, np.floor(np.arange(total_step) / decay_step))
    else:
        lrs = learning_rate * np.power(decay_rate, np.arange(total_step) / decay_step)
    return lrs.astype(np.float32)


def poly_decay_lr(learning_rate, end_learning_rate, decay_step, total_step, power, update_decay_step=False):
    """polynomial decay learning rate
    """
    lrs = []
    if update_decay_step:
        for step in range(total_step):
            tmp_decay_step = max(decay_step, decay_step * np.ceil(step / decay_step))
            lrs.append((learning_rate - end_learning_rate) * np.power(1 - step / tmp_decay_step, power)
                       + end_learning_rate)
    else:
        for step in range(total_step):
            step = min(step, decay_step)
            lrs.append((learning_rate - end_learning_rate) * np.power(1 - step / decay_step, power) + end_learning_rate)
    lrs = np.array(lrs)
    return lrs.astype(np.float32)
