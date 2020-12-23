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
"""Learning rate utilities."""
from math import ceil
import numpy as np

def polynomial_decay_scheduler(lr, min_lr, decay_steps, total_update_num, warmup_steps=1000, power=1.0):
    """
    Implements of polynomial decay learning rate scheduler which cycles by default.

    Args:
        lr (float): Initial learning rate.
        warmup_steps (int): Warmup steps.
        decay_steps (int): Decay steps.
        total_update_num (int): Total update steps.
        min_lr (float): Min learning.
        power (float): Power factor.

    Returns:
        np.ndarray, learning rate of each step.
    """
    lrs = np.zeros(shape=total_update_num, dtype=np.float32)

    if decay_steps <= 0:
        raise ValueError("`decay_steps` must larger than 1.")

    _start_step = 0
    if 0 < warmup_steps < total_update_num:
        warmup_end_lr = lr
        warmup_init_lr = 0 if warmup_steps > 0 else warmup_end_lr
        lrs[:warmup_steps] = np.linspace(warmup_init_lr, warmup_end_lr, warmup_steps)
        _start_step = warmup_steps

    decay_steps = decay_steps
    for step in range(_start_step, total_update_num):
        _step = step - _start_step
        ratio = ceil(_step / decay_steps)
        ratio = 1 if ratio < 1 else ratio
        _decay_steps = decay_steps * ratio
        lrs[step] = (lr - min_lr) * pow(1 - _step / _decay_steps, power) + min_lr

    return lrs
