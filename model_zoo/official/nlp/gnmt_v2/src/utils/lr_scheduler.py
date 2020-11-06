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
"""Learning scheduler."""
from math import ceil
import math
import numpy as np


def convert_float2int(values, total_steps):
    if isinstance(values, float):
        values = int(values * total_steps)
    return values


def square_root_schedule(lr, update_num, decay_start_step,
                         warmup_steps=2000,
                         min_lr=1e-5):
    """
    Decay the LR based on the ISR(inverse square root).

    During warm-up::
        lrs = np.linspace(0, lr, warmup_steps)

    After warm-up:
        decay_factor = lr * sqrt(warmup_steps)
        lr = decay_factor / sqrt(step) if step >= decay_start_step else lr

    Args:
        lr (float): Init learning rate.
        update_num (int): Total steps.
        decay_start_step (int): Decay begins after `decay_start_step` steps.
        warmup_steps (int): Warm up steps.
        min_lr (float): Min learning rate.

    Returns:
        np.ndarray, learning rate array.
    """
    warmup_end_lr = lr
    warmup_init_lr = 0 if warmup_steps > 0 else warmup_end_lr

    # If warmup_init_lr > lr, then lr_step is negative.
    # Otherwise, it's positive.
    lr_step = (warmup_end_lr - warmup_init_lr) / warmup_steps
    decay_factor = lr * warmup_steps ** 0.5

    lrs = np.empty(shape=update_num, dtype=np.float32)
    _start_step = 0
    if 0 < warmup_steps < update_num:
        lrs[:warmup_steps] = np.linspace(warmup_init_lr, warmup_end_lr, warmup_steps)
        _start_step = warmup_steps

    for step in range(_start_step, update_num):
        if step < warmup_steps:
            _lr = warmup_init_lr + step * lr_step
        elif step < decay_start_step:
            _lr = lr
        else:
            _lr = decay_factor * step ** -0.5
            if _lr < min_lr:
                _lr = min_lr
        lrs[step] = _lr

    return lrs


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
        _step = step - _start_step  # 2999
        ratio = ceil(_step / decay_steps)  # 3
        ratio = 1 if ratio < 1 else ratio
        _decay_steps = decay_steps * ratio  # 3000
        lrs[step] = (lr - min_lr) * pow(1 - _step / _decay_steps, power) + min_lr

    return lrs


def Warmup_MultiStepLR_scheduler(base_lr=0.002, total_update_num=200, warmup_steps=200, remain_steps=1.0,
                                 decay_interval=-1, decay_steps=4, decay_factor=0.5):
    """
    Implements of polynomial decay learning rate scheduler which cycles by default.

    Args:
        base_lr (float): Initial learning rate.
        total_update_num (int): Total update steps.
        warmup_steps (int or float): Warmup steps.
        remain_steps (int or float): start decay at 'remain_steps' iteration
        decay_interval (int): interval between LR decay steps
        decay_steps (int): Decay steps.
        decay_factor (float): decay factor

    Returns:
        np.ndarray, learning rate of each step.
    """

    if decay_steps <= 0:
        raise ValueError("`decay_steps` must larger than 1.")
    remain_steps = convert_float2int(remain_steps, total_update_num)
    warmup_steps = convert_float2int(warmup_steps, total_update_num)
    if warmup_steps > remain_steps:
        warmup_steps = remain_steps

    if decay_interval < 0:
        decay_iterations = total_update_num - remain_steps
        decay_interval = decay_iterations // decay_steps
        decay_interval = max(decay_interval, 1)
    else:
        decay_interval = convert_float2int(decay_interval, total_update_num)

    lrs = np.zeros(shape=total_update_num, dtype=np.float32)
    _start_step = 0
    for last_epoch in range(_start_step, total_update_num):
        if last_epoch < warmup_steps:
            if warmup_steps != 0:
                warmup_factor = math.exp(math.log(0.01) / warmup_steps)
            else:
                warmup_factor = 1.0
            inv_decay = warmup_factor ** (warmup_steps - last_epoch)
            lrs[last_epoch] = base_lr * inv_decay
        elif last_epoch >= remain_steps:
            decay_iter = last_epoch - remain_steps
            num_decay_step = decay_iter // decay_interval + 1
            num_decay_step = min(num_decay_step, decay_steps)
            lrs[last_epoch] = base_lr * (decay_factor ** num_decay_step)
        else:
            lrs[last_epoch] = base_lr
    return lrs
