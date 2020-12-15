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
import numpy as np

import mindspore.nn.learning_rate_schedule as lr_schedules
from mindspore import Tensor
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype

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


class BertLearningRate(lr_schedules.LearningRateSchedule):
    """
    Implements of warmup-polydecay learning rate scheduler.

    Args:
        learning_rate (float): The initial value of learning rate.
        end_learning_rate (float): The end value of learning rate.
        warmup_steps (int): The warm up steps of learning rate.
        decay_steps (int): A value used to calculate decayed learning rate.
        power (float): A value used to calculate decayed learning rate.

    Returns:
        Tensor. The learning rate value for the current step.
    """
    def __init__(self, learning_rate, end_learning_rate, warmup_steps, decay_steps, power):
        super(BertLearningRate, self).__init__()
        self.warmup_lr = lr_schedules.WarmUpLR(learning_rate, warmup_steps)
        self.decay_lr = lr_schedules.PolynomialDecayLR(learning_rate, end_learning_rate, decay_steps, power)
        self.warmup_steps = Tensor(np.array([warmup_steps]).astype(np.float32))

        self.greater = P.Greater()
        self.one = Tensor(np.array([1.0]).astype(np.float32))
        self.cast = P.Cast()

    def construct(self, global_step):
        is_warmup = self.cast(self.greater(self.warmup_steps, global_step), mstype.float32)
        warmup_lr = self.warmup_lr(global_step)
        decay_lr = self.decay_lr(global_step)
        lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        return lr
