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
"""learning rate generator"""
import math
import numpy as np


def _generate_steps_lr(lr_init, lr_max, total_steps, warmup_steps):
    """
    Applies three steps decay to generate learning rate array.

    Args:
       lr_init(float): init learning rate.
       lr_max(float): max learning rate.
       total_steps(int): all steps in training.
       warmup_steps(int): all steps in warmup epochs.

    Returns:
       np.array, learning rate array.
    """
    decay_epoch_index = [0.3 * total_steps, 0.6 * total_steps, 0.8 * total_steps]
    lr_each_step = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr = lr_init + (lr_max - lr_init) * i / warmup_steps
        else:
            if i < decay_epoch_index[0]:
                lr = lr_max
            elif i < decay_epoch_index[1]:
                lr = lr_max * 0.1
            elif i < decay_epoch_index[2]:
                lr = lr_max * 0.01
            else:
                lr = lr_max * 0.001
        lr_each_step.append(lr)
    return lr_each_step


def _generate_exponential_lr(lr_init, lr_max, total_steps, warmup_steps, steps_per_epoch):
    """
    Applies exponential decay to generate learning rate array.

    Args:
       lr_init(float): init learning rate.
       lr_max(float): max learning rate.
       total_steps(int): all steps in training.
       warmup_steps(int): all steps in warmup epochs.
       steps_per_epoch(int): steps of one epoch

    Returns:
       np.array, learning rate array.
    """
    lr_each_step = []
    if warmup_steps != 0:
        inc_each_step = (float(lr_max) - float(lr_init)) / float(warmup_steps)
    else:
        inc_each_step = 0
    for i in range(total_steps):
        if i < warmup_steps:
            lr = float(lr_init) + inc_each_step * float(i)
        else:
            decay_nums = math.floor((float(i - warmup_steps) / steps_per_epoch) / 2)
            decay_rate = pow(0.94, decay_nums)
            lr = float(lr_max) * decay_rate
            if lr < 0.0:
                lr = 0.0
        lr_each_step.append(lr)
    return lr_each_step


def _generate_cosine_lr(lr_init, lr_end, lr_max, total_steps, warmup_steps):
    """
    Applies cosine decay to generate learning rate array.

    Args:
       lr_init(float): init learning rate.
       lr_end(float): end learning rate
       lr_max(float): max learning rate.
       total_steps(int): all steps in training.
       warmup_steps(int): all steps in warmup epochs.

    Returns:
       np.array, learning rate array.
    """
    decay_steps = total_steps - warmup_steps
    lr_each_step = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr_inc = (float(lr_max) - float(lr_init)) / float(warmup_steps)
            lr = float(lr_init) + lr_inc * (i + 1)
        else:
            cosine_decay = 0.5 * (1 + math.cos(math.pi * (i-warmup_steps) / decay_steps))
            lr = (lr_max-lr_end)*cosine_decay + lr_end
        lr_each_step.append(lr)
    return lr_each_step


def _generate_liner_lr(lr_init, lr_end, lr_max, total_steps, warmup_steps):
    """
    Applies liner decay to generate learning rate array.

    Args:
       lr_init(float): init learning rate.
       lr_end(float): end learning rate
       lr_max(float): max learning rate.
       total_steps(int): all steps in training.
       warmup_steps(int): all steps in warmup epochs.

    Returns:
       np.array, learning rate array.
    """
    lr_each_step = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr = lr_init + (lr_max - lr_init) * i / warmup_steps
        else:
            lr = lr_max - (lr_max - lr_end) * (i - warmup_steps) / (total_steps - warmup_steps)
        lr_each_step.append(lr)
    return lr_each_step


def get_lr(lr_init, lr_end, lr_max, warmup_epochs, total_epochs, steps_per_epoch, lr_decay_mode):
    """
    generate learning rate array

    Args:
       lr_init(float): init learning rate
       lr_end(float): end learning rate
       lr_max(float): max learning rate
       warmup_epochs(int): number of warmup epochs
       total_epochs(int): total epoch of training
       steps_per_epoch(int): steps of one epoch
       lr_decay_mode(string): learning rate decay mode, including steps, steps_decay, cosine or liner(default)

    Returns:
       np.array, learning rate array
    """
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    if lr_decay_mode == 'steps':
        lr_each_step = _generate_steps_lr(lr_init, lr_max, total_steps, warmup_steps)
    elif lr_decay_mode == 'steps_decay':
        lr_each_step = _generate_exponential_lr(lr_init, lr_max, total_steps, warmup_steps, steps_per_epoch)
    elif lr_decay_mode == 'cosine':
        lr_each_step = _generate_cosine_lr(lr_init, lr_end, lr_max, total_steps, warmup_steps)
    else:
        lr_each_step = _generate_liner_lr(lr_init, lr_end, lr_max, total_steps, warmup_steps)
    learning_rate = np.array(lr_each_step).astype(np.float32)
    return learning_rate
