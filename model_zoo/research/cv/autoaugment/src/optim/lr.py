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
"""Utilities for generating learning rates."""

import math

import numpy as np


def _generate_cosine_lr(lr_init, lr_max, total_steps, warmup_steps):
    """
    Create an array of learning rates conforming to the cosine decay mode.

    Args:
       lr_init (float): Starting learning rate.
       lr_max (float): Maximum learning rate.
       total_steps (int): Total number of train steps.
       warmup_steps (int): Number of warmup steps.

    Returns:
       np.array, a learning rate array.
    """
    decay_steps = total_steps - warmup_steps
    lr_each_step = []
    lr_inc = (float(lr_max) - float(lr_init)) / float(warmup_steps)
    for i in range(total_steps):
        if i < warmup_steps:
            lr = float(lr_init) + lr_inc * (i + 1)
        else:
            linear_decay = (total_steps - i) / decay_steps
            cosine_decay = 0.5 * \
                (1 + math.cos(math.pi * 2 * 0.47 * i / decay_steps))
            decayed = linear_decay * cosine_decay + 0.00001
            lr = lr_max * decayed
        lr_each_step.append(lr)
    return lr_each_step


def get_lr(
        lr_init, lr_max,
        warmup_epochs, total_epochs, steps_per_epoch, lr_decay_mode='cosine',
):
    """

    Args:
       lr_init (float): Starting learning rate.
       lr_max (float): Maximum learning rate.
       warmup_epochs (int): Number of warmup epochs.
       total_epochs (int): Total number of train epochs.
       steps_per_epoch (int): Steps per epoch.
       lr_decay_mode (string): Learning rate decay mode.

    Returns:
       np.array, a learning rate array.
    """
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * warmup_epochs

    if lr_decay_mode == 'cosine':
        lr_each_step = _generate_cosine_lr(
            lr_init, lr_max, total_steps, warmup_steps,
        )
    else:
        assert False, 'lr_decay_mode {} is not supported'.format(lr_decay_mode)

    lr_each_step = np.array(lr_each_step).astype(np.float32)
    return lr_each_step
