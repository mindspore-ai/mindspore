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
import math
import numpy as np


def _generate_linear_lr(lr_init, lr_end, total_steps):
    """
    Applies liner decay to generate learning rate array.

    Args:
       lr_init(float): init learning rate.
       lr_end(float): end learning rate
       total_steps(int): all steps in training.

    Returns:
       np.array, learning rate array.
    """
    lr_each_step = []
    for i in range(total_steps):
        lr = lr_init - (lr_init - lr_end) * (i) / (total_steps)
        lr_each_step.append(lr)

    return lr_each_step

def _generate_cosine_lr(lr_init, total_steps):
    """
    Applies cosine decay to generate learning rate array.

    Args:
       lr_init(float): init learning rate.
       lr_end(float): end learning rate
       total_steps(int): all steps in training.
       warmup_steps(int): all steps in warmup epochs.

    Returns:
       np.array, learning rate array.
    """
    decay_steps = total_steps
    lr_each_step = []
    for i in range(total_steps):
        linear_decay = (total_steps - i) / decay_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * 2 * 0.47 * i / decay_steps))
        decayed = linear_decay * cosine_decay + 0.00001
        lr = lr_init * decayed
        lr_each_step.append(lr)
    return lr_each_step

def get_lr(lr_init, lr_end, total_epochs, steps_per_epoch, decay_mode):
    """
    generate learning rate array

    Args:
       lr_init(float): init learning rate
       lr_end(float): end learning rate
       total_epochs(int): total epoch of training
       steps_per_epoch(int): steps of one epoch

    Returns:
       np.array, learning rate array
    """

    total_steps = steps_per_epoch * total_epochs
    if decay_mode == "cosine":
        lr_each_step = _generate_cosine_lr(lr_init, total_steps)
    else:
        lr_each_step = _generate_linear_lr(lr_init, lr_end, total_steps)

    lr_each_step = np.array(lr_each_step).astype(np.float32)

    return lr_each_step
