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


def get_lr(lr_init, lr_end, lr_max, total_epochs, warmup_epochs,
           pretrain_epochs, steps_per_epoch, lr_decay_mode):
    """
    generate learning rate array

    Args:
        lr_init(float): init learning rate
        lr_end(float): end learning rate
        lr_max(float): max learning rate
        total_epochs(int): total epoch of training
        warmup_epochs(int): number of warmup epochs
        pretrain_epochs(int): number of pretrain epochs
        steps_per_epoch(int): steps of one epoch
        lr_decay_mode(string): learning rate decay mode,
                               including steps, poly, linear or cosine

    Returns:
        np.array, learning rate array
    """

    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    pretrain_steps = steps_per_epoch * pretrain_epochs
    decay_steps = total_steps - warmup_steps

    if lr_decay_mode == 'steps':
        decay_epoch_index = [
            0.3 * total_steps, 0.6 * total_steps, 0.8 * total_steps
        ]
        for i in range(total_steps):
            if i < decay_epoch_index[0]:
                lr = lr_max
            elif i < decay_epoch_index[1]:
                lr = lr_max * 0.1
            elif i < decay_epoch_index[2]:
                lr = lr_max * 0.01
            else:
                lr = lr_max * 0.001
            lr_each_step.append(lr)

    elif lr_decay_mode == 'poly':
        for i in range(total_steps):
            if i < warmup_steps:
                lr = linear_warmup_lr(i, warmup_steps, lr_max, lr_init)
            else:
                base = (1.0 - (i - warmup_steps) / decay_steps)
                lr = lr_max * base * base
            lr_each_step.append(lr)

    elif lr_decay_mode == 'linear':
        for i in range(total_steps):
            if i < warmup_steps:
                lr = linear_warmup_lr(i, warmup_steps, lr_max, lr_init)
            else:
                lr = lr_max - (lr_max - lr_end) * (i -
                                                   warmup_steps) / decay_steps
            lr_each_step.append(lr)

    elif lr_decay_mode == 'cosine':
        for i in range(total_steps):
            if i < warmup_steps:
                lr = linear_warmup_lr(i, warmup_steps, lr_max, lr_init)
            else:
                linear_decay = (total_steps - i) / decay_steps
                cosine_decay = 0.5 * (
                    1 + math.cos(math.pi * 2 * 0.47 *
                                 (i - warmup_steps) / decay_steps))
                decayed = linear_decay * cosine_decay + 0.00001
                lr = lr_max * decayed
            lr_each_step.append(lr)

    else:
        raise NotImplementedError(
            'Learning rate decay mode [{:s}] cannot be recognized'.format(
                lr_decay_mode))

    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[pretrain_steps:]

    return learning_rate


def linear_warmup_lr(current_step, warmup_steps, base_lr, init_lr):
    lr_inc = (base_lr - init_lr) / warmup_steps
    lr = init_lr + lr_inc * current_step
    return lr
