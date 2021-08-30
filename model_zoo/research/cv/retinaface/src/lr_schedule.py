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
"""learning rate schedule."""
import math
import numpy as np


def warmup_cosine_annealing_lr(lr5, steps_per_epoch, warmup_epochs, max_epoch, T_max, eta_min=0):
    """ warmup cosine annealing lr"""
    base_lr = lr5
    warmup_init_lr = 0
    total_steps = int(max_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)

    lr_each_step = []
    for i in range(total_steps):
        last_epoch = i // steps_per_epoch
        if i < warmup_steps:
            lr5 = linear_warmup_lr(i + 1, warmup_steps, base_lr, warmup_init_lr)
        else:
            lr5 = eta_min + (base_lr - eta_min) * (1. + math.cos(math.pi * last_epoch / T_max)) / 2
        lr_each_step.append(lr5)

    return np.array(lr_each_step).astype(np.float32)


def _linear_warmup_learning_rate(current_step, warmup_steps, base_lr, init_lr):
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    learning_rate = float(init_lr) + lr_inc * current_step
    return learning_rate


def _a_cosine_learning_rate(current_step, base_lr, warmup_steps, decay_steps):
    base = float(current_step - warmup_steps) / float(decay_steps)
    learning_rate = (1 + math.cos(base * math.pi)) / 2 * base_lr
    return learning_rate


def _dynamic_lr(base_lr, total_steps, warmup_steps, warmup_ratio=1 / 3):
    lr = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr.append(_linear_warmup_learning_rate(i, warmup_steps, base_lr, base_lr * warmup_ratio))
        else:
            lr.append(_a_cosine_learning_rate(i, base_lr, warmup_steps, total_steps))

    return lr


def adjust_learning_rate(initial_lr, gamma, stepvalues, steps_pre_epoch, total_epochs, warmup_epoch=5, lr_type1=None):
    """adjust_learning_rate"""
    if lr_type1 == 'dynamic_lr':
        return _dynamic_lr(initial_lr, total_epochs * steps_pre_epoch, warmup_epoch * steps_pre_epoch,
                           warmup_ratio=1 / 3)

    lr_each_step = []
    for epoch in range(1, total_epochs + 1):
        for _ in range(steps_pre_epoch):
            if epoch <= warmup_epoch:
                lr = 0.1 * initial_lr * (1.5849 ** (epoch - 1))
            else:
                if stepvalues[0] <= epoch <= stepvalues[1]:
                    lr = initial_lr * (gamma ** (1))
                elif epoch > stepvalues[1]:
                    lr = initial_lr * (gamma ** (2))
                else:
                    lr = initial_lr
            lr_each_step.append(lr)
    return lr_each_step
