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
"""Face detection learning rate scheduler."""
from collections import Counter
import math
import numpy as np


def linear_warmup_learning_rate(current_step, warmup_steps, base_lr, init_lr):
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    learning_rate = float(init_lr) + lr_inc * current_step
    return learning_rate


def warmup_step(args, gamma=0.1, lr_scale=1.0):
    '''warmup_step'''
    base_lr = args.lr
    warmup_init_lr = 0
    total_steps = int(args.max_epoch * args.steps_per_epoch)
    warmup_steps = int(args.warmup_epochs * args.steps_per_epoch)
    milestones = args.lr_epochs
    milestones_steps = []
    for milestone in milestones:
        milestones_step = milestone*args.steps_per_epoch
        milestones_steps.append(milestones_step)
    lr = base_lr
    milestones_steps_counter = Counter(milestones_steps)
    lr_each_step = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr = linear_warmup_learning_rate(
                i, warmup_steps, base_lr, warmup_init_lr)
        else:
            lr = lr_scale * lr * gamma**milestones_steps_counter[i]
        print('i:{} lr:{}'.format(i, lr))
        lr_each_step.append(lr)
    return np.array(lr_each_step).astype(np.float32)


def warmup_step_new(args, lr_scale=1.0):
    '''warmup_step_new'''
    warmup_lr = args.warmup_lr / args.batch_size
    lr_rates = [lr_rate / args.batch_size * lr_scale for lr_rate in args.lr_rates]
    total_steps = int(args.max_epoch * args.steps_per_epoch)
    lr_steps = args.lr_steps
    warmup_steps = lr_steps[0]
    lr_left = 0
    print('real warmup_lr', warmup_lr)
    print('real lr_rates', lr_rates)
    if args.max_epoch * args.steps_per_epoch > lr_steps[-1]:
        lr_steps.append(args.max_epoch * args.steps_per_epoch)
        lr_rates.append(lr_rates[-1])
    lr_each_step = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr = warmup_lr
        elif i < lr_steps[lr_left]:
            lr = lr_rates[lr_left]
        else:
            lr_left = lr_left + 1
        lr_each_step.append(lr)
    return np.array(lr_each_step).astype(np.float32)


def warmup_cosine_annealing_lr(lr, steps_per_epoch, warmup_epochs, max_epoch, t_max, eta_min=0):
    '''warmup_cosine_annealing_lr'''
    base_lr = lr
    warmup_init_lr = 0
    total_steps = int(max_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    lr_each_step = []
    for i in range(total_steps):
        last_epoch = i // steps_per_epoch
        if i < warmup_steps:
            lr = linear_warmup_learning_rate(i + 1, warmup_steps, base_lr, warmup_init_lr)
        else:
            lr = eta_min + (base_lr - eta_min) * (1. + math.cos(math.pi*last_epoch / t_max)) / 2
        lr_each_step.append(lr)

    return np.array(lr_each_step).astype(np.float32)
