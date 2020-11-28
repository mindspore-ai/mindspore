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
"""Face Quality Assessment learning rate scheduler."""
from collections import Counter


def linear_warmup_learning_rate(current_step, warmup_steps, base_lr, init_lr):
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    learning_rate = float(init_lr) + lr_inc * current_step
    return learning_rate


def warmup_step(args, gamma=0.1):
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
    for i in range(total_steps):
        if i < warmup_steps:
            lr = linear_warmup_learning_rate(i, warmup_steps, base_lr, warmup_init_lr)
        else:
            lr = lr * gamma**milestones_steps_counter[i]
        yield lr
