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


def adjust_learning_rate(initial_lr, gamma, stepvalues, steps_pre_epoch, total_epochs, warmup_epoch=5):
    return _dynamic_lr(initial_lr, total_epochs * steps_pre_epoch, warmup_epoch * steps_pre_epoch,
                       warmup_ratio=1 / 3)
