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

def get_lr_cifar10(current_step, lr_max, total_epochs, steps_per_epoch):
    """
    generate learning rate array

    Args:
       current_step(int): current steps of the training
       lr_max(float): max learning rate
       total_epochs(int): total epoch of training
       steps_per_epoch(int): steps of one epoch

    Returns:
       np.array, learning rate array
    """
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    decay_epoch_index = [0.8 * total_steps]
    for i in range(total_steps):
        if i < decay_epoch_index[0]:
            lr = lr_max
        else:
            lr = lr_max * 0.1
        lr_each_step.append(lr)
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]

    return learning_rate

def get_lr_imagenet(lr, epoch_size, steps_per_epoch):
    """generate learning rate array"""
    lr = warmup_cosine_annealing_lr(lr, epoch_size, steps_per_epoch)
    return lr

def linear_warmup_lr(current_step, warmup_steps, base_lr, init_lr):
    """Linear learning rate"""
    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    lr = float(init_lr) + lr_inc * current_step
    return lr

def warmup_cosine_annealing_lr(lr, max_epoch, steps_per_epoch, warmup_epochs=5, T_max=150, eta_min=0.0):
    """ Cosine annealing learning rate"""
    base_lr = lr
    warmup_init_lr = 0
    total_steps = int(max_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)

    lr_each_step = []
    for i in range(total_steps):
        last_epoch = i // steps_per_epoch
        if i < warmup_steps:
            lr = linear_warmup_lr(i + 1, warmup_steps, base_lr, warmup_init_lr)
        else:
            lr = eta_min + (base_lr - eta_min) * (1. + math.cos(math.pi*last_epoch / T_max)) / 2
        lr_each_step.append(lr)

    return np.array(lr_each_step).astype(np.float32)
