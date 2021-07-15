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
"""learning rate schedule"""

import math


def constant_lr(init_lr, total_epochs, steps_per_epoch):
    lr_each_step = [init_lr] * total_epochs * steps_per_epoch
    return lr_each_step


def cosine_lr(init_lr, total_epochs, steps_per_epoch, mode='epoch', start_from_epoch=0, warmup_epoch=0):
    """
    0.5+0.5*cos(k*pi/K)
    """
    lr_each_step = []
    if mode == 'epoch':
        for i in range(total_epochs):
            tmp_lr = (0.5 + 0.5 * (math.cos(math.pi * i / total_epochs))) * init_lr
            lr_each_step = lr_each_step + [tmp_lr] * steps_per_epoch
    elif mode == 'step':
        for i in range(total_epochs * steps_per_epoch):
            tmp_lr = (0.5 + 0.5 * (math.cos(math.pi * i / (total_epochs * steps_per_epoch)))) * init_lr
            lr_each_step.append(tmp_lr)
    lr_warm_up_step = []
    if warmup_epoch > 0:
        for i in range(warmup_epoch):
            tmp_lr = init_lr * (i + 1) / warmup_epoch
            lr_warm_up_step = lr_warm_up_step + [tmp_lr] * steps_per_epoch

    lr_each_step = lr_warm_up_step + lr_each_step
    return lr_each_step[start_from_epoch * steps_per_epoch:]


def step_cosine_lr(init_lr, total_epochs, epoch_stage, steps_per_epoch, mode='epoch', start_from_epoch=0,
                   warmup_epoch=0):
    """
    generate learning rate array by step cosine lr
    if mode = 'epoch'
        lr = lr * cos(7k*pi/15K) K = epoch_all, k = cur_epoch
    if mode = 'step'
        lr = lr * cos(7k*pi/15K) K = step_all, k = cur_step
    Args:
        init_lr(float): base learning rate
        total_epochs(int): total epoch of training
        epoch_stage(list): multiple stage for applying cosine lr, each stage starts with a new inti learning rate
    :return:
        list, learning rate list
    """
    lr_each_step = []

    assert sum(epoch_stage) == total_epochs

    if mode == 'epoch':
        cur_epoch = 0
        cur_stage_lr = init_lr
        for (stage, epochs) in enumerate(epoch_stage):
            denominator = (cur_epoch + epochs) * 15
            cur_stage_lr = cur_stage_lr * math.cos(7 * stage * math.pi / 15)

            for i in range(cur_epoch, cur_epoch + epochs):
                numerator = 7 * (i - cur_epoch) * math.pi
                tmp_lr = math.cos(numerator / denominator) * cur_stage_lr
                lr_each_step = lr_each_step + [tmp_lr] * steps_per_epoch

            cur_epoch = cur_epoch + epochs

    elif mode == 'step':
        cur_step = 0
        cur_stage_lr = init_lr
        for (stage, epochs) in enumerate(epoch_stage):
            denominator = (cur_step + epochs * steps_per_epoch) * 15
            cur_stage_lr = cur_stage_lr * math.cos(7 * stage * math.pi / 15)

            for i in range(cur_step, cur_step + epochs * steps_per_epoch):
                numerator = 7 * (i - cur_step) * math.pi
                tmp_lr = math.cos(numerator / denominator) * cur_stage_lr
                lr_each_step.append(tmp_lr)

            cur_step = cur_step + epochs * steps_per_epoch

    lr_warm_up_step = []
    if warmup_epoch > 0:
        for i in range(warmup_epoch):
            tmp_lr = init_lr * (i + 1) / warmup_epoch
            lr_warm_up_step = lr_warm_up_step + [tmp_lr] * steps_per_epoch

    lr_each_step = lr_warm_up_step + lr_each_step
    return lr_each_step[start_from_epoch * steps_per_epoch:]
