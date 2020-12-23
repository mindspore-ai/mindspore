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
"""training utils"""
import math
import numpy as np
from mindspore import dtype as mstype
from mindspore import Tensor


def get_lr(cfg, dataset_size):
    if cfg.cell == "lstm":
        lr = get_lr_lstm(0, cfg.lstm_lr_init, cfg.lstm_lr_end, cfg.lstm_lr_max, cfg.lstm_lr_warm_up_epochs,
                         cfg.lstm_num_epochs, dataset_size, cfg.lstm_lr_adjust_epochs)
        lr_ret = Tensor(lr, mstype.float32)
    else:
        lr_ret = cfg.lr
    return lr_ret


def get_lr_lstm(global_step, lr_init, lr_end, lr_max, warmup_epochs, total_epochs, steps_per_epoch, lr_adjust_epoch):
    """
    generate learning rate array

    Args:
       global_step(int): total steps of the training
       lr_init(float): init learning rate
       lr_end(float): end learning rate
       lr_max(float): max learning rate
       warmup_epochs(float): number of warmup epochs
       total_epochs(int): total epoch of training
       steps_per_epoch(int): steps of one epoch
       lr_adjust_epoch(int): lr adjust in lr_adjust_epoch, after that, the lr is lr_end

    Returns:
       np.array, learning rate array
    """
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    adjust_steps = lr_adjust_epoch * steps_per_epoch
    for i in range(total_steps):
        if i < warmup_steps:
            lr = lr_init + (lr_max - lr_init) * i / warmup_steps
        elif i < adjust_steps:
            lr = lr_end + \
                 (lr_max - lr_end) * \
                 (1. + math.cos(math.pi * (i - warmup_steps) / (adjust_steps - warmup_steps))) / 2.
        else:
            lr = lr_end
        if lr < 0.0:
            lr = 0.0
        lr_each_step.append(lr)

    current_step = global_step
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    learning_rate = lr_each_step[current_step:]

    return learning_rate
