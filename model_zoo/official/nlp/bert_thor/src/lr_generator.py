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
import numpy as np

from mindspore.common.tensor import Tensor


def get_poly_lr(global_step, lr_init, lr_end, lr_max, warmup_steps, total_steps, poly_power):
    """
    generate learning rate array

    Args:
       lr_init(float): init learning rate
       lr_end(float): end learning rate
       lr_max(float): max learning rate
       warmup_steps(int): number of warmup epochs
       total_steps(int): total epoch of training
       poly_power(int): poly learning rate power

    Returns:
       np.array, learning rate array
    """
    lr_each_step = []
    if warmup_steps != 0:
        inc_each_step = (float(lr_max) - float(lr_init)) / float(warmup_steps)
    else:
        inc_each_step = 0
    for i in range(total_steps):
        if i < warmup_steps:
            lr = float(lr_init) + inc_each_step * float(i)
        else:
            base = (1.0 - (float(i) - float(warmup_steps)) / (float(total_steps) - float(warmup_steps)))
            lr = float(lr_max - lr_end) * (base ** poly_power)
            lr = lr + lr_end
            if lr < 0.0:
                lr = 0.0
        lr_each_step.append(lr)

    learning_rate = np.array(lr_each_step).astype(np.float32)
    current_step = global_step
    learning_rate = learning_rate[current_step:]
    return learning_rate


# bert kfac hyperparam setting
def get_bert_lr():
    learning_rate = Tensor(
        get_poly_lr(global_step=0, lr_init=0.0, lr_end=1e-6, lr_max=3.1e-3, warmup_steps=0, total_steps=30000,
                    poly_power=1))
    return learning_rate


def get_bert_damping():
    damping = Tensor(
        get_poly_lr(global_step=0, lr_init=0.0, lr_end=1e-6, lr_max=5e-2, warmup_steps=0, total_steps=30000,
                    poly_power=1))
    return damping
