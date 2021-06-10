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
"""generate lr"""

import numpy as np
def _generate_steps_lr(lr_init, lr_max, total_steps, warmup_steps, global_step=0):
    """
    Applies three steps decay to generate learning rate array.

    Args:
       lr_init(float): init learning rate.
       lr_max(float): max learning rate.
       total_steps(int): all steps in training.
       warmup_steps(int): all steps in warmup epochs.

    Returns:
       np.array, learning rate array.
    """
    decay_epoch_index = [total_steps*i for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
    lr_each_step = []
    for i in range(total_steps):
        if i < warmup_steps:
            lr = lr_init + (lr_max - lr_init) * i / warmup_steps
        else:
            if i < decay_epoch_index[0]:
                lr = lr_max
            elif i < decay_epoch_index[1]:
                lr = lr_max * 0.5**1
            elif i < decay_epoch_index[2]:
                lr = lr_max * 0.5**2
            elif i < decay_epoch_index[3]:
                lr = lr_max * 0.5**3
            elif i < decay_epoch_index[4]:
                lr = lr_max * 0.5**4
            elif i < decay_epoch_index[5]:
                lr = lr_max * 0.5**5
            elif i < decay_epoch_index[6]:
                lr = lr_max * 0.5**6
            elif i < decay_epoch_index[7]:
                lr = lr_max * 0.5*7
            elif i < decay_epoch_index[8]:
                lr = lr_max * 0.5**8
            else:
                lr = lr_max * 0.5**9
        lr_each_step.append(lr)
    lr_each_step = np.array(lr_each_step).astype(np.float32)[global_step:]
    return lr_each_step
