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
# ===========================================================================
"""lr"""
import numpy as np


def _generate_linear_lr(lr_init, lr_end, total_steps, warmup_steps, useWarmup=False):
    """ warmup  lr"""
    lr_each_step = []
    if useWarmup:
        for i in range(0, total_steps):
            lrate = lr_init + (lr_end - lr_init) * i / warmup_steps
            if i >= warmup_steps:
                lrate = lr_end - (lr_end - lr_init) * (i - warmup_steps) / (total_steps - warmup_steps)
            lr_each_step.append(lrate)
    else:
        for i in range(total_steps):
            lrate = lr_end - (lr_end - lr_init) * i / total_steps
            lr_each_step.append(lrate)

    return lr_each_step


def get_warmup_linear_lr(lr_init, lr_end, total_steps, warmup_steps=10):
    lr_each_step = _generate_linear_lr(lr_init, lr_end, total_steps, warmup_steps)
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    return lr_each_step


if __name__ == '__main__':
    lr = get_warmup_linear_lr(0, 1e-4, 1000)
    print(lr.size)
    print(lr)
