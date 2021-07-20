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
"""
custom lr schedule
"""
from mindspore import Tensor
from mindspore import dtype as mstype
import mindspore.nn as nn


def dynamic_lr(num_epoch_per_decay, total_epochs, steps_per_epoch):
    """dynamic learning rate generator"""
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    decay_steps = steps_per_epoch * num_epoch_per_decay
    lr = nn.PolynomialDecayLR(1e-4, 1e-5, decay_steps, 0.5)
    for i in range(total_steps):
        if i < decay_steps:
            i = Tensor(i, mstype.int32)
            lr_each_step.append(lr(i).asnumpy())
        else:
            lr_each_step.append(1e-5)
    return lr_each_step
