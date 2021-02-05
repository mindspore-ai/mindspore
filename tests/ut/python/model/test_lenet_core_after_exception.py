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
"""test_lenet_core_after_exception"""
import numpy as np
import pytest

import mindspore.nn as nn
from mindspore.common.api import _executor
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from ....train_step_wrap import train_step_with_loss_warp


class LeNet5(nn.Cell):
    """LeNet5 definition"""

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, pad_mode="valid")
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode="valid")
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, 3)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2)
        self.flatten = P.Flatten()

    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test_lenet5_exception():
    in1 = np.ones([1, 1, 32, 32]).astype(np.float32) * 0.01
    in2 = np.zeros([1, 10]).astype(np.float32)
    predict = Tensor(in1)
    label = Tensor(in2)
    net = train_step_with_loss_warp(LeNet5())
    with pytest.raises(RuntimeError) as info:
        _executor.compile(net, predict, label)
    assert "x_shape[C_in] / group must equal to w_shape[C_in] = " in str(info.value)
