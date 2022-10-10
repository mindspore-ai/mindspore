# Copyright 2022 Huawei Technologies Co., Ltd
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
from mindspore.nn import Cell
from mindspore.common import Tensor, dtype, Parameter
import mindspore.ops.functional as F
import numpy as np
import pytest


class Net(Cell):

    def __init__(self):
        super().__init__()
        self.w = Parameter(Tensor([(- 2)], dtype.float32), name='weight')
        self.b = Parameter(Tensor([(- 1)], dtype.float32), name='bias')

    def construct(self, x, y):
        if self.b < 1:
            if x <= y:
                self.b = x * self.w
                y = 2 + y
                self.w = x - 1
            elif y <= self.w:
                x = x - x
            else:
                x = self.b - x
        elif self.w >= 4:
            x = y * 2
        if self.b <= 1:
            y = 3 + x
        elif self.b > 2:
            if self.w >= 5:
                x = 1 + y
                x = x * self.b
                self.w = self.w + x
            elif x <= self.w:
                self.w = y * y
            elif self.b == self.w:
                self.b = y - 3
            elif self.b <= 3:
                y = 1 / x
        return x + y


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_can_not_be_operator_err():
    """
    Feature: Control flow.
    Description: This test case failed before, add it to CI. Related issue: I5FLCG.
    Expectation: No exception raised.
    """
    x = np.array([5], np.float32)
    y = np.array([3], np.float32)
    net = Net()
    out = net(Tensor(x), Tensor(y))
    print('ms forward: ', out)
    grad_net = F.grad(net, grad_position=(0, 1))
    fgrad = grad_net(Tensor(x), Tensor(y))
    print('ms backward: ', fgrad)
