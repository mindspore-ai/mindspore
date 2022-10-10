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
import numpy as np
import pytest


class Net(Cell):

    def __init__(self):
        super().__init__()
        self.w = Parameter(Tensor([(- 1)], dtype.float32), name='weight')
        self.b = Parameter(Tensor([(- 5)], dtype.float32), name='bias')

    def construct(self, x, y):
        if y == x:
            for a in range(2):
                x = x - y
                self.w = a * x
                if self.w < 0:
                    return x
        elif self.b >= x:
            for a in range(2):
                x = x - x
                y = y - 3
        return x + y


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tuple_getitem_err():
    """
    Feature: Control flow.
    Description: This test case failed before, add it to CI. Related issue: I5G160.
    Expectation: No exception raised.
    """
    x = np.array([2], np.float32)
    y = np.array([1], np.float32)
    net = Net()
    out = net(Tensor(x), Tensor(y))
    assert out == Tensor([3], dtype.float32)
