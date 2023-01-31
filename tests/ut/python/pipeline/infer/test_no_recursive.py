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
import numpy as np
import mindspore as ms
from mindspore.common.api import _no_recursive as no_recursive

ms.set_context(mode=ms.GRAPH_MODE)


def mul(x, y):
    return x * y


def double(x):
    return mul(x, x)


def test_cell_no_recursive():
    """
    Feature: no_recursive
    Description: test no_recursive flag.
    Expectation: No exception.
    """
    @no_recursive
    class Net(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = ms.ops.Add()

        def construct(self, x, y):
            return self.add(double(x), double(y))

    x = ms.Tensor(np.array([2.0], np.float32))
    y = ms.Tensor(np.array([3.0], np.float32))
    net = Net()
    print(net(x, y))


@ms.jit
@no_recursive
def func(x, y):
    res = double(x) + double(y)
    print(res)
    return res


def test_ms_function_no_recursive():
    """
    Feature: no_recursive
    Description: test no_recursive flag.
    Expectation: No exception.
    """
    x = ms.Tensor(np.array([2.0], np.float32))
    y = ms.Tensor(np.array([3.0], np.float32))
    print(func(x, y))
