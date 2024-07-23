# Copyright 2021-2023 Huawei Technologies Co., Ltd
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
""" test syntax for logic expression """

import numpy as np
import mindspore.nn as nn
import mindspore
from mindspore import context, jit
from mindspore.common.tensor import Tensor
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.m = 1

    def construct(self, x, y):
        return x > y


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_compare_bool_vs_bool():
    """
    Feature: simple expression
    Description: test compare.
    Expectation: No exception
    """
    net = Net()
    ret = net(True, True)
    assert not ret


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_compare_bool_vs_int():
    """
    Feature: simple expression
    Description: test compare.
    Expectation: No exception
    """
    net = Net()
    ret = net(True, 1)
    assert not ret


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_compare_tensor_int_vs_tensor_float():
    """
    Feature: simple expression
    Description: test compare.
    Expectation: No exception
    """
    x = Tensor(1, mindspore.int32)
    y = Tensor(1.5, mindspore.float64)
    net = Net()
    ret = net(x, y)
    assert not ret


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_multiple_compare():
    """
    Feature: Test compare including multiple comparators.
    Description: test compare.
    Expectation: No exception
    """
    @jit
    def comp_func(x, y, z):
        return x < y < z

    a = Tensor([1])
    b = Tensor([2])
    c = Tensor([3])

    ret = comp_func(a, b, c)
    assert ret


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_multiple_compare_2():
    """
    Feature: Test compare including multiple comparators.
    Description: test compare.
    Expectation: No exception
    """
    @jit
    def comp_func(x, y, z):
        return x < y >= z

    a = Tensor([1])
    b = Tensor([2])
    c = Tensor([3])

    ret = comp_func(a, b, c)
    assert not ret


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_multiple_compare_3():
    """
    Feature: Test compare including multiple comparators.
    Description: test compare.
    Expectation: No exception
    """
    @jit
    def comp_func(x, y, z):
        return x == y == z

    a = Tensor([1])
    b = Tensor([1])
    c = Tensor([1])

    ret = comp_func(a, b, c)
    assert ret


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_multiple_compare_4():
    """
    Feature: Test compare including multiple comparators.
    Description: test compare.
    Expectation: No exception
    """
    @jit
    def comp_func():
        x = np.array([1])
        y = np.array([2])
        z = np.array([3])
        return x <= y <= z

    ret = comp_func()
    assert ret
