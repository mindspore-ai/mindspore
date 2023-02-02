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
""" test bitwise operator """
import numpy as np
import pytest
import mindspore.nn as nn
from mindspore import context

context.set_context(mode=context.GRAPH_MODE)


def test_bitwise_operator_1():
    """
    Feature: bitwise operator
    Description: test bitwise and, bitwise or, bitwise xor in normal cases
    Expectation: success
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            res1 = x & y
            res2 = x | y
            res3 = x ^ y
            return (res1, res2, res3)

    net = Net()
    int64_min = -9223372036854775808
    int64_max = 9223372036854775807
    for _ in range(1000):
        x = int(np.random.randint(int64_min, int64_max + 1))
        y = int(np.random.randint(int64_min, int64_max + 1))
        result = net(x, y)
        assert result == (x & y, x | y, x ^ y)


def test_bitwise_operator_2():
    """
    Feature: bitwise operator
    Description: test bitwise and, bitwise or, bitwise xor in special case
    Expectation: success
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.const_y = 2147483647

        def construct(self, x):
            res1 = x & self.const_y
            res2 = x | self.const_y
            res3 = x ^ self.const_y
            return (res1, res2, res3)

    x = 9223372036854775807
    net = Net()
    result = net(x)
    y = 2147483647
    assert result == (x & y, x | y, x ^ y)


def test_bitwise_operator_augassign():
    """
    Feature: bitwise operator
    Description: test bitwise and, bitwise or, bitwise xor about augassign
    Expectation: success
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.const_x = -2
            self.const_y = -9223372036854775808

        def construct(self):
            res1 = self.const_x
            res1 &= self.const_y
            res2 = self.const_x
            res2 |= self.const_y
            res3 = self.const_x
            res3 ^= self.const_y
            return (res1, res2, res3)

    net = Net()
    result = net()
    x = -2
    y = -9223372036854775808
    assert result == (x & y, x | y, x ^ y)


def test_bitwise_operator_error_list_input():
    """
    Feature: bitwise operator
    Description: test bitwise operator with lists
    Expectation: throw RuntimeError
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.const_x = [10]
            self.const_y = [11]

        def construct(self):
            res = self.const_x & self.const_y
            return res

    net = Net()
    with pytest.raises(RuntimeError) as err:
        net()
    assert "For operation 'MetaFuncGraph-bitwise_and'" in str(err.value)
    assert "The 1-th argument type 'List' is not supported now." in str(err.value)
    assert "<Tensor, Number>" in str(err.value)


def test_bitwise_operator_error_float_input():
    """
    Feature: bitwise operator
    Description: test bitwise operator with float numbers
    Expectation: throw TypeError
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.const_x = 10.0
            self.const_y = 11.0

        def construct(self):
            res = self.const_x | self.const_y
            return res

    net = Net()
    with pytest.raises(TypeError):
        net()


def test_bitwise_operator_error_too_large_number():
    """
    Feature: bitwise operator
    Description: test bitwise operator with too large numbers
    Expectation: throw TypeError
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            res = x ^ y
            return res

    x = 9223372036854775807 + 1
    y = 2
    net = Net()
    with pytest.raises(RuntimeError):
        net(x, y)
