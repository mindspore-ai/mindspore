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
import pytest
import mindspore.nn as nn
from mindspore import context

context.set_context(mode=context.GRAPH_MODE)


def test_bitwise_operator_1():
    """
    Feature: bitwise and operator
    Description: test bitwise and operator
    Expectation: success
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            res1 = x & y
            res2 = x | y
            return (res1, res2)

    x = 10
    y = 11
    net = Net()
    result = net(x, y)
    assert result == (10, 11)


def test_bitwise_operator_2():
    """
    Feature: bitwise and operator
    Description: test bitwise and operator in another case
    Expectation: success
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.const_y = 2147483647

        def construct(self, x):
            res1 = x & self.const_y
            res2 = x | self.const_y
            return (res1, res2)

    x = 9223372036854775807
    net = Net()
    result = net(x)
    assert result == (2147483647, 9223372036854775807)


def test_bitwise_operator_3():
    """
    Feature: bitwise and operator
    Description: test bitwise and operator with negative numbers
    Expectation: success
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.const_x = -2
            self.const_y = -9223372036854775808

        def construct(self):
            res1 = self.const_x & self.const_y
            res2 = self.const_x | self.const_y
            return (res1, res2)

    net = Net()
    result = net()
    assert result == (-9223372036854775808, -2)


def test_bitwise_operator_error_1():
    """
    Feature: bitwise and operator
    Description: test bitwise and operator with lists
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
    assert "operation does not support the type" in str(err.value)


def test_bitwise_operator_error_2():
    """
    Feature: bitwise and operator
    Description: test bitwise and operator with float numbers
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
    with pytest.raises(TypeError) as err:
        net()
    assert "Unsupported input type. For BitOr, only integer types are supported, but got" in str(err.value)


def test_bitwise_operator_error_3():
    """
    Feature: bitwise and operator
    Description: test bitwise and operator with too large numbers
    Expectation: throw TypeError
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            res1 = x & y
            res2 = x | y
            return (res1, res2)

    x = 9223372036854775807 + 1
    y = 2
    net = Net()
    with pytest.raises(RuntimeError):
        net(x, y)
