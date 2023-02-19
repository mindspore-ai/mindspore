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
""" test shift operator """
import pytest
import mindspore.nn as nn
from mindspore import context

context.set_context(mode=context.GRAPH_MODE)


def test_shift_operator():
    """
    Feature: shift operator
    Description: test left shift and right shift in normal cases
    Expectation: success
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            res1 = x << y
            res2 = x >> y
            return (res1, res2)

    net = Net()
    x = 2147483647
    y = 5
    result = net(x, y)
    assert result == (x << y, x >> y)


def test_bitwise_operator_augassign():
    """
    Feature: bitwise operator
    Description: test bitwise and, bitwise or, bitwise xor about augassign
    Expectation: success
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.const_x = -16
            self.const_y = 5

        def construct(self):
            res1 = self.const_x
            res1 <<= self.const_y
            res2 = self.const_x
            res2 >>= self.const_y
            return (res1, res2)

    net = Net()
    result = net()
    x = -16
    y = 5
    assert result == (x << y, x >> y)


def test_shift_operator_error_list_input():
    """
    Feature: shift operator
    Description: test shift operator with lists
    Expectation: throw RuntimeError
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.const_x = [10]
            self.const_y = [2]

        def construct(self):
            res = self.const_x << self.const_y
            return res

    net = Net()
    with pytest.raises(RuntimeError) as err:
        net()
    assert "For operation 'MetaFuncGraph-left_shift'" in str(err.value)
    assert "The 1-th argument type 'List' is not supported now." in str(err.value)
    assert "<Number, Number>" in str(err.value)


def test_shift_operator_error_float_input():
    """
    Feature: shift operator
    Description: test shift operator with float numbers
    Expectation: throw TypeError
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.const_x = 10.0
            self.const_y = 1.0

        def construct(self):
            res = self.const_x >> self.const_y
            return res

    net = Net()
    with pytest.raises(TypeError) as err:
        net()
    assert "only integer types are supported, but got" in str(err.value)


def test_shift_operator_error_too_large_number():
    """
    Feature: shift operator
    Description: test shift operator with too large numbers
    Expectation: throw RuntimeError
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            res = x << y
            return res

    x = 9223372036854775807 + 1
    y = 2
    net = Net()
    with pytest.raises(RuntimeError):
        net(x, y)


def test_shift_operator_error_negative_shift_count():
    """
    Feature: shift operator
    Description: test shift operator with negative shift count
    Expectation: throw ValueError
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            res = x >> y
            return res
    x = 16
    y = -2
    net = Net()
    with pytest.raises(ValueError) as err:
        net(x, y)
    assert "shift count must be a non-negative integer" in str(err.value)
