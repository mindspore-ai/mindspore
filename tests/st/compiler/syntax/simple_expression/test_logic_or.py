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

import pytest
import numpy as np

import mindspore.nn as nn
from mindspore import context
from mindspore.common.tensor import Tensor
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


class LogicOr(nn.Cell):
    def __init__(self):
        super(LogicOr, self).__init__()
        self.m = 1

    def construct(self, x, y):
        or_v = x or y
        return or_v


class LogicOrSpec(nn.Cell):
    def __init__(self, x, y):
        super(LogicOrSpec, self).__init__()
        self.x = x
        self.y = y

    def construct(self, x, y):
        or_v = self.x or self.y
        return or_v


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_logic_int_or_int():
    """
    Feature: simple expression
    Description: test logic or operator.
    Expectation: No exception
    """
    net = LogicOr()
    ret = net(1, 2)
    assert ret == 1


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_logic_float_or_float():
    """
    Feature: simple expression
    Description: test logic or operator.
    Expectation: No exception
    """
    net = LogicOr()
    ret = net(1.89, 1.99)
    assert ret == 1.89


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_logic_float_or_int():
    """
    Feature: simple expression
    Description: test logic or operator.
    Expectation: No exception
    """
    net = LogicOr()
    ret = net(1.89, 1)
    assert ret == 1.89


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_logic_tensor_1_int_or_tensor_1_int():
    """
    Feature: simple expression
    Description: test logic or operator.
    Expectation: No exception
    """
    net = LogicOr()
    x = Tensor(np.ones([1], np.int32))
    y = Tensor(np.zeros([1], np.int32))
    ret = net(x, y)
    assert (ret.asnumpy() == [1]).all()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_logic_tensor_1_float_or_tensor_1_int():
    """
    Feature: simple expression
    Description: test logic or operator.
    Expectation: No exception
    """
    with pytest.raises(TypeError, match="Cannot join the return values of different branches"):
        net = LogicOr()
        x = Tensor(np.ones([1], np.float))
        y = Tensor(np.zeros([1], np.int32))
        ret = net(x, y)
        print(ret)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_logic_tensor_2x2_int_or_tensor_2x2_int():
    """
    Feature: simple expression
    Description: test logic or operator.
    Expectation: No exception
    """
    with pytest.raises(ValueError) as err:
        net = LogicOr()
        x = Tensor(np.ones([2, 2], np.int32))
        y = Tensor(np.zeros([2, 2], np.int32))
        ret = net(x, y)
        print(ret)
    assert "Only tensor which shape is () or (1,) can be converted to bool, but got tensor shape is (2, 2)" in str(err)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_logic_int_or_str():
    """
    Feature: simple expression
    Description: test logic or operator.
    Expectation: No exception
    """
    net = LogicOr()
    ret = net(1, "cba")
    assert ret == 1


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_logic_int_or_str_2():
    """
    Feature: simple expression
    Description: test logic or operator.
    Expectation: No exception
    """
    net = LogicOrSpec(1, "cba")
    ret = net(1, 2)
    assert ret == 1


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_logic_str_or_str():
    """
    Feature: simple expression
    Description: test logic or operator.
    Expectation: No exception
    """
    net = LogicOrSpec("abc", "cba")
    ret = net(1, 2)
    assert ret == "abc"


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_logic_list_int_or_list_int():
    """
    Feature: simple expression
    Description: test logic or operator.
    Expectation: No exception
    """
    net = LogicOr()
    ret = net([1, 2, 3], [3, 2, 1])
    assert ret == [1, 2, 3]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_logic_list_int_or_int():
    """
    Feature: simple expression
    Description: test logic or operator.
    Expectation: No exception
    """
    net = LogicOr()
    ret = net([1, 2, 3], 1)
    assert ret == [1, 2, 3]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_logic_list_int_or_str():
    """
    Feature: simple expression
    Description: test logic or operator.
    Expectation: No exception
    """
    net = LogicOrSpec([1, 2, 3], "aaa")
    ret = net(1, 2)
    assert ret == [1, 2, 3]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_logic_list_int_or_list_str():
    """
    Feature: simple expression
    Description: test logic or operator.
    Expectation: No exception
    """
    net = LogicOrSpec([1, 2, 3], ["1", "2", "3"])
    ret = net(1, 2)
    assert ret == [1, 2, 3]
