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


class LogicNot(nn.Cell):
    def __init__(self):
        super(LogicNot, self).__init__()
        self.m = 1

    def construct(self, x):
        not_v = not x
        return not_v


class LogicNotSpec(nn.Cell):
    def __init__(self, x):
        super(LogicNotSpec, self).__init__()
        self.x = x

    def construct(self, x):
        not_v = not self.x
        return not_v


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_logic_not_int():
    """
    Feature: simple expression
    Description: test logic not operator.
    Expectation: No exception
    """
    net = LogicNot()
    ret = net(1)
    assert not ret


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_logic_not_float():
    """
    Feature: simple expression
    Description: test logic not operator.
    Expectation: No exception
    """
    net = LogicNot()
    ret = net(1.89)
    assert not ret


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_logic_tensor_1_int():
    """
    Feature: simple expression
    Description: test logic not operator.
    Expectation: No exception
    """
    net = LogicNot()
    x = Tensor(np.ones([1], np.int32))
    ret = net(x)
    assert not ret


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_logic_not_tensor_1_float():
    """
    Feature: simple expression
    Description: test logic not operator.
    Expectation: No exception
    """
    net = LogicNot()
    x = Tensor(np.ones([1], np.float))
    ret = net(x)
    assert not ret


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_logic_not_tensor_2x2_int():
    """
    Feature: simple expression
    Description: test logic not operator.
    Expectation: No exception
    """
    with pytest.raises(ValueError, match="Only tensor which shape is"):
        net = LogicNot()
        x = Tensor(np.ones([2, 2], np.int32))
        ret = net(x)
        print(ret)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_logic_not_tensor_2x2_float():
    """
    Feature: simple expression
    Description: test logic not operator.
    Expectation: No exception
    """
    with pytest.raises(ValueError, match="Only tensor which shape is"):
        net = LogicNot()
        x = Tensor(np.ones([2, 2], np.float))
        ret = net(x)
        print(ret)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_logic_not_str():
    """
    Feature: simple expression
    Description: test logic not operator.
    Expectation: No exception
    """
    net = LogicNotSpec("cba")
    ret = net(1)
    assert not ret


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_logic_not_list_int():
    """
    Feature: simple expression
    Description: test logic not operator.
    Expectation: No exception
    """
    net = LogicNot()
    ret = net([1, 2, 3])
    assert not ret


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_logic_not_list_float():
    """
    Feature: simple expression
    Description: test logic not operator.
    Expectation: No exception
    """
    net = LogicNot()
    ret = net([1.0, 2.0, 3.0])
    assert not ret


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_logic_not_list_str():
    """
    Feature: simple expression
    Description: test logic not operator.
    Expectation: No exception
    """
    net = LogicNotSpec(["1", "2", "3"])
    ret = net(1)
    assert not ret


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_logic_not_list_combine():
    """
    Feature: simple expression
    Description: test logic not operator.
    Expectation: No exception
    """
    net = LogicNotSpec([1, "2", 3])
    ret = net(1)
    assert not ret


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_logic_not_tuple_int():
    """
    Feature: simple expression
    Description: test logic not operator.
    Expectation: No exception
    """
    net = LogicNot()
    ret = net((1, 2, 3))
    assert not ret


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_logic_not_tuple_float():
    """
    Feature: simple expression
    Description: test logic not operator.
    Expectation: No exception
    """
    net = LogicNot()
    ret = net((1.0, 2.0, 3.0))
    assert not ret


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_logic_not_tuple_str():
    """
    Feature: simple expression
    Description: test logic not operator.
    Expectation: No exception
    """
    net = LogicNotSpec(("1", "2", "3"))
    ret = net(1)
    assert not ret


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_logic_not_tuple_combine():
    """
    Feature: simple expression
    Description: test logic not operator.
    Expectation: No exception
    """
    net = LogicNotSpec((1, "2", 3))
    ret = net(1)
    assert not ret
