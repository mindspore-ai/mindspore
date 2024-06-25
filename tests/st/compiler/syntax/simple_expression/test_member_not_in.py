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


class MemberNotIn(nn.Cell):
    def __init__(self):
        super(MemberNotIn, self).__init__()
        self.m = 1

    def construct(self, x, y):
        not_in_v = x not in y
        return not_in_v


class MemberNotInSpec(nn.Cell):
    def __init__(self, x, y):
        super(MemberNotInSpec, self).__init__()
        self.x = x
        self.y = y

    def construct(self):
        not_in_v = self.x not in self.y
        return not_in_v


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_ms_syntax_operator_int_not_in_int():
    """
    Feature: simple expression
    Description: test not in operator.
    Expectation: No exception
    """
    with pytest.raises(TypeError, match="argument of type 'int' is not iterable"):
        net = MemberNotIn()
        ret = net(1, 2)
        print(ret)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_int_not_in_list_int():
    """
    Feature: simple expression
    Description: test not in operator.
    Expectation: No exception
    """
    net = MemberNotIn()
    ret = net(1, [1, 2])
    assert not ret


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_int_not_in_list_str():
    """
    Feature: simple expression
    Description: test not in operator.
    Expectation: No exception
    """
    net = MemberNotInSpec(1, ["1", "2"])
    ret = net()
    assert ret


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_int_not_in_list_combine():
    """
    Feature: simple expression
    Description: test not in operator.
    Expectation: No exception
    """
    net = MemberNotInSpec(1, ["1", 2])
    ret = net()
    assert ret


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_int_not_in_tuple_int():
    """
    Feature: simple expression
    Description: test not in operator.
    Expectation: No exception
    """
    net = MemberNotIn()
    ret = net(1, (1, 2))
    assert not ret


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_int_not_in_tuple_str():
    """
    Feature: simple expression
    Description: test not in operator.
    Expectation: No exception
    """
    net = MemberNotInSpec(1, ("1", 2))
    ret = net()
    assert ret


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_int_not_in_dict_int():
    """
    Feature: simple expression
    Description: test not in operator.
    Expectation: No exception
    """
    dict_y = {1: 2, 2: 3}
    net = MemberNotInSpec(1, dict_y)
    ret = net()
    assert not ret


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_int_not_in_dict_str():
    """
    Feature: simple expression
    Description: test not in operator.
    Expectation: No exception
    """
    dict_y = {"1": 2, "2": 3}
    net = MemberNotInSpec(1, dict_y)
    ret = net()
    assert ret


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_str_not_in_dict_str():
    """
    Feature: simple expression
    Description: test not in operator.
    Expectation: No exception
    """
    dict_y = {"1": 2, "2": 3}
    net = MemberNotInSpec("1", dict_y)
    ret = net()
    assert not ret


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_str_not_in_dict_combine():
    """
    Feature: simple expression
    Description: test not in operator.
    Expectation: No exception
    """
    dict_y = {"1": 2, 2: 3}
    net = MemberNotInSpec("1", dict_y)
    ret = net()
    assert not ret


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_tensor_not_in_list_tensor():
    """
    Feature: simple expression
    Description: test not in operator.
    Expectation: No exception
    """
    net = MemberNotIn()
    x = Tensor(np.ones([2, 2], np.int32))
    y = Tensor(np.zeros([2, 2], np.int32))
    ret = net(x, [x, y])
    assert not ret


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_tensor_not_in_list_combine():
    """
    Feature: simple expression
    Description: test not in operator.
    Expectation: No exception
    """
    x = Tensor(np.ones([2, 2], np.int32))
    y = Tensor(np.zeros([2, 2], np.int32))
    net = MemberNotInSpec(x, [y, "a"])
    ret = net()
    assert ret


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_tensor_not_in_tuple_tensor():
    """
    Feature: simple expression
    Description: test not in operator.
    Expectation: No exception
    """
    net = MemberNotIn()
    x = Tensor(np.ones([2, 2], np.int32))
    y = Tensor(np.zeros([2, 2], np.int32))
    ret = net(x, (x, y))
    assert not ret


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_ms_syntax_operator_tensor_not_in_tuple_combine():
    """
    Feature: simple expression
    Description: test not in operator.
    Expectation: No exception
    """
    x = Tensor(np.ones([2, 2], np.int32))
    net = MemberNotInSpec(x, (x, "a"))
    ret = net()
    assert not ret
