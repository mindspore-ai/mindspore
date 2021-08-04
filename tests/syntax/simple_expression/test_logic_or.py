# Copyright 2021 Huawei Technologies Co., Ltd
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
from mindspore import context
from mindspore.common.tensor import Tensor

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


def test_ms_syntax_operator_logic_int_or_int():
    net = LogicOr()
    ret = net(1, 2)
    print(ret)


def test_ms_syntax_operator_logic_float_or_float():
    net = LogicOr()
    ret = net(1.89, 1.99)
    print(ret)


def test_ms_syntax_operator_logic_float_or_int():
    net = LogicOr()
    ret = net(1.89, 1)
    print(ret)


def test_ms_syntax_operator_logic_tensor_1_int_or_tensor_1_int():
    net = LogicOr()
    x = Tensor(np.ones([1], np.int32))
    y = Tensor(np.zeros([1], np.int32))
    ret = net(x, y)
    print(ret)


def test_ms_syntax_operator_logic_tensor_1_float_or_tensor_1_int():
    net = LogicOr()
    x = Tensor(np.ones([1], np.float))
    y = Tensor(np.zeros([1], np.int32))
    ret = net(x, y)
    print(ret)


def test_ms_syntax_operator_logic_tensor_2X2_int_or_tensor_2X2_int():
    net = LogicOr()
    x = Tensor(np.ones([2, 2], np.int32))
    y = Tensor(np.zeros([2, 2], np.int32))
    ret = net(x, y)
    print(ret)


def test_ms_syntax_operator_logic_int_or_str():
    net = LogicOr()
    ret = net(1, "cba")
    print(ret)


def test_ms_syntax_operator_logic_int_or_str_2():
    net = LogicOrSpec(1, "cba")
    ret = net(1, 2)
    print(ret)


def test_ms_syntax_operator_logic_str_or_str():
    net = LogicOrSpec("abc", "cba")
    ret = net(1, 2)
    print(ret)


def test_ms_syntax_operator_logic_list_int_or_list_int():
    net = LogicOr()
    ret = net([1, 2, 3], [3, 2, 1])
    print(ret)


def test_ms_syntax_operator_logic_list_int_or_int():
    net = LogicOr()
    ret = net([1, 2, 3], 1)
    print(ret)


def test_ms_syntax_operator_logic_list_int_or_str():
    net = LogicOrSpec([1, 2, 3], "aaa")
    ret = net(1, 2)
    print(ret)


def test_ms_syntax_operator_logic_list_int_or_list_str():
    net = LogicOrSpec([1, 2, 3], ["1", "2", "3"])
    ret = net(1, 2)
    print(ret)


def test_ms_syntax_operator_logic_list_int_or_list_str_var():
    left = [1, 2, 3]
    right = ["1", "2", "3"]
    net = LogicOrSpec(left, right)
    ret = net(1, 2)
    print(ret)
