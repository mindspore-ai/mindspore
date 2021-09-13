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


class MemberIn(nn.Cell):
    def __init__(self):
        super(MemberIn, self).__init__()
        self.m = 1

    def construct(self, x, y):
        in_v = x in y
        return in_v


class MemberInSpec(nn.Cell):
    def __init__(self, x, y):
        super(MemberInSpec, self).__init__()
        self.x = x
        self.y = y


    def construct(self, x, y):
        in_v = self.x in self.y
        return in_v


def test_ms_syntax_operator_int_in_int():
    net = MemberIn()
    ret = net(1, 2)
    print(ret)


def test_ms_syntax_operator_int_in_list_int():
    net = MemberIn()
    ret = net(1, [1, 2])
    print(ret)


def test_ms_syntax_operator_int_in_list_str():
    net = MemberInSpec(1, ["1", "2"])
    ret = net(1, 2)
    print(ret)


def test_ms_syntax_operator_int_in_list_combine():
    net = MemberInSpec(1, ["1", 2])
    ret = net(1, 2)
    print(ret)


def test_ms_syntax_operator_int_in_tuple_int():
    net = MemberIn()
    ret = net(1, (1, 2))
    print(ret)


def test_ms_syntax_operator_int_in_tuple_str():
    net = MemberInSpec(1, ("1", 2))
    ret = net(1, 2)
    print(ret)


def test_ms_syntax_operator_int_in_dict_str():
    dict_y = {"1": 2, "2": 3}
    net = MemberInSpec(1, dict_y)
    ret = net(1, 2)
    print(ret)


def test_ms_syntax_operator_str_in_dict_str():
    dict_y = {"1": 2, "2": 3}
    net = MemberInSpec("1", dict_y)
    ret = net(1, 2)
    print(ret)


def test_ms_syntax_operator_str_in_dict_combine():
    dict_y = {"1": 2, 2: 3}
    net = MemberInSpec("1", dict_y)
    ret = net(1, 2)
    print(ret)


def test_ms_syntax_operator_int_in_dict_combine():
    dict_y = {"1": 2, 2: 3}
    net = MemberInSpec(1, dict_y)
    ret = net(1, 2)
    print(ret)


def test_ms_syntax_operator_tensor_in_list_tensor():
    net = MemberIn()
    x = Tensor(np.ones([2, 2], np.int32))
    y = Tensor(np.zeros([2, 2], np.int32))
    ret = net(x, [x, y])
    print(ret)


def test_ms_syntax_operator_tensor_in_list_combine():
    x = Tensor(np.ones([2, 2], np.int32))
    y = Tensor(np.zeros([2, 2], np.int32))
    net = MemberInSpec(x, [y, "a"])
    ret = net(1, 2)
    print(ret)


def test_ms_syntax_operator_tensor_in_tuple_tensor():
    net = MemberIn()
    x = Tensor(np.ones([2, 2], np.int32))
    y = Tensor(np.zeros([2, 2], np.int32))
    ret = net(x, (x, y))
    print(ret)


def test_ms_syntax_operator_tensor_in_tuple_combine():
    x = Tensor(np.ones([2, 2], np.int32))
    net = MemberInSpec(x, (x, "a"))
    ret = net(1, 2)
    print(ret)
