# Copyright 2020 Huawei Technologies Co., Ltd
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
""" test_list """
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import Cell
from mindspore import ops


class Net1(Cell):
    def __init__(self, list1):
        super().__init__()
        self.list = list1
        self.fla = nn.Flatten()

    def construct(self, x):
        for _ in self.list:
            x = self.fla(x)
        return x


class Net2(Cell):
    def __init__(self, list1):
        super().__init__()
        self.list = list1
        self.addn = ops.AddN()

    def construct(self, x):
        x = self.addn(self.list[0::2])
        return x


def test_list1():
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)
    net = Net1([1])
    _cell_graph_executor.compile(net, input_me)


def test_list2():
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)
    net = Net1([1, 2])
    _cell_graph_executor.compile(net, input_me)


def test_list_slice():
    """
    Feature: Support List Slice
    Description: Test List Slice
    Expectation: No exception.
    """
    input_me = Tensor(8)
    net = Net2([Tensor(1), Tensor(2), Tensor(3), Tensor(4), Tensor(5), Tensor(6), Tensor(7)])
    _cell_graph_executor.compile(net, input_me)
