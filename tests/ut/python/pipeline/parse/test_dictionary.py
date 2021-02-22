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
""" test_dictionary """
import numpy as np

from mindspore import Tensor, context
from mindspore.nn import Cell

context.set_context(mode=context.GRAPH_MODE)


class Net1(Cell):
    def __init__(self):
        super().__init__()

    def construct(self, x):
        dic = {'x': 0, 'y': 1}
        output = []
        for i in dic.keys():
            output.append(i)
        for j in dic.values():
            output.append(j)
        return output


class Net2(Cell):
    def __init__(self):
        super().__init__()

    def construct(self, x):
        dic = {'x': x, 'y': 1}
        output = []
        for i in dic.keys():
            output.append(i)
        for j in dic.values():
            output.append(j)
        return output


class Net3(Cell):
    def __init__(self):
        super().__init__()

    def construct(self, x):
        dic = {'x': 0}
        dic['y'] = (0, 1)
        output = []
        for i in dic.keys():
            output.append(i)
        for j in dic.values():
            output.append(j)
        return output


def test_dict1():
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)
    net = Net1()
    out_me = net(input_me)
    assert out_me == ('x', 'y', 0, 1)


def test_dict2():
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)
    net = Net2()
    net(input_me)


def test_dict3():
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)
    net = Net3()
    out_me = net(input_me)
    assert out_me == ('x', 'y', 0, (0, 1))


def test_dict4():
    class Net(Cell):
        def __init__(self):
            super().__init__()

        def construct(self, tuple_x):
            output = tuple_x + tuple_x
            return output

    x = (1, Tensor([1, 2, 3]), {"a": Tensor([1, 2, 3]), "b": 1})
    net = Net()
    out_me = net(x)
    assert out_me == x + x
