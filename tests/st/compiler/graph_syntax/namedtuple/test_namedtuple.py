# Copyright 2023 Huawei Technologies Co., Ltd
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
""" test NamedTuple in graph mode """

import mindspore as ms
import mindspore.nn as nn
from mindspore import context, jit
from typing import NamedTuple
from collections import namedtuple
from mindspore import ops
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_namedtuple_get_attr1():
    """
    Feature: Support NamedTuple in graph mode.
    Description: Support NamedTuple in graph mode.
    Expectation: No exception.
    """
    class Data(NamedTuple):
        label1: ms.Tensor
        label2: ms.Tensor

    class Net(nn.Cell):
        def __init__(self, data):
            super().__init__()
            self.data = data

        def construct(self):
            label1 = self.data.label1
            label2 = self.data.label2
            return label1, label2

    data = Data(ops.randn(6, 5), ops.randn(6, 1))
    net = Net(data)
    label1, label2 = net()
    assert label1.shape == data.label1.shape
    assert label2.shape == data.label2.shape


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_namedtuple_get_attr2():
    """
    Feature: Support NamedTuple in graph mode.
    Description: Support NamedTuple in graph mode.
    Expectation: No exception.
    """
    class Data(NamedTuple):
        label1: ms.Tensor
        label2: ms.Tensor

    class Net(nn.Cell):
        def __init__(self, data):
            super().__init__()
            self.data = data

        def construct(self):
            label1 = self.data[0]
            label2 = self.data[1]
            return label1, label2

    data = Data(ops.randn(6, 5), ops.randn(6, 1))
    net = Net(data)
    label1, label2 = net()
    assert label1.shape == data.label1.shape
    assert label2.shape == data.label2.shape


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_namedtuple_get_attr3():
    """
    Feature: Support NamedTuple in graph mode.
    Description: Support NamedTuple in graph mode.
    Expectation: No exception.
    """
    class Data(NamedTuple):
        label1: ms.Tensor
        label2: ms.Tensor

    class Net(nn.Cell):
        def construct(self, data):
            label1 = data.label1
            label2 = data.label2
            return label1, label2

    data = Data(ops.randn(6, 5), ops.randn(6, 1))
    net = Net()
    label1, label2 = net(data)
    assert label1.shape == data.label1.shape
    assert label2.shape == data.label2.shape


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_namedtuple_get_attr4():
    """
    Feature: Support NamedTuple in graph mode.
    Description: Support NamedTuple in graph mode.
    Expectation: No exception.
    """
    class Data(NamedTuple):
        label1: ms.Tensor
        label2: ms.Tensor

    class Net(nn.Cell):
        def construct(self, data):
            label1 = data[0]
            label2 = data[1]
            return label1, label2

    data = Data(ops.randn(6, 5), ops.randn(6, 1))
    net = Net()
    label1, label2 = net(data)
    assert label1.shape == data.label1.shape
    assert label2.shape == data.label2.shape


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_namedtuple_get_attr5():
    """
    Feature: Support NamedTuple in graph mode.
    Description: Support NamedTuple in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, data):
            super().__init__()
            self.data = data

        def construct(self):
            label1 = self.data.label1
            label2 = self.data.label2
            return label1, label2


    Data = namedtuple('User', ['label1', 'label2'])
    data = Data(label1=ops.randn(6, 5), label2=ops.randn(6, 1))
    net = Net(data)
    label1, label2 = net()
    assert label1.shape == data.label1.shape
    assert label2.shape == data.label2.shape


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_namedtuple_get_attr6():
    """
    Feature: Support NamedTuple in graph mode.
    Description: Support NamedTuple in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, data):
            super().__init__()
            self.data = data

        def construct(self):
            label1 = self.data[0]
            label2 = self.data[1]
            return label1, label2

    Data = namedtuple('User', ['label1', 'label2'])
    data = Data(label1=ops.randn(6, 5), label2=ops.randn(6, 1))
    net = Net(data)
    label1, label2 = net()
    assert label1.shape == data.label1.shape
    assert label2.shape == data.label2.shape


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_namedtuple_get_attr7():
    """
    Feature: Support NamedTuple in graph mode.
    Description: Support NamedTuple in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, data):
            label1 = data.label1
            label2 = data.label2
            return label1, label2

    Data = namedtuple('User', ['label1', 'label2'])
    data = Data(label1=ops.randn(6, 5), label2=ops.randn(6, 1))
    net = Net()
    label1, label2 = net(data)
    assert label1.shape == data.label1.shape
    assert label2.shape == data.label2.shape


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_namedtuple_get_attr8():
    """
    Feature: Support NamedTuple in graph mode.
    Description: Support NamedTuple in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, data):
            label1 = data[0]
            label2 = data[1]
            return label1, label2

    Data = namedtuple('User', ['label1', 'label2'])
    data = Data(label1=ops.randn(6, 5), label2=ops.randn(6, 1))
    net = Net()
    label1, label2 = net(data)
    assert label1.shape == data.label1.shape
    assert label2.shape == data.label2.shape


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_create_namedtuple():
    """
    Feature: Support namedtuple in graph mode.
    Description: Support create namedtuple in graph mode.
    Expectation: No exception.
    """
    @ms.jit
    def _max():
        point = namedtuple('max', 'values, indices')
        rtl = point(1, 2)
        return rtl.values, rtl.indices
    output = _max()
    point = namedtuple('max', 'values, indices')
    expect = point(1, 2)
    assert output[0] == expect.values
    assert output[1] == expect.indices


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_create_and_return_namedtuple():
    """
    Feature: Support namedtuple in graph mode.
    Description: Support create and return namedtuple in graph mode.
    Expectation: No exception.
    """
    @jit
    def _max():
        point = namedtuple('max', 'values, indices')
        rtl = point(1, 2)
        return rtl
    output = _max()
    point = namedtuple('max', 'values, indices')
    expect = point(1, 2)
    assert output.values == expect.values
    assert output.indices == expect.indices
