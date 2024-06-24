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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore.context as context
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore import ops, Tensor


class ConstScalarAndTensorMinimum(Cell):
    def __init__(self):
        super(ConstScalarAndTensorMinimum, self).__init__()
        self.min = P.Minimum()
        self.x = 20

    def construct(self, y):
        return self.min(self.x, y)


class TwoTensorsMinimum(Cell):
    def __init__(self):
        super(TwoTensorsMinimum, self).__init__()
        self.min = P.Minimum()

    def construct(self, x, y):
        return self.min(x, y)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_minimum_constScalar_tensor_int():
    x = Tensor(np.array([[2, 3, 4], [100, 200, 300]]).astype(np.int32))
    expect = [[2, 3, 4], [20, 20, 20]]

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    min_op = ConstScalarAndTensorMinimum()
    output = min_op(x)
    assert np.all(output.asnumpy() == expect)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_minimum_two_tensors_Not_Broadcast_int():
    prop = 100 if np.random.random() > 0.5 else -100
    x = np.random.randn(3, 4, 5).astype(np.int32) * prop
    y = np.random.randn(3, 4, 5).astype(np.int32) * prop
    expect = np.minimum(x, y).astype(np.int32)

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    min_op = TwoTensorsMinimum()
    output = min_op(Tensor(x), Tensor(y))
    assert np.all(output.asnumpy() == expect)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_minimum_two_tensors_Broadcast_int():
    prop = 100 if np.random.random() > 0.5 else -100
    x = np.random.randn(3, 4, 5).astype(np.int32) * prop
    y = np.random.randn(3, 1, 1).astype(np.int32) * prop
    expect = np.minimum(x, y).astype(np.int32)

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    min_op = TwoTensorsMinimum()
    output = min_op(Tensor(x), Tensor(y))
    assert np.all(output.asnumpy() == expect)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_minimum_two_tensors_Broadcast_oneDimension_int():
    prop = 100 if np.random.random() > 0.5 else -100
    x = np.random.randn(3).astype(np.int32) * prop
    y = np.random.randn(3).astype(np.int32) * prop
    expect = np.minimum(x, y).astype(np.int32)

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    min_op = TwoTensorsMinimum()
    output = min_op(Tensor(x), Tensor(y))
    assert np.all(output.asnumpy() == expect)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_minimum_two_tensors_notBroadcast_all_oneDimension_int():
    x = Tensor(np.array([[2]]).astype(np.int32))
    y = Tensor(np.array([[100]]).astype(np.int32))
    expect = [[2]]

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    min_op = TwoTensorsMinimum()
    output = min_op(x, y)
    assert np.all(output.asnumpy() == expect)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_minimum_two_tensors_notBroadcast_float32():
    prop = 100 if np.random.random() > 0.5 else -100
    x = np.random.randn(3, 4, 5).astype(np.float32) * prop
    y = np.random.randn(3, 4, 5).astype(np.float32) * prop
    expect = np.minimum(x, y).astype(np.float32)
    error = np.ones(shape=expect.shape) * 1.0e-5

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    min_op = TwoTensorsMinimum()
    output = min_op(Tensor(x), Tensor(y))
    diff = output.asnumpy() - expect
    assert np.all(np.abs(diff) < error)
    assert output.shape == expect.shape


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_minimum_two_tensors_notBroadcast_float16():
    prop = 100 if np.random.random() > 0.5 else -100
    x = np.random.randn(3, 4, 5).astype(np.float16) * prop
    y = np.random.randn(3, 4, 5).astype(np.float16) * prop
    expect = np.minimum(x, y).astype(np.float16)
    error = np.ones(shape=expect.shape) * 1.0e-5

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    min_op = TwoTensorsMinimum()
    output = min_op(Tensor(x), Tensor(y))
    diff = output.asnumpy() - expect
    assert np.all(np.abs(diff) < error)
    assert output.shape == expect.shape


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_minimum_two_tensors_Broadcast_float16():
    prop = 100 if np.random.random() > 0.5 else -100
    x = np.random.randn(3, 4, 5).astype(np.float16) * prop
    y = np.random.randn(3, 4, 1).astype(np.float16) * prop
    expect = np.minimum(x, y).astype(np.float16)
    error = np.ones(shape=expect.shape) * 1.0e-5

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    min_op = TwoTensorsMinimum()
    output = min_op(Tensor(x), Tensor(y))
    diff = output.asnumpy() - expect
    assert np.all(np.abs(diff) < error)
    assert output.shape == expect.shape


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_minimum_two_tensors_notBroadcast_float64():
    prop = 100 if np.random.random() > 0.5 else -100
    x = np.random.randn(3, 4, 1).astype(np.float64) * prop
    y = np.random.randn(3, 4, 5).astype(np.float64) * prop
    expect = np.minimum(x, y).astype(np.float64)
    error = np.ones(shape=expect.shape) * 1.0e-5

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    min_op = TwoTensorsMinimum()
    output = min_op(Tensor(x), Tensor(y))
    diff = output.asnumpy() - expect
    assert np.all(np.abs(diff) < error)
    assert output.shape == expect.shape


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_minimum_two_tensors_functional_int8():
    """
    Feature: test minimum on cpu in graph mode
    Description: test interface functional
    Expectation: result match numpy result
    """
    prop = 100 if np.random.random() > 0.5 else 50
    x = np.random.randn(3, 4, 5).astype(np.int8) * prop
    y = np.random.randn(3, 4, 5).astype(np.int8) * prop
    expect = np.minimum(x, y).astype(np.int8)

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    output = ops.minimum(Tensor(x), Tensor(y))
    assert np.all(output.asnumpy() == expect)


class MinimumTensorNet(Cell):
    def construct(self, x, y):
        return x.minimum(y)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_minimum_two_tensors_tensor_graph_uint16():
    """
    Feature: test minimum on cpu in graph mode
    Description: test interface tensor
    Expectation: result match numpy result
    """
    prop = 100 if np.random.random() > 0.5 else 50
    x = np.random.randn(3, 4, 5).astype(np.uint16) * prop
    y = np.random.randn(3, 4, 5).astype(np.uint16) * prop
    expect = np.minimum(x, y).astype(np.uint16)

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    net = MinimumTensorNet()
    output = net(Tensor(x), Tensor(y))
    assert np.all(output.asnumpy() == expect)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_minimum_two_tensors_tensor_pynative_uint8():
    """
    Feature: test minimum on cpu in pynative mode
    Description: test interface tensor
    Expectation: result match numpy result
    """
    prop = 100 if np.random.random() > 0.5 else 50
    x = np.random.randn(3, 4, 5).astype(np.uint8) * prop
    y = np.random.randn(3, 4, 5).astype(np.uint8) * prop
    expect = np.minimum(x, y).astype(np.uint8)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    output = Tensor(x).minimum(Tensor(y))
    assert np.all(output.asnumpy() == expect)
