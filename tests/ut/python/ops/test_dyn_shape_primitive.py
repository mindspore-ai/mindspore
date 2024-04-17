# Copyright 2023-2024 Huawei Technologies Co., Ltd
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
# pylint: disable=unused-variable
import math
import pytest
import numpy as np

import mindspore as ms
from mindspore import nn, mutable
from mindspore.ops import auto_generate as ops


class AvgPoolNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.avg_pool = ops.AvgPool(kernel_size=1, strides=1, pad_mode="VALID", data_format="NCHW")

    def construct(self, x):
        return self.avg_pool(x)


class AvgPoolCreateInstanceNet(nn.Cell):
    def construct(self, x, kernel_size, strides, pad_mode, data_format):
        op = ops.AvgPool(kernel_size, strides, pad_mode, data_format)
        return op(x)


def test_avg_pool():
    """
    Feature: DynamicShape.
    Description: Test AvgPool with dynamic shape.
    Expectation: No exception.
    """
    ms.set_context(mode=ms.GRAPH_MODE, jit_syntax_level=ms.STRICT)
    x = ms.Tensor(np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4), ms.float32)
    AvgPoolNet()(x)


def test_avg_pool_create_instance_const_args():
    """
    Feature: DynamicShape.
    Description: Create AvgPool instance with constant arguaments.
    Expectation: No exception.
    """
    ms.set_context(mode=ms.GRAPH_MODE, jit_syntax_level=ms.STRICT)
    x = ms.Tensor(np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4), ms.float32)
    AvgPoolCreateInstanceNet()(x, 1, 1, "VALID", "NCHW")


def test_avg_pool_create_instance_var_args():
    """
    Feature: DynamicShape.
    Description: Create AvgPool instance with variable arguaments.
    Expectation: No exception.
    """
    ms.set_context(mode=ms.GRAPH_MODE, jit_syntax_level=ms.STRICT)
    x = ms.Tensor(np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4), ms.float32)
    AvgPoolCreateInstanceNet()(x, mutable(1), mutable(1), "VALID", "NCHW")


class PowNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.pow = ops.Pow()

    def construct(self, x, y):
        return self.pow(x, y)


class PowCreateInstanceNet(nn.Cell):
    def construct(self, x, y):
        return ops.Pow()(x, y)


def test_pow_type_cast():
    """
    Feature: DynamicShape.
    Description: Test type conversion for pow.
    Expectation: No exception.
    """
    ms.set_context(mode=ms.GRAPH_MODE, jit_syntax_level=ms.STRICT)
    PowNet()(1, 2)


def test_pow_create_instance_type_cast():
    """
    Feature: DynamicShape.
    Description: Test type conversion for pow.
    Expectation: No exception.
    """
    ms.set_context(mode=ms.GRAPH_MODE, jit_syntax_level=ms.STRICT)
    PowCreateInstanceNet()(1.0, 2)


class LSTM(nn.Cell):
    def __init__(self, input_s, hidden_s, num_layers, has_bias, batch_first, bidirectional, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_s, hidden_size=hidden_s, num_layers=num_layers, has_bias=has_bias,
                            batch_first=batch_first, bidirectional=bidirectional, dropout=dropout)

    def construct(self, inp, h0, c0):
        return self.lstm(inp, (h0, c0))


class LSTMWeightBias():
    def __init__(self, num_layers, has_bias, input_size, num_directions, hidden_size, bidirectional):
        self.num_layers = num_layers
        self.has_bias = has_bias
        self.input_size = input_size
        self.num_directions = num_directions
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

    def get_weight_bias(self):
        gate_size = 4 * self.hidden_size

        w_ih_list = []
        w_hh_list = []
        b_ih_list = []
        b_hh_list = []
        stdv = 1 / math.sqrt(self.hidden_size)
        for layer in range(self.num_layers):
            for direction in range(self.num_directions):
                layer_input_size = self.input_size if layer == 0 else self.hidden_size * self.num_directions
                suffix = '_reverse' if direction == 1 else ''

                w_ih_list.append(ms.Parameter(
                    ms.Tensor(np.random.uniform(-stdv, stdv, (gate_size, layer_input_size)).astype(np.float32)),
                    name='weight_ih_l{}{}'.format(layer, suffix)))
                w_hh_list.append(ms.Parameter(
                    ms.Tensor(np.random.uniform(-stdv, stdv, (gate_size, self.hidden_size)).astype(np.float32)),
                    name='weight_hh_l{}{}'.format(layer, suffix)))
                if self.has_bias:
                    b_ih_list.append(ms.Parameter(
                        ms.Tensor(np.random.uniform(-stdv, stdv, (gate_size)).astype(np.float32)),
                        name='bias_ih_l{}{}'.format(layer, suffix)))
                    b_hh_list.append(ms.Parameter(
                        ms.Tensor(np.random.uniform(-stdv, stdv, (gate_size)).astype(np.float32)),
                        name='bias_hh_l{}{}'.format(layer, suffix)))
        w_ih_list = ms.ParameterTuple(w_ih_list)
        w_hh_list = ms.ParameterTuple(w_hh_list)
        b_ih_list = ms.ParameterTuple(b_ih_list)
        b_hh_list = ms.ParameterTuple(b_hh_list)
        return w_ih_list, w_hh_list, b_ih_list, b_hh_list


def test_lstm_ops():
    """
    Feature: DynamicShape.
    Description: LSTM with input (3, 32, 32)
    Expectation: No exception.
    """
    ms.set_context(mode=ms.GRAPH_MODE, jit_syntax_level=ms.STRICT)
    input_s = 32
    hidden_s = 16
    has_bias = True
    bidirectional = False
    num_layers = 1
    num_directions = 1

    fact = LSTMWeightBias(num_layers, has_bias, input_s, num_directions, hidden_s, bidirectional)
    w_ih_list, w_hh_list, b_ih_list, b_hh_list = fact.get_weight_bias()

    h0 = ms.Tensor(np.random.randn(num_layers * 1, 32, 16).astype(np.float32))
    c0 = ms.Tensor(np.random.randn(num_layers * 1, 32, 16).astype(np.float32))
    input_ms = ms.Tensor(np.random.randn(3, 32, 32).astype(np.float32))

    net = LSTM(input_s=input_s, hidden_s=16, num_layers=num_layers, has_bias=has_bias, batch_first=False,
               bidirectional=bidirectional, dropout=0.0)
    net.lstm.w_ih_list = w_ih_list
    net.lstm.w_hh_list = w_hh_list
    net.lstm.b_ih_list = b_ih_list
    net.lstm.b_hh_list = b_hh_list
    net(input_ms, h0, c0)


def test_op_with_default_init_args():
    """
    Feature: DynamicShape.
    Description: Test default init args.
    Expectation: No exception.
    """
    @ms.jit
    def func(x, size):
        return ops.ResizeNearestNeighbor(size)(x)

    ms.set_context(mode=ms.GRAPH_MODE, jit_syntax_level=ms.STRICT)
    x = ms.Tensor(np.array([[[[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]]]), ms.float32)
    size = ms.mutable((2, 2))
    func(x, size)


def test_add_two_scalar():
    """
    Feature: DynamicShape.
    Description: Test add two scalar.
    Expectation: No exception.
    """
    @ms.jit
    def func(x, y):
        return ops.add(x, y)

    ms.set_context(mode=ms.GRAPH_MODE, jit_syntax_level=ms.STRICT)
    func(2.5, 1)


def test_primitive_init_keyword_argument():
    """
    Feature: DynamicShape.
    Description: Test keyword argument.
    Expectation: No exception.
    """
    @ms.jit
    def func(x, arg):
        return ops.AvgPool(arg, pad_mode="VALID", strides=arg)(x)

    ms.set_context(mode=ms.GRAPH_MODE, jit_syntax_level=ms.STRICT)
    x = ms.Tensor(np.random.rand(10, 36, 12, 12).astype(np.float32))
    func(x, 1)
    func(x, ms.mutable(1))


def test_primitive_call_keyword_argument():
    """
    Feature: DynamicShape.
    Description: Test keyword argument.
    Expectation: Raise TypeError.
    """
    @ms.jit
    def func(x, axis):
        return ops.Softmax(axis)(x=x)

    ms.set_context(mode=ms.GRAPH_MODE, jit_syntax_level=ms.STRICT)
    x = ms.Tensor(np.random.rand(10, 36, 12, 12).astype(np.float32))
    with pytest.raises(TypeError) as info1:
        func(x, (-1,))
    assert "only positional arguments as inputs are supported" in str(info1.value)
    with pytest.raises(TypeError) as info2:
        func(x, ms.mutable((-1,)))
    assert "only positional arguments as inputs are supported" in str(info2.value)


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_acos_unsupported_input_type(mode):
    """
    Feature: DynamicShape.
    Description: Test unsupported input type.
    Expectation: Raise TypeError.
    """
    class ACos(nn.Cell):
        def construct(self, x):
            return ops.acos(x)

    ms.set_context(mode=mode, jit_syntax_level=ms.STRICT)
    with pytest.raises(TypeError) as info:
        ACos()("str")
    assert "Failed calling ACos with \"ACos()(input=string)\"" in str(info.value)


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_diagonal_unsupported_input_type(mode):
    """
    Feature: DynamicShape.
    Description: Test unsupported input type.
    Expectation: Raise TypeError.
    """
    class Diagonal(nn.Cell):
        def construct(self, x, offset, dim1, dim2):
            return ops.diagonal(x, offset, dim1, dim2)

    ms.set_context(mode=mode, jit_syntax_level=ms.STRICT)
    with pytest.raises(TypeError) as info:
        Diagonal()(1.0, 1, 1, -3)
    assert "Failed calling Diagonal with \"Diagonal(offset=int, dim1=int, dim2=int)(input=float)\"" in str(info.value)
