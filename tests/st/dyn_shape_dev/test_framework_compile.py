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
# pylint: disable=unused-variable
import math
import pytest
import numpy as np
import test_utils

import mindspore as ms
from mindspore import nn, mutable
from mindspore.ops import auto_generate as ops


ms.set_context(jit_syntax_level=ms.STRICT)


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


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@test_utils.run_test_func
def test_avg_pool():
    """
    Feature: DynamicShape.
    Description: Test AvgPool with dynamic shape.
    Expectation: No exception.
    """
    ms.set_context(precompile_only=True, mode=ms.GRAPH_MODE)
    x = ms.Tensor(np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4), ms.float32)
    net = AvgPoolNet()
    out = net(x)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@test_utils.run_test_func
def test_avg_pool_create_instance_const_args():
    """
    Feature: DynamicShape.
    Description: Create AvgPool instance with constant arguaments.
    Expectation: No exception.
    """
    ms.set_context(precompile_only=True, mode=ms.GRAPH_MODE)
    x = ms.Tensor(np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4), ms.float32)
    net = AvgPoolCreateInstanceNet()
    out = net(x, 1, 1, "VALID", "NCHW")


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@test_utils.run_test_func
def test_avg_pool_create_instance_var_args():
    """
    Feature: DynamicShape.
    Description: Create AvgPool instance with variable arguaments.
    Expectation: No exception.
    """
    ms.set_context(precompile_only=True, mode=ms.GRAPH_MODE)
    x = ms.Tensor(np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4), ms.float32)
    net = AvgPoolCreateInstanceNet()
    out = net(x, mutable(1), mutable(1), "VALID", "NCHW")


class PowNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.pow = ops.Pow()

    def construct(self, x, y):
        return self.pow(x, y)


class PowCreateInstanceNet(nn.Cell):
    def construct(self, x, y):
        return ops.Pow()(x, y)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@test_utils.run_test_func
def test_pow_type_cast():
    """
    Feature: DynamicShape.
    Description: Test type conversion for pow.
    Expectation: No exception.
    """
    ms.set_context(precompile_only=True, mode=ms.GRAPH_MODE)
    net = PowNet()
    out = net(ms.Tensor(1), 2)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@test_utils.run_test_func
def test_pow_create_instance_type_cast():
    """
    Feature: DynamicShape.
    Description: Test type conversion for pow.
    Expectation: No exception.
    """
    ms.set_context(precompile_only=True, mode=ms.GRAPH_MODE)
    net = PowCreateInstanceNet()
    out = net(1.0, ms.Tensor(2))


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


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_lstm_ops():
    """
    Feature: DynamicShape.
    Description: LSTM with input (3, 32, 32)
    Expectation: No exception.
    """
    ms.set_context(precompile_only=True, mode=ms.GRAPH_MODE)
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
    out = net(input_ms, h0, c0)
