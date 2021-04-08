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
# ==============================================================================

import math
import pytest
import numpy as np
from mindspore import context
from mindspore import nn
from mindspore import Tensor
from mindspore.common.initializer import initializer
from mindspore.common.parameter import ParameterTuple
from mindspore.common.parameter import Parameter
from mindspore.ops import composite as c


class GradOfAllInputsAndParams(nn.Cell):
    def __init__(self, network, sens_param):
        super().__init__()
        self.grad = c.GradOperation(get_all=True, get_by_list=True, sens_param=sens_param)
        self.network = network
        self.params = ParameterTuple(self.network.trainable_params())

    def construct(self, *inputs):
        gout = self.grad(self.network, self.params)(*inputs)
        return gout


class LSTM(nn.Cell):
    def __init__(self, input_s, hidden_s, num_layers, has_bias, batch_first, bidirectional, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_s, hidden_size=hidden_s, num_layers=num_layers, has_bias=has_bias,
                            batch_first=batch_first, bidirectional=bidirectional, dropout=dropout)

    def construct(self, inp, h0, c0):
        return self.lstm(inp, (h0, c0))


class LSTMWeightBias():
    def __init__(self, num_layers, has_bias, input_s, num_directions, hidden_s, bidirectional):
        self.num_layers = num_layers
        self.has_bias = has_bias
        self.input_s = input_s
        self.num_directions = num_directions
        self.hidden_s = hidden_s
        self.bidirectional = bidirectional

    def get_weight_bias(self):
        stdv = 1 / math.sqrt(self.hidden_s)
        gate_size = 4 * self.hidden_s
        w_list_value = []
        b_list_value = []

        for i in range(self.num_layers):
            b0 = np.zeros(gate_size, dtype=np.float16)
            w_shape = self.input_s if i == 0 else (self.num_directions * self.hidden_s)
            w_np = np.random.uniform(-stdv, stdv, (w_shape + self.hidden_s, gate_size)).astype(np.float16)
            w_list_value.append(Parameter(initializer(Tensor(w_np), [w_shape + self.hidden_s, gate_size]),
                                          name="weight_fw" + str(i)))

            if self.has_bias:
                b_np = np.random.uniform(-stdv, stdv, gate_size).astype(np.float16)
                b_list_value.append(Parameter(initializer(Tensor(b_np), [gate_size]), name="bias_fw" + str(i)))
            else:
                b_list_value.append(Parameter(initializer(Tensor(b0), [gate_size]), name="bias_fw" + str(i)))

            if self.bidirectional:
                w_bw_np = np.random.uniform(-stdv, stdv, (w_shape + self.hidden_s, gate_size)).astype(np.float16)
                b_list_value.append(Parameter(initializer(Tensor(w_bw_np), [w_shape + self.hidden_s, gate_size]),
                                              name="weight_bw" + str(i)))
                b_bw_np = np.random.uniform(-stdv, stdv, (4 * self.hidden_s)).astype(
                    np.float16) if self.has_bias else b0
                b_list_value.append(Parameter(initializer(Tensor(b_bw_np), [gate_size]), name="bias_bw" + str(i)))
        w_list_value = ParameterTuple(w_list_value)
        b_list_value = ParameterTuple(b_list_value)
        return w_list_value, b_list_value


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sit_lstm_forward_input_3_32_32_is_32_hs_16():
    input_s = 32
    hidden_s = 16
    has_bias = True
    bidirectional = False
    num_layers = 1
    num_directions = 1

    fact = LSTMWeightBias(num_layers, has_bias, input_s, num_directions, hidden_s, bidirectional)
    w_list_value, b_list_value = fact.get_weight_bias()

    h0 = Tensor(np.random.randn(num_layers * 1, 32, 16).astype(np.float32))
    c0 = Tensor(np.random.randn(num_layers * 1, 32, 16).astype(np.float32))
    input_ms = Tensor(np.random.randn(3, 32, 32).astype(np.float32))

    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    net = LSTM(input_s=input_s, hidden_s=16, num_layers=num_layers, has_bias=has_bias, batch_first=False,
               bidirectional=bidirectional, dropout=0.0)
    net.lstm.w_list = w_list_value
    net.lstm.b_list = b_list_value
    out, (hy, cy) = net(input_ms, h0, c0)

    # pynative mode
    context.set_context(mode=context.PYNATIVE_MODE)
    net_pynative = LSTM(input_s=input_s, hidden_s=16, num_layers=num_layers, has_bias=has_bias, batch_first=False,
                        bidirectional=bidirectional, dropout=0.0)
    net_pynative.lstm.w_list = w_list_value
    net_pynative.lstm.b_list = b_list_value
    out_pynative, (hy_pynative, cy_pynative) = net_pynative(input_ms, h0, c0)

    assert np.allclose(out.asnumpy(), out_pynative.asnumpy(), 0.0001, 0.0001)
    assert np.allclose(hy.asnumpy(), hy_pynative.asnumpy(), 0.0001, 0.0001)
    assert np.allclose(cy.asnumpy(), cy_pynative.asnumpy(), 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_sit_lstm_grad_input_3_32_32_is_32_hs_16():
    input_s = 32
    hidden_s = 16
    has_bias = True
    bidirectional = False
    num_layers = 1
    num_directions = 1

    fact = LSTMWeightBias(num_layers, has_bias, input_s, num_directions, hidden_s, bidirectional)
    w_list_value, b_list_value = fact.get_weight_bias()

    h0 = Tensor(np.random.randn(num_layers * 1, 32, 16).astype(np.float32))
    c0 = Tensor(np.random.randn(num_layers * 1, 32, 16).astype(np.float32))
    input_ms = Tensor(np.random.randn(3, 32, 32).astype(np.float32))

    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    net = LSTM(input_s=input_s, hidden_s=16, num_layers=num_layers, has_bias=has_bias, batch_first=False,
               bidirectional=bidirectional, dropout=0.0)
    net.lstm.w_list = w_list_value
    net.lstm.b_list = b_list_value

    grad_net_inp = GradOfAllInputsAndParams(net, sens_param=False)
    grad_net_inp.set_train()
    out_grad, _ = grad_net_inp(input_ms, h0, c0)
    x_grad = out_grad[0].asnumpy()
    h_grad = out_grad[1].asnumpy()
    c_grad = out_grad[2].asnumpy()

    # pynative mode
    context.set_context(mode=context.PYNATIVE_MODE)
    net_pynative = LSTM(input_s=input_s, hidden_s=16, num_layers=num_layers, has_bias=has_bias, batch_first=False,
                        bidirectional=bidirectional, dropout=0.0)
    net_pynative.lstm.w_list = w_list_value
    net_pynative.lstm.b_list = b_list_value

    grad_net_inp_pynative = GradOfAllInputsAndParams(net_pynative, sens_param=False)
    grad_net_inp_pynative.set_train()
    out_grad_pynative, _ = grad_net_inp_pynative(input_ms, h0, c0)
    x_grad_pynative = out_grad_pynative[0].asnumpy()
    h_grad_pynative = out_grad_pynative[1].asnumpy()
    c_grad_pynative = out_grad_pynative[2].asnumpy()

    assert np.allclose(x_grad, x_grad_pynative, 0.0001, 0.0001)
    assert np.allclose(h_grad, h_grad_pynative, 0.0001, 0.0001)
    assert np.allclose(c_grad, c_grad_pynative, 0.0001, 0.0001)
