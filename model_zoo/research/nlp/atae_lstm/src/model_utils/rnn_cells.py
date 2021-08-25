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
"""LSTM cells"""
import math
import numpy as np

import mindspore
from mindspore import nn, Tensor, Parameter, ops as P
from mindspore.common.initializer import initializer, Uniform


def rnn_tanh_cell(input_x, hidden, w_ih, w_hh, b_ih, b_hh):
    if b_ih is None:
        igates = P.MatMul(False, True)(input_x, w_ih)
        hgates = P.MatMul(False, True)(hidden, w_hh)
    else:
        igates = P.MatMul(False, True)(input_x, w_ih) + b_ih
        hgates = P.MatMul(False, True)(hidden, w_hh) + b_hh
    return P.Tanh()(igates + hgates)

def rnn_relu_cell(input_x, hidden, w_ih, w_hh, b_ih, b_hh):
    if b_ih is None:
        igates = P.MatMul(False, True)(input_x, w_ih)
        hgates = P.MatMul(False, True)(hidden, w_hh)
    else:
        igates = P.MatMul(False, True)(input_x, w_ih) + b_ih
        hgates = P.MatMul(False, True)(hidden, w_hh) + b_hh
    return P.ReLU()(igates + hgates)

def lstm_cell(input_x, hidden, w_ih, w_hh, b_ih, b_hh):
    """lstm cell"""
    hx, cx = hidden
    if b_ih is None:
        gates = P.MatMul(False, True)(input_x, w_ih) + P.MatMul(False, True)(hx, w_hh)
    else:
        w_ih = P.Cast()(w_ih, mindspore.float16)
        w_hh = P.Cast()(w_hh, mindspore.float16)
        input_x = P.Cast()(input_x, mindspore.float16)
        hx = P.Cast()(hx, mindspore.float16)
        gates = P.MatMul(False, True)(input_x, w_ih) + P.MatMul(False, True)(hx, w_hh) + b_ih + b_hh
        gates = P.Cast()(gates, mindspore.float16)
    ingate, forgetgate, cellgate, outgate = P.Split(1, 4)(gates)

    ingate = P.Sigmoid()(ingate)
    forgetgate = P.Sigmoid()(forgetgate)
    cellgate = P.Tanh()(cellgate)
    outgate = P.Sigmoid()(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * P.Tanh()(cy)

    return hy, cy

def gru_cell(input_x, hidden, w_ih, w_hh, b_ih, b_hh):
    """gru cell"""
    if b_ih is None:
        gi = P.MatMul(False, True)(input_x, w_ih)
        gh = P.MatMul(False, True)(hidden, w_hh)
    else:
        gi = P.MatMul(False, True)(input_x, w_ih) + b_ih
        gh = P.MatMul(False, True)(hidden, w_hh) + b_hh
    i_r, i_i, i_n = P.Split(1, 3)(gi)
    h_r, h_i, h_n = P.Split(1, 3)(gh)

    resetgate = P.Sigmoid()(i_r + h_r)
    inputgate = P.Sigmoid()(i_i + h_i)
    newgate = P.Tanh()(i_n + resetgate * h_n)
    hy = newgate + inputgate * (hidden - newgate)

    return hy


class RNNCellBase(nn.Cell):
    """RNN cell"""
    def __init__(self, input_size, hidden_size, bias, num_chunks):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(Tensor(np.random.randn(num_chunks * hidden_size, input_size).astype(np.float16)))
        self.weight_hh = Parameter(Tensor(np.random.randn(num_chunks * hidden_size, hidden_size).astype(np.float16)))
        if bias:
            self.bias_ih = Parameter(Tensor(np.random.randn(num_chunks * hidden_size).astype(np.float16)))
            self.bias_hh = Parameter(Tensor(np.random.randn(num_chunks * hidden_size).astype(np.float16)))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1 / math.sqrt(self.hidden_size)
        for weight in self.get_parameters():
            weight.set_data(initializer(Uniform(stdv), weight.shape))

class RNNCell(RNNCellBase):
    """RNN cell"""
    _non_linearity = ['tanh', 'relu']
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh"):
        super().__init__(input_size, hidden_size, bias, num_chunks=1)
        self.nonlinearity = nonlinearity
        if nonlinearity not in self._non_linearity:
            raise ValueError("Unknown nonlinearity: {}".format(self.nonlinearity))

    def construct(self, input_x, hx):
        if self.nonlinearity == "tanh":
            ret = rnn_tanh_cell(input_x, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)
        else:
            ret = rnn_relu_cell(input_x, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)
        return ret

class LSTMCell(RNNCellBase):
    """lstm cell"""
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__(input_size, hidden_size, bias, num_chunks=4)
        self.support_non_tensor_inputs = True

    def construct(self, input_x, hx):
        return lstm_cell(input_x, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)

class GRUCell(RNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__(input_size, hidden_size, bias, num_chunks=3)

    def construct(self, input_x, hx):
        return gru_cell(input_x, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)
