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
'''RNN Cells module, include RNNCell, GRUCell, LSTMCell'''
import math
import numpy as np
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import Tensor, Parameter
from mindspore.common.initializer import initializer, Uniform

def rnn_tanh_cell(inputs, hidden, w_ih, w_hh, b_ih, b_hh):
    '''RNN cell function with tanh activation'''
    if b_ih is None:
        igates = P.MatMul(False, True)(inputs, w_ih)
        hgates = P.MatMul(False, True)(hidden, w_hh)
    else:
        igates = P.MatMul(False, True)(inputs, w_ih) + b_ih
        hgates = P.MatMul(False, True)(hidden, w_hh) + b_hh
    return P.Tanh()(igates + hgates)

def rnn_relu_cell(inputs, hidden, w_ih, w_hh, b_ih, b_hh):
    '''RNN cell function with relu activation'''
    if b_ih is None:
        igates = P.MatMul(False, True)(inputs, w_ih)
        hgates = P.MatMul(False, True)(hidden, w_hh)
    else:
        igates = P.MatMul(False, True)(inputs, w_ih) + b_ih
        hgates = P.MatMul(False, True)(hidden, w_hh) + b_hh
    return P.ReLU()(igates + hgates)

def lstm_cell(inputs, hidden, w_ih, w_hh, b_ih, b_hh):
    '''LSTM cell function'''
    hx, cx = hidden
    if b_ih is None:
        gates = P.MatMul(False, True)(inputs, w_ih) + P.MatMul(False, True)(hx, w_hh)
    else:
        gates = P.MatMul(False, True)(inputs, w_ih) + P.MatMul(False, True)(hx, w_hh) + b_ih + b_hh
    ingate, forgetgate, cellgate, outgate = P.Split(1, 4)(gates)

    ingate = P.Sigmoid()(ingate)
    forgetgate = P.Sigmoid()(forgetgate)
    cellgate = P.Tanh()(cellgate)
    outgate = P.Sigmoid()(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * P.Tanh()(cy)

    return hy, cy

def gru_cell(inputs, hidden, w_ih, w_hh, b_ih, b_hh):
    '''GRU cell function'''
    if b_ih is None:
        gi = P.MatMul(False, True)(inputs, w_ih)
        gh = P.MatMul(False, True)(hidden, w_hh)
    else:
        gi = P.MatMul(False, True)(inputs, w_ih) + b_ih
        gh = P.MatMul(False, True)(hidden, w_hh) + b_hh
    i_r, i_i, i_n = P.Split(1, 3)(gi)
    h_r, h_i, h_n = P.Split(1, 3)(gh)

    resetgate = P.Sigmoid()(i_r + h_r)
    inputgate = P.Sigmoid()(i_i + h_i)
    newgate = P.Tanh()(i_n + resetgate * h_n)
    hy = newgate + inputgate * (hidden - newgate)

    return hy

class RNNCellBase(nn.Cell):
    '''Basic class for RNN Cells'''
    def __init__(self, input_size: int, hidden_size: int, bias: bool, num_chunks: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(Tensor(np.random.randn(num_chunks * hidden_size, input_size).astype(np.float32)))
        self.weight_hh = Parameter(Tensor(np.random.randn(num_chunks * hidden_size, hidden_size).astype(np.float32)))
        if bias:
            self.bias_ih = Parameter(Tensor(np.random.randn(num_chunks * hidden_size).astype(np.float32)))
            self.bias_hh = Parameter(Tensor(np.random.randn(num_chunks * hidden_size).astype(np.float32)))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1 / math.sqrt(self.hidden_size)
        for weight in self.get_parameters():
            weight.set_data(initializer(Uniform(stdv), weight.shape))

class RNNCell(RNNCellBase):
    '''RNNCell operator class'''
    _non_linearity = ['tanh', 'relu']
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, nonlinearity: str = "tanh"):
        super().__init__(input_size, hidden_size, bias, num_chunks=1)
        if nonlinearity not in self._non_linearity:
            raise ValueError("Unknown nonlinearity: {}".format(nonlinearity))
        self.nonlinearity = nonlinearity

    def construct(self, inputs, hx):
        if self.nonlinearity == "tanh":
            ret = rnn_tanh_cell(inputs, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)
        else:
            ret = rnn_relu_cell(inputs, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)
        return ret

class LSTMCell(RNNCellBase):
    '''LSTMCell operator class'''
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__(input_size, hidden_size, bias, num_chunks=4)
        self.support_non_tensor_inputs = True

    def construct(self, inputs, hx):
        return lstm_cell(inputs, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)

class GRUCell(RNNCellBase):
    '''GRUCell operator class'''
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__(input_size, hidden_size, bias, num_chunks=3)

    def construct(self, inputs, hx):
        return gru_cell(inputs, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)
