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
'''RNN operators module, include RNN, GRU, LSTM'''
import math
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as P
from mindspore.ops.primitive import constexpr
from mindspore import Tensor, Parameter, ParameterTuple
from mindspore import log as logger
from mindspore import context
from src.rnn_cells import rnn_relu_cell, rnn_tanh_cell, lstm_cell, gru_cell
from src.utils import Reverse, ReverseSequence

@constexpr
def _init_state(shape, dtype, is_lstm):
    hx = Tensor(np.zeros(shape), dtype)
    cx = Tensor(np.zeros(shape), dtype)
    if is_lstm:
        return (hx, cx)
    return hx

class DynamicRNN(nn.Cell):
    '''Dynamic RNN module to compute RNN cell by timesteps'''
    def __init__(self, mode):
        super().__init__()
        if mode == "RNN_RELU":
            cell = rnn_relu_cell
        elif mode == "RNN_TANH":
            cell = rnn_tanh_cell
        elif mode == "LSTM":
            cell = lstm_cell
        elif mode == "GRU":
            cell = gru_cell
        else:
            raise ValueError("Unrecognized RNN mode: " + mode)
        self.cell = cell
        self.is_lstm = mode == "LSTM"

    def recurrent(self, x, h_0, w_ih, w_hh, b_ih, b_hh):
        '''recurrent steps without sequence length'''
        time_step = x.shape[0]
        outputs = []
        t = 0
        h = h_0
        while t < time_step:
            x_t = x[t:t+1:1]
            x_t = P.Squeeze(0)(x_t)
            h = self.cell(x_t, h, w_ih, w_hh, b_ih, b_hh)
            if self.is_lstm:
                outputs.append(h[0])
            else:
                outputs.append(h)
            t += 1
        outputs = P.Stack()(outputs)
        return outputs, h

    def variable_recurrent(self, x, h, seq_length, w_ih, w_hh, b_ih, b_hh):
        '''recurrent steps with sequence length'''
        time_step = x.shape[0]
        h_t = h
        if self.is_lstm:
            hidden_size = h[0].shape[-1]
            zero_output = P.ZerosLike()(h_t[0])
        else:
            hidden_size = h.shape[-1]
            zero_output = P.ZerosLike()(h_t)
        seq_length = P.Cast()(seq_length, mindspore.float32)
        seq_length = P.BroadcastTo((hidden_size, -1))(seq_length)
        seq_length = P.Cast()(seq_length, mindspore.int32)
        seq_length = P.Transpose()(seq_length, (1, 0))

        outputs = []
        state_t = h_t
        t = 0
        while t < time_step:
            x_t = x[t:t+1:1]
            x_t = P.Squeeze(0)(x_t)
            h_t = self.cell(x_t, state_t, w_ih, w_hh, b_ih, b_hh)
            seq_cond = seq_length > t
            if self.is_lstm:
                state_t_0 = P.Select()(seq_cond, h_t[0], state_t[0])
                state_t_1 = P.Select()(seq_cond, h_t[1], state_t[1])
                output = P.Select()(seq_cond, h_t[0], zero_output)
                state_t = (state_t_0, state_t_1)
            else:
                state_t = P.Select()(seq_cond, h_t, state_t)
                output = P.Select()(seq_cond, h_t, zero_output)
            outputs.append(output)
            t += 1
        outputs = P.Stack()(outputs)
        return outputs, state_t

    def construct(self, x, h, seq_length, w_ih, w_hh, b_ih, b_hh):
        if seq_length is None:
            return self.recurrent(x, h, w_ih, w_hh, b_ih, b_hh)
        return self.variable_recurrent(x, h, seq_length, w_ih, w_hh, b_ih, b_hh)

class RNNBase(nn.Cell):
    '''Basic class for RNN operators'''
    def __init__(self, mode, input_size, hidden_size, num_layers=1, has_bias=True,
                 batch_first=False, dropout=0, bidirectional=False):
        super().__init__()
        if not 0 <= dropout <= 1:
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")

        if dropout > 0 and num_layers == 1:
            logger.warning("dropout option adds dropout after all but last "
                           "recurrent layer, so non-zero dropout expects "
                           "num_layers greater than 1, but got dropout={} and "
                           "num_layers={}".format(dropout, num_layers))
        if mode == "LSTM":
            gate_size = 4 * hidden_size
        elif mode == "GRU":
            gate_size = 3 * hidden_size
        elif mode == "RNN_TANH":
            gate_size = hidden_size
        elif mode == "RNN_RELU":
            gate_size = hidden_size
        else:
            raise ValueError("Unrecognized RNN mode: " + mode)

        self.is_ascend = context.get_context("device_target") == "Ascend"
        if self.is_ascend:
            self.reverse = P.ReverseV2([0])
            self.reverse_sequence = P.ReverseSequence(0, 1)
        else:
            self.reverse = Reverse(0)
            self.reverse_sequence = ReverseSequence(0, 1)
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.num_layers = num_layers
        self.dropout = dropout
        self.dropout_op = nn.Dropout(float(1 - dropout))
        self.bidirectional = bidirectional
        self.has_bias = has_bias
        self.rnn = DynamicRNN(mode)
        num_directions = 2 if bidirectional else 1
        self.is_lstm = mode == "LSTM"

        self.w_ih_list = []
        self.w_hh_list = []
        self.b_ih_list = []
        self.b_hh_list = []
        stdv = 1 / math.sqrt(self.hidden_size)
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions
                suffix = '_reverse' if direction == 1 else ''

                self.w_ih_list.append(Parameter(
                    Tensor(np.random.uniform(-stdv, stdv, (gate_size, layer_input_size)).astype(np.float32)),
                    name='weight_ih_l{}{}'.format(layer, suffix)))
                self.w_hh_list.append(Parameter(
                    Tensor(np.random.uniform(-stdv, stdv, (gate_size, hidden_size)).astype(np.float32)),
                    name='weight_hh_l{}{}'.format(layer, suffix)))
                if has_bias:
                    self.b_ih_list.append(Parameter(
                        Tensor(np.random.uniform(-stdv, stdv, (gate_size)).astype(np.float32)),
                        name='bias_ih_l{}{}'.format(layer, suffix)))
                    self.b_hh_list.append(Parameter(
                        Tensor(np.random.uniform(-stdv, stdv, (gate_size)).astype(np.float32)),
                        name='bias_hh_l{}{}'.format(layer, suffix)))
        self.w_ih_list = ParameterTuple(self.w_ih_list)
        self.w_hh_list = ParameterTuple(self.w_hh_list)
        self.b_ih_list = ParameterTuple(self.b_ih_list)
        self.b_hh_list = ParameterTuple(self.b_hh_list)

    def _stacked_bi_dynamic_rnn(self, x, h, seq_length):
        """stacked bidirectional dynamic_rnn"""
        pre_layer = x
        h_n = ()
        c_n = ()
        output = 0
        for i in range(self.num_layers):
            offset = i * 2
            if self.has_bias:
                w_f_ih, w_f_hh, b_f_ih, b_f_hh = \
                    self.w_ih_list[offset], self.w_hh_list[offset], \
                    self.b_ih_list[offset], self.b_hh_list[offset]
                w_b_ih, w_b_hh, b_b_ih, b_b_hh = \
                    self.w_ih_list[offset + 1], self.w_hh_list[offset + 1], \
                    self.b_ih_list[offset + 1], self.b_hh_list[offset + 1]
            else:
                w_f_ih, w_f_hh = self.w_ih_list[offset], self.w_hh_list[offset]
                w_b_ih, w_b_hh = self.w_ih_list[offset + 1], self.w_hh_list[offset + 1]
                b_f_ih, b_f_hh, b_b_ih, b_b_hh = None, None, None, None
            if self.is_lstm:
                h_f_i = (h[0][offset], h[1][offset])
                h_b_i = (h[0][offset + 1], h[1][offset + 1])
            else:
                h_f_i = h[offset]
                h_b_i = h[offset + 1]
            if seq_length is None:
                x_b = self.reverse(pre_layer)
            else:
                x_b = self.reverse_sequence(pre_layer, seq_length)
            output_f, h_t_f = self.rnn(pre_layer, h_f_i, seq_length, w_f_ih, w_f_hh, b_f_ih, b_f_hh)
            output_b, h_t_b = self.rnn(x_b, h_b_i, seq_length, w_b_ih, w_b_hh, b_b_ih, b_b_hh)
            if seq_length is None:
                output_b = self.reverse(output_b)
            else:
                output_b = self.reverse_sequence(output_b, seq_length)
            output = P.Concat(2)((output_f, output_b))
            pre_layer = self.dropout_op(output) if (self.dropout != 0 and i < self.num_layers - 1) else output
            if self.is_lstm:
                h_n += (h_t_f[0], h_t_b[0],)
                c_n += (h_t_f[1], h_t_b[1],)
            else:
                h_n += (h_t_f, h_t_b,)
        if self.is_lstm:
            h_n = P.Concat(0)(h_n)
            c_n = P.Concat(0)(c_n)
            h_n = h_n.view(h[0].shape)
            c_n = c_n.view(h[1].shape)
            return output, (h_n.view(h[0].shape), c_n.view(h[1].shape))
        h_n = P.Concat(0)(h_n)
        return output, h_n.view(h.shape)

    def _stacked_dynamic_rnn(self, x, h, seq_length):
        """stacked mutil_layer dynamic_rnn"""
        pre_layer = x
        h_n = ()
        c_n = ()
        output = 0
        for i in range(self.num_layers):
            if self.has_bias:
                w_ih, w_hh, b_ih, b_hh = self.w_ih_list[i], self.w_hh_list[i], self.b_ih_list[i], self.b_hh_list[i]
            else:
                w_ih, w_hh = self.w_ih_list[i], self.w_hh_list[i]
                b_ih, b_hh = None, None
            if self.is_lstm:
                h_i = (h[0][i], h[1][i])
            else:
                h_i = h[i]
            output, h_t = self.rnn(pre_layer, h_i, seq_length, w_ih, w_hh, b_ih, b_hh)
            pre_layer = self.dropout_op(output) if (self.dropout != 0 and i < self.num_layers - 1) else output
            if self.is_lstm:
                h_n += (h_t[0],)
                c_n += (h_t[1],)
            else:
                h_n += (h_t,)
        if self.is_lstm:
            h_n = P.Concat(0)(h_n)
            c_n = P.Concat(0)(c_n)
            h_n = h_n.view(h[0].shape)
            c_n = c_n.view(h[1].shape)
            return output, (h_n.view(h[0].shape), c_n.view(h[1].shape))
        h_n = P.Concat(0)(h_n)
        return output, h_n.view(h.shape)

    def construct(self, x, h=None, seq_length=None):
        '''Defines the RNN like operators performed'''
        max_batch_size = x.shape[0] if self.batch_first else x.shape[1]
        num_directions = 2 if self.bidirectional else 1
        if h is None:
            h = _init_state((self.num_layers * num_directions, max_batch_size, self.hidden_size), x.dtype, self.is_lstm)
        if self.batch_first:
            x = P.Transpose()(x, (1, 0, 2))
        if self.bidirectional:
            x, h = self._stacked_bi_dynamic_rnn(x, h, seq_length)
        else:
            x, h = self._stacked_dynamic_rnn(x, h, seq_length)
        if self.batch_first:
            x = P.Transpose()(x, (1, 0, 2))
        return x, h

class RNN(RNNBase):
    '''RNN operator class'''
    def __init__(self, *args, **kwargs):
        if 'nonlinearity' in kwargs:
            if kwargs['nonlinearity'] == 'tanh':
                mode = 'RNN_TANH'
            elif kwargs['nonlinearity'] == 'relu':
                mode = 'RNN_RELU'
            else:
                raise ValueError("Unknown nonlinearity '{}'".format(
                    kwargs['nonlinearity']))
            del kwargs['nonlinearity']
        else:
            mode = 'RNN_TANH'

        super(RNN, self).__init__(mode, *args, **kwargs)

class GRU(RNNBase):
    '''GRU operator class'''
    def __init__(self, *args, **kwargs):
        mode = 'GRU'
        super(GRU, self).__init__(mode, *args, **kwargs)

class LSTM(RNNBase):
    '''LSTM operator class'''
    def __init__(self, *args, **kwargs):
        mode = 'LSTM'
        super(LSTM, self).__init__(mode, *args, **kwargs)
        self.support_non_tensor_inputs = True
