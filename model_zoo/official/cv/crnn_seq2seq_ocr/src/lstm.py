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
#" ============================================================================
"""lstm"""
import math
import numpy as np
from mindspore import nn, context, Tensor, Parameter, ParameterTuple
import mindspore.common.dtype as mstype
from mindspore.common.initializer import initializer
from mindspore.ops.primitive import constexpr
from mindspore.ops import operations as P


@constexpr
def _create_sequence_length(shape):
    num_step, batch_size, _ = shape
    sequence_length = Tensor(np.ones(batch_size, np.int32) * num_step, mstype.int32)
    return sequence_length

class LSTM(nn.Cell):
    """
    Stacked LSTM (Long Short-Term Memory) layers.

    Args:
        input_size (int): Number of features of input.
        hidden_size (int):  Number of features of hidden layer.
        num_layers (int): Number of layers of stacked LSTM . Default: 1.
        has_bias (bool): Whether the cell has bias `b_ih` and `b_hh`. Default: True.
        batch_first (bool): Specifies whether the first dimension of input is batch_size. Default: False.
        dropout (float, int): If not 0, append `Dropout` layer on the outputs of each
            LSTM layer except the last layer. Default 0. The range of dropout is [0.0, 1.0].
        bidirectional (bool): Specifies whether it is a bidirectional LSTM. Default: False.

    Inputs:
        - **input** (Tensor) - Tensor of shape (seq_len, batch_size, `input_size`) or
          (batch_size, seq_len, `input_size`).
        - **hx** (tuple) - A tuple of two Tensors (h_0, c_0) both of data type mindspore.float32 or
          mindspore.float16 and shape (num_directions * `num_layers`, batch_size, `hidden_size`).
          Data type of `hx` must be the same as `input`.

    Outputs:
        Tuple, a tuple contains (`output`, (`h_n`, `c_n`)).

        - **output** (Tensor) - Tensor of shape (seq_len, batch_size, num_directions * `hidden_size`).
        - **hx_n** (tuple) - A tuple of two Tensor (h_n, c_n) both of shape
          (num_directions * `num_layers`, batch_size, `hidden_size`).
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 has_bias=True,
                 batch_first=False,
                 dropout=0,
                 bidirectional=False):
        super(LSTM, self).__init__()
        self.is_ascend = context.get_context("device_target") == "Ascend"

        self.batch_first = batch_first
        self.transpose = P.Transpose()
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.lstm = P.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           has_bias=has_bias,
                           bidirectional=bidirectional,
                           dropout=float(dropout))

        weight_size = 0
        gate_size = 4 * hidden_size
        stdv = 1 / math.sqrt(hidden_size)
        num_directions = 2 if bidirectional else 1
        if self.is_ascend:
            self.reverse_seq = P.ReverseSequence(batch_dim=1, seq_dim=0)
            self.concat = P.Concat(axis=0)
            self.concat_2dim = P.Concat(axis=2)
            self.cast = P.Cast()
            self.shape = P.Shape()
            if dropout < 0 or dropout > 1:
                raise ValueError("For LSTM, dropout must be a number in range [0, 1], but got {}".format(dropout))
            if dropout == 1:
                self.dropout_op = P.ZerosLike()
            else:
                self.dropout_op = nn.Dropout(float(1 - dropout))
            b0 = np.zeros(gate_size, dtype=np.float32)
            self.w_list = []
            self.b_list = []
            self.rnns_fw = P.DynamicRNN(forget_bias=0.0)
            self.rnns_bw = P.DynamicRNN(forget_bias=0.0)

            for layer in range(num_layers):
                w_shape = input_size if layer == 0 else (num_directions * hidden_size)
                w_np = np.random.uniform(-stdv, stdv, (w_shape + hidden_size, gate_size)).astype(np.float32)
                self.w_list.append(Parameter(
                    initializer(Tensor(w_np), [w_shape + hidden_size, gate_size]), name='weight_fw' + str(layer)))
                if has_bias:
                    b_np = np.random.uniform(-stdv, stdv, gate_size).astype(np.float32)
                    self.b_list.append(Parameter(initializer(Tensor(b_np), [gate_size]), name='bias_fw' + str(layer)))
                else:
                    self.b_list.append(Parameter(initializer(Tensor(b0), [gate_size]), name='bias_fw' + str(layer)))
                if bidirectional:
                    w_bw_np = np.random.uniform(-stdv, stdv, (w_shape + hidden_size, gate_size)).astype(np.float32)
                    self.w_list.append(Parameter(initializer(Tensor(w_bw_np), [w_shape + hidden_size, gate_size]),
                                                 name='weight_bw' + str(layer)))
                    b_bw_np = np.random.uniform(-stdv, stdv, (4 * hidden_size)).astype(np.float32) if has_bias else b0
                    self.b_list.append(Parameter(initializer(Tensor(b_bw_np), [gate_size]),
                                                 name='bias_bw' + str(layer)))
            self.w_list = ParameterTuple(self.w_list)
            self.b_list = ParameterTuple(self.b_list)
        else:
            for layer in range(num_layers):
                input_layer_size = input_size if layer == 0 else hidden_size * num_directions
                increment_size = gate_size * input_layer_size
                increment_size += gate_size * hidden_size
                if has_bias:
                    increment_size += 2 * gate_size
                weight_size += increment_size * num_directions
            w_np = np.random.uniform(-stdv, stdv, (weight_size, 1, 1)).astype(np.float32)
            self.weight = Parameter(initializer(Tensor(w_np), [weight_size, 1, 1]), name='weight')

    def _stacked_bi_dynamic_rnn(self, x, init_h, init_c, weight, bias):
        """stacked bidirectional dynamic_rnn"""
        x_shape = self.shape(x)
        sequence_length = _create_sequence_length(x_shape)
        pre_layer = x
        hn = ()
        cn = ()
        output = x
        for i in range(self.num_layers):
            offset = i * 2
            weight_fw, weight_bw = weight[offset], weight[offset + 1]
            bias_fw, bias_bw = bias[offset], bias[offset + 1]
            init_h_fw, init_h_bw = init_h[offset:offset + 1, :, :], init_h[offset + 1:offset + 2, :, :]
            init_c_fw, init_c_bw = init_c[offset:offset + 1, :, :], init_c[offset + 1:offset + 2, :, :]
            bw_x = self.reverse_seq(pre_layer, sequence_length)
            y, h, c, _, _, _, _, _ = self.rnns_fw(pre_layer, weight_fw, bias_fw, None, init_h_fw, init_c_fw)
            y_bw, h_bw, c_bw, _, _, _, _, _ = self.rnns_bw(bw_x, weight_bw, bias_bw, None, init_h_bw, init_c_bw)
            y_bw = self.reverse_seq(y_bw, sequence_length)
            output = self.concat_2dim((y, y_bw))
            pre_layer = self.dropout_op(output) if self.dropout else output
            hn += (h[-1:, :, :],)
            hn += (h_bw[-1:, :, :],)
            cn += (c[-1:, :, :],)
            cn += (c_bw[-1:, :, :],)
        status_h = self.concat(hn)
        status_c = self.concat(cn)
        return output, status_h, status_c

    def _stacked_dynamic_rnn(self, x, init_h, init_c, weight, bias):
        """stacked mutil_layer dynamic_rnn"""
        pre_layer = x
        hn = ()
        cn = ()
        y = 0
        for i in range(self.num_layers):
            weight_fw, bias_bw = weight[i], bias[i]
            init_h_fw, init_c_bw = init_h[i:i + 1, :, :], init_c[i:i + 1, :, :]
            y, h, c, _, _, _, _, _ = self.rnns_fw(pre_layer, weight_fw, bias_bw, None, init_h_fw, init_c_bw)
            pre_layer = self.dropout_op(y) if self.dropout else y
            hn += (h[-1:, :, :],)
            cn += (c[-1:, :, :],)
        status_h = self.concat(hn)
        status_c = self.concat(cn)
        return y, status_h, status_c

    def construct(self, x, hx):
        if self.batch_first:
            x = self.transpose(x, (1, 0, 2))
        h, c = hx
        if self.is_ascend:
            x = self.cast(x, mstype.float16)
            h = self.cast(h, mstype.float16)
            c = self.cast(c, mstype.float16)
            if self.bidirectional:
                x, h, c = self._stacked_bi_dynamic_rnn(x, h, c, self.w_list, self.b_list)
            else:
                x, h, c = self._stacked_dynamic_rnn(x, h, c, self.w_list, self.b_list)
        else:
            x, h, c, _, _ = self.lstm(x, h, c, self.weight)
        if self.batch_first:
            x = self.transpose(x, (1, 0, 2))
        return x, (h, c)
