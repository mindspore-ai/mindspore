# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""lstm"""
import math
import numpy as np
import mindspore.context as context
import mindspore.common.dtype as mstype
from mindspore.ops.primitive import constexpr
from mindspore._checkparam import Validator as validator
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.common.tensor import Tensor
from mindspore.nn.cell import Cell
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F


__all__ = ['LSTM', 'LSTMCell']


@constexpr
def _create_sequence_length(shape):
    num_step, batch_size, _ = shape
    sequence_length = Tensor(np.ones(batch_size, np.int32) * num_step, mstype.int32)
    return sequence_length

@constexpr
def _check_input_dtype(input_dtype, param_name, allow_dtypes, cls_name):
    validator.check_type_name(param_name, input_dtype, allow_dtypes, cls_name)

@constexpr
def _check_input_3d(input_shape, param_name, func_name):
    if len(input_shape) != 3:
        raise ValueError(f"{func_name} {param_name} should be 3d, but got shape {input_shape}")

class LSTM(Cell):
    r"""
    Stacked LSTM (Long Short-Term Memory) layers.

    Apply LSTM layer to the input.

    There are two pipelines connecting two consecutive cells in a LSTM model; one is cell state pipeline
    and the other is hidden state pipeline. Denote two consecutive time nodes as :math:`t-1` and :math:`t`.
    Given an input :math:`x_t` at time :math:`t`, an hidden state :math:`h_{t-1}` and an cell
    state :math:`c_{t-1}` of the layer at time :math:`{t-1}`, the cell state and hidden state at
    time :math:`t` is computed using an gating mechanism. Input gate :math:`i_t` is designed to protect the cell
    from perturbation by irrelevant inputs. Forget gate :math:`f_t` affords protection of the cell by forgetting
    some information in the past, which is stored in :math:`h_{t-1}`. Output gate :math:`o_t` protects other
    units from perturbation by currently irrelevant memory contents. Candidate cell state :math:`\tilde{c}_t` is
    calculated with the current input, on which the input gate will be applied. Finally, current cell state
    :math:`c_{t}` and hidden state :math:`h_{t}` are computed with the calculated gates and cell states. The complete
    formulation is as follows.

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ix} x_t + b_{ix} + W_{ih} h_{(t-1)} + b_{ih}) \\
            f_t = \sigma(W_{fx} x_t + b_{fx} + W_{fh} h_{(t-1)} + b_{fh}) \\
            \tilde{c}_t = \tanh(W_{cx} x_t + b_{cx} + W_{ch} h_{(t-1)} + b_{ch}) \\
            o_t = \sigma(W_{ox} x_t + b_{ox} + W_{oh} h_{(t-1)} + b_{oh}) \\
            c_t = f_t * c_{(t-1)} + i_t * \tilde{c}_t \\
            h_t = o_t * \tanh(c_t) \\
        \end{array}

    Here :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product. :math:`W, b`
    are learnable weights between the output and the input in the formula. For instance,
    :math:`W_{ix}, b_{ix}` are the weight and bias used to transform from input :math:`x` to :math:`i`.
    Details can be found in paper `LONG SHORT-TERM MEMORY
    <https://www.bioinf.jku.at/publications/older/2604.pdf>`_ and
    `Long Short-Term Memory Recurrent Neural Network Architectures for Large Scale Acoustic Modeling
    <https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/43905.pdf>`_.

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

    Raises:
        TypeError: If `input_size`, `hidden_size` or `num_layers` is not an int.
        TypeError: If `has_bias`, `batch_first` or `bidirectional` is not a bool.
        TypeError: If `dropout` is neither a float nor an int.
        ValueError: If `dropout` is not in range [0.0, 1.0].

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> net = nn.LSTM(10, 16, 2, has_bias=True, batch_first=True, bidirectional=False)
        >>> input = Tensor(np.ones([3, 5, 10]).astype(np.float32))
        >>> h0 = Tensor(np.ones([1 * 2, 3, 16]).astype(np.float32))
        >>> c0 = Tensor(np.ones([1 * 2, 3, 16]).astype(np.float32))
        >>> output, (hn, cn) = net(input, (h0, c0))
        >>> print(output.shape)
        (3, 5, 16)
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
        validator.check_value_type("batch_first", batch_first, [bool], self.cls_name)
        validator.check_positive_int(hidden_size, "hidden_size", self.cls_name)
        validator.check_positive_int(num_layers, "num_layers", self.cls_name)
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
            b0 = np.zeros(gate_size, dtype=np.float16)
            self.w_list = []
            self.b_list = []
            self.rnns_fw = P.DynamicRNN(forget_bias=0.0)
            self.rnns_bw = P.DynamicRNN(forget_bias=0.0)

            for layer in range(num_layers):
                w_shape = input_size if layer == 0 else (num_directions * hidden_size)
                w_np = np.random.uniform(-stdv, stdv, (w_shape + hidden_size, gate_size)).astype(np.float16)
                self.w_list.append(Parameter(
                    initializer(Tensor(w_np), [w_shape + hidden_size, gate_size]), name='weight_fw' + str(layer)))
                if has_bias:
                    b_np = np.random.uniform(-stdv, stdv, gate_size).astype(np.float16)
                    self.b_list.append(Parameter(initializer(Tensor(b_np), [gate_size]), name='bias_fw' + str(layer)))
                else:
                    self.b_list.append(Parameter(initializer(Tensor(b0), [gate_size]), name='bias_fw' + str(layer)))
                if bidirectional:
                    w_bw_np = np.random.uniform(-stdv, stdv, (w_shape + hidden_size, gate_size)).astype(np.float16)
                    self.w_list.append(Parameter(initializer(Tensor(w_bw_np), [w_shape + hidden_size, gate_size]),
                                                 name='weight_bw' + str(layer)))
                    b_bw_np = np.random.uniform(-stdv, stdv, (4 * hidden_size)).astype(np.float16) if has_bias else b0
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
            x_dtype = F.dtype(x)
            h_dtype = F.dtype(h)
            c_dtype = F.dtype(c)
            _check_input_3d(F.shape(h), "h of hx", self.cls_name)
            _check_input_3d(F.shape(c), "c of hx", self.cls_name)
            _check_input_dtype(x_dtype, "x", [mstype.float32, mstype.float16], self.cls_name)
            _check_input_dtype(h_dtype, "h", [mstype.float32, mstype.float16], self.cls_name)
            _check_input_dtype(c_dtype, "c", [mstype.float32, mstype.float16], self.cls_name)
            x = self.cast(x, mstype.float16)
            h = self.cast(h, mstype.float16)
            c = self.cast(c, mstype.float16)
            if self.bidirectional:
                x, h, c = self._stacked_bi_dynamic_rnn(x, h, c, self.w_list, self.b_list)
            else:
                x, h, c = self._stacked_dynamic_rnn(x, h, c, self.w_list, self.b_list)
            x = self.cast(x, x_dtype)
            h = self.cast(h, h_dtype)
            c = self.cast(c, c_dtype)
        else:
            x, h, c, _, _ = self.lstm(x, h, c, self.weight)
        if self.batch_first:
            x = self.transpose(x, (1, 0, 2))
        return x, (h, c)


class LSTMCell(Cell):
    r"""
    LSTM (Long Short-Term Memory) layer.

    Apply LSTM layer to the input.

    There are two pipelines connecting two consecutive cells in a LSTM model; one is cell state pipeline
    and the other is hidden state pipeline. Denote two consecutive time nodes as :math:`t-1` and :math:`t`.
    Given an input :math:`x_t` at time :math:`t`, an hidden state :math:`h_{t-1}` and an cell
    state :math:`c_{t-1}` of the layer at time :math:`{t-1}`, the cell state and hidden state at
    time :math:`t` is computed using an gating mechanism. Input gate :math:`i_t` is designed to protect the cell
    from perturbation by irrelevant inputs. Forget gate :math:`f_t` affords protection of the cell by forgetting
    some information in the past, which is stored in :math:`h_{t-1}`. Output gate :math:`o_t` protects other
    units from perturbation by currently irrelevant memory contents. Candidate cell state :math:`\tilde{c}_t` is
    calculated with the current input, on which the input gate will be applied. Finally, current cell state
    :math:`c_{t}` and hidden state :math:`h_{t}` are computed with the calculated gates and cell states. The complete
    formulation is as follows.

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ix} x_t + b_{ix} + W_{ih} h_{(t-1)} + b_{ih}) \\
            f_t = \sigma(W_{fx} x_t + b_{fx} + W_{fh} h_{(t-1)} + b_{fh}) \\
            \tilde{c}_t = \tanh(W_{cx} x_t + b_{cx} + W_{ch} h_{(t-1)} + b_{ch}) \\
            o_t = \sigma(W_{ox} x_t + b_{ox} + W_{oh} h_{(t-1)} + b_{oh}) \\
            c_t = f_t * c_{(t-1)} + i_t * \tilde{c}_t \\
            h_t = o_t * \tanh(c_t) \\
        \end{array}

    Here :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product. :math:`W, b`
    are learnable weights between the output and the input in the formula. For instance,
    :math:`W_{ix}, b_{ix}` are the weight and bias used to transform from input :math:`x` to :math:`i`.
    Details can be found in paper `LONG SHORT-TERM MEMORY
    <https://www.bioinf.jku.at/publications/older/2604.pdf>`_ and
    `Long Short-Term Memory Recurrent Neural Network Architectures for Large Scale Acoustic Modeling
    <https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/43905.pdf>`_.

    LSTMCell is a single-layer RNN, you can achieve multi-layer RNN by stacking LSTMCell.

    Args:
        input_size (int): Number of features of input.
        hidden_size (int):  Number of features of hidden layer.
        has_bias (bool): Whether the cell has bias `b_ih` and `b_hh`. Default: True.
        batch_first (bool): Specifies whether the first dimension of input is batch_size. Default: False.
        dropout (float, int): If not 0, append `Dropout` layer on the outputs of each
            LSTM layer except the last layer. Default 0. The range of dropout is [0.0, 1.0].
        bidirectional (bool): Specifies whether this is a bidirectional LSTM. If set True,
            number of directions will be 2 otherwise number of directions is 1. Default: False.

    Inputs:
        - **input** (Tensor) - Tensor of shape (seq_len, batch_size, `input_size`).
        - **h** - data type mindspore.float32 or
          mindspore.float16 and shape (num_directions, batch_size, `hidden_size`).
        - **c** - data type mindspore.float32 or
          mindspore.float16 and shape (num_directions, batch_size, `hidden_size`).
          Data type of `h' and 'c' must be the same of `input`.
        - **w** - data type mindspore.float32 or
          mindspore.float16 and shape (`weight_size`, 1, 1).
          The value of `weight_size` depends on `input_size`, `hidden_size` and `bidirectional`

    Outputs:
        `output`, `h_n`, `c_n`, 'reserve', 'state'.

        - **output** (Tensor) - Tensor of shape (seq_len, batch_size, num_directions * `hidden_size`).
        - **h** - A Tensor with shape (num_directions, batch_size, `hidden_size`).
        - **c** - A Tensor with shape (num_directions, batch_size, `hidden_size`).
        - **reserve** - reserved
        - **state** - reserved

    Raises:
        TypeError: If `input_size` or `hidden_size` or `num_layers` is not an int.
        TypeError: If `has_bias` or `batch_first` or `bidirectional` is not a bool.
        TypeError: If `dropout` is neither a float nor an int.
        ValueError: If `dropout` is not in range [0.0, 1.0].

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> net = nn.LSTMCell(10, 12, has_bias=True, batch_first=True, bidirectional=False)
        >>> input = Tensor(np.ones([3, 5, 10]).astype(np.float32))
        >>> h = Tensor(np.ones([1, 3, 12]).astype(np.float32))
        >>> c = Tensor(np.ones([1, 3, 12]).astype(np.float32))
        >>> w = Tensor(np.ones([1152, 1, 1]).astype(np.float32))
        >>> output, h, c, _, _ = net(input, h, c, w)
        >>> print(output.shape)
        (3, 5, 12)
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 has_bias=True,
                 batch_first=False,
                 dropout=0,
                 bidirectional=False):
        super(LSTMCell, self).__init__()
        self.batch_first = validator.check_value_type("batch_first", batch_first, [bool], self.cls_name)
        self.transpose = P.Transpose()
        self.lstm = P.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=1,
                           has_bias=has_bias,
                           bidirectional=bidirectional,
                           dropout=float(dropout))

    def construct(self, x, h, c, w):
        if self.batch_first:
            x = self.transpose(x, (1, 0, 2))
        x, h, c, _, _ = self.lstm(x, h, c, w)
        if self.batch_first:
            x = self.transpose(x, (1, 0, 2))
        return x, h, c, _, _
