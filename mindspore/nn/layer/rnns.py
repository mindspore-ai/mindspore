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
'''RNN operators module, include RNN, GRU'''
import math
import numpy as np
import mindspore.ops as P
import mindspore.common.dtype as mstype
from mindspore.ops.primitive import constexpr
from mindspore.common.initializer import initializer, Uniform
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import ParameterTuple, Parameter
from mindspore.nn.cell import Cell
from mindspore import nn
from mindspore import log as logger
from mindspore._checkparam import Validator as validator

__all__ = ['GRU', 'RNN', 'GRUCell', 'RNNCell']

@constexpr
def _init_state(shape, dtype, is_lstm):
    hx = Tensor(np.zeros(shape), dtype)
    cx = Tensor(np.zeros(shape), dtype)
    if is_lstm:
        return (hx, cx)
    return hx

@constexpr
def _check_input_dtype(input_dtype, param_name, allow_dtypes, cls_name):
    validator.check_type_name(param_name, input_dtype, allow_dtypes, cls_name)

def _rnn_tanh_cell(inputs, hidden, w_ih, w_hh, b_ih, b_hh):
    '''RNN cell function with tanh activation'''
    if b_ih is None:
        igates = P.MatMul(False, True)(inputs, w_ih)
        hgates = P.MatMul(False, True)(hidden, w_hh)
    else:
        igates = P.MatMul(False, True)(inputs, w_ih) + b_ih
        hgates = P.MatMul(False, True)(hidden, w_hh) + b_hh
    return P.Tanh()(igates + hgates)

def _rnn_relu_cell(inputs, hidden, w_ih, w_hh, b_ih, b_hh):
    '''RNN cell function with relu activation'''
    if b_ih is None:
        igates = P.MatMul(False, True)(inputs, w_ih)
        hgates = P.MatMul(False, True)(hidden, w_hh)
    else:
        igates = P.MatMul(False, True)(inputs, w_ih) + b_ih
        hgates = P.MatMul(False, True)(hidden, w_hh) + b_hh
    return P.ReLU()(igates + hgates)

def _lstm_cell(inputs, hidden, w_ih, w_hh, b_ih, b_hh):
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

def _gru_cell(inputs, hidden, w_ih, w_hh, b_ih, b_hh):
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

class _DynamicRNN(Cell):
    '''Dynamic RNN module to compute RNN cell by timesteps'''
    def __init__(self, mode):
        super().__init__()
        if mode == "RNN_RELU":
            cell = _rnn_relu_cell
        elif mode == "RNN_TANH":
            cell = _rnn_tanh_cell
        elif mode == "LSTM":
            cell = _lstm_cell
        elif mode == "GRU":
            cell = _gru_cell
        else:
            raise ValueError(f"For '{self.cls_name}', the 'mode' should be in ['RNN_RELU', 'RNN_TANH', 'LSTM', 'GRU'], "
                             f"but got {mode}.")
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
        seq_length = P.Cast()(seq_length, mstype.float32)
        seq_length = P.BroadcastTo((hidden_size, -1))(seq_length)
        seq_length = P.Cast()(seq_length, mstype.int32)
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

class _RNNBase(Cell):
    '''Basic class for RNN operators'''
    def __init__(self, mode, input_size, hidden_size, num_layers=1, has_bias=True,
                 batch_first=False, dropout=0, bidirectional=False):
        super().__init__()
        validator.check_positive_int(hidden_size, "hidden_size", self.cls_name)
        validator.check_positive_int(input_size, "input_size", self.cls_name)
        validator.check_positive_int(num_layers, "num_layers", self.cls_name)
        validator.check_is_float(dropout, "dropout", self.cls_name)
        validator.check_value_type("has_bias", has_bias, [bool], self.cls_name)
        validator.check_value_type("batch_first", batch_first, [bool], self.cls_name)
        validator.check_value_type("bidirectional", bidirectional, [bool], self.cls_name)

        if not 0 <= dropout < 1:
            raise ValueError(f"For '{self.cls_name}', the 'dropout' should be a number in range [0, 1) "
                             f"representing the probability of an element being zeroed, but got {dropout}.")

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
            raise ValueError(f"For '{self.cls_name}', the 'mode' should be in ['RNN_RELU', 'RNN_TANH', 'LSTM', 'GRU'], "
                             f"but got {mode}.")

        self.reverse = P.ReverseV2([0])
        self.reverse_sequence = P.ReverseSequence(0, 1)
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.num_layers = num_layers
        self.dropout = dropout
        self.dropout_op = nn.Dropout(float(1 - dropout))
        self.bidirectional = bidirectional
        self.has_bias = has_bias
        self.rnn = _DynamicRNN(mode)
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

    def construct(self, x, hx=None, seq_length=None):
        '''Defines the RNN like operators performed'''
        x_dtype = P.dtype(x)
        hx_dtype = P.dtype(hx)
        _check_input_dtype(x_dtype, "x", [mstype.float32], self.cls_name)
        _check_input_dtype(hx_dtype, "hx", [mstype.float32], self.cls_name)
        if seq_length is not None:
            seq_length_dtype = P.dtype(seq_length)
            _check_input_dtype(seq_length_dtype, "seq_length", [mstype.int32, mstype.int64], self.cls_name)

        max_batch_size = x.shape[0] if self.batch_first else x.shape[1]
        num_directions = 2 if self.bidirectional else 1
        if hx is None:
            hx = _init_state((self.num_layers * num_directions, max_batch_size, self.hidden_size),
                             x.dtype, self.is_lstm)
        if self.batch_first:
            x = P.Transpose()(x, (1, 0, 2))
        if self.bidirectional:
            x, h = self._stacked_bi_dynamic_rnn(x, hx, seq_length)
        else:
            x, h = self._stacked_dynamic_rnn(x, hx, seq_length)
        if self.batch_first:
            x = P.Transpose()(x, (1, 0, 2))
        return x, h

class RNN(_RNNBase):
    r"""
    Stacked Elman RNN layers.

    Apply RNN layer with :math:`\tanh` or :math:`\text{ReLU}` non-linearity to the input.

    For each element in the input sequence, each layer computes the following function:

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

    Here :math:`h_t` is the hidden state at time `t`, :math:`x_t` is
    the input at time `t`, and :math:`h_{(t-1)}` is the hidden state of the
    previous layer at time `t-1` or the initial hidden state at time `0`.
    If :attr:`nonlinearity` is ``'relu'``, then :math:`\text{ReLU}` is used instead of :math:`\tanh`.

    Args:
        input_size (int): Number of features of input.
        hidden_size (int):  Number of features of hidden layer.
        num_layers (int): Number of layers of stacked RNN. Default: 1.
        nonlinearity (str): The non-linearity to use. Can be either ``'tanh'`` or ``'relu'``. Default: ``'tanh'``
        has_bias (bool): Whether the cell has bias `b_ih` and `b_hh`. Default: True.
        batch_first (bool): Specifies whether the first dimension of input `x` is batch_size. Default: False.
        dropout (float): If not 0.0, append `Dropout` layer on the outputs of each
            RNN layer except the last layer. Default 0.0. The range of dropout is [0.0, 1.0).
        bidirectional (bool): Specifies whether it is a bidirectional RNN,
            num_directions=2 if bidirectional=True otherwise 1. Default: False.

    Inputs:
        - **x** (Tensor) - Tensor of data type mindspore.float32 and
          shape (seq_len, batch_size, `input_size`) or (batch_size, seq_len, `input_size`).
        - **hx** (Tensor) - Tensor of data type mindspore.float32 and
          shape (num_directions * `num_layers`, batch_size, `hidden_size`). Data type of `hx` must be the same as `x`.
        - **seq_length** (Tensor) - The length of each batch.
          Tensor of shape :math:`(\text{batch_size})`. Default: None.

    Outputs:
        Tuple, a tuple contains (`output`, `h_n`).

        - **output** (Tensor) - Tensor of shape (seq_len, batch_size, num_directions * `hidden_size`).
        - **hx_n** (Tensor) - Tensor of shape (num_directions * `num_layers`, batch_size, `hidden_size`).

    Raises:
        TypeError: If `input_size`, `hidden_size` or `num_layers` is not an int.
        TypeError: If `has_bias`, `batch_first` or `bidirectional` is not a bool.
        TypeError: If `dropout` is neither a float nor an int.
        ValueError: If `dropout` is not in range [0.0, 1.0).
        ValueError: If `nonlinearity` is not in ['tanh', 'relu'].

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> net = nn.RNN(10, 16, 2, has_bias=True, batch_first=True, bidirectional=False)
        >>> x = Tensor(np.ones([3, 5, 10]).astype(np.float32))
        >>> h0 = Tensor(np.ones([1 * 2, 3, 16]).astype(np.float32))
        >>> output, hn = net(x, h0)
        >>> print(output.shape)
        (3, 5, 16)
    """
    def __init__(self, *args, **kwargs):
        if 'nonlinearity' in kwargs:
            if kwargs['nonlinearity'] == 'tanh':
                mode = 'RNN_TANH'
            elif kwargs['nonlinearity'] == 'relu':
                mode = 'RNN_RELU'
            else:
                raise ValueError(f"For '{self.cls_name}', the 'nonlinearity' should be in ['tanh', 'relu'], "
                                 f"but got {kwargs['nonlinearity']}.")
            del kwargs['nonlinearity']
        else:
            mode = 'RNN_TANH'

        super(RNN, self).__init__(mode, *args, **kwargs)

class GRU(_RNNBase):
    r"""
    Stacked GRU (Gated Recurrent Unit) layers.

    Apply GRU layer to the input.

    There are two gates in a GRU model; one is update gate and the other is reset gate.
    Denote two consecutive time nodes as :math:`t-1` and :math:`t`.
    Given an input :math:`x_t` at time :math:`t`, an hidden state :math:`h_{t-1}`, the update and reset gate at
    time :math:`t` is computed using an gating mechanism. Update gate :math:`z_t` is designed to protect the cell
    from perturbation by irrelevant inputs and past hidden state. Reset gate :math:`r_t` determines how much
    information should be reset from old hidden state. New memory state :math:`{n}_t` is
    calculated with the current input, on which the reset gate will be applied. Finally, current hidden state
    :math:`h_{t}` is computed with the calculated update grate and new memory state. The complete
    formulation is as follows.

    .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}

    Here :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product. :math:`W, b`
    are learnable weights between the output and the input in the formula. For instance,
    :math:`W_{ir}, b_{ir}` are the weight and bias used to transform from input :math:`x` to :math:`r`.
    Details can be found in paper
    `Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation
    <https://aclanthology.org/D14-1179.pdf>`_.

    Args:
        input_size (int): Number of features of input.
        hidden_size (int):  Number of features of hidden layer.
        num_layers (int): Number of layers of stacked GRU. Default: 1.
        has_bias (bool): Whether the cell has bias `b_ih` and `b_hh`. Default: True.
        batch_first (bool): Specifies whether the first dimension of input `x` is batch_size. Default: False.
        dropout (float): If not 0.0, append `Dropout` layer on the outputs of each
            GRU layer except the last layer. Default 0.0. The range of dropout is [0.0, 1.0).
        bidirectional (bool): Specifies whether it is a bidirectional GRU,
            num_directions=2 if bidirectional=True otherwise 1. Default: False.

    Inputs:
        - **x** (Tensor) - Tensor of data type mindspore.float32 and
          shape (seq_len, batch_size, `input_size`) or (batch_size, seq_len, `input_size`).
        - **hx** (Tensor) - Tensor of data type mindspore.float32 and
          shape (num_directions * `num_layers`, batch_size, `hidden_size`). Data type of `hx` must be the same as `x`.
        - **seq_length** (Tensor) - The length of each batch.
          Tensor of shape :math:`(\text{batch_size})`. Default: None.

    Outputs:
        Tuple, a tuple contains (`output`, `h_n`).

        - **output** (Tensor) - Tensor of shape (seq_len, batch_size, num_directions * `hidden_size`).
        - **hx_n** (Tensor) - Tensor of shape (num_directions * `num_layers`, batch_size, `hidden_size`).

    Raises:
        TypeError: If `input_size`, `hidden_size` or `num_layers` is not an int.
        TypeError: If `has_bias`, `batch_first` or `bidirectional` is not a bool.
        TypeError: If `dropout` is neither a float nor an int.
        ValueError: If `dropout` is not in range [0.0, 1.0).

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> net = nn.GRU(10, 16, 2, has_bias=True, batch_first=True, bidirectional=False)
        >>> x = Tensor(np.ones([3, 5, 10]).astype(np.float32))
        >>> h0 = Tensor(np.ones([1 * 2, 3, 16]).astype(np.float32))
        >>> output, hn = net(x, h0)
        >>> print(output.shape)
        (3, 5, 16)
    """
    def __init__(self, *args, **kwargs):
        mode = 'GRU'
        super(GRU, self).__init__(mode, *args, **kwargs)

class _RNNCellBase(Cell):
    '''Basic class for RNN Cells'''
    def __init__(self, input_size: int, hidden_size: int, has_bias: bool, num_chunks: int):
        super().__init__()
        validator.check_value_type("has_bias", has_bias, [bool], self.cls_name)
        validator.check_positive_int(hidden_size, "hidden_size", self.cls_name)
        validator.check_positive_int(input_size, "input_size", self.cls_name)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = has_bias
        self.weight_ih = Parameter(Tensor(np.random.randn(num_chunks * hidden_size, input_size).astype(np.float32)))
        self.weight_hh = Parameter(Tensor(np.random.randn(num_chunks * hidden_size, hidden_size).astype(np.float32)))
        if has_bias:
            self.bias_ih = Parameter(Tensor(np.random.randn(num_chunks * hidden_size).astype(np.float32)))
            self.bias_hh = Parameter(Tensor(np.random.randn(num_chunks * hidden_size).astype(np.float32)))
        else:
            self.bias_ih = None
            self.bias_ih = None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1 / math.sqrt(self.hidden_size)
        for weight in self.get_parameters():
            weight.set_data(initializer(Uniform(stdv), weight.shape))

class RNNCell(_RNNCellBase):
    r"""
    An Elman RNN cell with tanh or ReLU non-linearity.

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

    Here :math:`h_t` is the hidden state at time `t`, :math:`x_t` is
    the input at time `t`, and :math:`h_{(t-1)}` is the hidden state of the
    previous layer at time `t-1` or the initial hidden state at time `0`.
    If :attr:`nonlinearity` is ``'relu'``, then :math:`\text{ReLU}` is used instead of :math:`\tanh`.

    Args:
        input_size (int): Number of features of input.
        hidden_size (int):  Number of features of hidden layer.
        has_bias (bool): Whether the cell has bias `b_ih` and `b_hh`. Default: True.
        nonlinearity (str): The non-linearity to use. Can be either ``'tanh'`` or ``'relu'``. Default: ``'tanh'``

    Inputs:
        - **x** (Tensor) - Tensor of shape (batch_size, `input_size`).
        - **hx** (Tensor) - Tensor of data type mindspore.float32 and shape (batch_size, `hidden_size`).
          Data type of `hx` must be the same as `x`.

    Outputs:
        - **h'** (Tensor) - Tensor of shape (batch_size, `hidden_size`).

    Raises:
        TypeError: If `input_size`, `hidden_size` is not an int.
        TypeError: If `has_bias` is not a bool.
        ValueError: If `nonlinearity` is not in ['tanh', 'relu'].

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> net = nn.RNNCell(10, 16)
        >>> x = Tensor(np.ones([5, 3, 10]).astype(np.float32))
        >>> hx = Tensor(np.ones([3, 16]).astype(np.float32))
        >>> output = []
        >>> for i in range(5):
        >>>     hx = net(x[i], hx)
        >>>     output.append(hx)
        >>> print(output[0].shape)
        (3, 16)
    """
    _non_linearity = ['tanh', 'relu']
    def __init__(self, input_size: int, hidden_size: int, has_bias: bool = True, nonlinearity: str = "tanh"):
        super().__init__(input_size, hidden_size, has_bias, num_chunks=1)
        if nonlinearity not in self._non_linearity:
            raise ValueError(f"For '{self.cls_name}', the 'nonlinearity' should be in ['tanh', 'relu'], "
                             f"but got {nonlinearity}.")
        self.nonlinearity = nonlinearity

    def construct(self, x, hx):
        x_dtype = P.dtype(x)
        hx_dtype = P.dtype(hx)
        _check_input_dtype(x_dtype, "x", [mstype.float32], self.cls_name)
        _check_input_dtype(hx_dtype, "hx", [mstype.float32], self.cls_name)

        if self.nonlinearity == "tanh":
            ret = _rnn_tanh_cell(x, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)
        else:
            ret = _rnn_relu_cell(x, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)
        return ret

class GRUCell(_RNNCellBase):
    r"""
    A GRU(Gated Recurrent Unit) cell.

    .. math::

        \begin{array}{ll}
        r = \sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\
        z = \sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\
        n = \tanh(W_{in} x + b_{in} + r * (W_{hn} h + b_{hn})) \\
        h' = (1 - z) * n + z * h
        \end{array}

    Here :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product. :math:`W, b`
    are learnable weights between the output and the input in the formula. For instance,
    :math:`W_{ir}, b_{ir}` are the weight and bias used to transform from input :math:`x` to :math:`r`.
    Details can be found in paper
    `Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation
    <https://aclanthology.org/D14-1179.pdf>`_.

    Args:
        input_size (int): Number of features of input.
        hidden_size (int):  Number of features of hidden layer.
        has_bias (bool): Whether the cell has bias `b_ih` and `b_hh`. Default: True.

    Inputs:
        - **x** (Tensor) - Tensor of shape (batch_size, `input_size`).
        - **hx** (Tensor) - Tensor of data type mindspore.float32 and shape (batch_size, `hidden_size`).
          Data type of `hx` must be the same as `x`.

    Outputs:
        - **h'** (Tensor) - Tensor of shape (batch_size, `hidden_size`).

    Raises:
        TypeError: If `input_size`, `hidden_size` is not an int.
        TypeError: If `has_bias` is not a bool.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> net = nn.GRUCell(10, 16)
        >>> x = Tensor(np.ones([5, 3, 10]).astype(np.float32))
        >>> hx = Tensor(np.ones([3, 16]).astype(np.float32))
        >>> output = []
        >>> for i in range(5):
        >>>     hx = net(x[i], hx)
        >>>     output.append(hx)
        >>> print(output[0].shape)
        (3, 16)
    """
    def __init__(self, input_size: int, hidden_size: int, has_bias: bool = True):
        super().__init__(input_size, hidden_size, has_bias, num_chunks=3)

    def construct(self, x, hx):
        x_dtype = P.dtype(x)
        hx_dtype = P.dtype(hx)
        _check_input_dtype(x_dtype, "x", [mstype.float32], self.cls_name)
        _check_input_dtype(hx_dtype, "hx", [mstype.float32], self.cls_name)

        return _gru_cell(x, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)
