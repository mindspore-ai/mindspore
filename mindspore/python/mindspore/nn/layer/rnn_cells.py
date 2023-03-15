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
"""RNN Cells module, include RNNCell, GRUCell, LSTMCell."""
from __future__ import absolute_import

import math
import numpy as np

import mindspore.ops as P
import mindspore.common.dtype as mstype
from mindspore import log as logger
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer, Uniform
from mindspore.ops.primitive import constexpr
from mindspore.nn.cell import Cell
from mindspore._checkparam import Validator as validator

__all__ = ['LSTMCell', 'GRUCell', 'RNNCell']


@constexpr
def _check_input_dtype(input_dtype, param_name, allow_dtypes, cls_name):
    validator.check_type_name(param_name, input_dtype, allow_dtypes, cls_name)


@constexpr(check=False)
def _check_is_tensor(param_name, input_data, cls_name):
    """Internal function, used to check whether the input data is Tensor."""
    if input_data is not None and not isinstance(P.typeof(input_data), mstype.tensor_type):
        raise TypeError(f"For '{cls_name}', the '{param_name}' must be '{mstype.tensor_type}', "
                        f"but got '{P.typeof(input_data)}'")


@constexpr
def _check_is_tuple(param_name, input_data, cls_name):
    """Internal function, used to check whether the input data is Tensor."""
    if input_data is not None and not isinstance(P.typeof(input_data), mstype.Tuple):
        raise TypeError(f"For '{cls_name}', the '{param_name}' must be '{mstype.Tuple}', "
                        f"but got '{P.typeof(input_data)}'")


@constexpr
def _check_tuple_length(param_name, input_data, length, cls_name):
    """Internal function, used to check whether the input data is Tensor."""
    if input_data is not None and len(input_data) != length:
        raise TypeError(f"For '{cls_name}', the length of '{param_name}' must be '{length}', "
                        f"but got '{len(input_data)}'")


def _check_lstmcell_init(func):
    def wrapper(*args, **kwargs):
        logger.warning(f"LSTMCell has been changed from 'single LSTM layer' to 'single LSTM cell', "
                       f"if you still need use single LSTM layer, please use `nn.LSTM` instead.")
        if len(args) > 4 or 'batch_size' in kwargs or \
            'dropout' in kwargs or 'bidirectional' in kwargs:
            raise ValueError(f"The arguments of `nn.LSTMCell` from old MindSpore version(<1.6) are detected, "
                             f"if you still need use single LSTM layer, please use `nn.LSTM` instead.")
        return func(*args, **kwargs)
    return wrapper


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


class RNNCellBase(Cell):
    '''Basic class for RNN Cells'''
    def __init__(self, input_size: int, hidden_size: int, has_bias: bool, num_chunks: int):
        super().__init__()
        validator.check_value_type("has_bias", has_bias, [bool], self.cls_name)
        validator.check_positive_int(hidden_size, "hidden_size", self.cls_name)
        validator.check_positive_int(input_size, "input_size", self.cls_name)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.has_bias = has_bias
        self.weight_ih = Parameter(Tensor(np.random.randn(num_chunks * hidden_size, input_size).astype(np.float32)))
        self.weight_hh = Parameter(Tensor(np.random.randn(num_chunks * hidden_size, hidden_size).astype(np.float32)))
        if has_bias:
            self.bias_ih = Parameter(Tensor(np.random.randn(num_chunks * hidden_size).astype(np.float32)))
            self.bias_hh = Parameter(Tensor(np.random.randn(num_chunks * hidden_size).astype(np.float32)))
        else:
            self.bias_ih = None
            self.bias_hh = None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1 / math.sqrt(self.hidden_size)
        for weight in self.get_parameters():
            weight.set_data(initializer(Uniform(stdv), weight.shape))


class RNNCell(RNNCellBase):
    r"""
    An Elman RNN cell with tanh or ReLU non-linearity.

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

    Here :math:`h_t` is the hidden state at time `t`, :math:`x_t` is
    the input at time `t`, and :math:`h_{(t-1)}` is the hidden state of the
    previous layer at time :math:`t-1` or the initial hidden state at time `0`.
    If `nonlinearity` is `relu`, then `relu` is used instead of `tanh`.

    Args:
        input_size (int): Number of features of input.
        hidden_size (int):  Number of features of hidden layer.
        has_bias (bool): Whether the cell has bias `b_ih` and `b_hh`. Default: True.
        nonlinearity (str): The non-linearity to use. Can be either `tanh` or `relu`. Default: `tanh`.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(batch\_size, input\_size)` .
        - **hx** (Tensor) - Tensor of data type mindspore.float32 and shape :math:`(batch\_size, hidden\_size)` .
          Data type of `hx` must be the same as `x`.

    Outputs:
        - **hx'** (Tensor) - Tensor of shape :math:`(batch\_size, hidden\_size)` .

    Raises:
        TypeError: If `input_size` or `hidden_size` is not an int or not greater than 0.
        TypeError: If `has_bias` is not a bool.
        ValueError: If `nonlinearity` is not in ['tanh', 'relu'].

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> net = nn.RNNCell(10, 16)
        >>> x = Tensor(np.ones([5, 3, 10]).astype(np.float32))
        >>> hx = Tensor(np.ones([3, 16]).astype(np.float32))
        >>> output = []
        >>> for i in range(5):
        ...     hx = net(x[i], hx)
        ...     output.append(hx)
        >>> print(output[0].shape)
        (3, 16)
    """
    _non_linearity = ['tanh', 'relu']

    def __init__(self, input_size: int, hidden_size: int, has_bias: bool = True, nonlinearity: str = "tanh"):
        super().__init__(input_size, hidden_size, has_bias, num_chunks=1)
        validator.check_value_type("nonlinearity", nonlinearity, [str], self.cls_name)
        validator.check_string(nonlinearity, self._non_linearity, "nonlinearity", self.cls_name)
        self.nonlinearity = nonlinearity

    def construct(self, x, hx):
        _check_is_tensor('x', x, self.cls_name)
        _check_is_tensor('hx', hx, self.cls_name)
        _check_input_dtype(x.dtype, "x", [mstype.float32, mstype.float16], self.cls_name)
        _check_input_dtype(hx.dtype, "hx", [mstype.float32, mstype.float16], self.cls_name)

        if self.nonlinearity == "tanh":
            ret = _rnn_tanh_cell(x, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)
        else:
            ret = _rnn_relu_cell(x, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)
        return ret


class LSTMCell(RNNCellBase):
    r"""
    A LSTM (Long Short-Term Memory) cell.

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

    The encapsulated LSTMCell can be simplified to the following formula:

    .. math::
        h^{'},c^{'} = LSTMCell(x, (h_0, c_0))

    Args:
        input_size (int): Number of features of input.
        hidden_size (int):  Number of features of hidden layer.
        has_bias (bool): Whether the cell has bias `b_ih` and `b_hh`. Default: True.

    Inputs:
        - **x** (Tensor) - Tensor of shape (batch_size, `input_size`).
        - **hx** (tuple) - A tuple of two Tensors (h_0, c_0) both of data type mindspore.float32
          and shape (batch_size, `hidden_size`). The data type of `hx` must be the same as `x`.

    Outputs:
        - **hx'** (Tensor) - A tuple of two Tensors (h', c') both of data shape (batch_size, `hidden_size`).

    Raises:
        TypeError: If `input_size`, `hidden_size` is not an int.
        TypeError: If `has_bias` is not a bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> net = nn.LSTMCell(10, 16)
        >>> x = Tensor(np.ones([5, 3, 10]).astype(np.float32))
        >>> h = Tensor(np.ones([3, 16]).astype(np.float32))
        >>> c = Tensor(np.ones([3, 16]).astype(np.float32))
        >>> output = []
        >>> for i in range(5):
        ...     hx = net(x[i], (h, c))
        ...     output.append(hx)
        >>> print(output[0][0].shape)
        (3, 16)
    """
    @_check_lstmcell_init
    def __init__(self, input_size: int, hidden_size: int, has_bias: bool = True):
        super().__init__(input_size, hidden_size, has_bias, num_chunks=4)
        self.support_non_tensor_inputs = True

    def construct(self, x, hx):
        _check_is_tensor('x', x, self.cls_name)
        _check_is_tuple('hx', hx, self.cls_name)
        _check_tuple_length('hx', hx, 2, self.cls_name)
        _check_is_tensor('hx[0]', hx[0], self.cls_name)
        _check_is_tensor('hx[1]', hx[1], self.cls_name)
        _check_input_dtype(x.dtype, "x", [mstype.float32, mstype.float16], self.cls_name)
        _check_input_dtype(hx[0].dtype, "hx[0]", [mstype.float32, mstype.float16], self.cls_name)
        _check_input_dtype(hx[1].dtype, "hx[1]", [mstype.float32, mstype.float16], self.cls_name)
        return _lstm_cell(x, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)

    def _check_construct_args(self, *inputs, **kwargs):
        if len(inputs) == 4:
            raise ValueError(f"For '{self.cls_name}', the number of input args of construct is {len(inputs)}, if you "
                             f"are using the implementation of `nn.LSTMCell` from old MindSpore version(<1.6), "
                             f"please notice that: LSTMCell has been changed from 'single LSTM layer' to "
                             f"'single LSTM cell', if you still need use single LSTM layer, "
                             f"please use `nn.LSTM` instead.")
        return super()._check_construct_args(*inputs, **kwargs)


class GRUCell(RNNCellBase):
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

    The LSTMCell can be simplified in NN layer, the following formula:

    .. math::
        h^{'},c^{'} = LSTMCell(x, (h_0, c_0))

    Args:
        input_size (int): Number of features of input.
        hidden_size (int):  Number of features of hidden layer.
        has_bias (bool): Whether the cell has bias `b_in` and `b_hn`. Default: True.

    Inputs:
        - **x** (Tensor) - Tensor of shape (batch_size, `input_size`).
        - **hx** (Tensor) - Tensor of data type mindspore.float32 and shape (batch_size, `hidden_size`).
          Data type of `hx` must be the same as `x`.

    Outputs:
        - **hx'** (Tensor) - Tensor of shape (batch_size, `hidden_size`).

    Raises:
        TypeError: If `input_size`, `hidden_size` is not an int.
        TypeError: If `has_bias` is not a bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> net = nn.GRUCell(10, 16)
        >>> x = Tensor(np.ones([5, 3, 10]).astype(np.float32))
        >>> hx = Tensor(np.ones([3, 16]).astype(np.float32))
        >>> output = []
        >>> for i in range(5):
        ...     hx = net(x[i], hx)
        ...     output.append(hx)
        >>> print(output[0].shape)
        (3, 16)
    """
    def __init__(self, input_size: int, hidden_size: int, has_bias: bool = True):
        super().__init__(input_size, hidden_size, has_bias, num_chunks=3)

    def construct(self, x, hx):
        _check_is_tensor('x', x, self.cls_name)
        _check_is_tensor('hx', hx, self.cls_name)
        _check_input_dtype(x.dtype, "x", [mstype.float32, mstype.float16], self.cls_name)
        _check_input_dtype(hx.dtype, "hx", [mstype.float32, mstype.float16], self.cls_name)
        return _gru_cell(x, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)
