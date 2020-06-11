# Copyright 2020 Huawei Technologies Co., Ltd
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
import mindspore.nn as nn
from mindspore import context
from mindspore._checkparam import Validator as validator
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter, ParameterTuple
from mindspore.common.tensor import Tensor
from mindspore.nn.cell import Cell
from mindspore.ops import operations as P

__all__ = ['LSTM', 'LSTMCell']


class LSTM(Cell):
    r"""
    LSTM (Long Short-Term Memory) layer.

    Applies a LSTM to the input.

    There are two pipelines connecting two consecutive cells in a LSTM model; one is cell state pipeline
    and another is hidden state pipeline. Denote two consecutive time nodes as :math:`t-1` and :math:`t`.
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
        has_bias (bool): Specifies whether has bias `b_ih` and `b_hh`. Default: True.
        batch_first (bool): Specifies whether the first dimension of input is batch_size. Default: False.
        dropout (float, int): If not 0, append `Dropout` layer on the outputs of each
            LSTM layer except the last layer. Default 0. The range of dropout is [0.0, 1.0].
        bidirectional (bool): Specifies whether this is a bidirectional LSTM. If set True,
            number of directions will be 2 otherwise number of directions is 1. Default: False.

    Inputs:
        - **input** (Tensor) - Tensor of shape (seq_len, batch_size, `input_size`).
        - **hx** (tuple) - A tuple of two Tensors (h_0, c_0) both of data type mindspore.float32 or
          mindspore.float16 and shape (num_directions * `num_layers`, batch_size, `hidden_size`).
          Data type of `hx` should be the same of `input`.

    Outputs:
        Tuple, a tuple constains (`output`, (`h_n`, `c_n`)).

        - **output** (Tensor) - Tensor of shape (seq_len, batch_size, num_directions * `hidden_size`).
        - **hx_n** (tuple) - A tuple of two Tensor (h_n, c_n) both of shape
          (num_directions * `num_layers`, batch_size, `hidden_size`).

    Examples:
        >>> class LstmNet(nn.Cell):
        >>>     def __init__(self, input_size, hidden_size, num_layers, has_bias, batch_first, bidirectional):
        >>>         super(LstmNet, self).__init__()
        >>>         self.lstm = nn.LSTM(input_size=input_size,
        >>>                             hidden_size=hidden_size,
        >>>                             num_layers=num_layers,
        >>>                             has_bias=has_bias,
        >>>                             batch_first=batch_first,
        >>>                             bidirectional=bidirectional,
        >>>                             dropout=0.0)
        >>>
        >>>     def construct(self, inp, h0, c0):
        >>>         return self.lstm(inp, (h0, c0))
        >>>
        >>> net = LstmNet(10, 12, 2, has_bias=True, batch_first=True, bidirectional=False)
        >>> input = Tensor(np.ones([3, 5, 10]).astype(np.float32))
        >>> h0 = Tensor(np.ones([1 * 2, 3, 12]).astype(np.float32))
        >>> c0 = Tensor(np.ones([1 * 2, 3, 12]).astype(np.float32))
        >>> output, (hn, cn) = net(input, h0, c0)
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
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.has_bias = has_bias
        self.batch_first = validator.check_value_type("batch_first", batch_first, [bool], self.cls_name)
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        if self.batch_first:
            self.transpose1 = P.Transpose()
            self.transpose2 = P.Transpose()
        num_directions = 2 if self.bidirectional else 1
        self.cpu_target = False
        if context.get_context("device_target") == "CPU":
            self.cpu_target = True
        if not self.cpu_target:
            self.lstm = P.LSTM(input_size=self.input_size,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               has_bias=self.has_bias,
                               bidirectional=self.bidirectional,
                               dropout=self.dropout)
            weight_size = 0
            gate_size = 4 * self.hidden_size
            for layer in range(self.num_layers):
                input_layer_size = self.input_size if layer == 0 else self.hidden_size * num_directions
                increment_size = gate_size * input_layer_size
                increment_size += gate_size * self.hidden_size
                if self.has_bias:
                    increment_size += 2 * gate_size
                weight_size += increment_size * num_directions
            stdv = 1 / math.sqrt(hidden_size)
            w_np = np.random.uniform(-stdv, stdv, (weight_size, 1, 1)).astype(np.float32)
            self.weight = Parameter(initializer(Tensor(w_np), [weight_size, 1, 1]), name='weight')
        else:
            input_size_list = []
            input_size_list.append(self.input_size)
            for i in range(self.num_layers - 1):
                input_size_list.append(self.hidden_size * num_directions)
            weights = []
            layers = []
            bias_size = 0 if not self.has_bias else num_directions * self.hidden_size * 4
            stdv = 1 / math.sqrt(hidden_size)
            for i in range(num_layers):
                weight_size = (input_size_list[i] + self.hidden_size) * num_directions * self.hidden_size * 4
                if has_bias:
                    weight_size = weight_size + bias_size
                w_np = np.random.uniform(-stdv, stdv, (weight_size, 1, 1)).astype(np.float32)
                weights.append(Parameter(initializer(Tensor(w_np), w_np.shape), name='weight' + str(i)))
                layers.append(nn.LSTMCell(input_size=input_size_list[i],
                                          hidden_size=self.hidden_size,
                                          has_bias=self.has_bias,
                                          bidirectional=self.bidirectional,
                                          dropout=self.dropout))
            self.lstms = layers
            self.weight = ParameterTuple(tuple(weights))
        self.fill = P.Fill()
        self.shape = P.Shape()

    def construct(self, x, hx):
        if self.batch_first:
            x = self.transpose1(x, (1, 0, 2))
        if not self.cpu_target:
            h, c = hx
            output, h, c, _, _ = self.lstm(x, h, c, self.weight)
            if self.batch_first:
                output = self.transpose2(output, (1, 0, 2))
            return (output, (h, c))
        h, c = hx
        output, hn, cn, _, _ = self.lstms[0](x, h[0], c[0], self.weight[0])
        for i in range(1, self.num_layers):
            output, hn, cn, _, _ = self.lstms[i](output, h[i], c[i], self.weight[i])
        if self.batch_first:
            output = self.transpose2(output, (1, 0, 2))
        return (output, (hn, cn))


class LSTMCell(Cell):
    r"""
    LSTM (Long Short-Term Memory) layer.

    Applies a LSTM layer to the input.

    There are two pipelines connecting two consecutive cells in a LSTM model; one is cell state pipeline
    and another is hidden state pipeline. Denote two consecutive time nodes as :math:`t-1` and :math:`t`.
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
        layer_index (int): index of current layer of stacked LSTM . Default: 0.
        has_bias (bool): Specifies whether has bias `b_ih` and `b_hh`. Default: True.
        batch_first (bool): Specifies whether the first dimension of input is batch_size. Default: False.
        dropout (float, int): If not 0, append `Dropout` layer on the outputs of each
            LSTM layer except the last layer. Default 0. The range of dropout is [0.0, 1.0].
        bidirectional (bool): Specifies whether this is a bidirectional LSTM. If set True,
            number of directions will be 2 otherwise number of directions is 1. Default: False.

    Inputs:
        - **input** (Tensor) - Tensor of shape (seq_len, batch_size, `input_size`).
        - **h** - data type mindspore.float32 or
          mindspore.float16 and shape (num_directions * `num_layers`, batch_size, `hidden_size`).
        - **c** - data type mindspore.float32 or
          mindspore.float16 and shape (num_directions * `num_layers`, batch_size, `hidden_size`).
          Data type of `h' and 'c' should be the same of `input`.

    Outputs:
        `output`, `h_n`, `c_n`, 'reserve', 'state'.

        - **output** (Tensor) - Tensor of shape (seq_len, batch_size, num_directions * `hidden_size`).
        - **h** - A Tensor with shape (num_directions * `num_layers`, batch_size, `hidden_size`).
        - **c** - A Tensor with shape (num_directions * `num_layers`, batch_size, `hidden_size`).
        - **reserve** - reserved
        - **state** - reserved

    Examples:
        >>> class LstmNet(nn.Cell):
        >>>     def __init__(self, input_size, hidden_size, layer_index, has_bias, batch_first, bidirectional):
        >>>         super(LstmNet, self).__init__()
        >>>         self.lstm = nn.LSTMCell(input_size=input_size,
        >>>                             hidden_size=hidden_size,
        >>>                             layer_index=layer_index,
        >>>                             has_bias=has_bias,
        >>>                             batch_first=batch_first,
        >>>                             bidirectional=bidirectional,
        >>>                             dropout=0.0)
        >>>
        >>>     def construct(self, inp, h0, c0):
        >>>         return self.lstm(inp, (h0, c0))
        >>>
        >>> net = LstmNet(10, 12, 2, has_bias=True, batch_first=True, bidirectional=False)
        >>> input = Tensor(np.ones([3, 5, 10]).astype(np.float32))
        >>> h0 = Tensor(np.ones([1 * 2, 3, 12]).astype(np.float32))
        >>> c0 = Tensor(np.ones([1 * 2, 3, 12]).astype(np.float32))
        >>> output, hn, cn, _, _ = net(input, h0, c0)
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 has_bias=True,
                 batch_first=False,
                 dropout=0,
                 bidirectional=False):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.has_bias = has_bias
        self.batch_first = validator.check_value_type("batch_first", batch_first, [bool], self.cls_name)
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        self.num_directions = 1
        if self.bidirectional:
            self.num_directions = 2
        if self.batch_first:
            self.transpose1 = P.Transpose()
            self.transpose2 = P.Transpose()

        self.lstm = P.LSTM(input_size=self.input_size,
                           hidden_size=self.hidden_size,
                           num_layers=1,
                           has_bias=self.has_bias,
                           bidirectional=self.bidirectional,
                           dropout=self.dropout)

    def construct(self, x, h, c, w):
        if self.batch_first:
            x = self.transpose1(x, (1, 0, 2))
        output, hn, cn, _, _ = self.lstm(x, h, c, w)
        if self.batch_first:
            output = self.transpose2(output, (1, 0, 2))
        return output, hn, cn, _, _
