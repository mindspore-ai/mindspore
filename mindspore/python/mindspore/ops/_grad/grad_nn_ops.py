# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

"""Define the grad rules of neural network related operations."""
from mindspore.ops._grad.grad_base import bprop_getters
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops.composite.multitype_ops.zeros_like_impl import zeros_like
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops.operations import _rl_inner_ops as rl_ops


@bprop_getters.register(P.DepthwiseConv2dNative)
def get_bprop_depthwise_conv2d_native(self):
    """Grad definition for `DepthwiseConv2dNative` operation."""
    input_grad = G.DepthwiseConv2dNativeBackpropInput(
        self.channel_multiplier, self.kernel_size, self.pad_mode, self.pad, self.pad_list, self.mode, self.stride,
        self.dilation, self.group
    )
    filter_grad = G.DepthwiseConv2dNativeBackpropFilter(
        self.channel_multiplier, self.kernel_size, self.pad_mode, self.pad, self.pad_list, self.mode, self.stride,
        self.dilation, self.group
    )
    get_shape = P.Shape()

    def bprop(x, w, out, dout):
        dx = input_grad(get_shape(x), w, dout)

        dw = filter_grad(x, get_shape(w), dout)
        return dx, dw

    return bprop


@bprop_getters.register(P.SparseSoftmaxCrossEntropyWithLogits)
def get_bprop_sparse_softmax_cross_entropy_with_logits(self):
    """Grad definition for `SparseSoftmaxCrossEntropyWithLogits` operation."""
    is_grad = self.is_grad
    grad_op = P.SparseSoftmaxCrossEntropyWithLogits(is_grad=True)

    def bprop(logits, labels, out, dout):
        grad = out[0]
        if not is_grad:
            # if construct use loss
            grad = grad_op(logits, labels)
            grad = F.depend(grad, out)
            grad = grad * dout
        return grad, zeros_like(labels)

    return bprop


@bprop_getters.register(rl_ops.GRUV2)
def get_bppro_gru_v2(self):
    """Grad definition for `GRUV2` operation."""
    gru_grad_v2 = G.GRUV2Grad(
        self.input_size,
        self.hidden_size,
        self.num_layers,
        self.has_bias,
        self.bidirectional,
        self.dropout
    )

    def bpro(x, hx, w, seq_length, out, dout):
        y, hy, reverse, _ = out
        dy, dhy, _, _ = dout
        dx, dhx, dw = gru_grad_v2(x, hx, w, seq_length, y, hy, dy, dhy, reverse)
        dx_all = (dx, dhx, dw, (0))
        return dx_all

    return bpro


@bprop_getters.register(rl_ops.CudnnGRU)
def get_bprop_gru(self):
    """Grad definition for `GRU` operation."""
    gru_grad_data = G.GruGradData(
        input_size=self.input_size,
        hidden_size=self.hidden_size,
        num_layers=self.num_layers,
        has_bias=self.has_bias,
        bidirectional=self.bidirectional,
        dropout=self.dropout
    )

    gru_grad_weight = G.GruGradWeight(
        input_size=self.input_size,
        hidden_size=self.hidden_size,
        num_layers=self.num_layers,
        has_bias=self.has_bias,
        bidirectional=self.bidirectional,
        dropout=self.dropout
    )

    def bprop(x, hx, w, out, dout):
        y, _, reserve, state = out
        dy, dhy, _, _ = dout
        dx, dhx = gru_grad_data(y, dy, dhy, w, hx, reserve, state)
        dw = gru_grad_weight(F.depend(x, dx), hx, y, reserve, state)
        return dx, dhx, dw

    return bprop


@bprop_getters.register(P.BasicLSTMCell)
def get_bprop_basic_lstm_cell(self):
    """Grad definition for `BasicLSTMCell` operation."""
    basic_lstm_cell_cstate_grad = G.BasicLSTMCellCStateGrad(
        forget_bias=self.forget_bias,
        activation=self.activation
    )

    basic_lstm_cell_weight_grad = G.BasicLSTMCellWeightGrad()

    basic_lstm_cell_input_grad = G.BasicLSTMCellInputGrad(keep_prob=self.keep_prob)

    def bprop(x, h, c, w, b, out, dout):
        _, _, it, jt, ft, ot, tanhct = out
        dct, dht, _, _, _, _, _ = dout
        dgate, dct_1 = basic_lstm_cell_cstate_grad(c, dht, dct, it, jt, ft, ot, tanhct)
        dxt, dht = basic_lstm_cell_input_grad(dgate, w)
        dw, db = basic_lstm_cell_weight_grad(F.depend(x, dxt), h, dgate)
        return dxt, dht, dct_1, dw, db

    return bprop
