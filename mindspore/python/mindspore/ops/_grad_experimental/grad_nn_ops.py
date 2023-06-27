# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
from __future__ import absolute_import

from mindspore.common import dtype as mstype
from mindspore.ops._grad_experimental.grad_base import bprop_getters
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.composite.multitype_ops.zeros_like_impl import zeros_like
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops.operations.nn_ops import FractionalAvgPool
from mindspore.ops.operations._grad_ops import FractionalAvgPoolGrad
from mindspore.ops.operations.nn_ops import FractionalMaxPool
from mindspore.ops.operations._grad_ops import FractionalMaxPoolGrad
from mindspore.ops.operations.nn_ops import FractionalMaxPool3DWithFixedKsize
from mindspore.ops.operations._grad_ops import FractionalMaxPool3DGradWithFixedKsize
from mindspore.ops.operations.nn_ops import AvgPoolV1
from mindspore.ops.operations._grad_ops import AvgPoolGradV1
from mindspore.ops.operations.nn_ops import MaxPoolWithArgmaxV2
from mindspore.ops.operations.nn_ops import FractionalMaxPoolWithFixedKsize
from mindspore.ops.operations._grad_ops import FractionalMaxPoolGradWithFixedKsize
from mindspore.ops.operations.nn_ops import WKV
from mindspore.ops.operations._grad_ops import WKVGrad
from mindspore.ops.operations.nn_ops import GLU
from mindspore.ops.operations.nn_ops import AdaptiveMaxPool3D
from mindspore.ops.operations._sequence_ops import TupleToTensor
from mindspore.ops.operations import _rl_inner_ops as rl_ops


@bprop_getters.register(FractionalMaxPool)
def get_bprop_fractional_max_pool(self):
    """Grad definition for `FractionalMaxPool` operation."""
    fractional_max_pool_grad = FractionalMaxPoolGrad(self.overlapping)

    def bprop(x, out, dout):
        dx = fractional_max_pool_grad(x, out[0], dout[0], out[1], out[2])
        return (dx,)

    return bprop


@bprop_getters.register(FractionalMaxPool3DWithFixedKsize)
def get_bprop_fractional_max_pool3d_with_fixed_ksize(self):
    """Grad definition for `FractionalMaxPool3DWithFixedKsize` operation."""
    fractional_max_pool3d_grad_with_fixed_ksize = FractionalMaxPool3DGradWithFixedKsize(data_format=self.data_format)

    def bprop(x, random_samples, out, dout):
        dx = fractional_max_pool3d_grad_with_fixed_ksize(x, dout[0], out[1])
        return (dx, zeros_like(random_samples))

    return bprop


@bprop_getters.register(FractionalAvgPool)
def get_bprop_fractional_avg_pool(self):
    """Grad definition for `FractionalAvgPool` operation."""
    fractional_avg_pool_grad = FractionalAvgPoolGrad(overlapping=self.overlapping)
    shape_op = P.Shape()
    tuple_to_tensor_op = TupleToTensor()

    def bprop(x, out, dout):
        dx = fractional_avg_pool_grad(tuple_to_tensor_op(shape_op(x), mstype.int64), dout[0], out[1], out[2])
        return (dx,)

    return bprop


@bprop_getters.register(AdaptiveMaxPool3D)
def get_bprop_adaptive_max_pool_3d(self):
    """Grad definition for `AdaptiveMaxPool3D` operation."""
    grad = G.AdaptiveMaxPool3DGrad()

    def bprop(x, output_size, out, dout):
        dx = grad(dout[0], x, out[1])
        return (dx, P.ZerosLike()(output_size))

    return bprop


@bprop_getters.register(AvgPoolV1)
def get_bprop_avg_pool_v1_grad(self):
    """Grad definition for `AvgPoolV1` operation."""
    avgpool_grad_v1 = AvgPoolGradV1(
        kernel_size=self.kernel_size,
        strides=self.strides,
        pad_mode=self.pad_mode,
        data_format=self.format)
    to_arr = P.TupleToArray()
    get_shape = P.Shape()

    def bprop(x, out, dout):
        orig_input_shape = to_arr(get_shape(x))
        dx = avgpool_grad_v1(orig_input_shape, dout)
        return (dx,)

    return bprop


@bprop_getters.register(GLU)
def get_bprop_glu(self):
    """Grad definition for `Glu` operation."""
    input_grad = G.GluGrad(self.axis)

    def bprop(x, out, dout):
        dx = input_grad(dout, x)
        return (dx,)

    return bprop


@bprop_getters.register(MaxPoolWithArgmaxV2)
def get_bprop_maxpoolwithargmaxv2(self):
    """Grad definition for `MaxPoolWithArgmaxV2` operation."""
    maxpoolwithargmaxv2_grad = G.MaxPoolGradWithArgmaxV2(
        kernel_size=self.kernel_size,
        strides=self.strides,
        pads=self.pads,
        dilation=self.dilation,
        ceil_mode=self.ceil_mode,
        argmax_type=self.argmax_type)

    def bprop(x, out, dout):
        dx = maxpoolwithargmaxv2_grad(x, dout[0], out[1])
        return (dx,)

    return bprop


@bprop_getters.register(FractionalMaxPoolWithFixedKsize)
def get_bprop_fractional_max_pool_with_fixed_ksize(self):
    """Grad definition for 'FractionalMaxPoolWithFixedKsize' operation."""
    input_grad = FractionalMaxPoolGradWithFixedKsize(data_format=self.data_format)

    def bprop(x, random_samples, out, dout):
        dx = input_grad(x, dout[0], out[1])
        return (dx, zeros_like(random_samples))

    return bprop


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
def get_bprop_gru_v2(self):
    """Grad definition for `GRUV2` operation."""
    gru_grad_v2 = G.GRUV2Grad(
        self.input_size,
        self.hidden_size,
        self.num_layers,
        self.has_bias,
        self.bidirectional,
        self.dropout
    )

    def bprop(x, hx, w, seq_length, out, dout):
        y, hy, reverse, _ = out
        dy, dhy, _, _ = dout
        dx, dhx, dw = gru_grad_v2(x, hx, w, seq_length, y, hy, dy, dhy, reverse)
        return tuple((dx, dhx, dw, (0)))

    return bprop


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
        res = (dxt, dht, dct_1, dw, db)
        return res

    return bprop


@bprop_getters.register(WKV)
def get_bprop_wkv(self):
    """Grad definition for `wkv` operation."""
    wkv_backward = WKVGrad()

    def bpro(w, u, k, v, sp, sq, sm, out, dout):
        gw, gu, gk, gv = wkv_backward(w, u, k, v, dout[0])
        gw = F.sum(gw, 0)
        gu = F.sum(gu, 0)
        res = (gw, gu, gk, gv, zeros_like(sp), zeros_like(sq), zeros_like(sm))
        return res

    return bpro
