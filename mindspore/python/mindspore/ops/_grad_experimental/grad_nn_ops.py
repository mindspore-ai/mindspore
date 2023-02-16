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

from mindspore import Tensor
import mindspore.numpy as mnp
from mindspore.ops import matmul
from mindspore.ops.operations.nn_ops import GridSampler2D
from mindspore.ops.operations.nn_ops import GridSampler3D
from mindspore.ops.primitive import constexpr
from mindspore.common import dtype as mstype
from mindspore.ops._grad.grad_base import bprop_getters
from mindspore.ops import operations as P
from mindspore.ops.composite.multitype_ops.zeros_like_impl import zeros_like
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops.operations.nn_ops import MaxUnpool2D
from mindspore.ops.operations.nn_ops import MaxUnpool3D
from mindspore.ops.operations.nn_ops import Dilation2D
from mindspore.ops.operations.nn_ops import FractionalMaxPool
from mindspore.ops.operations.nn_ops import SparseSoftmaxCrossEntropyWithLogitsV2
from mindspore.ops.operations._grad_ops import FractionalMaxPoolGrad
from mindspore.ops.operations.nn_ops import FractionalMaxPool3DWithFixedKsize
from mindspore.ops.operations._grad_ops import FractionalMaxPool3DGradWithFixedKsize
from mindspore.ops.operations.nn_ops import FractionalAvgPool
from mindspore.ops.operations._grad_ops import FractionalAvgPoolGrad
from mindspore.ops.operations.nn_ops import InstanceNormV2
from mindspore.ops.operations._grad_ops import InstanceNormV2Grad
from mindspore.ops.operations.nn_ops import MultiMarginLoss
from mindspore.ops.operations.nn_ops import MultilabelMarginLoss
from mindspore.ops.operations.nn_ops import NthElement
from mindspore.ops.operations.nn_ops import UpsampleTrilinear3D
from mindspore.ops.operations._grad_ops import UpsampleTrilinear3DGrad
from mindspore.ops.operations.nn_ops import Pdist
from mindspore.ops.operations._grad_ops import PdistGrad
from mindspore.ops.operations.nn_ops import PSROIPooling
from mindspore.ops.operations._grad_ops import PSROIPoolingGrad
from mindspore.ops.operations.nn_ops import PadV3
from mindspore.ops.operations._grad_ops import PadV3Grad
from mindspore.ops.operations.nn_ops import AvgPoolV1
from mindspore.ops.operations._grad_ops import AvgPoolGradV1
from mindspore.ops.operations.nn_ops import AdaptiveAvgPool2DV1
from mindspore.ops.operations._grad_ops import AdaptiveAvgPool2DGradV1
from mindspore.ops.operations.nn_ops import UpsampleNearest3D
from mindspore.ops.operations._grad_ops import UpsampleNearest3DGrad
from mindspore.ops.operations.nn_ops import MaxPoolV1
from mindspore.ops.operations._grad_ops import MaxPoolGradV1
from mindspore.ops.operations.nn_ops import ReLUV3
from mindspore.ops.operations._grad_ops import ReluGrad
from mindspore.ops.operations.image_ops import ResizeLinear1D
from mindspore.ops.operations.nn_ops import MaxPool3DWithArgmax
from mindspore.ops.operations.nn_ops import FractionalMaxPoolWithFixedKsize
from mindspore.ops.operations._grad_ops import FractionalMaxPoolGradWithFixedKsize
from mindspore.ops.operations.nn_ops import AdaptiveAvgPool3D
from mindspore.ops.operations.nn_ops import GLU
from mindspore.ops.operations.nn_ops import AdaptiveMaxPool3D


@bprop_getters.register(P.CTCLossV2)
def get_bprop_ctc_loss_v2(self):
    """Grad definition for `CTCLossV2` operation"""
    ctc_loss_grad = P.CTCLossV2Grad(self.blank, self.reduction, self.zero_infinity)

    def bprop(log_probs, targets, input_lengths, target_lengths, out, dout):
        grad = ctc_loss_grad(dout[0], log_probs, targets, input_lengths, target_lengths, out[0], out[1])
        return grad, zeros_like(targets), zeros_like(input_lengths), zeros_like(target_lengths)

    return bprop


@bprop_getters.register(InstanceNormV2)
def get_bprop_instance_norm_v2(self):
    """Grad definition for `InstanceNormV2` operation."""
    grad_ops = InstanceNormV2Grad(self.is_training, self.epsilon)

    def bprop(x, gamma, beta, mean, variance, out, dout):
        saved_mean = out[1]
        saved_variance = out[2]
        grad_ops_out = grad_ops(dout[0], x, gamma, mean, variance, saved_mean, saved_variance)
        dx = grad_ops_out[0]
        dgamma = grad_ops_out[1]
        dbeta = grad_ops_out[2]
        pd_res = (dx, dgamma, dbeta, zeros_like(mean), zeros_like(variance))
        return pd_res

    return bprop


@bprop_getters.register(P.SoftMarginLoss)
def get_bprop_soft_margin_loss(self):
    """Grad definition for `SoftMarginLoss` operation."""
    grad = G.SoftMarginLossGrad(reduction=self.reduction)

    def bprop(predict, label, out, dout):
        dx = grad(predict, label, dout)
        dy = grad(label, predict, dout)
        return dx, dy

    return bprop


@bprop_getters.register(P.SoftShrink)
def get_bprop_softshrink(self):
    """Grad definition for `SoftShrink` operation."""
    input_grad = G.SoftShrinkGrad(self.lambd)

    def bprop(input_x, out, dout):
        dx = input_grad(dout, input_x)
        return (dx,)

    return bprop


@bprop_getters.register(P.HShrink)
def get_bprop_hshrink(self):
    """Grad definition for `HShrinkGrad` operation."""
    grad = G.HShrinkGrad(self.lambd)

    def bprop(features, out, gradients):
        dx = grad(gradients, features)
        return (dx,)

    return bprop


@bprop_getters.register(PadV3)
def get_bprop_pad_v3(self):
    """Grad definition for `PadV3` operation."""
    mode = self.mode
    if mode == 'constant':
        pad_v3_grad = PadV3(mode, self.paddings_contiguous)
    else:
        pad_v3_grad = PadV3Grad(mode, self.paddings_contiguous)

    def bprop(x, paddings, constant_values, out, dout):
        if mode == 'constant':
            if isinstance(paddings, (list, tuple)):
                neg_paddings = tuple(-x for x in paddings)
            else:
                neg_paddings = -paddings
            dx = pad_v3_grad(dout, neg_paddings, zeros_like(constant_values))
        else:
            dx = pad_v3_grad(dout, paddings)
        return (dx, zeros_like(paddings), zeros_like(constant_values))

    return bprop


@bprop_getters.register(MultilabelMarginLoss)
def get_bprop_multilabel_margin_loss(self):
    """Grad definition for `MultilabelMarginLoss` operation."""
    input_grad = G.MultilabelMarginLossGrad(reduction=self.reduction)

    def bprop(x, target, out, dout):
        dx = input_grad(dout[0], x, target, out[1])
        return dx, zeros_like(target)

    return bprop


@bprop_getters.register(Dilation2D)
def get_bprop_dilation2d(self):
    """Grad definition for `Dilation2D` operation"""

    input_grad = G.Dilation2DBackpropInput(stride=self.stride, dilation=self.dilation,
                                           pad_mode=self.pad_mode, data_format=self.data_format)
    filter_grad = G.Dilation2DBackpropFilter(stride=self.stride, dilation=self.dilation,
                                             pad_mode=self.pad_mode, data_format=self.data_format)

    def bprop(x, _filter, out, dout):
        dx = input_grad(x, _filter, dout)
        dfilter = filter_grad(x, _filter, dout)
        return dx, dfilter

    return bprop


@bprop_getters.register(P.CeLU)
def get_bprop_celu(self):
    """Grad definition for `CeLU` operation."""
    alpha = self.alpha
    greater_equal = P.GreaterEqual()

    def bprop(x, out, dout):
        condition = greater_equal(x, 0.0)
        dx = dout * mnp.where(condition, 1.0, out / alpha + 1.0)
        return (dx,)

    return bprop


@bprop_getters.register(Pdist)
def get_bprop_pdist(self):
    """Generate bprop for Pdist"""
    input_grad = PdistGrad(p=self.p)

    def bprop(x, out, dout):
        dx = input_grad(dout, x, out)
        return (dx,)

    return bprop


@bprop_getters.register(MultiMarginLoss)
def get_bprop_multi_margin_loss(self):
    """Grad definition for `MultiMarginLoss` operation."""
    input_grad = G.MultiMarginLossGrad(p=self.p, margin=self.margin, reduction=self.reduction)

    def bprop(x, target, weight, out, dout):
        dx = input_grad(dout, x, target, weight)
        return dx, zeros_like(target), zeros_like(weight)

    return bprop


@bprop_getters.register(UpsampleNearest3D)
def get_bprop_upsample_nearest_3d(self):
    """Grad definition for `UpsampleNearest3D` operation."""
    output_size = self.output_size
    scales = self.scales

    def bprop(x, out, dout):
        x_shape = P.Shape()(x)
        grad_op = UpsampleNearest3DGrad(x_shape, output_size, scales)
        dx = grad_op(dout)
        return (dx,)

    return bprop


@bprop_getters.register(UpsampleTrilinear3D)
def get_bprop_upsample_trilinear_3d(self):
    """Grad definition for `UpsampleTrilinear3D` operation."""
    output_size = self.output_size
    scales = self.scales
    align_corners = self.align_corners

    def bprop(x, out, dout):
        input_size = P.Shape()(x)
        grad_op = UpsampleTrilinear3DGrad(input_size, output_size, scales, align_corners)
        dx = grad_op(dout)
        return (dx,)

    return bprop


@bprop_getters.register(GridSampler3D)
def get_bprop_grid_sampler_3d(self):
    """Grad definition for `GridSampler3D` operation."""
    grad = G.GridSampler3DGrad(self.interpolation_mode, self.padding_mode, self.align_corners)

    def bprop(input_x, grid, out, dout):
        dx, dgrid = grad(dout, input_x, grid)
        return dx, dgrid
    return bprop


@bprop_getters.register(ReLUV3)
def get_bprop_relu(self):
    """Grad definition for `ReLUV3` operation."""
    input_grad = ReluGrad()

    def bprop(x, out, dout):
        dx = input_grad(dout, out)
        return (dx,)

    return bprop


@bprop_getters.register(MaxUnpool2D)
def get_bprop_maxunpool2d(self):
    """Grad definition for `MaxUnpool2D` operation."""
    maxunpool2d_grad = G.MaxUnpool2DGrad(
        ksize=self.ksize,
        strides=self.strides,
        pads=self.pads,
        output_shape=self.output_shape,
        data_format=self.data_format)

    def bprop(x, argmax, out, dout):
        dx = maxunpool2d_grad(x, dout, argmax)
        dargmax = zeros_like(argmax)
        return (dx, dargmax)

    return bprop


@bprop_getters.register(MaxUnpool3D)
def get_bprop_maxunpool3d(self):
    """Grad definition for `MaxUnpool3D` operation."""
    maxunpool3d_grad = G.MaxUnpool3DGrad(
        ksize=self.ksize,
        strides=self.strides,
        pads=self.pads,
        output_shape=self.output_shape,
        data_format=self.data_format)

    def bprop(x, argmax, out, dout):
        dx = maxunpool3d_grad(x, dout, argmax)
        dargmax = zeros_like(argmax)
        return (dx, dargmax)

    return bprop


@bprop_getters.register(NthElement)
def get_bprop_nth_element(self):
    """Grad definition for `NthElement` operation."""
    expand_dims = P.ExpandDims()
    cast = P.Cast()
    equal = P.Equal()
    reduce_sum = P.ReduceSum()
    divide = P.Div()

    def bprop(input_x, n, out, dout):
        indicators = cast(equal(expand_dims(out, -1), input_x), mstype.float32)
        dout = expand_dims(dout, -1)
        num_select = expand_dims(reduce_sum(indicators, -1), -1)
        return cast(divide(indicators, num_select) * dout, input_x.dtype), None

    return bprop


@bprop_getters.register(AdaptiveAvgPool3D)
def get_bprop_adaptiveavgpool3d(self):
    """Grad definition for `AdaptiveAvgPool3D` operation."""
    grad = G.AdaptiveAvgPool3DGrad()

    def bprop(x, out, dout):
        get_x_shape = P.TensorShape()
        x_shape = get_x_shape(x).astype(mstype.int32)
        dx = grad(dout, x_shape)
        return (dx,)

    return bprop


@bprop_getters.register(AdaptiveAvgPool2DV1)
def get_bprop_adaptive_avg_pool_2d_v1(self):
    """Grad definition for `AdaptiveAvgPool2DV1` operation."""

    shape = P.Shape()
    def bprop(x, out, dout):
        adaptive_avgpool_grad = AdaptiveAvgPool2DGradV1(orig_input_shape=shape(x))
        dx = adaptive_avgpool_grad(dout)
        return (dx,)

    return bprop


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


@constexpr
def _create_tensor(x_shape):
    return Tensor(x_shape, mstype.int64)


@bprop_getters.register(FractionalAvgPool)
def get_bprop_fractional_avg_pool(self):
    """Grad definition for `FractionalAvgPool` operation."""
    fractional_avg_pool_grad = FractionalAvgPoolGrad(overlapping=self.overlapping)

    def bprop(x, out, dout):
        dx = fractional_avg_pool_grad(_create_tensor(x.shape), dout[0], out[1], out[2])
        return (dx,)

    return bprop


@bprop_getters.register(PSROIPooling)
def get_bprop_p_s_r_o_i_pooling(self):
    """Grad definition for `PSROIPooling` operation."""
    spatial_scale = self.spatial_scale
    group_size = self.group_size
    output_dim = self.output_dim

    def bprop(x, rois, out, dout):
        shape = P.Shape()(x)
        p_s_r_o_i_pooling_grad = PSROIPoolingGrad((shape[2:]), spatial_scale, group_size, output_dim)
        dx = p_s_r_o_i_pooling_grad(dout, rois)
        return (dx, zeros_like(rois))

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


@bprop_getters.register(MaxPoolV1)
def get_bprop_max_pool_v1_grad(self):
    """Grad definition for `MaxPoolV1` operation."""
    maxpool_grad_v1 = MaxPoolGradV1(
        kernel_size=self.kernel_size,
        strides=self.strides,
        pad_mode=self.pad_mode,
        data_format=self.format)

    def bprop(x, out, dout):
        dx = maxpool_grad_v1(x, out, dout)
        return (dx,)
    return bprop


@bprop_getters.register(SparseSoftmaxCrossEntropyWithLogitsV2)
def get_bprop_sparse_softmax_cross_entropy_with_logits_v2(self):
    """Grad definition for `SparseSoftmaxCrossEntropyWithLogitsV2` operation."""
    mul = P.Mul()
    squeeze = P.Squeeze(axis=1)
    softmax_op = P.Softmax()
    expand_dims = P.ExpandDims()

    def bprop(logits, labels, out, dout):
        grad_loss = dout[0]
        softmax_grad = out[1]
        grad_loss = expand_dims(grad_loss, -1)
        grad = mul(grad_loss, softmax_grad)
        if dout[1] is not None:
            softmax = softmax_op(logits)
            grad += ((dout[1] - squeeze(matmul(expand_dims(dout[1], 1), expand_dims(softmax, 2)))) * softmax)
        return grad, zeros_like(labels)

    return bprop


@bprop_getters.register(GridSampler2D)
def get_bprop_grid_sampler_2d(self):
    """Grad definition for `GridSampler2D` operation."""
    grad = G.GridSampler2DGrad(self.interpolation_mode, self.padding_mode, self.align_corners)

    def bprop(input_x, grid, out, dout):
        dx, dgrid = grad(dout, input_x, grid)
        return dx, dgrid

    return bprop


@bprop_getters.register(ResizeLinear1D)
def get_bprop_resize_bilinear(self):
    """Grad definition for `ResizeLinear1D` operation."""
    resize_grad = G.ResizeLinear1DGrad(self.coordinate_transformation_mode)

    def bprop(input_x, size, out, dout):
        dx = resize_grad(dout, input_x)
        return (dx, zeros_like(size))

    return bprop


@bprop_getters.register(GLU)
def get_bprop_glu(self):
    """Grad definition for `Glu` operation."""
    input_grad = G.GluGrad(self.axis)

    def bprop(x, out, dout):
        dx = input_grad(dout, x)
        return (dx,)

    return bprop


@bprop_getters.register(MaxPool3DWithArgmax)
def get_bprop_maxpool3dwithargmax(self):
    """Grad definition for `MaxPool3DWithArgmax` operation."""
    maxpool3dwithargmax_grad = G.MaxPool3DGradWithArgmax(
        ksize=self.ksize,
        strides=self.strides,
        pads=self.pads,
        dilation=self.dilation,
        ceil_mode=self.ceil_mode,
        data_format=self.data_format)

    def bprop(x, out, dout):
        dx = maxpool3dwithargmax_grad(x, dout[0], out[1])
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
