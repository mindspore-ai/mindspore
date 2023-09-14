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

"""inner_ops"""
from __future__ import absolute_import

from mindspore.ops.operations import _inner_ops as inner
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops.composite.multitype_ops.zeros_like_impl import zeros_like
from mindspore.ops._grad_experimental.grad_base import bprop_getters, sum_grad_reduce_axis
import mindspore as ms

reshape = P.Reshape()


@bprop_getters.register(inner.TensorCopySlices)
def get_bprop_tensor_copy_slices(self):
    """Generate bprop for TensorCopySlices"""
    tensor_copy_slices = inner.TensorCopySlices()

    def bprop(x, update, begin, end, stride, out, dout):
        x_grad = tensor_copy_slices(dout, zeros_like(update), begin, end, stride)
        update_grad = F.strided_slice(dout, begin, end, stride)
        res = (x_grad, update_grad, zeros_like(begin), zeros_like(end), zeros_like(stride))
        return res

    return bprop


@bprop_getters.register(inner.DynamicResizeNearestNeighbor)
def get_bprop_dynamic_resize_nearest_neighbor(self):
    """Generate bprop for DynamicResizeNearestNeighbor"""
    op = G.ResizeNearestNeighborGrad(self.align_corners)
    shape_op = P.Shape()

    def bprop(inputs, size, out, dout):
        shp = shape_op(inputs)
        # 2 and 3 represent the height and width
        shp = (shp[2:])
        return (op(dout, shp), zeros_like(size))

    return bprop


@bprop_getters.register(inner.ParallelResizeBilinear)
def get_bprop_parallel_resize_bilinear(self):
    """Grad definition for `ParallelResizeBilinear` operation."""
    grad = G.ParallelResizeBilinearGrad(self.ori_image_size, self.src_start_w, self.dst_start_w,
                                        self.align_corners)

    def bprop(x, size, out, dout):
        dx = grad(dout, x, size)
        return dx, zeros_like(size)

    return bprop


@bprop_getters.register(inner.ConvertToDynamic)
def get_bprop_gpu_convert_to_dynamic_rank(self):
    """Get backprop for ConvertToDynamic."""

    def bprop(x, out, dout):
        return (dout,)
    return bprop


@bprop_getters.register(inner.PsROIPooling)
def get_bprop_ps_roi_pooling(self):
    """Grad definition for `PsROIPooling` operation."""
    shape_op = P.Shape()
    pooled_height = self.pooled_height
    pooled_width = self.pooled_width
    spatial_scale = self.spatial_scale
    out_dim = self.out_dim
    num_rois = self.num_rois

    def bprop(inputs, rois, out, dout):
        mapping_channel = out[1]
        inputs_shape = shape_op(inputs)
        batch_size = inputs_shape[0]
        channels = inputs_shape[1]
        height = inputs_shape[2]
        width = inputs_shape[3]

        dx = G.PsROIPoolingGrad(
            batch_size,
            channels,
            height,
            width,
            num_rois,
            pooled_height,
            pooled_width,
            spatial_scale,
            out_dim
        )(dout[0], rois, mapping_channel)

        return dx, zeros_like(rois)

    return bprop


@bprop_getters.register(inner.DynamicBroadcastTo)
def get_bprop_dynamic_broadcast_to(self):
    """Generate bprop for DynamicBroadcastTo"""
    shape_op = P.Shape()

    def bprop(x, shp, out, dout):
        x_shape = shape_op(x)
        broadcast_shape = shape_op(out)

        _, reduction_axes = inner.DynamicBroadcastGradientArgs()(broadcast_shape, x_shape)
        out_type = dout.dtype
        if out_type in (ms.int16, ms.int32, ms.int64):
            dout = P.Cast()(dout, ms.float32)
            reduced_grad = sum_grad_reduce_axis(dout, reduction_axes, keep_dims=True)
            reduced_grad = P.Cast()(reduced_grad, out_type)
        else:
            reduced_grad = sum_grad_reduce_axis(dout, reduction_axes, keep_dims=True)
        dx = reshape(reduced_grad, x_shape)
        return dx, zeros_like(shp)

    return bprop


@bprop_getters.register(inner.SiLU)
def get_bprop_silu(self):
    """Generate bprop for SiLU"""
    sigmoid_grad = G.SigmoidGrad()
    mul_func = P.Mul()

    def bprop(x, out, dout):
        sigmoid_input = P.Sigmoid()(x)
        bc_dx = mul_func(x, dout)
        bc_dy = mul_func(sigmoid_input, dout)
        dx = sigmoid_grad(sigmoid_input, bc_dx)
        return (dx+bc_dy,)

    return bprop
