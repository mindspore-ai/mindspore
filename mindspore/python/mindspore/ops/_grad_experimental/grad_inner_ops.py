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

from mindspore.ops.operations.comm_ops import _VirtualPipelineEnd
from mindspore.ops._grad.grad_base import bprop_getters
from mindspore.ops.operations import _inner_ops as inner
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops.composite.multitype_ops.zeros_like_impl import zeros_like


@bprop_getters.register(inner.TensorCopySlices)
def get_bprop_tensor_copy_slices(self):
    """Generate bprop for TensorCopySlices"""
    tensor_copy_slices = inner.TensorCopySlices()

    def bprop(x, update, begin, end, stride, out, dout):
        x_grad = tensor_copy_slices(dout, zeros_like(update), begin, end, stride)
        update_grad = F.strided_slice(dout, begin, end, stride)
        return x_grad, update_grad, zeros_like(begin), zeros_like(end), zeros_like(stride)

    return bprop


@bprop_getters.register(_VirtualPipelineEnd)
def get_bprop_virtual_pipeline_end(self):
    """Backpropagator for _VirtualPipelineEnd."""
    grad = _VirtualPipelineEnd()

    def bprop(x, out, dout):
        dx = grad(dout)
        return (dx,)
    return bprop


@bprop_getters.register(inner.DynamicResizeNearestNeighbor)
def get_bprop_dynamic_resize_nearest_neighbor(self):
    """Generate bprop for DynamicResizeNearestNeighbor"""
    op = G.ResizeNearestNeighborGrad(self.align_corners)
    tensor_shape = P.TensorShape()
    shape_op = P.Shape()

    def bprop(inputs, size, out, dout):
        if F.is_sequence_value_unknown(shape_op(inputs)):
            shp = tensor_shape(inputs)
        else:
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
