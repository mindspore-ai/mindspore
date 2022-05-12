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
from mindspore import context
from .._grad.grad_base import bprop_getters
from ..operations import _inner_ops as inner
from ..operations import _grad_ops as G
from .. import functional as F
from .. import operations as P
from ..composite.multitype_ops.zeros_like_impl import zeros_like
from ...common import dtype as mstype



@bprop_getters.register(inner.TensorCopySlices)
def get_bprop_tensor_copy_slices(self):
    """Generate bprop for TensorCopySlices"""
    tensor_copy_slices = inner.TensorCopySlices()

    def bprop(x, update, begin, end, stride, out, dout):
        x_grad = tensor_copy_slices(dout, zeros_like(update), begin, end, stride)
        update_grad = F.strided_slice(dout, begin, end, stride)
        return x_grad, update_grad, zeros_like(begin), zeros_like(end), zeros_like(stride)

    return bprop


@bprop_getters.register(inner.Roll)
def get_bprop_roll(self):
    """Generate bprop for Roll"""
    if context.get_context("device_target") == "GPU":
        shift = []
        axis = self.axis
        for tmp in enumerate(self.shift):
            shift.append(-tmp[1])
        roll_grad = inner.Roll(shift, axis)
    else:
        shift = self.shift
        axis = self.axis
        roll_grad = inner.Roll(-shift, axis)

    def bprop(x_input, out, dout):
        dx = roll_grad(dout)
        return (dx,)

    return bprop


@bprop_getters.register(inner.DynamicResizeNearestNeighbor)
def get_bprop_dynamic_resize_nearest_neighbor(self):
    """Generate bprop for DynamicResizeNearestNeighbor"""
    op = G.ResizeNearestNeighborGrad(self.align_corners)
    shape_op = P.Shape()

    def bprop(inputs, size, out, dout):
        shp = shape_op(inputs)
        # 2 and 3 represent the height and width
        shp = (shp[2], shp[3])
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


@bprop_getters.register(inner.FillV2)
def get_bprop_fill_v2(self):
    """Generate bprop for FillV2"""
    sum_op = P.ReduceSum()
    cast_op = P.Cast()

    def bprop(shape, value, out, dout):
        dout_type = F.dtype(dout)
        type_list = [mstype.int8, mstype.int16, mstype.int32, mstype.int64,
                     mstype.uint8, mstype.uint16, mstype.uint32, mstype.uint64, mstype.float16]
        if dout_type in type_list:
            dout = cast_op(dout, mstype.float32)
        if dout_type == mstype.float64:
            dout = cast_op(dout, mstype.float32)
        dvalue = sum_op(dout)
        return zeros_like(shape), cast_op(dvalue, dout_type)

    return bprop
