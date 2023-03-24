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

"""Define the grad rules of math related operations."""

from mindspore.common import dtype as mstype
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops.composite.multitype_ops.zeros_like_impl import zeros_like
from mindspore.ops.functional import broadcast_gradient_args
from mindspore.ops._grad.grad_base import bprop_getters
from mindspore.ops._grad.grad_base import sum_grad_reduce_axis
from mindspore.ops._grad.grad_base import dyn_ones, dyn_rank_1d
from mindspore.ops.operations._inner_ops import DynamicBroadcastGradientArgs


shape_op = P.Shape()
dyn_shape_op = P.TensorShape()
reduce_sum = P.ReduceSum()
reshape = P.Reshape()


def dyn_binop_grad_common(x, y, dx, dy):
    """
    Common grad definition for binary operations when the input is dynamic shape.

    The function is usually used in backprop op to reduce additional dimensions created by broadcasting.
    """
    shape_of_x = dyn_shape_op(x)
    shape_of_y = dyn_shape_op(y)
    rx, ry = DynamicBroadcastGradientArgs()(shape_of_x, shape_of_y)
    dx_origin_dtype = dx.dtype
    if dx_origin_dtype in (mstype.int16, mstype.int32, mstype.int64):
        dx = F.cast(dx, mstype.float32)
        dx = sum_grad_reduce_axis(dx, rx)
        dx = F.cast(dx, dx_origin_dtype)
    else:
        dx = sum_grad_reduce_axis(dx, rx)
    dy_origin_dtype = dy.dtype
    if dy_origin_dtype in (mstype.int16, mstype.int32, mstype.int64):
        dy = F.cast(dy, mstype.float32)
        dy = sum_grad_reduce_axis(dy, ry)
        dy = F.cast(dy, dy_origin_dtype)
    else:
        dy = sum_grad_reduce_axis(dy, ry)
    reduce_dx = reshape(dx, shape_of_x)
    reduce_dy = reshape(dy, shape_of_y)
    return reduce_dx, reduce_dy


def dyn_binop_grad_common_with_shift(x, y, dx, dy, shift):
    """
    Common grad definition for binary operations with shift when the input is dynamic shape.

    The function is usually used in backprop op to reduce additional dimensions created by broadcasting.
    """
    shape_of_x = dyn_shape_op(x)
    shape_of_y = dyn_shape_op(y)
    broadcast_shape_of_x = shape_of_x[:-shift]
    broadcast_shape_of_y = shape_of_y[:-shift]
    rx, ry = DynamicBroadcastGradientArgs()(broadcast_shape_of_x, broadcast_shape_of_y)
    dx = sum_grad_reduce_axis(dx, rx)
    dy = sum_grad_reduce_axis(dy, ry)
    reduce_dx = reshape(dx, shape_of_x)
    reduce_dy = reshape(dy, shape_of_y)
    return reduce_dx, reduce_dy


def _reduce_sum_with_cast(dx, axis):
    dx_origin_dtype = dx.dtype
    # Currently, for Ascend and GPU, the reduce_sum's input does not support int16, int32 and int64.
    if dx_origin_dtype in (mstype.int16, mstype.int32, mstype.int64):
        dx = F.cast(dx, mstype.float32)
        dx = reduce_sum(dx, axis)
        return F.cast(dx, dx_origin_dtype)
    return reduce_sum(dx, axis)


def binop_grad_common(x, y, dx, dy):
    """
    Common grad definition for binary operations.

    The function is usually used in backprop op to reduce additional dimensions created by broadcasting.
    """
    shape_of_x = shape_op(x)
    shape_of_y = shape_op(y)
    # if input shape is the same as dout shape, do not need to reduce
    reduce_dx = dx
    reduce_dy = dy
    if not (F.is_sequence_value_unknown(shape_of_x) or F.is_sequence_value_unknown(shape_of_y)):
        rx = broadcast_gradient_args(shape_of_x, shape_of_y)
        if rx[0]:
            # if dx is scalar whose shape is (), do not need reduce
            if shape_op(dx):
                dx = _reduce_sum_with_cast(dx, rx[0])
            reduce_dx = reshape(dx, shape_of_x)
        if rx[1]:
            # if dy is scalar whose shape is (), do not need reduce
            if shape_op(dy):
                dy = _reduce_sum_with_cast(dy, rx[1])
            reduce_dy = reshape(dy, shape_of_y)
        return reduce_dx, reduce_dy

    if not isinstance(shape_of_x, tuple) or not isinstance(shape_of_y, tuple):
        # x or y is scalar
        if not isinstance(shape_of_x, tuple):
            reduce_dx = _reduce_sum_with_cast(dx, ())
        if not isinstance(shape_of_y, tuple):
            reduce_dy = _reduce_sum_with_cast(dy, ())
        return reduce_dx, reduce_dy

    return dyn_binop_grad_common(x, y, dx, dy)


def binop_grad_common_with_shift(x, y, dx, dy, shift):
    """
    Common grad definition for binary operations with shift.

    The function is usually used in backprop op to reduce additional dimensions created by broadcasting.
    """
    shape_of_x = shape_op(x)
    shape_of_y = shape_op(y)
    broadcast_shape_of_x = shape_of_x[:-shift]
    broadcast_shape_of_y = shape_of_y[:-shift]
    # if input shape is the same as dout shape, do not need to reduce
    reduce_dx = dx
    reduce_dy = dy
    if not (F.is_sequence_value_unknown(broadcast_shape_of_x) or F.is_sequence_value_unknown(broadcast_shape_of_y)):
        rx = broadcast_gradient_args(broadcast_shape_of_x, broadcast_shape_of_y)
        if rx[0]:
            # if dx is scalar whose shape is (), do not need reduce
            if shape_op(dx):
                dx = _reduce_sum_with_cast(dx, rx[0])
            reduce_dx = reshape(dx, shape_of_x)
        if rx[1]:
            # if dy is scalar whose shape is (), do not need reduce
            if shape_op(dy):
                dy = _reduce_sum_with_cast(dy, rx[1])
            reduce_dy = reshape(dy, shape_of_y)
        return reduce_dx, reduce_dy

    if not isinstance(shape_of_x, tuple) or not isinstance(shape_of_y, tuple):
        # x or y is scalar
        if not isinstance(shape_of_x, tuple):
            reduce_dx = _reduce_sum_with_cast(dx, ())
        if not isinstance(shape_of_y, tuple):
            reduce_dy = _reduce_sum_with_cast(dy, ())
        return reduce_dx, reduce_dy

    return dyn_binop_grad_common_with_shift(x, y, dx, dy, shift)


def _onehot_with_neg_axis(axis, indices, depth, on_value_dtype):
    """onehot support tensor axis"""
    depth_range = P.Range()(F.cast(0, depth.dtype), depth, F.cast(1, depth.dtype))
    indices_expand = P.ExpandDims()(indices, axis)
    indices_expand_rank = dyn_rank_1d(indices_expand)
    broad_shape = dyn_ones(indices_expand_rank, mstype.int64)
    # It should use int64 dtype, but the TensorScatterUpdate op does not support the int64
    # dtype on Ascend device, so the float32 dtype is used here.
    update_dtype = mstype.float32
    broad_shape = dyn_ones(indices_expand_rank, update_dtype)
    broad_shape[axis] = F.cast(depth, update_dtype)
    broad_shape = F.cast(broad_shape, mstype.int64)
    depth_broad = P.Reshape()(depth_range, broad_shape)
    one_hot_bool = P.Equal()(indices_expand, depth_broad)
    one_hot_res = F.cast(one_hot_bool, on_value_dtype)
    return one_hot_res


@bprop_getters.register(P.TensorAdd)
def get_bprop_tensor_add(self):
    """Grad definition for `Add` operation."""

    def bprop(x, y, out, dout):
        return binop_grad_common(x, y, dout, dout)

    return bprop


@bprop_getters.register(P.BitwiseAnd)
def get_bprop_bitwiseand(self):
    """Grad definition for `BitwiseAnd` operation."""

    def bprop(x, y, out, dout):
        return zeros_like(x), zeros_like(y)

    return bprop


@bprop_getters.register(P.BitwiseOr)
def get_bprop_bitwiseor(self):
    """Grad definition for `BitwiseOr` operation."""

    def bprop(x, y, out, dout):
        return zeros_like(x), zeros_like(y)

    return bprop


@bprop_getters.register(P.BitwiseXor)
def get_bprop_bitwisexor(self):
    """Grad definition for `BitwiseXor` operation."""

    def bprop(x, y, out, dout):
        return zeros_like(x), zeros_like(y)

    return bprop


@bprop_getters.register(P.InplaceUpdate)
def get_bprop_inplace_update(self):
    """Grad definition for `InplaceUpdate` operation."""

    def bprop(x, v, out, dout):
        return zeros_like(x), zeros_like(v)

    return bprop


@bprop_getters.register(P.InplaceUpdateV2)
def get_bprop_inplace_update_v2(self):
    """Grad definition for `InplaceUpdateV2` operation."""

    def bprop(x, indices, v, out, dout):
        return zeros_like(x), zeros_like(indices), zeros_like(v)

    return bprop


@bprop_getters.register(P.InplaceSub)
def get_bprop_inplace_sub(self):
    """Grad definition for `InplaceSub` operation."""

    def bprop(x, input_v, out, dout):
        return zeros_like(x), zeros_like(input_v)

    return bprop


@bprop_getters.register(P.InplaceAdd)
def get_bprop_inplace_add(self):
    """Grad definition for `InplaceAdd` operation."""

    def bprop(x, input_v, out, dout):
        return zeros_like(x), zeros_like(input_v)

    return bprop
