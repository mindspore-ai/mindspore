# Copyright 2022 Huawei Technologies Co., Ltd
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


"""Define the grad rules of clip operations."""
from __future__ import absolute_import
from mindspore.ops._grad.grad_base import bprop_getters, dyn_fill
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.operations import _inner_ops as inner
from mindspore.ops.operations import _grad_ops as G
from mindspore.common import dtype as mstype
from mindspore.ops._grad.grad_math_ops import _sum_grad
from mindspore.ops._grad.grad_math_ops import binop_grad_common


@bprop_getters.register(inner.ClipByNorm)
def get_bprop_clip_by_norm(self):
    """Grad definition for `ClipByNorm` operation."""
    neg_op = P.Neg()
    mul_op = P.Mul()
    div_op = P.Div()
    fill_op = P.Fill()
    type_op = P.DType()
    shape_op = P.Shape()
    dyn_shape_op = P.TensorShape()
    cast_op = P.Cast()
    sqrt_op = P.Sqrt()
    max_op = P.Maximum()
    square_op = P.Square()
    reduce_sum_op = P.ReduceSum(keep_dims=True)
    sqrt_grad_op = G.SqrtGrad()
    max_grad_op = G.MaximumGrad()
    reduce_sum_axis = self.get_attr_dict()["axis"]

    def bprop(x, clip_norm, out, dout):
        cast_x = cast_op(x, mstype.float32)
        cast_clip_norm = cast_op(clip_norm, mstype.float32)
        square_out = square_op(cast_x)
        reduce_sum_out = reduce_sum_op(square_out, reduce_sum_axis)
        sqrt_out = sqrt_op(reduce_sum_out)
        max_out = max_op(sqrt_out, cast_clip_norm)
        mul_out = mul_op(cast_x, cast_clip_norm)
        # grad for div operation
        div_bc_x = div_op(dout, max_out)
        div_bc_y = neg_op(mul_op(div_bc_x, out))
        div_dout_x, div_dout_y = binop_grad_common(mul_out, max_out, div_bc_x, div_bc_y)
        # grad for mul operation
        mul_bc_x = mul_op(cast_clip_norm, div_dout_x)
        mul_bc_y = mul_op(cast_x, div_dout_x)
        mul_dout_x, mul_dout_y = binop_grad_common(cast_x, cast_clip_norm, mul_bc_x, mul_bc_y)
        # grad for max operation
        max_dout_x, max_dout_y = max_grad_op(sqrt_out, cast_clip_norm, div_dout_y)
        # grad for sqrt operation
        sqrt_dout_x = sqrt_grad_op(sqrt_out, max_dout_x)
        # grad for reduce_sum operation
        reduce_sum_dout_x = _sum_grad(square_out, reduce_sum_axis, sqrt_dout_x)
        # grad for square operation
        temp_num = 2.0
        temp_out = mul_op(reduce_sum_dout_x, cast_x)
        shape_cast_x = shape_op(cast_x)
        if F.is_sequence_value_unknown(shape_cast_x):
            fill_x = dyn_fill(type_op(temp_out), dyn_shape_op(cast_x), temp_num)
        else:
            fill_x = fill_op(type_op(temp_out), shape_cast_x, temp_num)
        square_dout_x = mul_op(fill_x, temp_out)
        # grad for cast operation
        x_dout = cast_op((mul_dout_x + square_dout_x), type_op(x))
        clip_norm_dout = cast_op((mul_dout_y + max_dout_y), type_op(clip_norm))
        return x_dout, clip_norm_dout

    return bprop
