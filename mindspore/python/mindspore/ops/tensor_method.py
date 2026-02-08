# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Tensor method for overload."""
# pylint: disable=C0413
# pylint: disable=W1309
from mindspore import _checkparam as validator
from mindspore import log as logger
from mindspore import ops
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.composite.multitype_ops import _compile_utils as utils
from mindspore._checkparam import check_axis_in_range
from mindspore.ops.composite.multitype_ops._compile_utils import (
    sequence_to_tensor, _tensor_sub, _tensor_pow, _tensor_div, _tensor_floordiv, _tensor_mod
)
from mindspore.ops.auto_generate.gen_ops_prim import (
    inplace_scatter_src_op, inplace_scatter_src_reduce_op, inplace_scatter_value_op, inplace_scatter_value_reduce_op
)
from mindspore.ops.auto_generate.gen_ops_prim import (
    floor_div_op, floor_div_scalar_op
)
from mindspore.ops.auto_generate import (mul, muls)
# 1 common import

# 2 common import
from mindspore import Tensor
# 3 common import
from mindspore.common import dtype as mstype
# 4 common import
from mindspore.common import COOTensor, CSRTensor

# 5 common import
from mindspore.ops.function.nn_func import gelu
# 6 common import

# 7 common import


# 1 to
from mindspore.ops.auto_generate import cast
# 2 masked_fill

# 3 abs
from mindspore.ops.auto_generate import abs
# 4 __abs__

# 5 add
from mindspore.ops.auto_generate import add_ext, add

# 6 all
from mindspore.ops.auto_generate import all
# 7 allclose

# 8 any
from mindspore.ops.function.math_func import any
# 9 arctan2
from mindspore.ops.function.math_func import arctan2
# 10 argmax
from mindspore.ops.function.array_func import argmax
# 11 argmin
from mindspore.ops.function.math_func import argmin
# 12 argsort
from mindspore.ops.function.array_func import argsort
# 13 atan2
from mindspore.ops.function.math_func import atan2
# 14 bfloat16

# 15 bmm

# 16 bool

# 17 broadcast_to
from mindspore.ops.auto_generate import broadcast_to
# 18 byte

# 19 ceil
from mindspore.ops.function.math_func import ceil
# 20 chunk
from mindspore.ops.function.array_func import chunk
# 21 clamp
from mindspore.ops.auto_generate import clamp_tensor, clamp_scalar
# 22 clip

# 23 cos
from mindspore.ops.function.math_func import cos
# 24 cumprod

# 25 cumsum
from mindspore.ops.function.math_func import cumsum
# 26 dim

# 27 div
from mindspore.ops.function.math_func import div
# 28 divide

# 29 eq
from mindspore.ops.function.math_func import eq
# 30 erf
from mindspore.ops.auto_generate import erf
# 31 exp
from mindspore.ops.auto_generate import exp
# 32 expand

# 33 expand_as

# 34 flatten
from mindspore.ops.function.array_func import flatten

# 35 flip

# 36 float

# 37 floor
from mindspore.ops.function.math_func import floor
# 38 gather
from mindspore.ops.auto_generate import gather
from mindspore.ops.function.array_func import gather_ext
# 39 greater
from mindspore.ops.function.math_func import greater
# 40 greater_equal
from mindspore.ops.function.math_func import greater_equal
# 41 gt

# 42 half

# 43 index_put

# 44 index_select
from mindspore.ops.function.array_func import index_select
# 45 int

# 46 inverse
from mindspore.ops.function.math_func import inverse
# 47 is_contiguous

# 48 isclose
from mindspore.ops.function.math_func import isclose
# 49 isfinite
from mindspore.ops.auto_generate import isfinite
# 50 isnan

# 51 item

# 52 le
from mindspore.ops.function.math_func import le
# 53 less

# 54 less_equal

# 55 log

# 56 log2
from mindspore.ops.function.math_func import log2
# 57 logical_and
from mindspore.ops.function.math_func import logical_and
# 58 logical_not
from mindspore.ops.function.math_func import logical_not
# 59 logical_or
from mindspore.ops.function.math_func import logical_or
# 60 long

# 61 lt

# 62 masked_fill
from mindspore.ops.auto_generate import masked_fill
# 63 masked_select

# 64 matmul
from mindspore.ops.auto_generate import matmul_ext
# 65 max
from mindspore.ops.auto_generate import max_
from mindspore.ops.function.array_func import max as max_func
# 66 maximum

# 67 mean
from mindspore.ops.auto_generate import mean_ext
from mindspore.ops.function.math_func import mean
# 68 min
from mindspore.ops.auto_generate import min_
from mindspore.ops.function.array_func import min as min_func
# 69 minimum

# 70 mul

# 71 nan_to_num

# 72 narrow

# 73 ne

# 74 neg

# 75 negative

# 76 nonzero

# 77 norm

# 78 numel

# 79 numpy

# 80 outer

# 81 permute

# 82 pow
from mindspore.ops.auto_generate import pow
# 83 prod
from mindspore.ops.auto_generate import prod_ext
# 84 reciprocal
from mindspore.ops.function.math_func import reciprocal
# 85 remainder
from mindspore.ops.function.math_func import remainder
# 86 repeat

# 87 repeat_interleave
from mindspore.ops.function.array_func import repeat_interleave, repeat_interleave_ext
# 88 reshape
from mindspore.ops.auto_generate import reshape
# 89 round
from mindspore.ops.function.math_func import round
# 90 rsqrt
from mindspore.ops.auto_generate import rsqrt
# 91 scatter
from mindspore.ops.function.array_func import scatter
# 92 scatter_add
from mindspore.ops.function.array_func import tensor_scatter_add
# 93 select
from mindspore.ops.auto_generate import select, select_ext_view
# 94 sigmoid
from mindspore.ops.auto_generate import sigmoid
from mindspore.ops.auto_generate import inplace_sigmoid as sigmoid_
# 95 sin
from mindspore.ops.auto_generate import sin
# 96 size

# 97 sort
from mindspore.ops.function.array_func import sort
# 98 split
from mindspore.ops.function.array_func import split
# 99 sqrt
from mindspore.ops.auto_generate import sqrt
# 100 square
from mindspore.ops.auto_generate import square
# 101 squeeze

# 102 std

# 103 sub
from mindspore.ops.auto_generate import sub, sub_ext
# 104 sum
from mindspore.ops.function.math_func import sum
# 105 swapaxes

# 106 t
from mindspore.ops.function.math_func import t
# 107 tanh
from mindspore.ops.auto_generate import tanh
# 108 tile
from mindspore.ops.operations.manually_defined import tile
# 109 tolist

# 110 topk
from mindspore.ops.function.array_func import topk
# 111 transpose
from mindspore.ops.auto_generate import transpose, transpose_ext_view
# 112 tril
from mindspore.ops.function.array_func import tril
# 113 trunc

# 114 type

# 115 type_as

# 116 unbind

# 117 unfold

# 118 unique
from mindspore.ops.auto_generate import UniqueDim, Unique2
# 119 unsqeeze

# 121 contiguous

# 122 where
from mindspore.ops.function.array_func import where as where_func

# 123 div_

# 124 fill_

# 125 floor_

# 126 masked_fill_

# 127 mul_

# 128 normal_

# 129 requires_grad_

# 130 sub_

# 131 uniform_

# 132 absolute

# 133 bincount
from mindspore.ops.function.math_func import bincount, roll

# 134 diff

# 135 double

# 136 lcm

# 137 mm

# 138 ravel

# 139 nelement

# 140 stride

# 141 indices

# 142 view_as
from mindspore.ops.auto_generate import view_as
# 143 values

# 144 index_copy

# 145 element_size

# 146 gcd
from mindspore.ops.auto_generate import gcd

# 147 isinf
from mindspore.ops.auto_generate import isinf
# 148 not_equal

# 149 triu

# 150 __eq__

# 151 fmod
from mindspore.ops.function.math_func import fmod
# 152
from mindspore.ops.auto_generate import logaddexp2
# 153
from mindspore.ops.auto_generate import acos_ext, acosh_ext, asin_ext, asinh_ext, atan_ext, dot
# 154 isneginf

# 155
from mindspore.ops.function.math_func import median

# 156
from mindspore.ops.function.math_func import permute
# 157
from mindspore.ops.auto_generate import xlogy_op

# 158

# 159 histc
from mindspore.ops.function.math_func import histc

# 160 frac
from mindspore.ops.function.math_func import frac

# 161 bitwise_not
from mindspore.ops.auto_generate.gen_ops_prim import bitwise_not_op
from mindspore.ops.function.math_func import bitwise_or, bitwise_and, bitwise_xor
from mindspore.ops.auto_generate import logical_xor_op

# 162 log10
from mindspore.ops.function.math_func import log10

from mindspore.ops.auto_generate import clone
from mindspore.ops.function.array_func import new_ones
from mindspore.ops.function.array_func import new_zeros

# 163
from mindspore.ops.auto_generate import cosh
from mindspore.ops.auto_generate import sinc
from mindspore.ops.auto_generate import sinh
from mindspore.ops.function.array_func import unsqueeze

# 186
from mindspore.ops.function.math_func import addcdiv
from mindspore.ops.auto_generate import addcdiv_ext_op

# 204 erfc
from mindspore.ops.auto_generate import erfc

# 207 expm1
from mindspore.ops.auto_generate import expm1

# 220 hardshrink
from mindspore.ops.auto_generate import hardshrink

# 931
from mindspore.ops.function.math_func import nansum

# 244 log1p
from mindspore.ops.auto_generate import log1p

# 501
from mindspore.ops.function.math_func import addbmm
# 502
from mindspore.ops.function.math_func import addmm
# 846
from mindspore.ops.function.math_func import count_nonzero

# 880
from mindspore.ops.auto_generate import lerp, lerp_scalar

# 790 addmv
from mindspore.ops.function.math_func import addmv

# 916 index_add
from mindspore.ops.primitive import constexpr
from mindspore._checkparam import check_is_number

# 1028
from mindspore.ops.function.math_func import var_ext

# 1029 exp_
from mindspore.ops.auto_generate.gen_ops_prim import inplace_exp_op

# 1030 log_
from mindspore.ops.auto_generate.gen_ops_prim import inplace_log_op

# 1031 masked_scatter
from mindspore.ops.auto_generate import masked_scatter


########################################functions########################################
def place_holder():
    logger.error(
        "This is a place holder function and should not be called. Please check the implementation.")


unique_dim_ = UniqueDim()
unique2_ = Unique2()
tuple_slice = validator.tuple_slice
expanded_shape = validator.expanded_shape


# 1 to
def tensor_to(input, dtype):
    return cast(input, dtype)


# 2 masked_fill
def tensor_masked_fill(input_x, mask, value):
    return masked_fill(input_x, mask, value)


# 3 abs
def tensor_abs(input):
    return abs(input)


# 4 __abs__

# 5 add
def tensor_add_ext(input, other, *, alpha=1):
    return add_ext(input, other, alpha=alpha)


def deprecated_tensor_add(input, other):
    if isinstance(other, COOTensor):
        return other + input
    if isinstance(other, (tuple, list)):
        other = sequence_to_tensor(other, F.dtype(input))
    return add(input, other)


# 6 all
def tensor_all(x, axis=None, keep_dims=False):
    return all(x, axis, keep_dims)


def deprecated_tensor_all(x, dim=None, keepdim=False):
    return all(x, dim, keepdim)


# 7 allclose
def tensor_allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    return isclose(input, other, rtol, atol, equal_nan).all().item()


# 8 any
def reduce_tensor_any(x, axis=None, keep_dims=False):
    if axis is None:
        axis = ()
    return any(x, axis, keep_dims)


def tensor_any(input, dim=None, keepdim=False):
    if dim is None:
        dim = ()
    return any(input, dim, keepdim)


# 9 arctan2
def tensor_arctan2(input, other):
    return arctan2(input, other)


# 10 argmax
def tensor_argmax(input, dim=None, keepdim=False):
    return argmax(input, dim, keepdim)


def deprecated_tensor_argmax(input, axis=None, keepdims=False):
    return argmax(input, axis, keepdims)


# 11 argmin
def tensor_argmin(input, dim=None, keepdim=False):
    return argmin(input, dim, keepdim)


def deprecated_tensor_argmin(input, axis=None, keepdims=False):
    return argmin(input, axis, keepdims)


# 12 argsort
def tensor_argsort(input, dim=-1, descending=False, stable=False):
    return argsort(input, dim, descending)


def deprecated_tensor_argsort(input, axis=-1, descending=False):
    return argsort(input, axis, descending)


# 13 atan2
def tensor_atan2(input, other):
    return atan2(input, other)


# 14 bfloat16

# 15 bmm

# 16 bool

# 17 broadcast_to

# 18 byte

# 19 ceil
def tensor_ceil(input):
    return ceil(input)


# 20 chunk
def deprecated_tensor_chunk(input, chunks, axis=0):
    return chunk(input, chunks, axis)


def tensor_chunk(input, chunks, dim=0):
    return chunk(input, chunks, dim)


# 21 clamp
def tensor_clamp_tensor(input, min=None, max=None):
    return clamp_tensor(input, min, max)


def tensor_clamp_scalar(input, min=None, max=None):
    return clamp_scalar(input, min, max)


# 22 clip

# 23 cos
def tensor_cos(input):
    return cos(input)


# 24 cumprod

# 25 cumsum
def deprecated_tensor_cumsum(x, axis=None, dtype=None):
    r"""
    For details, please refer to :func:`mindspore.ops.cumsum`.
    """
    original_dtype = x.dtype
    # If original tensor is int, and has precision less then int32, convert to int32
    if x.dtype in (mstype.bool_, mstype.int8, mstype.int16, mstype.uint8, mstype.int16):
        x = x.astype(mstype.int32)
    if axis is None:
        x = x.ravel()
        axis = 0
    validator.check_axis_in_range(axis, x.ndim)
    if dtype is not None and original_dtype != dtype:
        return cumsum(x, axis).astype(dtype, copy=False)
    return cumsum(x, axis)


def tensor_cumsum(input, dim, *, dtype=None):
    return deprecated_tensor_cumsum(input, dim, dtype)


# 26 dim

# 27 div
def tensor_div(input, value, *, rounding_mode=None):
    return div(input, value, rounding_mode=rounding_mode)


def tensor_div_deal_sequence(input, other, *, rounding_mode=None):
    if isinstance(input, (tuple, list)):
        input = sequence_to_tensor(input, F.dtype(other))
    if isinstance(other, (tuple, list)):
        other = sequence_to_tensor(other, F.dtype(input))
    return div(input, other, rounding_mode=rounding_mode)

# 28 divide

# 29 eq
def tensor_eq(input, other):
    return eq(input, other)


# 30 erf
def tensor_erf(input):
    return erf(input)


# 31 exp
def tensor_exp(input):
    return exp(input)


# 32 expand

# 33 expand_as
def tensor_expand_as(input, other):
    return broadcast_to(input, other.shape)


def deprecated_tensor_expand_as(input, x):
    return broadcast_to(input, x.shape)


# 34 flatten
def deprecated_tensor_flatten(input, order='C', *, start_dim=0, end_dim=-1):
    return flatten(input, order, start_dim=start_dim, end_dim=end_dim)


def tensor_flatten(input, start_dim=0, end_dim=-1):
    return flatten(input, start_dim=start_dim, end_dim=end_dim)


# 35 flip

# 36 float

# 37 floor
def tensor_floor(input):
    return floor(input)


# 38 gather
def tensor_gather_ext(input, dim, index):
    return gather_ext(input, dim, index)


def deprecated_tensor_gather(input, input_indices, axis, batch_dims=0):
    r"""
    For details, please refer to :func:`mindspore.ops.gather`.
    """
    validator.check_is_int(axis, 'axis')
    validator.check_is_int(batch_dims, "batch_dims")
    return gather(input, input_indices, axis, batch_dims)


# 39 greater
def tensor_greater(input, other):
    return greater(input, other)


# 40 greater_equal
def tensor_greater_equal(input, other):
    return greater_equal(input, other)


def deprecated_tensor_greater_equal(input, other):
    return greater_equal(input, other)


# 41 gt

# 42 half

# 43 index_put

# 44 index_select
def tensor_index_select(input, dim, index):
    return index_select(input, dim, index)


def deprecated_tensor_index_select(input, axis, index):
    return index_select(input, axis, index)


# 45 int

# 46 inverse
def tensor_inverse(input):
    return inverse(input)


def deprecated_tensor_inverse(input):
    return inverse(input)


# 47 is_contiguous

# 48 isclose
def deprecated_tensor_isclose(input, x2, rtol=1e-05, atol=1e-08, equal_nan=False):
    return isclose(input, x2, rtol, atol, equal_nan)


# 49 isfinite
def tensor_isfinite(input):
    return isfinite(input)


# 50 isnan

# 51 item

# 52 le
def tensor_le(input, other):
    return le(input, other)


# 53 less
def tensor_less(input, other):
    return F.less(input, other)


# 54 less_equal

# 55 log
def tensor_log(input):
    return F.log(input)


# 56 log2
def tensor_log2(input):
    return log2(input)


# 57 logical_and
def tensor_logical_and(input, other):
    return logical_and(input, other)


# 58 logical_not
def tensor_logical_not(input):
    return logical_not(input)


# 59 logical_or
def tensor_logical_or(input, other):
    return logical_or(input, other)


# 60 long

# 61 lt

# 62 masked_fill

# 63 masked_select
def tensor_masked_select(tensor, mask):
    return F.masked_select(tensor, mask)


# 64 matmul
def tensor_matmul(input, mat2):
    return matmul_ext(input, mat2)


def deprecated_tensor_matmul(input, tensor2):
    return F.matmul(input, tensor2)


# 65 max
def tensor_max(input):
    return max_(input)


def tensor_maxdim(input, dim, keepdim=False):
    argmax_with_value_op = P.ArgMaxWithValue(dim, keepdim)
    indices, values = argmax_with_value_op(input)
    return values, indices


def deprecated_tensor_max(input, axis=None, keepdims=False, *, initial=None, where=True, return_indices=False):
    r"""
    For details, please refer to :func:`mindspore.ops.max`.
    """
    if isinstance(axis, (list, tuple)):
        reduce_max = P.ReduceMax
        maximum = F.maximum
        return utils.reduce_(input, reduce_max(keepdims), cmp_fn=maximum, axis=axis, keepdims=keepdims,
                             initial=initial, where=where)
    values, indices = max_func(input, axis, keepdims, initial=initial, where=where)
    if not return_indices:
        return values
    return values, indices


# 66 maximum
def tensor_maximum(input, other):
    return F.maximum(input, other)


# 67 mean
def tensor_mean_ext(input, axis=None, keep_dims=False, *, dtype=None):
    return mean_ext(input, axis, keep_dims, dtype=dtype)


def deprecated_tensor_mean(input, axis=None, keep_dims=False):
    return mean(input, axis, keep_dims)


# 68 min
def tensor_min(input):
    return min_(input)


def tensor_mindim(input, dim, keepdim=False):
    argmin_with_value_op = P.ArgMinWithValue(dim, keepdim)
    indices, values = argmin_with_value_op(input)
    return values, indices


def deprecated_tensor_min(input, axis=None, keepdims=False, *, initial=None, where=True, return_indices=False):
    r"""
    For details, please refer to :func:`mindspore.ops.min`.
    """
    if isinstance(axis, (list, tuple)):
        reduce_min = P.ReduceMin
        minimum = F.minimum
        return utils.reduce_(input, reduce_min(keepdims), cmp_fn=minimum, axis=axis, keepdims=keepdims,
                             initial=initial, where=where)
    values, indices = min_func(input, axis, keepdims, initial=initial, where=where)
    if not return_indices:
        return values
    return values, indices


# 69 minimum
def tensor_minimum(input, other):
    return F.minimum(input, other)


# 70 mul
def tensor_mul(input, other):
    if isinstance(input, Tensor) and isinstance(other, (int, float, bool)):
        return muls(input, other)
    return mul(input, other)


def deprecated_tensor_mul(input, other):
    r"""
    Deprecated Tensor multiplication implementation.

    This function keeps legacy behavior for special input types:
    - If `other` is a sparse tensor (COOTensor/CSRTensor), dispatches to sparse * dense.
    - If `other` is a Python sequence (tuple/list), converts it to a tensor with `input` dtype.

    Args:
        input (Tensor): The left operand.
        other (Union[Tensor, COOTensor, CSRTensor, tuple, list]): The right operand.

    Returns:
        Tensor: The multiplication result.
    """
    if isinstance(other, (COOTensor, CSRTensor)):
        return other * input
    return tensor_mul(input, other)


# 71 nan_to_num
def tensor_nan_to_num(input, nan=0.0, posinf=None, neginf=None):
    return F.nan_to_num(input, nan, posinf, neginf)


# 72 narrow
def deprecated_tensor_narrow(input, axis, start, length):
    return F.narrow(input, axis, start, length)


# 73 ne
def tensor_ne(input, other):
    return F.ne(input, other)


# 74 neg
def tensor_neg(input):
    return F.neg(input)


# 75 negative

# 76 nonzero

# 77 norm

# 78 numel

# 79 numpy

# 80 outer
def deprecated_tensor_outer(input, vec2):
    return F.outer(input, vec2)


# 81 permute

# 82 pow
def tensor_pow_tensor_tensor(input, exponent):
    return pow(input, exponent)


def deprecated_tensor_pow(input, exponent):
    return pow(input, exponent)


# 83 prod
def tensor_prod(input, axis=None, keep_dims=False, dtype=None):
    return prod_ext(input, axis, keep_dims, dtype)


def deprecated_tensor_prod(input, dim=None, keepdim=False, dtype=None):
    return prod_ext(input, dim, keepdim, dtype)


# 84 reciprocal
def tensor_reciprocal(input):
    return reciprocal(input)


# 85 remainder
def tensor_remainder(input, other):
    return remainder(input, other)


def deprecated_tensor_remainder(input, divisor):
    return remainder(input, divisor)


def deprecated_tensor_mod(input, other):
    return _tensor_mod(input, other)


# 86 repeat
def tensor_repeat(input, *repeats):
    raise RuntimeError("'repeat' is not supported on this device.")


# 87 repeat_interleave
def deprecated_tensor_repeat_interleave(input, repeats, dim=None):
    return repeat_interleave(input, repeats, dim)


def tensor_repeat_interleave_ext(input, repeats, dim=None, *, output_size=None):
    return repeat_interleave_ext(input, repeats, dim, output_size)


# 88 reshape
def tensor_reshape(input, *shape):
    new_shape = validator.check_reshape_shp(shape)
    return reshape(input, new_shape)


# 89 round
def tensor_round(input, decimals=0):
    return round(input, decimals=decimals)


# 90 rsqrt
def tensor_rsqrt(input):
    return rsqrt(input)


# 91 scatter
def tensor_scatter(input, dim, index, src):
    return scatter(input, dim, index, src)


def deprecated_tensor_scatter(input, axis, index, src):
    return scatter(input, axis, index, src)


# 92 scatter_add
def tensor_scatter_add_empty(input, dim, index, src):
    raise ValueError("should not come here for scatter_add method.")


def deprecated_tensor_scatter_add(input, indices, updates):
    return tensor_scatter_add(input, indices, updates)


# 93 select
def tensor_select_ext(input, dim, index):
    return select_ext_view(input, dim, index)


def deprecated_tensor_select(input, condition, y):
    r"""
    For details, please refer to :func:`mindspore.ops.select`.
    """
    if not isinstance(condition, Tensor):
        raise TypeError(f"For 'Tensor.select', the argument 'condition' should be Tensor,"
                        f" but got {type(condition)}.")
    if not isinstance(y, (Tensor, int, float)):
        raise TypeError(f"For 'Tensor.select', the argument 'y' should be Tensor, int or float,"
                        f" but got {type(y)}.")
    if isinstance(y, int) and input.dtype != mstype.int32:
        raise TypeError(f"For 'Tensor.select', if the argument 'y' is int,"
                        f" then the tensor type should be int32 but got {input.dtype}")
    if isinstance(y, float) and input.dtype != mstype.float32:
        raise TypeError(f"For 'Tensor.select', if the argument 'y' is float,"
                        f" then the tensor type should be float32 but got {input.dtype}")
    input_y = y
    if isinstance(y, (int, float)):
        zeros_like = F.zeros_like
        cast_f = F.cast
        input_y = zeros_like(input) + y
        if isinstance(y, int):
            input_y = cast_f(input_y, mstype.int32)
        else:
            input_y = cast_f(input_y, mstype.float32)
    return select(condition, input, input_y)


# 94 sigmoid
def tensor_sigmoid(input):
    return sigmoid(input)


def tensor_sigmoid_(input):
    return sigmoid_(input)


# 95 sin
def tensor_sin(input):
    return sin(input)


# 96 size

# 97 sort
def deprecated_tensor_sort(input, axis=-1, descending=False):
    return sort(input, axis, descending)


def tensor_sort(input, dim=-1, descending=False, stable=False):
    return sort(input, dim, descending)


# 98 split
def deprecated_tensor_split(input, split_size_or_sections, axis=0):
    return split(input, split_size_or_sections, axis)


def tensor_split_tensor(input, split_size, dim=0):
    return deprecated_tensor_split(input, split_size, dim)


def tensor_split_with_size(input, split_size, dim=0):
    return deprecated_tensor_split(input, split_size, dim)


# 99 sqrt
def tensor_sqrt(x):
    return sqrt(x)


# 100 square
def tensor_square(input):
    return square(input)


# 101 squeeze

# 102 std
def tensor_std(input, dim=None, *, correction=0, keepdim=False):
    x_var = input.var(dim, correction, keepdim)
    return F.tensor_pow(x_var, 0.5)


def deprecated_tensor_std(self, axis=None, ddof=0, keepdims=False):
    """
    For details, please refer to :func:`mindspore.ops.std`.
    """
    x_var = self.var(axis, ddof, keepdims)
    return F.tensor_pow(x_var, 0.5)


# 103 sub
def tensor_sub_ext(input, other, *, alpha=1):
    return sub_ext(input, other, alpha=alpha)


def deprecated_tensor_sub(input, y):
    if isinstance(y, COOTensor):
        return F.tensor_scatter_sub(input, y.indices, y.values)
    if isinstance(input, (tuple, list)):
        input = sequence_to_tensor(input, F.dtype(y))
    if isinstance(y, (tuple, list)):
        y = sequence_to_tensor(y, F.dtype(input))
    return sub(input, y)


def deprecated_tensor_sub_(input, y):
    if isinstance(y, COOTensor):
        return F.tensor_scatter_sub(input, y.indices, y.values)
    if isinstance(input, (tuple, list)):
        input = sequence_to_tensor(input, F.dtype(y))
    if isinstance(y, (tuple, list)):
        y = sequence_to_tensor(y, F.dtype(input))
    return sub(input, y)


# 104 sum
def tensor_sum_ext(input, dim=None, keepdim=False, *, dtype=None):
    return sum(input, dim, keepdim, dtype=dtype)


def deprecated_tensor_sum(input, axis=None, dtype=None, keepdims=False, initial=None):
    if initial is None:
        res = sum(input, axis, keepdims, dtype=dtype)
    else:
        res = sum(input, axis, keepdims, dtype=dtype) + initial
    if dtype is not None and (dtype == mstype.bool_):
        res = res.astype(mstype.bool_)
    return res


# 105 swapaxes


# 106 t
def tensor_t(input):
    return t(input)


def deprecated_tensor_t(input):
    r"""
    For details, please refer to :func:`mindspore.ops.t`.
    """
    return t(input)


# 107 tanh
def tensor_tanh(input):
    return tanh(input)


# 108 tile
def tensor_tile(input, dims):
    return tile(input, dims)


def deprecated_tensor_tile(input, reps):
    return tile(input, reps)


# 109 tolist


# 110 topk
def tensor_topk(input, k, dim=-1, largest=True, sorted=True):
    return topk(input, k, dim, largest, sorted)


def deprecated_tensor_topk(input, k, dim=None, largest=True, sorted=True):
    return topk(input, k, dim, largest, sorted)


# 111 transpose
def tensor_transpose_ext(input, dim0, dim1):
    return transpose_ext_view(input, dim0, dim1)


def deprecated_tensor_transpose(input, *axes):
    perm = validator.check_transpose_axis(axes, input.ndim)
    return transpose(input, perm)


def deprecated_tensor_permute(input, *axis):
    perm = validator.check_transpose_axis(axis, input.ndim)
    return permute(input, perm)


# 112 tril
def deprecated_tensor_tril(input, diagonal=0):
    return tril(input, diagonal)


# 113 trunc
def tensor_trunc(input):
    return F.trunc(input)


# 114 type


# 115 type_as
def deprecated_tensor_type_as(input, other):
    return input.astype(other.dtype)


# 116 unbind
def deprecated_tensor_unbind(input, dim=0):
    r"""
    For details, please refer to :func:`mindspore.ops.unbind`.
    """
    return F.unstack(input, dim)


# 117 unfold

# 118 unique
def deprecated_tensor_unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None):
    """
    Function for computing the unique elements of a tensor along a specified dimension or over the entire tensor.

    Args:
        input (Tensor): The input tensor from which to find unique elements.
        sorted (bool, optional): If True, the unique elements will be sorted. Default is True.
        return_inverse (bool, optional): Return the indices of the unique elements of the input tensor if true.
        return_counts (bool, optional): Return the count of each unique element of the input tensor if true.
        dim (int, optional): The dimension along which to find the unique elements.

    Returns:
        Tensor or tuple:
            - If `return_inverse` is False and `return_counts` is False, returns a tensor of unique elements.
            - If `return_inverse` is True, returns a tuple (unique_elements, inverse_indices).
            - If `return_counts` is True, returns a tuple (unique_elements, counts).
            - If both `return_inverse` and `return_counts` are True,
                returns a tuple (unique_elements, inverse_indices, counts).

    Raises:
        ValueError: If `return_inverse` or `return_counts` are mutable (non-constant).
        TypeError: If the `return_counts` argument is not a boolean when `dim` is specified.
    """
    if not F.isconstant(return_inverse) or not F.isconstant(return_counts):
        raise ValueError(
            "For 'unique_ext', 'return_inverse' and 'return_counts' cannot be mutable")
    if dim is None:
        y, inverse_, counts = unique2_(
            input, sorted, return_inverse, return_counts)
    else:
        validator.check_value_type(
            "return_counts", return_counts, [bool], "unique_ext")
        y, inverse_, counts = unique_dim_(input, sorted, return_inverse, dim)
    if return_inverse and return_counts:
        return y, inverse_, counts
    if return_inverse:
        return y, inverse_
    if return_counts:
        return y, counts
    return y


# 119 unsqeeze

# 120 view
def tensor_view_dtype(self, dtype):
    r"""
    Placeholder Tensor method for viewing a tensor with a different dtype.

    This API is currently not supported and will raise `NotImplementedError` when called.

    Args:
        self (Tensor): The input tensor.
        dtype (mindspore.dtype): Target dtype to view.

    Raises:
        NotImplementedError: Currently not supported.
    """
    raise NotImplementedError("Currently not supported.")


# 121 contiguous

# 122 where
def tensor_where(condition, input, other):
    return where_func(condition, input, other)


def deprecated_tensor_where(input, condition, y):
    return where_func(condition, input, y)


# 123 div_

# 124 fill_
def tensor_inplace_fill_scalar_empty(input, value):
    raise ValueError("should not come here for fill_scalar method.")


def tensor_inplace_fill_tensor_empty(input, value):
    raise ValueError("should not come here for fill_tensor method.")


def tensor_inplace_fill_diagonal(input, fill_value, wrap=False):
    raise ValueError("should not come here for fill_diagonal method.")


# 125 floor_

# 126 masked_fill_
def tensor_inplace_masked_fill_scalar_empty(input, masked, value):
    raise ValueError("should not come here for masked_fill_scalar method.")


def tensor_inplace_masked_fill_tensor_empty(input, masked, value):
    raise ValueError("should not come here for masked_fill_tensor method.")


# 127 mul_
def tensor_inplace_mul(input, other):
    return F.mul(input, other)

# 128 normal_

# 129 requires_grad_

# 130 sub_

# 131 uniform_

# 132 absolute

# 133 bincount
def tensor_bincount(input, weights=None, minlength=0):
    return bincount(input, weights, minlength)


def tensor_roll(input, shifts, dims=None):
    return roll(input, shifts, dims)

# 134 diff

# 135 double

# 136 lcm


# 137 mm
def tensor_mm(input, mat2):
    return F.mm(input, mat2)


def deprecated_tensor_mm(input, mat2):
    return F.mm(input, mat2)


# 138 ravel

# 139 nelement

# 140 stride

# 141 indices

# 142 view_as
def tensor_view_as(input, other):
    shape = other.shape
    return reshape(input, shape)


def deprecated_tensor_view_as(input, other):
    return view_as(input, other)


# 143 values

# 144 index_copy

# 145 element_size

# 146 gcd
def tensor_gcd(input, other):
    return gcd(input, other)


# 147 isinf
def tensor_isinf(input):
    return isinf(input)


# 148 not_equal
def tensor_not_equal(input, other):
    return F.ne(input, other)


# 149 triu
def tensor_triu(input, diagonal=0):
    return F.triu(input, diagonal)


# 150 __eq__


# 151 scatter_
def tensor_inplace_scatter_src(input, dim, index, src):
    return inplace_scatter_src_op(input, dim, index, src)


def tensor_inplace_scatter_src_reduce(input, dim, index, src, *, reduce):
    return inplace_scatter_src_reduce_op(input, dim, index, src, reduce=reduce)


def tensor_inplace_scatter_value(input, dim, index, value):
    return inplace_scatter_value_op(input, dim, index, value)


def tensor_inplace_scatter_value_reduce(input, dim, index, value, *, reduce):
    return inplace_scatter_value_reduce_op(input, dim, index, value, reduce=reduce)


# 152 fmod
def fmod_tensor(input, other):
    return fmod(input, other)


def fmod_scalar(input, other):
    return fmod(input, other)


def deprecated_tensor_fmod(input, other):
    return fmod(input, other)


# 153 acos, arccos; acosh, arccosh; asin, arcsin; asinh, arcsinh; atan, arctanh, dot
def tensor_acos(input):
    return acos_ext(input)


def deprecated_tensor_acos(input):
    return F.acos(input)


def tensor_acosh(input):
    return acosh_ext(input)


def deprecated_tensor_acosh(input):
    return F.acosh(input)


def tensor_asin(input):
    return asin_ext(input)


def deprecated_tensor_asin(input):
    return F.asin(input)


def tensor_asinh(input):
    return asinh_ext(input)


def deprecated_tensor_asinh(input):
    return F.asinh(input)


def tensor_atan(input):
    return atan_ext(input)


def deprecated_tensor_atan(input):
    return F.atan(input)


def tensor_atanh(input):
    return F.atanh(input)


def tensor_copy_(input, src, non_blocking=False):
    raise ValueError("should not come here for copy_ method")


def tensor_tan(input):
    return F.tan(input)


def tensor_dot(input, other):
    return dot(input, other)


def deprecated_tensor_dot(input, other):
    return F.dot(input, other)


def deprecated_tensor_logsumexp(input, dim, keepdim=False):
    return F.logsumexp(input, dim, keepdim)


# 154
def tensor_isneginf(input):
    inf_tensor = isinf(input)
    neg_tensor = input < 0
    return logical_and(inf_tensor, neg_tensor)


def deprecated_tensor_isneginf(input):
    inf_tensor = isinf(input)
    neg_tensor = input < 0
    return logical_and(inf_tensor, neg_tensor)


# 155
def deprecated_tensor_median(input, axis=-1, keepdims=False):
    return median(input, axis, keepdims)


def tensor_median(input):
    return median(input)


def tensor_median_dim(input, dim=-1, keepdim=False):
    return median(input, dim, keepdim)


# 156
def tensor_logaddexp2(input, other):
    return logaddexp2(input, other)


def deprecated_tensor_logaddexp2(input, other):
    return F.logaddexp2(input, other)


# 157
def tensor_empty(*size, dtype=None, device=None, pin_memory=False):
    r"""
    For details, please refer to :func:`mindspore.mint.empty`.
    """
    logger.error(
        "This is a function for empty not should be called. Please check the implementation.")


def tensor_empty_like(input, *, dtype=None, device=None, pin_memory=False):
    """
    For details, please refer to :func:`mindspore.mint.empty_like`.
    """
    raise NotImplementedError(
        "This is a function for empty_like should not be called. Please check the implementation.")


def tensor_new_empty(input, size, *, dtype=None, device=None):
    raise NotImplementedError(
        "This is a function for new_empty should not be called. Please check the implementation.")


def deprecated_tensor_logaddexp(input, other):
    return F.logaddexp(input, other)


def tensor_xlogy(input, other):
    if isinstance(other, (float, int, bool)):
        other = F.scalar_to_tensor(other)
    return xlogy_op(input, other)


# 158


# 159 histc
def tensor_histc(input, bins=100, min=0, max=0):
    return histc(input, bins, min, max)


# 160 frac
def tensor_frac(input):
    return frac(input)


# 161 bitwise_not baddbmm bitwise_or bitwise_and bitwise_xor logical_xor
def deprecated_baddbmm(input, batch1, batch2, *, beta=1, alpha=1):
    return F.baddbmm(input, batch1, batch2, beta=beta, alpha=alpha)


def tensor_bitwise_not(input):
    return bitwise_not_op(input)


def deprecated_bitwise_or(input, other):
    return bitwise_or(input, other)


def deprecated_bitwise_and(input, other):
    return bitwise_and(input, other)


def deprecated_bitwise_xor(input, other):
    return bitwise_xor(input, other)


def tensor_logical_xor(input, other):
    return logical_xor_op(input, other)


# 162
def tensor_log10(input):
    return log10(input)


# 186
def deprecated_tensor_addcdiv(input, tensor1, tensor2, value=1):
    return addcdiv(input, tensor1, tensor2, value=value)


def tensor_addcdiv_ext(input, tensor1, tensor2, *, value=1):
    return addcdiv_ext_op(input, tensor1, tensor2, value=value)


# 501
def tensor_addbmm(input, batch1, batch2, *, beta=1, alpha=1):
    return addbmm(input, batch1, batch2, beta=beta, alpha=alpha)


def deprecated_tensor_addbmm(input, batch1, batch2, *, beta=1, alpha=1):
    r"""
    For details, please refer to :func:`mindspore.ops.addbmm`.
    """
    return addbmm(input, batch1, batch2, beta=beta, alpha=alpha)


# 502
def tensor_addmm(input, mat1, mat2, *, beta=1, alpha=1):
    return addmm(input, mat1, mat2, beta=beta, alpha=alpha)


def deprecated_tensor_addmm(input, mat1, mat2, *, beta=1, alpha=1):
    r"""
    For details, please refer to :func:`mindspore.ops.addmm`.
    """
    return addmm(input, mat1, mat2, beta=beta, alpha=alpha)


# 543
def tensor_put_(input, index, source, accumulate=False):
    raise RuntimeError("There is no branch to go function tensor_put_!")


# 790
def tensor_addmv(input, mat, vec, *, beta=1, alpha=1):
    return addmv(input, mat, vec, beta=beta, alpha=alpha)


def deprecated_tensor_addmv(input, mat, vec, beta=1, alpha=1):
    r"""
    For details, please refer to :func:`mindspore.ops.addmv`.
    """
    return addmv(input, mat, vec, beta=beta, alpha=alpha)


# 846
def deprecated_tensor_count_nonzero(input,
                                    axis=(),
                                    keep_dims=False,
                                    dtype=None):
    if dtype is None:
        return count_nonzero(input,
                             axis=axis,
                             keep_dims=keep_dims,
                             dtype=mstype.int32)
    return count_nonzero(input, axis=axis, keep_dims=keep_dims, dtype=dtype)


# 732
def tensor_take(input, index):
    return deprecated_tensor_take(input, index)


def deprecated_tensor_take(x, indices, axis=None, mode='clip'):
    """
    Takes elements from a tensor along an axis.
    """
    if mode not in ('raise', 'wrap', 'clip'):
        raise ValueError(f"For 'Tensor.take', the argument 'mode' should be one of in ['raise', 'wrap', 'clip'],"
                         f" but got {mode}.")
    if axis is None:
        a = x.ravel()
        axis = 0
    else:
        a = x
    ndim = a.ndim
    axis = check_axis_in_range(axis, ndim)

    shape_a = a.shape
    shape_indices = indices.shape
    size_indices = indices.size
    indices = utils.check_indices(shape_a[axis], indices, mode)

    # reshapes indices to shape (Ni..., Nj..., Nk)
    shape_ni = tuple_slice(shape_a, None, axis)
    shape_nk = tuple_slice(shape_a, axis + 1, None)
    shape_out = shape_ni + shape_indices + shape_nk
    shape_indices = expanded_shape(ndim, size_indices, axis)
    indices = indices.reshape(shape_indices)
    shape_indices = shape_ni + (indices.size,) + shape_nk
    indices = F.broadcast_to(indices, shape_indices)

    res = F.gather_d(a, axis, indices)
    return res.reshape(shape_out)


def tensor_clone(input):
    return clone(input)


def tensor_new_ones(input, size, dtype=None):
    return new_ones(input, size, dtype=dtype)


def tensor_new_zeros(input, size, dtype=None):
    return new_zeros(input, size, dtype=dtype)


def tensor_cosh(input):
    return cosh(input)


def tensor_sinh(input):
    return sinh(input)


def tensor_sinc(input):
    return sinc(input)


def tensor_unsqueeze(input, dim):
    return


def deprecated_tensor_unsqueeze(input, dim):
    return unsqueeze(input, dim)


# 204 erfc
def tensor_erfc(input):
    return erfc(input)


# 207 expm1
def tensor_expm1(input):
    return expm1(input)


# 880
def tensor_lerp(input, end, weight):
    return lerp(input, end, weight)


def tensor_lerp_scalar(input, end, weight):
    return lerp_scalar(input, end, weight)


# 220 hardshrink
def tensor_hardshrink(input, lambd=0.5):
    return hardshrink(input, lambd)


# 931
def deprecated_tensor_nansum(input, axis=(), keepdims=False, *, dtype=None):
    return nansum(input, axis, keepdims, dtype=dtype)


# 244 log1p
def tensor_log1p(input):
    return log1p(input)


def tensor_diag(input, diagonal=0):
    if diagonal != 0:
        raise ValueError(f"For 'Tensor.diag', the argument 'diagonal' should be '0', but got {diagonal}.")
    return F.diag(input)


def deprecated_tensor_diag(input):
    return F.diag(input)


def deprecated_einsum(equation, operands):
    raise NotImplementedError('einsum only supports Ascend.')


# 916 index_add
@constexpr
def _check_index_add_alpha(alpha):
    check_is_number(alpha, (int, float))


def tensor_index_add(input, dim, index, source, *, alpha=1):
    _check_index_add_alpha(alpha)
    source = source * alpha
    return F.index_add(input, indices=index, y=source, axis=dim)


def deprecated_tensor_index_add(input, indices, y, axis, use_lock=True, check_index_bound=True):
    return F.index_add(input, indices, y, axis, use_lock, check_index_bound)


# 1028
def tensor_var(input, dim=None, *, correction=1, keepdim=False):
    return var_ext(input, dim, correction=correction, keepdim=keepdim)


def deprecated_tensor_var(input, axis=None, ddof=0, keepdims=False):
    r"""
    For details, please refer to :func:`mindspore.ops.var`.
    """
    if 0 in input.shape:
        return Tensor(float('nan'), input.dtype)
    if not isinstance(ddof, int):
        raise TypeError("For 'Tensor.var', the type of the argument 'ddof' must be int, but got "
                        "{}.".format(type(ddof)))
    if not isinstance(keepdims, bool):
        raise TypeError("For 'Tensor.var', the type of the argument 'keepdims' must be bool, but "
                        "got {}.".format(type(keepdims)))

    if axis is None:
        axis = ()
    else:
        axis = validator.check_and_canonicalize_axes(axis, input.ndim)
    x_mean = mean(input, axis, True)
    x_sub = _tensor_sub(input, x_mean)
    x_pow = _tensor_pow(x_sub, 2)
    x_sum = P.ReduceSum(bool(keepdims))(x_pow, axis)
    nums = 1
    if axis == ():
        nums = input.size
    else:
        for ax in axis:
            nums *= input.shape[ax]
    return _tensor_div(x_sum, nums - ddof)


# 1222
def tensor_index_fill_(input, dim, index, value):
    raise NotImplementedError('Tensor.index_fill_ only supports Ascend.')


def tensor_kthvalue(input, k, dim=-1, keepdim=False):
    raise ValueError("should not come here for kthvalue py_method.")


def tensor_index_copy_(input, dim, index, tensor):
    raise NotImplementedError('Tensor.index_copy_ only supports Ascend.')


def tensor_sub_empty_(input, other, alpha=1):
    raise ValueError("should not come here for sub_ method.")


def tensor_inplace_sub(input, other, *, alpha=1):
    if alpha == 1:
        return sub(input, other)
    return sub_ext(input, other, alpha=alpha)


def tensor_new_full(input, size, fill_value, *, dtype=None):
    raise NotImplementedError("new_full method support Ascend only")


def tensor_div_empty_(input, other, rounding_mode=None):
    raise ValueError("should not come here for div_ method.")


def tensor_subtract(input, other, *, alpha=1):
    return tensor_sub_ext(input, other, alpha=alpha)


def tensor_true_divide(input, other):
    return div(input, other)


def all_gather_matmul(
        input,
        x2,
        group,
        world_size,
        *,
        bias=None,
        gather_index=0,
        gather_output=True,
        comm_turn=0,
        trans_input=False,
        trans_x2=False,
    ):
    """
    For details, please refer to :func:`mindspore.ops.all_gather_matmul`.
    """
    raise NotImplementedError('all_gather_matmul only supports Ascend.')


def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    raise NotImplementedError('conv1d only supports Ascend.')


def conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    raise NotImplementedError('conv3d only supports Ascend.')


def tensor_remainder_(input, other):
    return _tensor_mod(input, other)


def tensor_floor_divide_(input, other):
    return _tensor_floordiv(input, other)


def matmul_reduce_scatter(
        input,
        x2,
        group,
        world_size,
        *,
        reduce_op=ops.ReduceOp.SUM,
        bias=None,
        comm_turn=0,
        trans_input=False,
        trans_x2=False,
    ):
    """
    For details, please refer to :func:`mindspore.ops.matmul_reduce_scatter`.
    """
    raise NotImplementedError('matmul_reduce_scatter only supports Ascend.')


# 1030
def tensor_log_(input):
    return inplace_log_op(input)


def tensor_floor_div(input, other):
    return floor_div_op(input, other)


def tensor_floor_div_scalar(input, other):
    return floor_div_scalar_op(input, other)


# 1029
def tensor_exp_(input):
    return inplace_exp_op(input)


def tensor_gelu(input, *, approximate):
    return gelu(input, approximate)


def tensor_bernoulli_(input, p, seed, offset):
    raise RuntimeError("'bernoulli_' is not supported on this device.")


def deprecated_pixel_shuffle(input, upscale_factor):
    return F.pixel_shuffle(input, upscale_factor)


def tensor_quant_matmul(x1, x2, scale, *, offset=None, pertoken_scale=None, bias=None, output_dtype=None,
                        x1_dtype=None, x2_dtype=None, pertoken_scale_dtype=None, scale_dtype=None, group_sizes=None):
    r"""
    For details, please refer to :func:`mindspore.ops.auto_generate.quant_matmul`.
    """
    raise NotImplementedError('quant_matmul only supports Ascend.')


def tensor_index(input, value):
    raise NotImplementedError("index only supports Ascend.")


def tensor_gmm(x, weight, *, bias=None, group_list=None, group_type=0, group_list_type=0):
    raise NotImplementedError("gmm has not been implemented by python.")


def raise_func(*args, **kwargs):
    raise NotImplementedError("this func has not been implemented.")


def tensor_masked_scatter(input, mask, source):
    return masked_scatter(input, mask, source)


def tensor_inplace_masked_scatter(input, mask, source):
    return F.inplace_masked_scatter(input, mask, source)


def tensor_nsa_select_attention(query, key, value, topk_indices, scale_value, head_num, select_block_size,
                                select_block_count, *, atten_mask=None, actual_seq_qlen=None, actual_seq_kvlen=None):
    r"""
    Placeholder Tensor method for NSA Select Attention.

    This API is used to compute NSA Select Attention. The implementation in this file is a placeholder
    for **non-Ascend devices** and will raise `NotImplementedError` at runtime.

    Args:
        query (Tensor): The Query tensor.
        key (Tensor): The Key tensor.
        value (Tensor): The Value tensor.
        topk_indices (Tensor): TopK indices for selected blocks/tokens.
        scale_value (Union[float, int, Tensor]): Scaling factor for attention.
        head_num (int): Number of attention heads.
        select_block_size (int): Size of a selected block.
        select_block_count (int): Number of selected blocks.

    Keyword Args:
        atten_mask (Tensor, optional): Attention mask. Default: ``None``.
        actual_seq_qlen (Tensor, optional): Actual sequence length info for Query. Default: ``None``.
        actual_seq_kvlen (Tensor, optional): Actual sequence length info for Key/Value. Default: ``None``.

    Raises:
        NotImplementedError: Currently only supported on Ascend devices.
    """
    raise NotImplementedError("nsa_select_attention only supports Ascend.")


def tensor_broadcast_to(x, shape):
    return F.broadcast_to(x, shape)


def tensor_squeeze(input, axis=None):
    return F.squeeze(input, axis)


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    raise NotImplementedError('conv2d only supports Ascend.')


def tensor_real(input):
    r"""
    For details, please refer to :func:`mindspore.ops.real`.
    """
    return ops.real(input)


def tensor_imag(input):
    r"""
    For details, please refer to :func:`mindspore.ops.imag`.
    """
    return ops.imag(input)


def _tensor_nsa_compress(input, weight, compress_block_size, compress_stride, *, actual_seq_len=None):
    """Placeholder for unsupported devices: nsa_compress."""
    raise RuntimeError("'nsa_compress' is only supported on Ascend.")


def _tensor_nsa_compress_attention(
        query,
        key,
        value,
        scale_value,
        head_num,
        compress_block_size,
        compress_stride,
        select_block_size,
        select_block_count,
        *,
        topk_mask=None,
        atten_mask=None,
        actual_seq_qlen=None,
        actual_cmp_seq_kvlen=None,
        actual_sel_seq_kvlen=None,
    ):
    """Placeholder for unsupported devices: nsa_compress_attention."""
    raise RuntimeError("'nsa_compress_attention' is only supported on Ascend.")
