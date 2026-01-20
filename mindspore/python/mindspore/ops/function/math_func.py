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

# pylint: disable=unused-import
"""Defines math operators with functional form."""

import collections
from functools import cmp_to_key
import math
import numbers
import numpy as np

import mindspore as ms
from mindspore import log as logger
from mindspore import ops
from mindspore.common import dtype as mstype
from mindspore.common._decorator import deprecated
from mindspore.common.generator import default_generator
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops.composite.multitype_ops import _constexpr_utils as const_utils
from mindspore.ops.primitive import constexpr, _primexpr
from mindspore.ops.operations._inner_ops import TileSize
from mindspore.ops.auto_generate import Cummin, BatchMatMul, BernoulliExt, lin_space_ext_op, BitwiseAndScalar, \
    BitwiseAndTensor, BitwiseOrScalar, BitwiseOrTensor, BitwiseXorScalar, BitwiseXorTensor, RemainderTensorTensor, \
    RemainderTensorScalar, RemainderScalarTensor, std_mean_op, var_mean_op, InplaceErfinv
from mindspore.ops import auto_generate
from mindspore.ops.operations.math_ops import STFT
from mindspore.ops.operations.math_ops import LuUnpack
from mindspore.ops.auto_generate import addcmul_ext_op
from mindspore.ops.auto_generate.pyboost_inner_prim import roll_impl, cross_impl
from mindspore.ops.auto_generate.pyboost_inner_prim import reduce_max_impl, reduce_min_impl
from mindspore.ops.operations.math_ops import Ormqr
from mindspore.ops.operations.math_ops import DivMod
from mindspore.ops.auto_generate import multi_scale_deformable_attn_op
from mindspore.ops.operations.array_ops import MatrixSetDiagV3
# 1
from mindspore.ops.auto_generate import (minimum, maximum, mul, muls, sin, sinc, sinh, cummax, real, conj, add, sub,
                                         cos,
                                         cosh, nan_to_num, norm_op, lp_norm_v2_op, linalg_vector_norm_op, std_op,
                                         matrix_exp, sqrt, rsqrt, square, trace, nextafter, abs, acos, acosh, angle,
                                         asin, asinh, atan, atan2, atanh, ceil, equal, erf, erfc, erfinv, exp, expm1,
                                         floor, floor_divide, floor_mod, gcd, greater, greater_equal, less, less_equal,
                                         log, log1p, neg, not_equal, round_op, isfinite, argmax_ext, mean_ext_op,
                                         sum_ext_op, prod_ext_op, all, matrix_inverse_ext, atan2_ext, sign, acos_ext,
                                         acosh_ext, asin_ext, asinh_ext, atan_ext, tan, median_ext_op, median_dim_op,
                                         xlogy_op, xlogy_scalar_other_op, xlogy_scalar_self_op, trunc, histc_ext, roll,
                                         bincount_ext, rotated_iou_op, cat, narrow, var_op, pow, inplace_erfinv_op,
                                         frac_ext, pow_tensor_scalar_op, not_equal_op, isinf, addmv_op, cdist,
                                         addbmm_op, addmm_op, pow_scalar_tensor_op, transpose_op)
# 2
from mindspore.ops.functional_overload import gmm
# 3

# 4

# 5

# 6

# 7

# 8

# 9

# 10

# 11

# 12

# 13

# 14

# 15

# 16

# 17

# 18

# 19

# 20

from mindspore.ops.auto_generate.gen_ops_def import add_ext, sub_ext, bmm_ext
from mindspore.ops.auto_generate import tanh, tanh_
from mindspore.nn import layer
from mindspore._checkparam import check_is_number
from mindspore import _checkparam as validator
from mindspore.ops.operations.math_ops import (
    Bernoulli,
    BesselI0,
    BesselI1,
    BesselJ0,
    BesselJ1,
    BesselK0,
    BesselK0e,
    BesselY0,
    BesselY1,
    BesselK1,
    BesselK1e,
    CumulativeLogsumexp,
    LuSolve,
    MatrixExp,
    MatrixSolve,
    Median,
    Fmax,
    Orgqr,
    Fmin,
    Renorm,
    Hypot,
    Heaviside,
    Lcm,
    Gcd,
    Quantile,
    NanToNum,
    SparseSegmentMean,
    TrilIndices,
    TriuIndices,
    InplaceIndexAdd,
    InplaceUpdateV2,
    Igamma,
    Igammac,
    Polar,
    Angle,
    FFTWithSize,
)
from mindspore.common.tensor import Tensor
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore._c_expression import TensorPy as Tensor_
import mindspore.ops.function as F
from mindspore.ops.operations._sequence_ops import TupleToTensor


@constexpr
def _make_tensor(val, dtype):
    """Returns the tensor with value `val` and dtype `dtype`."""
    return Tensor(val, dtype)


def get_x_shape(x_shape):
    s = 1
    for i in x_shape:
        s = s * i
    return (s,)


#####################################
# Public Operation Functions.
#####################################
absolute_ = P.Abs()
cast_ = P.Cast()
tensor_add = P.Add()
tensor_ceil = P.Ceil()
tensor_div = P.RealDiv()
tensor_exp = P.Exp()
tensor_expm1 = P.Expm1()
tensor_floordiv = P.FloorDiv()
floordiv = tensor_floordiv
tensor_ge = P.GreaterEqual()
tensor_gt = greater
tensor_le = P.LessEqual()
tensor_lt = P.Less()
tensor_mod = P.FloorMod()
floormod = tensor_mod
tensor_mul = P.Mul()
tensor_muls = muls
tensor_pow = P.Pow()
pows = tensor_pow
tensor_sub = P.Sub()
xdivy_ = P.Xdivy()
tensor_div_ = P.Div()
tensor_divmod_ = DivMod()
generator_step_ = Tensor(12, mstype.int64)
tuple_to_tensor_ = TupleToTensor()

#####################################
# Private Operation Functions.
#####################################
_accumulate_prim = None


def _get_accumulate_prim():
    """Lazy init to avoid import-time deprecated warnings."""
    global _accumulate_prim
    if _accumulate_prim is None:
        _accumulate_prim = P.AccumulateNV2()
    return _accumulate_prim


acos_ = P.ACos()
acosh_ = P.Acosh()
addcdiv_ = P.Addcdiv()
addcuml_ = P.Addcmul()
addn_ = P.AddN()
angle_ = Angle()
asin_ = P.Asin()
asinh_ = P.Asinh()
atan2_ = P.Atan2()
atan_ = P.Atan()
atanh_ = P.Atanh()
batch_matmul_ = BatchMatMul()
bernoulli_ext_ = BernoulliExt()
bessel_i0_ = BesselI0()
bessel_i0e_ = P.BesselI0e()
bessel_i1_ = BesselI1()
bessel_i1e_ = P.BesselI1e()
bessel_j0_ = BesselJ0()
bessel_j1_ = BesselJ1()
bessel_k0_ = BesselK0()
bessel_k0e_ = BesselK0e()
bessel_k1_ = BesselK1()
bessel_k1e_ = BesselK1e()
bessel_y0_ = BesselY0()
bessel_y1_ = BesselY1()
bitwise_and_ = P.BitwiseAnd()
bitwise_or_ = P.BitwiseOr()
bitwise_xor_ = P.BitwiseXor()
conj_ = P.Conj()
cumprod_ = P.CumProd()
cumsum_ = P.CumSum()
cumulative_logsumexp_ = CumulativeLogsumexp()
digamma_ = P.Digamma()
dtype_ = P.DType()
eps_ = P.Eps()
erf_ = P.Erf()
erfc_ = P.Erfc()
erfinv_ext_ = P.Erfinv()
exp2_ = P.Pow()
expand_dims_ = P.ExpandDims()
fill_v2_ = P.FillV2()
floor_ = P.Floor()
gcd_ = Gcd()
igamma_ = Igamma()
igammac_ = Igammac()
imag_ = P.Imag()
inv_ = P.math_ops.Inv()
invert_ = P.Invert()
isnan_ = P.IsNan()
lcm_ = Lcm()
lerp_ = P.Lerp()
lgamma_ = P.Lgamma()
linspace_ = P.LinSpace()
log1p_ = P.Log1p()
log_ = P.Log()
log_matrix_determinant_ = P.LogMatrixDeterminant()
logical_and_ = P.LogicalAnd()
logical_not_ = P.LogicalNot()
logical_or_ = P.LogicalOr()
logical_xor_ = P.LogicalXor()
lu_solve_ = LuSolve()
lu_unpack_ = LuUnpack()
matmul_ = P.MatMul()
matrix_determinant_ = P.MatrixDeterminant()
matrix_inverse_ = P.MatrixInverse()
mod_ = P.Mod()
nextafter_ = P.NextAfter()
ones_ = P.Ones()
polar_ = Polar()
poly_gamma_ = P.Polygamma()
rank_ = P.Rank()
reciprocal_ = P.Reciprocal()
reduce_sum_ = P.ReduceSum()
reshape_ = P.Reshape()
select_ = P.Select()
slice_ = P.Slice()
size_ = P.Size()
scalar_to_tensor_ = P.ScalarToTensor()
shape_ = P.Shape()
_sparse_segment_mean_prim = None


def _get_sparse_segment_mean_prim():
    """Lazy init to avoid import-time deprecated warnings."""
    global _sparse_segment_mean_prim
    if _sparse_segment_mean_prim is None:
        _sparse_segment_mean_prim = SparseSegmentMean()
    return _sparse_segment_mean_prim


tensor_round_ = P.Round()
tile_ = P.Tile()
tile_size_ = TileSize()
trunc_ = P.Trunc()
truncate_div_ = P.TruncateDiv()
truncate_mod_ = P.TruncateMod()
xlogy_ = P.Xlogy()
zeros_ = P.Zeros()
zeta_ = P.Zeta()
bitwise_and_scalar_ = BitwiseAndScalar()
bitwise_and_tensor_ = BitwiseAndTensor()
bitwise_or_scalar_ = BitwiseOrScalar()
bitwise_or_tensor_ = BitwiseOrTensor()
bitwise_xor_scalar_ = BitwiseXorScalar()
bitwise_xor_tensor_ = BitwiseXorTensor()

#####################################
# Element-wise Operation Functions.
#####################################
remainder_tensor_tensor_ = RemainderTensorTensor()
remainder_tensor_scalar_ = RemainderTensorScalar()
remainder_scalar_tensor_ = RemainderScalarTensor()


def addn(x):
    """
    Return the element-wise sum of all input tensors.

    Args:
        x (Union(tuple[Tensor], list[Tensor])): List of tensors or tuple of tensors.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([1, 2, 3])
        >>> y = mindspore.tensor([4, 5, 6])
        >>> mindspore.ops.addn([x, y, x, y])
        Tensor(shape=[3], dtype=Int64, value= [10, 14, 18])
    """
    return addn_(x)


def absolute(input):
    """
    Alias for :func:`mindspore.ops.abs` .

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    return abs(input)


def addcdiv(input, tensor1, tensor2, value=1):
    r"""
    Divide `tensor1` by `tensor2` element-wise, multiply the result by
    the scalar `value` , and add it to `input` .

    .. math::
        y[i] = input[i] + value[i] * (tensor1[i] / tensor2[i])

    Args:
        input (Tensor): The input tensor.
        tensor1 (Tensor): Tensor1, the numerator.
        tensor2 (Tensor): Tensor2, the denominator.
        value (Union[Tensor, number]): The multiplier for ( `tensor1` / `tensor2` ). Default ``1`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([1, 1, 1, 1], mindspore.float32)
        >>> x1 = mindspore.tensor([1, 2, 3, 4], mindspore.float32)
        >>> x2 = mindspore.tensor([4, 4, 2, 1], mindspore.float32)
        >>> y = mindspore.ops.addcdiv(x, x1, x2, 0.1)
        >>> print(y)
        [1.025 1.05  1.15  1.4  ]
    """
    return addcdiv_(input, tensor1, tensor2, Tensor(value))


def addcmul(input, tensor1, tensor2, value=1):
    r"""
    Multiply `tensor1` by `tensor2` element-wise, scale the result by
    the scalar `value` , and add it to `input` .

    .. math::
        output[i] = input[i] + value[i] * (tensor1[i] * tensor2[i])

    Args:
        input (Tensor): The input tensor.
        tensor1 (Tensor): The first tensor to be multiplied.
        tensor2 (Tensor): The second tensor to be multiplied.
        value (Union[Tensor, number]): The multiplier for ( `tensor1` * `tensor2` ). Default ``1`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor(([1, 1, 1]), mindspore.float32)
        >>> x1 = mindspore.tensor([[1], [2], [3]], mindspore.float32)
        >>> x2 = mindspore.tensor([[1, 2, 3]], mindspore.float32)
        >>> value = mindspore.tensor([1], mindspore.float32)
        >>> y = mindspore.ops.addcmul(x, x1, x2, value)
        >>> print(y)
        [[ 2.  3.  4.]
         [ 3.  5.  7.]
         [ 4.  7. 10.]]
    """
    return addcuml_(input, tensor1, tensor2, Tensor(value))


def addcmul_ext(input, tensor1, tensor2, *, value=1):
    r"""
    Performs the element-wise product of tensor tensor1 and tensor tensor2,
    multiply the result by the scalar value and add it to input data.

    .. math::
        output[i] = input[i] + value * (tensor1[i] * tensor2[i])

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): The tensor to be added.
        tensor1 (Tensor): The tensor to be multiplied.
        tensor2 (Tensor): The tensor to be multiplied.

    Keyword Args:
        value (Number, optional): The multiplier for tensor1*tensor2. Default: ``1`` .

    Returns:
        Tensor, has the same shape and dtype as tensor1*tensor2.

    Raises:
        TypeError: If dtype of `tensor1`, `tensor2`, `input` is not Tensor.
        ValueError: If `tensor1` could not be broadcast to a tensor with shape of `tensor2`.
        ValueError: If `input` could not be broadcast to tensors with shapes of `tensor1*tensor2`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input_data = Tensor(np.array([1, 1, 1]), mindspore.float32)
        >>> x1 = Tensor(np.array([[1], [2], [3]]), mindspore.float32)
        >>> x2 = Tensor(np.array([[1, 2, 3]]), mindspore.float32)
        >>> y = ops.addcmul(input_data, x1, x2, value=1)
        >>> print(y)
        [[ 2.  3.  4.]
         [ 3.  5.  7.]
         [ 4.  7. 10.]]
    """
    return addcmul_ext_op(input, tensor1, tensor2, value=value)


def bincount(input, weights=None, minlength=0):
    """
    Count the frequency of each value in the input tensor of non-negative ints.

    If you don't specify `minlength`, the length of output tensor the length of the output tensor is max( `input` ) + 1.

    If `minlength` is specified, the length of the output tensor is max([max( `input` ) + 1, `minlength`]).

    If 'weights' is specified, the output results are weighted. If `n` is the value at position `i`,
    i.e ``out[n] += weight[i]`` instead of ``out[n] += 1``.

    Note:
        If `input` contains negative value, the result will be undefined.

    Args:
        input (Tensor): 1-D input tensor.
        weights (Tensor, optional): Weights. Default ``None`` .
        minlength (int, optional): A minimum number of bins for the output tensor. Default ``0`` .

    Returns:
        Tensor, a tensor of shape [max(input)+1] if input is non-empty, otherwise, the shape is [0].

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([2, 4, 1, 0, 0], dtype=mindspore.int64)
        >>> print(mindspore.ops.bincount(x, minlength=7))
        [2. 1. 1. 0. 1. 0. 0.]
        >>> weights = mindspore.tensor([0, 0.25, 0.5, 0.75, 1], dtype=mindspore.float32)
        >>> print(mindspore.ops.bincount(x, weights=weights))
        [1.75 0.5  0.   0.   0.25]
    """
    if not isinstance(input, Tensor):
        raise TypeError("For math function 'bincount', 'input' must be Tensor.")
    if weights is not None and not isinstance(weights, Tensor):
        raise TypeError(f"For math function 'bincount', 'weights' must be Tensor, but got {type(weights)}.")
    if not isinstance(minlength, int) or isinstance(minlength, bool):
        raise TypeError(f"For math function 'bincount', 'minlength' must be int but got {type(minlength)}.")
    if rank_(input) != 1:
        raise ValueError("For math function 'bincount', 'input' should be one-dimensional tensor.")
    if input.shape[0] == 0:
        return Tensor_([])
    if minlength < 0:
        raise ValueError(f"For 'bincount', 'minlength' should be >= 0 but got {minlength}.")
    if input.astype(mstype.float32).max() > minlength - 1:
        length = (input.astype(mstype.float32).max() + 1).astype(mstype.int32)
    else:
        length = cast_(minlength, mstype.int32)
    idx = F.arange(length).expand_dims(-1)
    idx_mapping = equal(input, idx.astype(input.dtype))
    if weights is not None:
        if input.shape != weights.shape:
            raise ValueError('for bincount `input` and `weights` must have the same length')
        idx_mapping = weights * idx_mapping
    return reduce_sum_(idx_mapping.astype(mstype.float32), 1).ravel()


def bucketize(input, boundaries, *, right=False):
    r"""
    Return the indices of the buckets to which each element in the input tensor belongs. If `right` is ``False``, the
    left boundary is open. For each element x in `input`, the returned index satisfies the following rules:

    .. math::

        \begin{cases}
        boundaries[i-1] < x <= boundaries[i], & \text{if right} = False\\
        boundaries[i-1] <= x < boundaries[i], & \text{if right} = True
        \end{cases}

    Args:
        input (Tensor): The input tensor.
        boundaries (list): A sorted ascending list of bucket boundary values.

    Keyword Args:
        right (bool, optional): if ``False``, gets the lower bound index for each value in input from boundaries;
            If ``True``, gets the upper bound index instead. Default: ``False``.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[3, 6, 9], [3, 6, 9]])
        >>> boundaries = [1., 3., 5., 7., 9.]
        >>> output = mindspore.ops.bucketize(input, boundaries, right=True)
        >>> output
        Tensor(shape=[2, 3], dtype=Int32, value=
        [[2, 3, 5],
         [2, 3, 5]])
    """

    bucketize_op = _get_cache_prim(P.Bucketize)
    epsilon_ = 0. if right else 1.e-6
    boundaries = [boundary + epsilon_ for boundary in boundaries]
    return bucketize_op(boundaries)(input)


def exp2(input):
    """
    Compute base two exponential of the input tensor element-wise.

    .. math::
        out_i = 2^{input_i}

    Args:
        input (Tensor): The input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([0.0, 1.0, 3.0], mindspore.float32)
        >>> output = mindspore.ops.exp2(input)
        >>> print(output)
        [1. 2. 8.]
    """

    tensor_2 = Tensor(np.array(2.0).astype(np.float32))
    if input.dtype == mstype.float16:
        tensor_2 = Tensor(np.array(2.0).astype(np.float16))
    return exp2_(tensor_2, input)


def argmin(input, axis=None, keepdims=False):
    """
    Return the indices of the minimum values along a specified dimension of the tensor.

    Args:
        input (Tensor): The input tensor.
        axis (Union[int, None], optional): Specify the dimension for computation. If ``None`` , compute all elements in
            the `input` . Default ``None`` .
        keepdims (bool, optional): Whether the output tensor has dim retained. Default ``False`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[2, 5, 1, 6],
        ...                           [3, -7, -2, 4],
        ...                           [8, -4, 1, -3]])
        >>> # case 1: By default, compute the minimum indice of all elements.
        >>> mindspore.ops.argmin(input)
        Tensor(shape=[], dtype=Int32, value= 5)
        >>>
        >>> # case 2: Compute the minimum indices along axis 1.
        >>> mindspore.ops.argmin(input, axis=1)
        Tensor(shape=[3], dtype=Int32, value= [2, 1, 1])
        >>>
        >>> # case 3: If keepdims=True, the output shape will be same of that of the input.
        >>> mindspore.ops.argmin(input, axis=1, keepdims=True)
        Tensor(shape=[3, 1], dtype=Int32, value=
        [[2],
         [1],
         [1]])
    """
    if not input.shape:
        return Tensor(0)
    is_axis_none = False
    if axis is None:
        input = reshape_(input, (-1,))
        axis = 0
        is_axis_none = True
    out = _get_cache_prim(P.Argmin)(axis)(input)
    if keepdims and not is_axis_none:
        out = expand_dims_(out, axis)
    return out


def negative(input):
    r"""
    Alias for :func:`mindspore.ops.neg` .

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    return neg(input)


def positive(input):
    r"""
    Return self tensor.

    Args:
        input (Tensor): The input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([-5.0, 1.5, 3.0, 100.0], mindspore.float32)
        >>> print(mindspore.ops.positive(x))
        [ -5.    1.5   3.  100. ]
    """
    _check_is_tensor("input", input, "positive")
    return input


def numel(input):
    r"""
    Return the total number of elements in the tensor.

    Args:
        input (Tensor): The input tensor.

    Returns:
        The total number of elements in tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input_x = mindspore.tensor([[2, 2], [2, 2]], mindspore.float32)
        >>> print(mindspore.ops.numel(input_x))
        4
    """
    _check_is_tensor("input", input, "numel")
    return input.size


def permute(input, axis):
    """
    Permute the input tensor along the specified axis.

    Args:
        input (Tensor): The input tensor.
        axis (tuple(int)): The axis in a specified order.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input_x = mindspore.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], mindspore.float32)
        >>> input_perm = (0, 2, 1)
        >>> print(mindspore.ops.permute(input_x, input_perm))
        [[[ 1.  4.]
          [ 2.  5.]
          [ 3.  6.]]
         [[ 7. 10.]
          [ 8. 11.]
          [ 9. 12.]]]
    """
    return transpose_op(input, axis)


def subtract(input, other, *, alpha=1):
    r"""
    Subtract `other` scaled by `alpha` from `input`.

    Support implicit type conversion and type promotion.

    .. math::
        output[i] = input[i] - alpha * other[i]

    Args:
        input (Union[Tensor, number.Number]): The first input.
        other (Union[Tensor, number.Number]): The second input.

    Keyword Args:
        alpha (number): The multiplier for :math:`other`. Default ``1`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([4, 5, 6], mindspore.float32)
        >>> y = mindspore.tensor([1, 2, 3], mindspore.float32)
        >>> z = mindspore.ops.subtract(input, y, alpha=1)
        >>> print(z)
        [3. 3. 3.]
    """
    return tensor_sub(input, alpha * other)


def multiply(input, other):
    r"""
    Alias for :func:`mindspore.ops.asinh`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    return tensor_mul(input, other)


def div(input, other, *, rounding_mode=None):
    r"""
    Divide the first input tensor by the second input tensor in floating-point type element-wise.

    .. math::

        out_{i} = input_{i} / other_{i}

    Note:
        - The two input tensors must be broadcastable.
        - The two input tensors can not be bool type at the same time,
        - The two input tensors comply with the implicit type conversion rules to make the data types
          consistent.

    Args:
        input (Union[Tensor, Number, bool]): The first input tensor.
        other (Union[Tensor, Number, bool]): The second input tensor.

    Keyword Args:
        rounding_mode (str, optional): Type of rounding applied to the result. Default ``None`` .
            Three types are defined as:

            - None: the same as true division in Python or `true_divide` in NumPy.

            - "floor": rounds the results of the division down, which is the same as floor division in Python
              or `floor_divide` in NumPy.

            - "trunc": rounds the results of the division towards zero, which is the same as C-style integer division.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[-0.3711, -1.9353, -0.4605, -0.2917],
        ...                           [ 0.1815, -1.0111,  0.9805, -1.5923],
        ...                           [ 0.1062,  1.4581,  0.7759, -1.2344],
        ...                           [-0.1830, -0.0313,  1.1908, -1.4757]])
        >>> other = mindspore.tensor([ 0.8032,  0.2930, -0.8113, -0.2308])
        >>> output = mindspore.ops.div(input, other)
        >>> print(output)
        [[-0.4620269  -6.605119    0.5676076   1.2638649 ]
         [ 0.22597112 -3.4508533  -1.2085541   6.899047  ]
         [ 0.13222112  4.97645    -0.95636636  5.348354  ]
         [-0.22783864 -0.10682594 -1.4677677   6.3938475 ]]
        >>> output = mindspore.ops.div(input, other, rounding_mode='floor')
        >>> print(output)
        [[-1. -7.  0.  1.]
         [ 0. -4. -2.  6.]
         [ 0.  4. -1.  5.]
         [-1. -1. -2.  6.]]
    """
    if rounding_mode is not None and rounding_mode not in ['floor', 'trunc']:
        raise ValueError("For ops.div, rounding_mode value should be None, 'floor' or 'trunc'.")
    if rounding_mode:
        output = tensor_divmod_(input, other, rounding_mode)
    else:
        output = tensor_div_(input, other)
    return output


def true_divide(dividend, divisor):
    r"""
    Alias for :func:`mindspore.ops.div` with  ``rounding_mode=None`` .

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    return div(dividend, divisor)


def divide(input, other, *, rounding_mode=None):
    """
    Alias for :func:`mindspore.ops.div` .

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    return div(input, other, rounding_mode=rounding_mode)


def float_power(input, exponent):
    """
    Computes the first input to the power of the second input.
    For the real number type, cast `input` and `exponent` to mindspore.float64 to calculate.

    .. note::
        Currently, complex type calculation is not supported.

    Args:
        input (Union[Tensor, Number]): The first input.
        exponent (Union[Tensor, Number]): The second input, if the first input is Number, it must be a Tensor.

    Returns:
        Tensor whose data type is mindspore.float64.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> # case 1: When exponent is scalar:
        >>> input = mindspore.tensor([-1.5, 0., 2.])
        >>> output = mindspore.ops.float_power(input, 2)
        >>> print(output)
        [2.25 0.   4.  ]
        >>>
        >>> # case 2: When exponent is a tensor:
        >>> exponent = mindspore.tensor([0, 1, 2])
        >>> output = mindspore.ops.float_power(input, exponent)
        >>> print(output)
        [1. 0. 4.]
    """
    if not (isinstance(input, Tensor) or isinstance(exponent, Tensor)):
        raise TypeError("At least one of the types of inputs must be tensor, " + \
                        f"but the type of 'input' got is {type(input)}, " + \
                        f"and the type of 'exponent' is {type(exponent)}.")
    if not isinstance(input, (Tensor, numbers.Number)):
        raise TypeError(f"The type of 'input' must be Tensor or Number, but got {type(input)}.")
    if not isinstance(exponent, (Tensor, numbers.Number)):
        raise TypeError(f"The type of 'exponent' must be Tensor or Number, but got {type(exponent)}.")

    if (isinstance(input, Tensor) and is_complex(input)) or \
            (isinstance(exponent, Tensor) and is_complex(exponent)) or \
            isinstance(input, complex) or isinstance(exponent, complex):
        input = cast_(input, mstype.complex128)
        exponent = cast_(exponent, mstype.complex128)
    else:
        input = cast_(input, mstype.float64)
        exponent = cast_(exponent, mstype.float64)
    return pow(input, exponent)


def float_power_ext(input, exponent):
    """
    Computes `input` to the power of `exponent` element-wise in double precision, and always
    returns a mindspore.float64 tensor.

    .. math::

        out_{i} = input_{i} ^ {exponent_{i}}

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Note:
        Unlike `ops.pow`, this function always uses double precision for calculations, while
        the precision of `ops.pow` depends on type promotion.
        Currently, this function does not support complex number calculations.
        Since float64 calculations are significantly slower on ascend devices compared to other data
        types, it is strongly recommended to use this function only in scenarios where double precision
        is required and performance is not a priority. Otherwise, using `ops.pow` is a better choice.

    Args:
        input (Union[Tensor, Number]): The first input is a tensor or a number.
        exponent (Union[Tensor, Number]): The second input, if the first input is Tensor,
            the second input can be Number or Tensor. Otherwise, it must be a Tensor.

    Returns:
        Tensor, the shape is the same as the one after broadcasting, the return value type
        is mindspore.float64.

    Raises:
        TypeError: If neither `input` nor `exponent` is a Tensor.
        TypeError: If the data type of `input` or `exponent` is neither a tensor nor a number,
            or it contains complex numbers.
        ValueError: If `input` and `exponent` have different shapes and cannot be broadcasted
            to each other.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor, ops
        >>> input = Tensor([1, 2, 3])
        >>> ops.function.math_func.float_power_ext(input, 2)
        Tensor(shape=[3], dtype=Float64, value= [ 1.00000000e+00,  4.00000000e+00,  9.00000000e+00])
        >>>
        >>> exp = Tensor([2, -3, -4])
        >>> ops.function.math_func.float_power_ext(input, exp)
        Tensor(shape=[3], dtype=Float64, value= [ 1.00000000e+00,  1.25000000e-01,  1.23456790e-02])
    """
    if not (isinstance(input, Tensor) or isinstance(exponent, Tensor)):
        raise TypeError("At least one of the types of inputs must be tensor, " +
                        f"but the type of 'input' got is {type(input)}, " +
                        f"and the type of 'exponent' is {type(exponent)}.")
    if (not isinstance(input, (Tensor, int, float, bool))) or \
            (isinstance(input, Tensor) and is_complex(input)):
        raise TypeError("The type of 'input' must be Tensor or Number (excluding complex), " +
                        f"but got {type(input)}.")
    if (not isinstance(exponent, (Tensor, int, float, bool))) or \
            (isinstance(exponent, Tensor) and is_complex(exponent)):
        raise TypeError("The type of 'exponent' must be Tensor or Number (excluding complex), " +
                        f"but got {type(exponent)}.")

    op = pow
    if isinstance(input, Tensor) and isinstance(exponent, numbers.Number):
        op = pow_tensor_scalar_op
    if isinstance(input, numbers.Number) and isinstance(exponent, Tensor):
        op = pow_scalar_tensor_op

    if isinstance(input, Tensor) and input.dtype != mstype.float64:
        input = cast_(input, mstype.float64)
    if isinstance(exponent, Tensor) and exponent.dtype != mstype.float64:
        exponent = cast_(exponent, mstype.float64)
    return op(input, exponent)


def floor_div(x, y):
    """
    Alias for :func:`mindspore.ops.floor_divide` .

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    return tensor_floordiv(x, y)


def fmod(input, other):
    """
    Compute the remainder of element-wise division of first input by the second input.

    Support broadcasting and type promotion.

    .. math::

        out = input - n * other

    Where :math:`n` is :math:`input/other` with its fractional part truncated.
    The returned value has the same sign as `input` and is less than `other` in magnitude.

    Args:
        input (Union[Tensor, Number]): the dividend.
        other (Union[Tensor, Number]): the divisor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> # case 1: When other is scalar:
        >>> input = mindspore.tensor([-3., -2, -1, 3, 4, 5])
        >>> output = mindspore.ops.fmod(input, 2)
        >>> print(output)
        [-1.  0. -1.  1.  0.  1.]
        >>>
        >>> # case 2: When other is a tensor:
        >>> other = mindspore.tensor([-1., 1, 2, 2.5, 1.5, 1.7])
        >>> output = mindspore.ops.fmod(input, other)
        >>> print(output)
        [ 0.         0.        -1.         0.5        1.         1.5999999]
    """
    if not (isinstance(input, (Tensor, Tensor_)) or isinstance(other, (Tensor, Tensor_))):
        raise TypeError("At least one of the types of inputs must be tensor, " + \
                        f"but the type of 'input' got is {type(input)}, " + \
                        f"and the type of 'other' is {type(other)}.")
    return input - div(input, other, rounding_mode="trunc") * other


def logdet(input):
    r"""
    Calculate log determinant of one or a batch of square matrices.

    Args:
        input (Tensor): The input tensor of shape :math:`(*, N, N)` where :math:`*` means batch dimensions.

    Returns:
        Tensor

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import mindspore
        >>> a = mindspore.tensor([[[8., 9.], [1., 2.]], [[5., 6.], [3., 4.]]])
        >>> mindspore.ops.logdet(a)
        Tensor(shape=[2], dtype=Float32, value= [ 1.94591010e+00,  6.93146825e-01])
    """
    det_x = det(input)
    return log_(det_x)


def i0(input):
    r"""
    For details, please refer to :func:`mindspore.ops.bessel_i0`.
    The parameter `input` of the current interface is the same as the parameter `x` of the reference interface.

    Supported Platforms:
        ``GPU`` ``CPU``
    """
    return bessel_i0(input)


def inplace_update(x, v, indices):
    """
    Updates `x` to `v` according to the indices.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    for each `i, ..., j` in `indices` :

    .. math::
        x[\text{indices}[i, ..., j]] = v[i, ..., j]

    Args:
        x (Tensor): The input tensor.
        v (Tensor): The input tensor to update to `x`.
        indices (Union[int, tuple[int], Tensor]): Indices into the input `x` along the 0th dimension.

    Returns:
        Tensor

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> indices = (0, 1)
        >>> x = mindspore.tensor([[1, 2], [3, 4], [5, 6]], mindspore.float32)
        >>> v = mindspore.tensor([[0.5, 1.0], [1.0, 1.5]], mindspore.float32)
        >>> output = mindspore.ops.inplace_update(x, v, indices)
        >>> print(output)
        [[0.5 1. ]
         [1.  1.5]
         [5.  6. ]]
    """
    inplace_update_inner = InplaceUpdateV2()
    return inplace_update_inner(x, indices, v)


def inplace_add(x, v, indices):
    r"""
    Add `x` to `v` according to the indices.

    for each `i, ..., j` in `indices` :

    .. math::
        x[\text{indices}[i, ..., j]] \mathrel{+}= v[i, ..., j]

    Args:
        x (Tensor): The input tensor.
        v (Tensor): The input tensor to add to `x`.
        indices (Union[int, tuple]): Indices into the input `x` along the 0th dimension.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> indices = (0, 1)
        >>> x = mindspore.tensor([[1, 2], [3, 4], [5, 6]], mindspore.float32)
        >>> input_v = mindspore.tensor([[0.5, 1.0], [1.0, 1.5]], mindspore.float32)
        >>> output = mindspore.ops.inplace_add(x, input_v, indices)
        >>> print(output)
        [[1.5 3. ]
         [4.  5.5]
         [5.  6. ]]
    """
    inplace_add_inner = _get_cache_prim(P.InplaceAdd)(indices)
    return inplace_add_inner(x, v)


def inplace_index_add(var, indices, updates, axis):  # pylint: disable=redefined-outer-name
    r"""
    Add `updates` to `var` according to the indices and axis.

    for each `i, ..., j` in `indices` :

    .. math::
        x[:, \text{indices}[i, ..., j], :] \mathrel{+}= v[:, i, ..., j, :]

    where `i` is the index of the element in `indices`, and the axis of `indices[i]` is determined by the input `axis`.

    Args:
        var (Union[Parameter, Tensor]): The input parameter or tensor.
        indices (Tensor): The specified indices, a 1-D tensor.
        updates (Tensor): The input tensor to add to `var`.
        axis (int): The specified axis.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore
        >>> var = mindspore.Parameter(mindspore.tensor([[1, 2], [3, 4], [5, 6]], mindspore.float32))
        >>> indices = mindspore.tensor([0, 1], mindspore.int32)
        >>> updates = mindspore.tensor([[0.5, 1.0], [1.0, 1.5]], mindspore.float32)
        >>> mindspore.ops.inplace_index_add(var, indices, updates, axis=0)
        >>> print(var.asnumpy())
        [[1.5 3. ]
         [4.  5.5]
         [5.  6. ]]
    """

    inplace_index_add_ = InplaceIndexAdd(axis)
    return inplace_index_add_(var, indices, updates)


def inplace_sub(x, v, indices):
    r"""
    Subtract `v` in `x` according to the indices.

    for each `i, ..., j` in `indices` :

    .. math::
        x[\text{indices}[i, ..., j]] \mathrel{-}= v[i, ..., j]

    Args:
        x (Tensor): The input tensor.
        v (Tensor): The input tensor to subtract from `x` .
        indices (Union[int, tuple]): Indices into the input `x` along the 0th dimension.

    Returns:
        Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> indices = (0, 1)
        >>> x = mindspore.tensor([[1, 2], [3, 4], [5, 6]], mindspore.float32)
        >>> input_v = mindspore.tensor([[0.5, 1.0], [1.0, 1.5]], mindspore.float32)
        >>> output = mindspore.ops.inplace_sub(x, input_v, indices)
        >>> print(output)
        [[0.5 1. ]
         [2.  2.5]
         [5.  6. ]]
    """
    inplace_sub_inner = _get_cache_prim(P.InplaceSub)(indices)
    return inplace_sub_inner(x, v)


def logical_not(input):
    """
    Compute the "logical NOT" of the input tensor element-wise.

    .. math::

        out_{i} = \\neg input_{i}

    Args:
        input (Tensor): The input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([True, False, True], mindspore.bool)
        >>> output = mindspore.ops.logical_not(x)
        >>> print(output)
        [False  True False]
    """
    return logical_not_(input)


def logical_or(input, other):
    r"""
    Compute the "logical OR" of two tensors element-wise.

    .. math::

        out_{i} = input_{i} \vee other_{i}

    .. note::
        - Broadcasting is supported.
        - Support implicit type conversion.

    Args:
        input (Union[Tensor, bool]): The first input.
        other (Union[Tensor, bool]): The second input.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([True, False, True], mindspore.bool)
        >>> y = mindspore.tensor([True, True, False], mindspore.bool)
        >>> output = mindspore.ops.logical_or(x, y)
        >>> print(output)
        [ True  True  True]
        >>> x = mindspore.tensor(1, mindspore.bool)
        >>> y = mindspore.tensor(0, mindspore.bool)
        >>> output = mindspore.ops.logical_or(x, y)
        >>> print(output)
        True
        >>> x = True
        >>> y = mindspore.tensor(0, mindspore.bool)
        >>> output = mindspore.ops.logical_or(x, y)
        >>> print(output)
        True
        >>> x = True
        >>> y = mindspore.tensor([True, False], mindspore.bool)
        >>> output = mindspore.ops.logical_or(x, y)
        >>> print(output)
        [True True]
    """
    return logical_or_(input, other)


def logical_and(input, other):
    r"""
    Compute the "logical AND" of two tensors element-wise.

    .. math::

        out_{i} = input_{i} \wedge other_{i}

    .. note::
        - Broadcasting is supported.
        - Support implicit type conversion.

    Args:
        input (Union[Tensor, bool]): The first input.
        other (Union[Tensor, bool]): The second input.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([True, False, True], mindspore.bool)
        >>> y = mindspore.tensor([True, True, False], mindspore.bool)
        >>> output = mindspore.ops.logical_and(x, y)
        >>> print(output)
        [ True False False]
        >>> x = mindspore.tensor(1, mindspore.bool)
        >>> y = mindspore.tensor(0, mindspore.bool)
        >>> output = mindspore.ops.logical_and(x, y)
        >>> print(output)
        False
        >>> x = True
        >>> y = mindspore.tensor(0, mindspore.bool)
        >>> output = mindspore.ops.logical_and(x, y)
        >>> print(output)
        False
        >>> x = True
        >>> y = mindspore.tensor([True, False], mindspore.bool)
        >>> output = mindspore.ops.logical_and(x, y)
        >>> print(output)
        [True False]
    """
    return logical_and_(input, other)


def signbit(input):
    r"""
    Determine the symbol of each element. If the element value is less than 0,
    the corresponding output position is True; otherwise, it is False.

    Args:
        input (Tensor): The input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([0.3, 1.2, 0., -2.5])
        >>> output = mindspore.ops.signbit(input)
        >>> print(output)
        [False False False  True]
    """
    if not isinstance(input, Tensor):
        raise TypeError(f"For signbit, the input must be a Tensor, but got {type(input)}")
    res = ops.less(input, 0)
    return res


def sgn(input):
    r"""
    Extension of :func:`mindspore.ops.sign` in complex domain.
    For real number input, this function is the same as :func:`mindspore.ops.sign`.
    For complex input, this function is calculated according to the following formula.

    .. math::
        \text{out}_{i} = \begin{cases}
                        0 & |\text{input}_i| == 0 \\
                        \frac{{\text{input}_i}}{|{\text{input}_i}|} & \text{otherwise}
                        \end{cases}

    Args:
        input (Tensor): The input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[3+4j, 7-24j, 0, 6+8j, 8], [15+20j, 7-24j, 0, 3+4j, 20]], mindspore.complex64)
        >>> output = mindspore.ops.sgn(input)
        >>> print(output)
        [[0.6 +0.8j  0.28-0.96j 0.  +0.j   0.6 +0.8j  1.  +0.j  ]
         [0.6 +0.8j  0.28-0.96j 0.  +0.j   0.6 +0.8j  1.  +0.j  ]]
    """
    if not isinstance(input, Tensor):
        raise TypeError(f"For sgn, the input must be a Tensor, but got {type(input)}")
    if not ops.is_complex(input):
        return ops.sign(input)
    modulus = ops.ComplexAbs()(input)
    zeros_mask = modulus.equal(0)
    non_zero_modulus = ops.masked_fill(modulus, zeros_mask, ops.cast(1, modulus.dtype))
    zeros_modulus = ops.zeros_like(non_zero_modulus)
    complex_modulus = ops.Complex()(non_zero_modulus, zeros_modulus)
    res = tensor_div(input, complex_modulus)
    return res


def cosine_similarity(x1, x2, dim=1, eps=1e-08):
    r"""
    Calculate cosine similarity between two input tensors along the specified dimension.

    .. math::
        \text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}

    Note:
        Currently, broadcast of input is not supported.

    Args:
        x1 (Tensor): The first input tensor.
        x2 (Tensor): The second input tensor.
        dim (int, optional): Specify the dimension for computation. Default ``1`` .
        eps (float, optional): Minimal value to avoid division by zero. Default ``1e-08`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[-0.0256, 0.0127, -0.2475, 0.2316, 0.8037],
        ...                           [0.5809, -1.2712, -0.7038, -0.2558, 0.7494]])
        >>> other = mindspore.tensor([[-0.6115, -0.1965, -0.8484, 0.2389, 0.2409],
        ...                           [1.8940, -2.1997, 0.1915, 0.0856, 0.7542]])
        >>> output = mindspore.ops.cosine_similarity(input, other)
        >>> print(output)
        [0.4843164  0.81647635]
    """
    molecule = ops.sum(x1 * x2, dim=dim)
    denominator = (ops.norm(x1, dim=dim, ord=2) * ops.norm(x2, dim=dim, ord=2)).clip(min=eps)
    output = molecule / denominator
    return output


def _check_cov_weights(weights, weights_name, num_observations, valid_type, valid_type_name):
    """check cov weights valid"""
    if weights.ndim > 1:
        raise ValueError(
            f"For cov, the {weights_name} must have one or fewer dimensions, but got {weights.ndim} dimensions.")
    if weights.dtype not in valid_type:
        raise TypeError(
            f"For cov, the dtype of {weights_name} must be {valid_type_name} type, but got type {weights.dtype}")
    if ops.numel(weights) != num_observations:
        raise ValueError(
            f"For cov, the numel of {weights_name} must equal the number of columns of input, "
            f"but got numel:{ops.numel(weights)}, number of columns of input:{num_observations}.")
    return 0


def _get_default_div_type(param):
    """get the default type when div"""
    if param.dtype == mstype.float64:
        return param
    return param.astype(mstype.float32)


def cov(input, *, correction=1, fweights=None, aweights=None):
    r"""
    Return the covariance matrix of the input tensor, where the rows of the input tensor represent variables and the
    columns represent observations. The diagonal of the covariance matrix contains the variance of each variable in the
    input tensor, while the off-diagonal elements represent the covariance between pairs of variables.
    If the input is a 0-D or 1-D tensor, its variance is returned.

    The unbiased sample covariance of the variables :math:`a` and :math:`b` is given by the following formula:

    .. math::
        \text{cov}_w(a,b) = \frac{\sum^{N}_{i = 1}(a_{i} - \bar{a})(b_{i} - \bar{b})}{N~-~1}

    where :math:`\bar{a}` and :math:`\bar{b}` are the simple means of the :math:`a` and :math:`b` respectively.

    If `fweights` and/or `aweights` are provided, the unbiased weighted covariance
    is calculated, which is given by:

    .. math::
        \text{cov}_w(a,b) = \frac{\sum^{N}_{i = 1}w_i(a_{i} - \mu_a^*)(b_{i} - \mu_b^*)}{\sum^{N}_{i = 1}w_i~-~1}

    where :math:`w` denotes `fweights` or `aweights` based on whichever is provided, or
    :math:`w = fweights \times aweights` if both are provided, and
    :math:`\mu_x^* = \frac{\sum^{N}_{i = 1}w_ix_{i} }{\sum^{N}_{i = 1}w_i}` is the weighted mean of the variable.

    .. warning::
        The values of `fweights` and `aweights` cannot be negative, and the negative weight scene result is undefined.

    .. note::
        Currently, complex number is not supported.

    Args:
        input (Tensor): The 0-D, 1-D or 2-D tensor.

    Keyword Args:
        correction (int, optional): The difference between sample size and sample degrees of freedom.
            `correction = 0` will return the simple average. Defaults to Bessel's correction, `correction = 1` which
            returns the unbiased estimate, even if both `fweights` and `aweights` are specified.
        fweights (Tensor, optional): 0D or 1D tensor containing integer frequency weight, indicating
            the number of repetition of each observation vector. Its numel must equal the number of columns of `input`.
            Ignored if `None`. Default ``None`` .
        aweights (Tensor, optional): A scalar or 1D Tensor containing float observation weights represents
            the importance of each observation vector. The higher the importance, the greater the corresponding value.
            Its numel must equal the number of columns of `input`. Must have floating point dtype. Ignored if `None`.
            Default ``None`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[0., 5., 7.],
        ...                           [3., 5., 0.]])
        >>> output = mindspore.ops.cov(input)
        >>> print(output)
        [[13.        -3.5      ]
         [-3.5        6.3333335]]
        >>> output = mindspore.ops.cov(input, correction=0)
        >>> print(output)
        [[ 8.666667  -2.3333333]
         [-2.3333333  4.2222223]]
        >>> fw = mindspore.tensor([5, 2, 4], dtype=mindspore.int64)
        >>> aw = mindspore.tensor([0.4588, 0.9083, 0.7616], mindspore.float32)
        >>> output = mindspore.ops.cov(input, fweights=fw, aweights=aw)
        >>> print(output)
        [[10.146146 -3.47241 ]
         [-3.47241   4.716825]]
    """
    if input.ndim > 2:
        raise ValueError(f"For cov, the input must have two or fewer dimensions, but got {input.ndim} dimensions.")
    if input.dtype == mstype.bool_:
        raise TypeError("For cov, the input dtype can not be bool.")

    # View input tensor as 2D
    input_x = input.view((1, -1)) if input.ndim < 2 else input
    num_observations = input_x.shape[1]
    if fweights is not None:
        _check_cov_weights(fweights, "fweights", num_observations, mstype.int_type, "an integer")

    if aweights is not None:
        _check_cov_weights(aweights, "aweights", num_observations, mstype.float_type, "a floating point")

    if fweights is not None and aweights is None:
        w = fweights
    elif fweights is None and aweights is not None:
        w = aweights
    elif fweights is not None and aweights is not None:
        w = fweights * aweights
    else:
        w = None

    if w is not None:
        w_sum = w.sum()
        avg = (input_x * w).sum(1) / _get_default_div_type(w_sum)
    else:
        w_sum = ops.cast(num_observations, mstype.int64)
        avg = input_x.sum(1) / _get_default_div_type(w_sum)

    if w is not None and aweights is not None and correction != 0:
        norm_factor = w_sum - correction * (w * aweights).sum() / w_sum
    else:
        norm_factor = w_sum - correction

    norm_factor = norm_factor.clip(min=0)

    input_x = input_x - avg.unsqueeze(1)
    c = ops.mm(input_x, (input_x * w if w is not None else input_x).T)
    norm_factor = norm_factor.astype(mstype.float32)
    return ops.true_divide(c, _get_default_div_type(norm_factor)).squeeze()


def t(input):
    r"""
    Transpose a 2-D tensor. 0-D and 1-D tensor are returned as it is.

    Args:
        input (Tensor): The input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> # case 1: Input is a 0-D tensor
        >>> mindspore.ops.t(mindspore.tensor(6))
        Tensor(shape=[], dtype=Int64, value= 6)
        >>>
        >>> # case 2: Input is a 1-D tensor
        >>> mindspore.ops.t(mindspore.tensor([1, 2]))
        Tensor(shape=[2], dtype=Int64, value= [1, 2])
        >>>
        >>> # case 3: Input is a 2-D tensor
        >>> mindspore.ops.t(mindspore.tensor([[1, 2, 3], [2, 3, 4]]))
        Tensor(shape=[3, 2], dtype=Int64, value=
        [[1, 2],
         [2, 3],
         [3, 4]])
    """
    if input.ndim == 2:
        return transpose_op(input, (1, 0))
    return input


def xlogy(input, other):
    r"""
    Compute `input` multiplied by the logarithm of `other` element-wise.

    .. math::

        out_i = input_{i} * ln{other_{i}}

    .. note::
        - Support broadcast, support implicit type conversion and type promotion.

    .. warning::
        On Ascend, the data type of `input` and `other` must be float16 or float32.

    Args:
        input (Union[Tensor, numbers.Number, bool]): The first input tensor.
        other (Union[Tensor, numbers.Number, bool]): The second input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> output = mindspore.ops.xlogy(mindspore.tensor([-5., 0., 4.]), mindspore.tensor([2., 2., 2.]))
        >>> print(output)
        [-3.465736   0.         2.7725887]
    """
    if isinstance(input, (float, int, bool)):
        input = scalar_to_tensor_(input)
    if isinstance(other, (float, int, bool)):
        other = scalar_to_tensor_(other)
    if isinstance(input, Tensor) and isinstance(other, Tensor) and input.dtype == mstype.bool_ \
            and other.dtype == mstype.bool_:
        input = input.astype(mstype.float32)
        other = other.astype(mstype.float32)
    return xlogy_(input, other)


def arccosh(input):
    r"""
    Alias for :func:`mindspore.ops.acosh`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """
    return acosh_(input)


def arccosh_ext(input):
    r"""
    Alias for :func:`mindspore.ops.acosh_ext`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    return acosh_ext(input)


def arcsin(x):
    r"""
    Alias for :func:`mindspore.ops.asin`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    return asin_(x)


def arcsin_ext(x):
    r"""
    Alias for :func:`mindspore.ops.asin_ext`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    return asin_ext(x)


def arctan(input):
    r"""
    Alias for :func:`mindspore.ops.atan`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """
    return atan_(input)


def arctan_ext(input):
    r"""
    Alias for :func:`mindspore.ops.atan_ext`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    return atan_ext(input)


def arctan2(input, other):
    r"""
    Alias for :func:`mindspore.ops.atan2`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """
    return atan2_(input, other)


def arctan2_ext(input, other):
    r"""
    Alias for :func:`mindspore.ops.atan2_ext`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, ops
        >>> import numpy as np
        >>> x = Tensor(np.array([0, 1]), mindspore.float32)
        >>> y = Tensor(np.array([1, 1]), mindspore.float32)
        >>> output = ops.arctan2_ext(x, y)
        >>> print(output)
        [0.        0.7853982]
    """
    return atan2_ext(input, other)


def polar(abs, angle):  # pylint: disable=redefined-outer-name
    r"""
    Converts polar coordinates to Cartesian coordinates.

    Returns a complex tensor, its elements are Cartesian coordinates constructed with the polar
    coordinates which is specified by radial distance `abs` and polar angle `angle`.

    .. math::

        y_{i} =  abs_{i} * \cos(angle_{i}) + abs_{i} * \sin(angle_{i}) * j

    Args:
        abs (Tensor, float): Radial distance.
        angle (Tensor, float): Polar angle.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> abs = mindspore.tensor([1, 2], mindspore.float32)
        >>> angle = mindspore.tensor([np.pi / 2, 5 * np.pi / 4], mindspore.float32)
        >>> output = mindspore.ops.polar(abs, angle)
        >>> print(output)
        [ -4.3711388e-08+1.j         -1.4142137e+00-1.4142134j]
    """
    return polar_(abs, angle)


def pow_ext(input, exponent):
    """
    Calculates the `exponent` power of each element in `input`.

    When `exponent` is a Tensor, the shapes of `input` and `exponent` must be broadcastable.

    .. math::

        out_{i} = input_{i} ^{ exponent_{i}}

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Union[Tensor, Number]): The first input is a Number or a tensor whose data type is
            `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.dtype.html>`_ or
            `bool <https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.dtype.html>`_.
        exponent (Union[Tensor, Number]): The second input is a Number or a tensor whose data type is
            `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.dtype.html>`_ or
            `bool <https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.dtype.html>`_.

    Returns:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If types of `input` and `exponent` are bool.
        TypeError: The `input` is tensor and of type int or bool, while the `exponent` is negative int.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
        >>> exponent = 3.0
        >>> output = ops.pow(input, exponent)
        >>> print(output)
        [ 1.  8. 64.]
        >>>
        >>> input = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
        >>> exponent = Tensor(np.array([2.0, 4.0, 3.0]), mindspore.float32)
        >>> output = ops.pow(input, exponent)
        >>> print(output)
        [ 1. 16. 64.]
    """
    if isinstance(input, Tensor) and isinstance(exponent, numbers.Number):
        return pow_tensor_scalar_op(input, exponent)
    if isinstance(input, numbers.Number) and isinstance(exponent, Tensor):
        return pow_scalar_tensor_op(input, exponent)
    return pow(input, exponent)


def arccos(input):
    """
    Alias for :func:`mindspore.ops.acos` .

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    return acos(input)


def arccos_ext(input):
    """
    Alias for :func:`mindspore.ops.acos_ext` .

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    return acos_ext(input)


def arcsinh(input):
    r"""
    Alias for :func:`mindspore.ops.asinh`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    return asinh(input)


def arcsinh_ext(input):
    r"""
    Alias for :func:`mindspore.ops.asinh_ext`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    return asinh_ext(input)


def arctanh(input):
    r"""
    Alias for :func:`mindspore.ops.atanh`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    return atanh(input)


def bitwise_and(input, other):
    r"""
    Compute the bitwise AND of two input tensors.
    If `input` and `other` have different data types, the implicit type conversion rules and type promotion rules are
    followed.

    Args:
        input (Tensor): The first input tensor.
        other (Tensor): The second input tensor.

    Returns:
        Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([-1, -2, 3])
        >>> other = mindspore.tensor([1, 0, 3])
        >>> mindspore.ops.bitwise_and(input, other)
        Tensor(shape=[3], dtype=Int64, value= [1, 0, 3])
        >>> # Same as calling via the | operator:
        >>> input & other
        Tensor(shape=[3], dtype=Int64, value= [1, 0, 3])
    """
    return bitwise_and_(input, other)


def bitwise_and_ext(input, other):
    r"""
    Returns bitwise `and` of two tensors element-wise.

    .. math::

        out_i = input_{i} \wedge other_{i}

    Note:
        Args of `input` and `other` comply with the implicit type conversion rules to
        make the data types consistent.
        If they have different data types, the lower precision data type will be converted to
        the relatively highest precision data type.

    Args:
        input (Tensor): The input tensor.
        other (Tensor, Number.number): The input tensor or scalar. It has the same shape
            with `input` or its shape is able to broadcast with `input`.

    Returns:
        Tensor, the shape is the same as the one after broadcasting, and the data type is same as `input`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input = Tensor(np.array([0, 0, 1, -1, 1, 1, 1]), mindspore.int16)
        >>> other = Tensor(np.array([0, 1, 1, -1, -1, 2, 3]), mindspore.int16)
        >>> output = ops.bitwise_and_ext(input, other)
        >>> print(output)
        [ 0  0  1 -1  1  0  1]
    """
    if not isinstance(other, Tensor):
        return bitwise_and_scalar_(input, other)
    return bitwise_and_tensor_(input, other)


def bitwise_or(input, other):
    r"""
    Compute the bitwise OR of two input tensors.
    If `input` and `other` have different data types, the implicit type conversion rules and type promotion rules are
    followed.

    Args:
        input (Tensor): The first input tensor.
        other (Tensor): The second input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([-1, -2, 3])
        >>> other = mindspore.tensor([1, 0, 3])
        >>> mindspore.ops.bitwise_or(input, other)
        Tensor(shape=[3], dtype=Int64, value= [-1, -2,  3])
        >>> # Same as calling via the | operator:
        >>> input | other
        Tensor(shape=[3], dtype=Int64, value= [-1, -2,  3])
    """
    return bitwise_or_(input, other)


def bitwise_or_ext(input, other):
    r"""
    Returns bitwise `or` of two tensors element-wise.

    .. math::

        out_i = input_{i} \mid other_{i}

    Note:
        Args of `input` and `other` comply with the implicit type conversion rules to
        make the data types consistent.
        If they have different data types, the lower precision data type will be converted to
        the relatively highest precision data type.

    Args:
        input (Tensor): The input tensor.
        other (Tensor, Number.number): The input tensor or scalar. It has the same shape
            with `input` or its shape is able to broadcast with `input`.

    Returns:
        Tensor, the shape is the same as the one after broadcasting, and the data type is same as `input`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input = Tensor(np.array([0, 0, 1, -1, 1, 1, 1]), mindspore.int16)
        >>> other = Tensor(np.array([0, 1, 1, -1, -1, 2, 3]), mindspore.int16)
        >>> output = ops.bitwise_or_ext(input, other)
        >>> print(output)
        [ 0  1  1 -1 -1  3  3]
    """
    if not isinstance(other, Tensor):
        return bitwise_or_scalar_(input, other)
    return bitwise_or_tensor_(input, other)


def bitwise_xor(input, other):
    r"""
    Compute the bitwise XOR of two input tensors.
    If `input` and `other` have different data types, the implicit type conversion rules and type promotion rules are
    followed.

    Args:
        input (Tensor): The first input tensor.
        other (Tensor): The second input tensor.

    Returns:
        Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([-1, -2, 3])
        >>> other = mindspore.tensor([1, 0, 3])
        >>> mindspore.ops.bitwise_xor(input, other)
        Tensor(shape=[3], dtype=Int64, value= [-2, -2,  0])
        >>> # Same as calling via the ^ operator:
        >>> input ^ other
        Tensor(shape=[3], dtype=Int64, value= [-2, -2,  0])
    """
    return bitwise_xor_(input, other)


def bitwise_xor_ext(input, other):
    r"""
    Returns bitwise `xor` of two tensors element-wise.

    .. math::

        out_i = input_{i} \oplus other_{i}

    Note:
        Args of `input` and `other` comply with the implicit type conversion rules to
        make the data types consistent.
        If they have different data types, the lower precision data type will be converted to
        the relatively highest precision data type.

    Args:
        input (Tensor): The input tensor.
        other (Tensor, Number.number): The input tensor or scalar. It has the same shape
            with `input` or its shape is able to broadcast with `input`.

    Returns:
        Tensor, the shape is the same as the one after broadcasting, and the data type is same as `input`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input = Tensor(np.array([0, 0, 1, -1, 1, 1, 1]), mindspore.int16)
        >>> other = Tensor(np.array([0, 1, 1, -1, -1, 2, 3]), mindspore.int16)
        >>> output = ops.bitwise_xor_ext(input, other)
        >>> print(output)
        [ 0  1  0  0 -2  3  2]
    """
    if not isinstance(other, Tensor):
        return bitwise_xor_scalar_(input, other)
    return bitwise_xor_tensor_(input, other)


def bitwise_left_shift(input, other):
    r"""
    Perform a left bitwise shift operation on the `input` element-wise, where the number of bits to shift is
    specified by `other`.

    .. math::

        \begin{aligned}
        &out_{i} =input_{i} << other_{i}
        \end{aligned}

    Args:
        input (Union[Tensor, int, bool]): The input to be left shifted.
        other (Union[Tensor, int, bool]): The number of bit to be applied on left arithmetic shift.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([1, 2, 4, 8])
        >>> mindspore.ops.bitwise_left_shift(input, 1)
        Tensor(shape=[4], dtype=Int64, value= [ 2,  4,  8, 16])
    """
    if not isinstance(input, Tensor) and not isinstance(other, Tensor):
        raise TypeError("For 'bitwise_left_shift', at least one of the inputs should be a Tensor.")

    cast = ops.Cast()
    if isinstance(input, numbers.Number):
        if not isinstance(input, int):
            raise TypeError(f"For 'bitwise_left_shift', 'input' must be an integer, but got input:{type(input)}.")
        input = cast(input, other.dtype)
    elif isinstance(other, numbers.Number):
        if not isinstance(other, int):
            raise TypeError(f"For 'bitwise_left_shift', 'other' must be an integer, but got other:{type(other)}.")
        other = cast(other, input.dtype)
    ls = ops.LeftShift()
    return ls(input, other)


def bitwise_right_shift(input, other):
    r"""
    Perform a right bitwise shift operation on the `input` element-wise, where the number of bits to shift is
    specified by `other`.

    .. math::

        \begin{aligned}
        &out_{i} =input_{i} >> other_{i}
        \end{aligned}

    Args:
        input (Union[Tensor, int, bool]): The input to be right shifted.
        other (Union[Tensor, int, bool]): The number of bit to be applied on right arithmetic shift.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([1, 2, 4, 8])
        >>> mindspore.ops.bitwise_right_shift(input, 1)
        Tensor(shape=[4], dtype=Int64, value= [0, 1, 2, 4])
    """
    if not isinstance(input, Tensor) and not isinstance(other, Tensor):
        raise TypeError("For 'bitwise_left_shift', at least one of the inputs should be a Tensor.")
    cast = ops.Cast()
    if isinstance(input, numbers.Number):
        if not isinstance(input, int):
            raise TypeError(f"For 'bitwise_left_shift', 'input' must be an integer, but got input:{type(input)}.")
        input = cast(input, other.dtype)
    elif isinstance(other, numbers.Number):
        if not isinstance(other, int):
            raise TypeError(f"For 'bitwise_left_shift', 'other' must be an integer, but got other:{type(other)}.")
        other = cast(other, input.dtype)
    rs = ops.RightShift()
    return rs(input, other)


def inv(x):
    r"""
    Computes Reciprocal of input tensor element-wise.

    .. math::
        out_i = \frac{1}{x_{i} }

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([0.25, 0.4, 0.31, 0.52])
        >>> output = mindspore.ops.inv(input)
        >>> print(output)
        [4.        2.5       3.2258065 1.923077 ]
    """
    return inv_(x)


def inverse(input):
    """
    Compute the inverse of the input matrix.

    Note:
        The `input` must be at least two dimensions, and the size of the last two dimensions must be the same size.
        The matrix must be invertible. Dtype of complex numbers is not supported.

    Args:
        input (Tensor): The input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> print(mindspore.ops.inverse(mindspore.tensor([[1., 2.], [3., 4.]])))
        [[-2.   1. ]
         [ 1.5 -0.5]]
    """
    _check_is_tensor("input", input, "inverse")
    return matrix_inverse_(input)


def inverse_ext(input):
    """
    Compute the inverse of the input matrix.

    Args:
        input (Tensor): A matrix to be calculated. Input `input` must be at least 2-D, at most 6-D, and the size of
            the last two dimensions must be the same size. And the matrix must be invertible.

    Returns:
        Tensor, has the same type and shape as input `input`.

    Raises:
        ValueError: If the size of the last two dimensions of `input` is not the same.
        ValueError: If `input` is not empty and its dimensions are less than 2.
        ValueError: If the dimensions of `input` are larger than 6.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([[1., 2.], [3., 4.]], mindspore.float32)
        >>> print(mindspore.ops.inverse_ext(x))
        [[-2.0000002   1.0000001 ]
         [ 1.5000001  -0.50000006]]
    """
    return matrix_inverse_ext(input)


def invert(x):
    r"""
    Flip all bits of input tensor element-wise. For example, 01010101 becomes 10101010.

    .. math::
        out_i = \sim x_{i}

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> mindspore.ops.invert(mindspore.tensor([25, 4, 13, 9], mindspore.int16))
        Tensor(shape=[4], dtype=Int16, value= [-26,  -5, -14, -10])
    """
    return invert_(x)


def bessel_j0(x):
    r"""
    Computes the zeroth order Bessel function of the first kind for each element input.

    .. math::
        \begin{array}{ll} \\
            J_{0}(x) = \frac{1}{\pi} \int_{0}^{\pi} \cos (x \sin \theta) d \theta
            =\sum_{m=0}^{\infty} \frac{(-1)^{m} x^{2 m}}{2^{2 m} (m !)^2}
        \end{array}

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([0.5, 1., 2., 4.])
        >>> output = mindspore.ops.bessel_j0(x)
        >>> print(output)
        [0.93846981  0.76519769  0.22389078  -0.39714981]
    """
    return bessel_j0_(x)


def bessel_j1(x):
    r"""
    Computes the first order Bessel function of the first kind for each element input.

    .. math::
        \begin{array}{ll} \\
            J_{1}(x) = \frac{1}{\pi} \int_{0}^{\pi} \cos (x \sin \theta- \theta) d \theta
            =\sum_{m=0}^{\infty} \frac{(-1)^{m} x^{2 m+1}}{2^{2 m+1} m !(m+1) !}
        \end{array}

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([0.5, 1., 2., 4.])
        >>> output = mindspore.ops.bessel_j1(x)
        >>> print(output)
        [0.24226846  0.44005059  0.57672481 -0.06604333]
    """
    return bessel_j1_(x)


def bessel_i0(x):
    r"""
    Computes the zeroth order modified Bessel function of the first kind for each element input.

    .. math::
        \begin{array}{ll} \\
            I_{0}(x)=J_{0}(\mathrm{i} x)=\sum_{m=0}^{\infty}
            \frac{x^{2 m}}{2^{2 m} (m !)^{2}}
        \end{array}

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([-1., -0.5, 0.5, 1.])
        >>> output = mindspore.ops.bessel_i0(x)
        >>> print(output)
        [1.266066  1.0634835 1.0634835 1.266066]
    """
    return bessel_i0_(x)


def bessel_i0e(x):
    r"""
    Computes the exponentially scaled zeroth order modified Bessel function of the
    first kind for each element input.

    .. math::
        \begin{array}{ll} \\
            \text I_{0}e(x)=e^{(-|x|)} * I_{0}(x)=e^{(-|x|)} * \sum_{m=0}^
            {\infty} \frac{x^{2 m}}{2^{2 m} (m !)^{2}}
        \end{array}

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([-1., -0.5, 0.5, 1.])
        >>> output = mindspore.ops.bessel_i0e(x)
        >>> print(output)
        [0.46575961  0.64503527  0.64503527  0.46575961]
    """
    return bessel_i0e_(x)


def bessel_k0(x):
    r"""
    Computes the zeroth order modified Bessel function of the second kind for each element input.

    .. math::
        \begin{array}{ll} \\
            K_{0}(x)= \lim_{\nu \to 0} \left(\frac{\pi}{2}\right) \frac
            {I_{-\nu}(x)-I_{\nu}(x)}{\sin (\nu \pi)} = \int_{0}^{\infty} e^{-x \cosh t} d t
        \end{array}

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([0.5, 1., 2., 4.])
        >>> output = mindspore.ops.bessel_k0(x)
        >>> print(output)
        [0.92441907  0.42102444  0.11389387  0.01115968]
    """
    return bessel_k0_(x)


def bessel_k0e(x):
    r"""
    Computes the exponentially scaled zeroth order modified Bessel function of the
    second kind for each element input.

    .. math::
        \begin{array}{ll} \\
            K_{0}e(x)= e^{(-|x|)} * K_{0}(x) = e^{(-|x|)} * \int_{0}^
            {\infty} e^{-x \cosh t} d t
        \end{array}

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([0.5, 1., 2., 4.])
        >>> output = mindspore.ops.bessel_k0e(x)
        >>> print(output)
        [1.52410939  1.14446308  0.84156822  0.60929767]
    """
    return bessel_k0e_(x)


def bessel_y0(x):
    r"""
    Computes the zeroth order Bessel function of the second kind for each element input.

    .. math::
        \begin{array}{ll} \\
            Y_{0}(x)=\lim_{n \to 0} \frac{J_{n}(x) \cos n \pi-J_{-n}(x)}{\sin n \pi}
        \end{array}

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([0.5, 1., 2., 4.])
        >>> output = mindspore.ops.bessel_y0(x)
        >>> print(output)
        [-0.44451874  0.08825696  0.51037567  -0.01694074]
    """
    return bessel_y0_(x)


def bessel_y1(x):
    r"""
    Computes the first order Bessel function of the second kind for each element input.

    .. math::
        \begin{array}{ll} \\
            Y_{1}(x)=\lim_{n \to 1} \frac{J_{n}(x) \cos n \pi-J_{-n}(x)}{\sin n \pi}
        \end{array}

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([0.5, 1., 2., 4.])
        >>> output = mindspore.ops.bessel_y1(x)
        >>> print(output)
        [-1.47147239  -0.78121282  -0.10703243  0.39792571]
    """
    return bessel_y1_(x)


def eps(x):
    r"""
    Create a tensor with the same data type and shape as input, and the element value is the minimum value that the
    corresponding data type can express.

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([4, 1, 2, 3], mindspore.float32)
        >>> mindspore.ops.eps(x)
        Tensor(shape=[4], dtype=Float32, value= [ 1.19209290e-07,  1.19209290e-07,  1.19209290e-07,  1.19209290e-07])
    """
    return eps_(x)


def erfinv_(input):
    r"""
    Update the `input` tensor in-place by computing the inverse error function with `input`, which is defined in the
    range `(-1, 1)` as:

    .. math::


        erfinv(erf(input)) = input

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): The input tensor to compute with.

    Returns:

        Tensor.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If `input.dtype` is not one of: bool, int8, int16, int32, int64, uint8, float16, float32, bfloat16.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input = Tensor(np.array([0, 0.5, -0.9]), mindspore.float32)
        >>> output = ops.erfinv_(input)
        >>> print(output)
        [ 0.          0.47693613 -1.1630869 ]
    """
    return inplace_erfinv_op(input)


def linspace(start, end, steps):
    r"""
    Generate a one-dimensional tensor with `steps` elements, evenly distributed in the interval [start, end].

    .. math::
        \begin{aligned}
        &step = (end - start)/(steps - 1)\\
        &output = [start, start+step, start+2*step, ... , end]
        \end{aligned}

    Args:
        start (Union[Tensor, int, float]): Start value of interval.
        end (Union[Tensor, int, float]): Last value of interval.
        steps (Union[Tensor, int]): Number of elements.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> output = mindspore.ops.linspace(3, 10, 5)
        >>> print(output)
        [ 3.    4.75  6.5   8.25 10.  ]
        >>> output = mindspore.ops.linspace(-10, 10, 5)
        >>> print(output)
        [-10.  -5.   0.   5.  10.]
        >>> output = mindspore.ops.linspace(-10, 10, 1)
        >>> print(output)
        [-10.]
    """
    if not isinstance(start, Tensor):
        start = Tensor(start, mstype.float32)
    if not isinstance(end, Tensor):
        end = Tensor(end, mstype.float32)
    return linspace_(start, end, steps)


def linspace_ext(start, end, steps, *, dtype=None):
    r"""
    Generate a one-dimensional tensor with `steps` elements, evenly distributed in the interval [start, end].

    .. math::
        \begin{aligned}
        &step = (end - start)/(steps - 1)\\
        &output = [start, start+step, start+2*step, ... , end]
        \end{aligned}

    .. warning::
        Atlas training series does not support int16 dtype currently.

    Args:
        start (Union[float, int]): Start value of interval.
        end (Union[float, int]): Last value of interval.
        steps (int): Number of elements.

    Keyword Args:
        dtype (mindspore.dtype, optional): The data type returned.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> output = mindspore.ops.function.math_func.linspace_ext(3, 10, 5)
        >>> print(output)
        [ 3.    4.75  6.5   8.25 10.  ]
        >>> output = mindspore.ops.function.math_func.linspace_ext(-10, 10, 5)
        >>> print(output)
        [-10.  -5.   0.   5.  10.]
        >>> output = mindspore.ops.function.math_func.linspace_ext(-10, 10, 1)
        >>> print(output)
        [-10.]
    """
    return lin_space_ext_op(start, end, steps, dtype)


def det(input):
    r"""
    Computes the determinant of one or more square matrices.

    Args:
        input (Tensor): A matrix to be calculated, its shape should be :math:`[..., M, M]` who must
          have at least two dimensions, and the last two
          dimensions must be the same size. Data type must be float32, float64, complex64 or complex128.

    Returns:
        Tensor. The shape is :math:`input.shape[:-2]`, and the dtype is same as `input`.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If dtype of `input` not float32, float64, complex64 or complex128.
        ValueError: If the last two dimensions of `input` is not same size.
        ValueError: If the dimension of `input` is less than 2.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input = Tensor(np.array([[[-4.5, -1.5], [7.0, 6.0]], [[2.5, 0.5], [3.0, 9.0]]]), mindspore.float32)
        >>> output = ops.det(input)
        >>> print(output)
        [-16.5 21. ]

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    return matrix_determinant_(input)


def matrix_determinant(input):
    r"""
    `matrix_determinant` is deprecated, please use `det` instead.
    """
    logger.warning("matrix_determinant is deprecated, please use `det` instead.")
    return matrix_determinant_(input)


def log_matrix_determinant(input):
    r"""
    `log_matrix_determinant` is deprecated, please use `matrix_solve` instead.
    """
    logger.warning("`log_matrix_determinant` is deprecated, please use `matrix_solve` instead.")
    return log_matrix_determinant_(input)


def lu_solve(b, LU_data, LU_pivots):
    r"""
    Compute the solution to a system of linear equations :math:`Ay = b`.

    .. note::
        - `b` of shape :math:`(*, m, k)` , `LU_data` of shape :math:`(*, m, m)` ,
          `LU_pivots` of shape :math:`(*, m)` , where :math:`*` means batch dimensions.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        b (Tensor): Column vector `b` in the above equation.
        LU_data (Tensor): LU decomposition. The `A` in the formula above.
        LU_pivots (Tensor): Permutation matrix P of LU decomposition.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> b = mindspore.tensor([[1.], [3.], [3.]])
        >>> LU_data = mindspore.tensor([[2., 1., 1.], [0.5, 1., 1.5], [0.5, 0., 2.5]])
        >>> LU_pivots = mindspore.tensor(([2, 2, 3]), mindspore.int32)
        >>> y = mindspore.ops.lu_solve(b, LU_data, LU_pivots)
        >>> print(y)
        [[ 1.9000001]
         [-1.4000001]
         [ 0.6      ]]
    """
    out = lu_solve_(b, LU_data, LU_pivots)
    return out


def matrix_solve(matrix, rhs, adjoint=False):  # pylint: disable=redefined-outer-name
    r"""
    Solves systems of linear equations.

    .. math::
        \begin{aligned}
        &matrix[..., M, M] * x[..., M, K] = rhs[..., M, K]\\
        &adjoint(matrix[..., M, M]) * x[..., M, K] = rhs[..., M, K]
        \end{aligned}

    .. warning::
        On GPU, if the matrix is irreversible, an error may be reported or an unknown result may be returned.

    Args:
        matrix (Tensor): The first input tensor.
        rhs (Tensor): The second input tensor.
        adjoint(bool, optional): Indicating whether to solve with matrix or its (block-wise) adjoint.
            Default ``False`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore
        >>> matrix = mindspore.tensor([[5., 4.], [3., 1.]])
        >>> rhs = mindspore.tensor([[7.], [2.]])
        >>> result = mindspore.ops.matrix_solve(matrix, rhs)
        >>> print(result)
        [[0.14285707]
         [1.5714287 ]]
    """
    matrix_solve_ = _get_cache_prim(MatrixSolve)(adjoint=adjoint)
    return matrix_solve_(matrix, rhs)


def slogdet(input):
    r"""
    Computes the sign and the log of the absolute value of the determinant of one or more square matrices.

    Note:
        The type of output always be real-value, even `input` is complex.

    Args:
        input (Tensor): The input tensor, shape is :math:`(..., M, M)`.

    Returns:
        Tuple of 2 tensors which are sign and the log of the absolute value.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[1., 2], [3, 4]])
        >>> sign, value = mindspore.ops.slogdet(input)
        >>> print(sign)
        -1.0
        >>> print(value)
        0.6931472
        >>> input = mindspore.tensor([[[-4.5, -1.5], [7.0, 6.0]], [[2.5, 0.5], [3.0, 9.0]]])
        >>> sign, value = mindspore.ops.slogdet(input)
        >>> print(sign)
        [-1.  1.]
        >>> print(value)
        [2.8033605 3.0445223]
    """
    return log_matrix_determinant_(input)


def truncate_div(x, y):
    """
    Divide the first input tensor `x` by the second input tensor `y` element-wise and rounds the results
    of division towards zero.

    Note:
        Support implicit type conversion and broadcasting.

    Args:
        x(Union[Tensor, Number, bool]): The first input tensor.
        y(Union[Tensor, Number, bool]): The second input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> mindspore.ops.truncate_div(mindspore.tensor([2, 4, -1]), mindspore.tensor([3, 3, 3]))
        Tensor(shape=[3], dtype=Int64, value= [0, 1, 0])
    """
    return truncate_div_(x, y)


def truncate_mod(x, y):
    r"""
    Return the remainder of division element-wise.

    Support implicit type conversion and broadcasting.

    .. warning::
        - The input data does not support 0.
        - When the elements of input exceed 2048 , the accuracy of operator cannot guarantee the requirement of
          double thousandths in the mini form.
        - Due to different architectures, the calculation results of this operator on NPU and CPU may be inconsistent.
        - If shape is expressed as (D1,D2... ,Dn), then D1\*D2... \*DN<=1000000,n<=8.

    Args:
        x(Union[Tensor, Number, bool]): The first input tensor.
        y(Union[Tensor, Number, bool]): The second input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> output = mindspore.ops.truncate_mod(mindspore.tensor([2., 4., -1.]), mindspore.tensor([3., 3., 3.]))
        >>> print(output)
        [ 2.  1. -1.]
    """
    return truncate_mod_(x, y)


def ldexp(x, other):
    """
    Multiplies input tensor by :math:`2^{other}` element-wise.

    Typically this function is used to construct floating point numbers by multiplying mantissas in `x` with integral
    powers of two created from the exponents in `other`:

    .. math::

        out_{i} = x_{i} * ( 2 ^{other_{i}} )

    Args:
        x (Tensor): Mantissas tensor.
        other (Tensor): Exponents tensor, typically integers.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([1.], mindspore.float32)
        >>> other = mindspore.tensor([1, 2, 3, 4], mindspore.int32)
        >>> out = mindspore.ops.ldexp(x, other)
        >>> print(out)
        [ 2.  4.  8. 16.]
        >>> x = mindspore.tensor([[1.], [2]], mindspore.float32)
        >>> other = mindspore.tensor([[1.], [2]], mindspore.int32)
        >>> out = mindspore.ops.ldexp(x, other)
        >>> print(out)
        [[2.]
         [8.]]
    """
    out = tensor_mul(x, tensor_pow(2.0, other))
    return out


def logit(input, eps=None):
    r"""
    Calculate the logit of a tensor element-wise.

    The formula is defined as:

    .. math::
        \begin{align}
        y_{i} & = \ln(\frac{z_{i}}{1 - z_{i}}) \\
        z_{i} & = \begin{cases}
        input_{i} & \text{if eps is None} \\
        \text{eps} & \text{if } input_{i} \lt \text{eps} \\
        input_{i} & \text{if } \text{eps} \leq input_{i} \leq 1 - \text{eps} \\
        1 - \text{eps} & \text{if } input_{i} \gt 1 - \text{eps}
        \end{cases}
        \end{align}

    Args:
        input (Tensor): The input tensor.
        eps (float, optional): The epsilon for input clamp bound. If eps is not None,
            the input clamp bound is defined as [eps, 1-eps], otherwise, the `input` is not clamped. Default ``None`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([0.1, 0.2, 0.3], mindspore.float32)
        >>> output = mindspore.ops.logit(x, eps=1e-5)
        >>> print(output)
        [-2.1972246 -1.3862944 -0.8472978]
    """
    if eps is None:
        eps = -1.0
    logit_ = _get_cache_prim(P.Logit)(eps)
    return logit_(input)


#####################################
# Comparison Operation Functions.
#####################################


def lt(input, other):
    """
    Alias for :func:`mindspore.ops.less` .

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    return less(input, other)


def le(input, other):
    r"""
    Compute the value of :math:`input <= other` element-wise.

    .. math::

        out_{i} =\begin{cases}
            & \text{True,    if } input_{i}<=other_{i} \\
            & \text{False,   if } input_{i}>other_{i}
            \end{cases}

    .. note::
        - Support implicit type conversion.
        - The inputs must be two tensors or one tensor and one scalar.
        - When the inputs are one tensor and one scalar, the scalar could only be a constant.

    Args:
        input (Union[Tensor, Number, bool]): The first input.
        other (Union[Tensor, Number, bool]): The second input.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> # case 1: The shape of two inputs are different
        >>> input = mindspore.tensor([1, 2, 3], mindspore.float32)
        >>> output = mindspore.ops.le(input, 2.0)
        >>> print(output)
        [True  True False]
        >>> # case 2: The shape of two inputs are the same
        >>> input = mindspore.tensor([1, 2, 3], mindspore.int32)
        >>> other = mindspore.tensor([1, 2, 4], mindspore.int32)
        >>> output = mindspore.ops.le(input, other)
        >>> print(output)
        [ True  True  True]
    """
    return tensor_le(input, other)


def gt(input, other):
    r"""
    Compute the value of :math:`input > other` element-wise.

    .. math::

        out_{i} =\begin{cases}
            & \text{True,    if } input_{i}>other_{i} \\
            & \text{False,   if } input_{i}<=other_{i}
            \end{cases}

    Note:
        - Support implicit type conversion.
        - The inputs must be two tensors or one tensor and one scalar.
        - When the inputs are two tensors, dtypes of them cannot be bool at the same time,
          and the shapes of them can be broadcast.
        - When the inputs are one tensor and one scalar, the scalar could only be a constant.
        - Broadcasting is supported.
        - If the input Tensor can be broadcast, the low dimension will be extended to the corresponding high dimension
          in another input by copying the value of the dimension.

    Args:
        input (Union[Tensor, number.Number, bool]): The first input.
        other (Union[Tensor, number.Number, bool]): The second input.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> # case 1: The shape of two inputs are different
        >>> input = mindspore.tensor([1, 2, 3], mindspore.float32)
        >>> output = mindspore.ops.gt(input, 2.0)
        >>> print(output)
        [False False True]
        >>> # case 2: The shape of two inputs are the same
        >>> input = mindspore.tensor([1, 2, 3], mindspore.int32)
        >>> other = mindspore.tensor([1, 2, 4], mindspore.int32)
        >>> output = mindspore.ops.gt(input, other)
        >>> print(output)
        [ False False False]
    """
    return tensor_gt(input, other)


def ge(input, other):
    r"""
    Compute the value of :math:`input >= other` element-wise.

    Note:
        - Support implicit type conversion.
        - The inputs must be two tensors or one tensor and one scalar.
        - When the inputs are two tensors, dtypes of them cannot be bool at the same time,
          and the shapes of them can be broadcast.
        - When the inputs are one tensor and one scalar, the scalar could only be a constant.
        - Broadcasting is supported.
        - If the input Tensor can be broadcast, the low dimension will be extended to the corresponding high dimension
          in another input by copying the value of the dimension.

    .. math::

        out_{i} =\begin{cases}
            & \text{True,    if } input_{i}>=other_{i} \\
            & \text{False,   if } input_{i}<other_{i}
            \end{cases}

    Args:
        input (Union[Tensor, Number, bool]): The first input.
        other (Union[Tensor, Number, bool]): The second input.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> # case 1: The shape of two inputs are different
        >>> input = mindspore.tensor([1, 2, 3], mindspore.float32)
        >>> output = mindspore.ops.ge(input, 2.0)
        >>> print(output)
        [False  True True]
        >>> # case 2: The shape of two inputs are the same
        >>> input = mindspore.tensor([1, 2, 3], mindspore.int32)
        >>> other = mindspore.tensor([1, 2, 4], mindspore.int32)
        >>> output = mindspore.ops.ge(input, other)
        >>> print(output)
        [ True  True False]
    """
    return tensor_ge(input, other)


def eq(input, other):
    r"""
    Compute the equivalence of the two inputs element-wise.

    .. math::

        out_{i} =\begin{cases}
            & \text{True,    if } input_{i} = other_{i} \\
            & \text{False,   if } input_{i} \ne other_{i}
            \end{cases}

    Note:
        - Support implicit type conversion.
        - The input must be two Tensors, or a Tensor and a Scalar.
        - The shapes of the inputs can be broadcasted to each other.

    Args:
        input (Union[Tensor, Number]): The first input.
        other (Union[Tensor, Number]): The second input.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> # case 1: The shape of two inputs are different
        >>> x = mindspore.tensor([1, 2, 3], mindspore.float32)
        >>> output = mindspore.ops.eq(x, 2.0)
        >>> print(output)
        [False  True False]
        >>> # case 2: The shape of two inputs are the same
        >>> x = mindspore.tensor([1, 2, 3], mindspore.int32)
        >>> y = mindspore.tensor([1, 2, 4], mindspore.int32)
        >>> output = mindspore.ops.eq(x, y)
        >>> print(output)
        [ True  True False]
    """
    return equal(input, other)


def ne(input, other):
    r"""
    Compute the non-equivalence of two inputs element-wise.

    .. math::

        out_{i} =\begin{cases}
        & \text{True,    if } input_{i} \ne other_{i} \\
        & \text{False,   if } input_{i} = other_{i}
        \end{cases}

    Note:
        - Support implicit type conversion.
        - When the inputs are two tensors, the shapes of them could be broadcast.
        - When the inputs are one tensor and one scalar, the scalar could only be a constant.
        - Broadcasting is supported.

    Args:
        input (Union[Tensor, Number, bool]): The first input.
        other (Union[Tensor, Number, bool]): The second input.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> # case 1: The shape of two inputs are different
        >>> input = mindspore.tensor([1, 2, 3], mindspore.float32)
        >>> output = mindspore.ops.ne(input, 2.0)
        >>> print(output)
        [True False  True]
        >>> # case 2: The shape of two inputs are the same
        >>> input = mindspore.tensor([1, 2, 3], mindspore.int32)
        >>> other = mindspore.tensor([1, 2, 4], mindspore.int32)
        >>> output = mindspore.ops.ne(input, other)
        >>> print(output)
        [ False False  True]
    """
    return not_equal(input, other)

@deprecated("2.8.0", "Tensor.isclose", False, "ops.")
def approximate_equal(x, y, tolerance=1e-5):
    r"""
    `ops.approximate_equal` is deprecated from version 2.8.0 and will be removed in a future version,
    please use :func:`mindspore.Tensor.isclose` instead.

    Return a boolean tensor where two tensors are element-wise equal within a tolerance.

    Support implicit type conversion and type promotion.

    Math function is defined as:

    .. math::

        out_i = \begin{cases}
        & \text{ if } \left | x_{i} - y_{i} \right | < \text{tolerance},\ \ True  \\
        & \text{ if } \left | x_{i} - y_{i} \right | \ge \text{tolerance},\ \  False
        \end{cases}

    Two infinite values and two NaN values are not considered equal.

    Args:
        x (Tensor): The first input tensor.
        y (Tensor): The second input tensor.
        tolerance (float): The maximum deviation that two elements can be considered equal. Default ``1e-5`` .

    Returns:
        Tensor

    Supported Platforms:
        Deprecated

    Examples:
        >>> import mindspore
        >>> mindspore.ops.approximate_equal(mindspore.tensor([1e6, 2e6, float("inf"), float("-inf"), float("nan")]),
        ...                                 mindspore.tensor([1e6, 2e7, float("inf"), float("-inf"), float("nan")]))
        Tensor(shape=[5], dtype=Bool, value= [ True, False, False, False, False])
        >>>
        >>> mindspore.ops.approximate_equal(mindspore.tensor([1e6, 2e6, 3e6]),
        ...                                 mindspore.tensor([1.00001e6, 2.00002e6, 3.00009e6]), tolerance=1e3)
        Tensor(shape=[3], dtype=Bool, value= [ True,  True,  True])
        >>>
        >>> # If `x` and `y` have different datatypes, the lower precision data type will be converted to the
            relatively highest precision data type.
        >>> mindspore.ops.approximate_equal(mindspore.tensor([1, 2], mindspore.int32),
        ...                                 mindspore.tensor([1., 2], mindspore.float32))
        Tensor(shape=[2], dtype=Bool, value= [ True,  True])
    """
    return _get_cache_prim(P.ApproximateEqual)(tolerance)(x, y)


def isnan(input):
    r"""
    Return a boolean tensor indicating which elements are NaN.

    Args:
        input (Tensor): The input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([-1, 3, float("inf"), float("-inf"), float("nan")])
        >>> mindspore.ops.isnan(input)
        Tensor(shape=[5], dtype=Bool, value= [False, False, False, False,  True])
    """
    return isnan_(input)


def isclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    """
    Return a boolean tensor where two tensors are element-wise equal within a tolerance. Math function is defined as:

    .. math::
        |input-other| ≤ atol + rtol × |other|

    Two Infinite values are considered equal if they have the same sign, Two NaN values are considered equal if
    `equal_nan` is ``True`` .

    Args:
        input (Tensor): The first input tensor.
        other (Tensor): The second input tensor.
        rtol (Union[float, int, bool], optional): Relative tolerance. Default ``1e-05`` .
        atol (Union[float, int, bool], optional): Absolute tolerance. Default ``1e-08`` .
        equal_nan (bool, optional): Whether two NaNs are considered equal. Default ``False`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> mindspore.ops.isclose(mindspore.tensor([2e6, float("inf"), float("-inf"), float("inf"), float("nan")]),
        ...                       mindspore.tensor([2e7, float("inf"), float("-inf"), float("-inf"), float("nan")]))
        Tensor(shape=[5], dtype=Bool, value= [False,  True,  True, False, False])
        >>>
        >>> mindspore.ops.isclose(mindspore.tensor([1e6, 2e6, 3e6]),
        ...                       mindspore.tensor([1.00008e6, 2.00008e7, 3.00008e8]), rtol=1e3)
        Tensor(shape=[3], dtype=Bool, value= [ True,  True,  True])
        >>>
        >>> mindspore.ops.isclose(mindspore.tensor([1e6, 2e6, 3e6]),
        ...                       mindspore.tensor([1.00001e6, 2.00002e6, 3.00009e6]), atol=1e3)
        Tensor(shape=[3], dtype=Bool, value= [ True,  True,  True])
        >>> mindspore.ops.isclose(mindspore.tensor([float("nan"), 1, 2]),
        ...                       mindspore.tensor([float("nan"), 1, 2]), equal_nan=True)
        Tensor(shape=[3], dtype=Bool, value= [ True,  True,  True])
    """
    is_close = _get_cache_prim(P.IsClose)(rtol=rtol, atol=atol, equal_nan=equal_nan)
    return is_close(input, other)


def isreal(input):
    """
    Return a boolean tensor indicating which elements are real.

    A complex value is considered real when its imaginary part is 0.

    Args:
        input (Tensor): The input tensor.

    Returns:
       Tensor

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([False, 0j, 1, 2.1, 1+2j])
        >>> mindspore.ops.isreal(input)
        Tensor(shape=[5], dtype=Bool, value= [ True,  True,  True,  True, False])
    """

    _check_is_tensor("input", input, "isreal")

    # Note: Integral and Floating tensor values are always real
    value = Tensor(1, mstype.bool_)
    real_dtype = mstype.int_type + mstype.uint_type + mstype.float_type + (mstype.bool_,)
    if input.dtype in real_dtype:
        return fill_v2_(input.shape, value)
    return imag_(input) == 0


def is_complex(input):
    '''
    Return a boolean tensor indicating which elements are complex.

    Args:
        input (Tensor): The input tensor.

    Returns:
        Bool

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([1, 1+1j, 2+2j], mindspore.dtype.complex64)
        >>> mindspore.ops.is_complex(input)
        True
        >>>
        >>> input = mindspore.tensor([1, 1+1j, 2+2j], mindspore.dtype.complex128)
        >>> mindspore.ops.is_complex(input)
        True
        >>> input = mindspore.tensor([1, 1+1j, 2+2j], mindspore.dtype.int32)
        >>> mindspore.ops.is_complex(input)
        False
    '''
    if not isinstance(input, (Tensor, Tensor_)):
        raise TypeError("The input must be Tensor!")
    return input.dtype in mstype.complex_type


def fmax(input, other):
    r"""
    Compute the maximum of input tensors element-wise.

    .. math::
        output_i = \max(x1_i, x2_i)

    Note:
        - Support implicit type conversion and type promotion.
        - Shapes of `input` and `other` should be able to broadcast.
        - If one of the elements to be compared is NaN, another element is returned.

    Args:
        input (Tensor): The first input.
        other (Tensor): The second input.

    Returns:
        Tensor

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import mindspore
        >>> x1 = mindspore.tensor([1.0, 5.0, 3.0], mindspore.float32)
        >>> x2 = mindspore.tensor([4.0, 2.0, 6.0], mindspore.float32)
        >>> output = mindspore.ops.fmax(x1, x2)
        >>> print(output)
        [4. 5. 6.]
    """
    fmax_ = Fmax()
    return fmax_(input, other)


def fmin(input, other):
    r"""
    Computes the minimum of input tensors element-wise.

    .. math::
        output_i = min(input_i, other_i)

    Note:
        - Inputs of `input` and `other` comply with the implicit type conversion rules to make the data types
          consistent.
        - Shapes of `input` and `other` should be able to broadcast.
        - If one of the elements to be compared is NaN, another element is returned.

    Args:
        input (Tensor): The first tensor. The supported dtypes are: float16, float32, float64, int32, int64.
        other (Tensor): The second tensor. The supported dtypes are: float16, float32, float64, int32, int64.

    Returns:
        A Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `input` or `other` is not Tensor.
        TypeError: If dtype of `input` or `other` is not one of: float16, float32, float64, int32, int64.
        ValueError: If the shape of  `input` and `other` can not broadcast.

    Supported Platforms:


    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> from mindspore import dtype as mstype
        >>> input = Tensor(np.array([1.0, 5.0, 3.0]), mstype.float32)
        >>> other = Tensor(np.array([4.0, 2.0, 6.0]), mstype.float32)
        >>> output = ops.fmin(input, other)
        >>> print(output)
        [1. 2. 3.]
    """
    fmin_ = Fmin()
    return fmin_(input, other)


def median_ext(input, dim=None, keepdim=False):
    r"""
    Return the median(s) and indice(s) of the tensor along the specified dimension.

    Args:
        input (Tensor): The input tensor.
        dim (int, optional): Specify the dimension for calculation. Default ``None`` .
        keepdim (bool, optional): Whether the output tensor has dim retained.
            Default ``False`` .

    Returns:
        Tuple(median, median_indices) of 2 tensors.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> x = mindspore.tensor(np.array([[0.57, 0.11, 0.21],[0.38, 0.50, 0.57], 
        ...                                [0.36, 0.16, 0.44]]).astype(np.float32))
        >>> y = mindspore.ops.function.math_func.median_ext(x, dim=0, keepdim=False)
        >>> print(y)
        (Tensor(shape=[3], dtype=Float32, value= [ 3.79999995e-01,  1.59999996e-01,  4.39999998e-01]),
        Tensor(shape=[3], dtype=Int64, value= [1, 2, 2]))
    """
    if dim is None:
        return median_ext_op(input)
    return median_dim_op(input, dim, keepdim)


def median(input, axis=-1, keepdims=False):
    r"""
    Return the median(s) and indice(s) of the tensor along the specified axis.

    .. warning::
        - `indices` does not necessarily contain the first occurrence of each median value found in the `input`,
          unless it is unique. The specific implementation of this API is device-specific.
          The results may be different on CPU and GPU.

    Args:
        input (Tensor): The input tensor.
        axis (int, optional): Specify the axis for computation. Default ``-1`` .
        keepdims (bool, optional): Whether the output tensor has dim retained. Default ``False`` .

    Returns:
        Tuple(median, median_indices) of 2 tensors.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[9, 3, 4, 5],
        ...                           [5, 2, 7, 4],
        ...                           [8, 1, 3, 6]])
        >>> # case 1: By default, compute the median along axis -1.
        >>> mindspore.ops.median(input)
        (Tensor(shape=[3], dtype=Int64, value= [4, 4, 3]),
         Tensor(shape=[3], dtype=Int64, value= [2, 3, 2]))
        >>>
        >>> # case 2: Compute the median along axis 0.
        >>> mindspore.ops.median(input, axis=0)
        (Tensor(shape=[4], dtype=Int64, value= [8, 2, 4, 5]),
         Tensor(shape=[4], dtype=Int64, value= [2, 1, 0, 0]))
        >>>
        >>> # case 3: If keepdims=True, the output shape will be same of that of the input.
        >>> mindspore.ops.median(input, axis=0, keepdims=True)
        (Tensor(shape=[1, 4], dtype=Int64, value=
         [[8, 2, 4, 5]]),
         Tensor(shape=[1, 4], dtype=Int64, value=
         [[2, 1, 0, 0]]))
    """
    median_ = _get_cache_prim(Median)(global_median=False, axis=axis, keep_dims=keepdims, ignore_nan=False)
    return median_(input)


def nanmedian(input, axis=-1, keepdims=False):
    r"""
    Computes the median and indices of `input` in specified dimension, ignoring NaN.

    .. warning::
        `indices` does not necessarily contain the first occurrence of each median value found in the `input`,
        unless it is unique.

    Args:
        input (Tensor): The input tensor.
        axis (int, optional): The specified axis for computation. Default ``-1`` .
        keepdims (bool, optional): Whether the output tensor needs to retain dimension or not. Default ``False`` .

    Returns:
        Tuple(median, median_indices)

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([[0.57, 0.11, float("nan")],
        ...             [0.38, float("nan"), float("nan")],
        ...             [0.36, 0.16, float("nan")]], mindspore.float32)
        >>> y, idx = mindspore.ops.nanmedian(x, axis=0, keepdims=False)
        >>> print(y)
        [0.38 0.11  nan]
        >>> print(idx)
        [1 0 0]
    """
    nanmedian_ = _get_cache_prim(Median)(global_median=False, axis=axis, keep_dims=keepdims, ignore_nan=True)
    return nanmedian_(input)


def nanmean(input, axis=None, keepdims=False, *, dtype=None):
    r"""
    Computes the mean of `input` in specified dimension, ignoring NaN.

    Args:
        input (Tensor): The input tensor.
        axis (int, optional): The specified axis for computation. Default ``None`` .
        keepdims (bool, optional): Whether the output tensor has to dim retained. Default ``False`` .

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): The data type returned. Default ``None`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([[0.5, -1.1, float('nan')], [3.4, float('nan'), float('nan')]], mindspore.float32)
        >>> y = mindspore.ops.nanmean(x, axis=0, keepdims=False)
        >>> print(y)
        [ 1.95 -1.1    nan]
    """
    _check_is_tensor("input", input, "nanmean")
    _check_repeat_in_axis(axis, input.ndim, "nanmean")
    if input.dtype not in mstype.float_type:
        raise TypeError(f"For 'nanmean', input should be floating point dtype, but got {type(input)}.")
    nan_sum = nansum(input, axis, keepdims)
    is_num = isnan(input).logical_not()
    is_num = is_num.sum(axis=axis, keepdims=keepdims)
    out = nan_sum / is_num
    if dtype is not None:
        return out.astype(dtype)
    return out


def orgqr(input, input2):
    r"""
    Compute the first :math:`N` columns of a product of Householder matrices.

    Usually used to calculate the explicit representation of the orthogonal matrix :math:`Q`
    returned by :class:`mindspore.ops.Geqrf`.

    Take the case of input without batch dimension as an example:
    Suppose input `input` is a matrix of size :math:`(M, N)` after householder transformation.
    When the diagonal of `input` is set to 1, every column of lower triangular in `input` is
    denoted as :math:`w_j` for :math:`j` for
    :math:`j=1, \ldots, M`, this function returns the first :math:`N` columns of the matrix

    .. math::
        H_{1} H_{2} \ldots H_{k} \quad \text { with } \quad H_{j}=\mathrm{I}_{M}-\tau_{j} w_{j} w_{j}^{\mathrm{H}}

    where :math:`\mathrm{I}_{M}` is the :math:`M`-dimensional identity matrix. And when :math:`w` is complex,
    :math:`w^{\mathrm{H}}` is the conjugate transpose, otherwise the transpose.
    The output matrix is the same size as the input matrix `input`.
    :math:`tau` is corresponding to `input2`.

    Args:
        input (Tensor): 2-D or 3-D input tensor, householder vectors, shape :math:`(*, M, N)`.
        input2 (Tensor): 1-D or 2-D input tensor, householder reflection coefficients, shape :math:`(*, K)`, where `K`
            is less than or equal to `N`, indicating.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[-2.0, -1.0], [1.0, 2.0]])
        >>> y, tau = mindspore.ops.geqrf(input)
        >>> mindspore.ops.orgqr(y, tau)
        Tensor(shape=[2, 2], dtype=Float32, value=
        [[-8.94427061e-01,  4.47213590e-01],
         [ 4.47213590e-01,  8.94427180e-01]])
    """

    orgqr_ = Orgqr()
    return orgqr_(input, input2)


def ormqr(input, tau, other, left=True, transpose=False):
    r"""
    Calculates the product of a matrix `other` and a matrix Q (represented by Householder vectors `input` and
    Householder reflection coefficients `tau`).

    If `left` is True, computes Q \* `other` , otherwise, compute `other` \* Q.

    Args:
        input (Tensor): The input tensor, shape :math:`(*, mn, k)`, when `left` is True, mn equals to m,
            otherwise, mn equals to n.
        tau (Tensor): The input tensor, shape :math:`(*, min(mn, k))` .
        other (Tensor): The input tensor, shape :math:`(*, m, n)` .
        left (bool, optional): The order of computation. Default ``True`` .
        transpose (bool, optional): Whether the matrix Q is conjugate transposed or not. Default ``False`` .

    Returns:
        Tensor

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor(([[-114.6, 10.9, 1.1], [-0.304, 38.07, 69.38], [-0.45, -0.17, 62]]),
        ...                             mindspore.float32)
        >>> tau = mindspore.tensor(([1.55, 1.94, 3.0]), mindspore.float32)
        >>> other = mindspore.tensor(([[-114.6, 10.9, 1.1],
        ...                            [-0.304, 38.07, 69.38],
        ...                            [-0.45, -0.17, 62]]), mindspore.float32)
        >>> output = mindspore.ops.ormqr(input, tau, other)
        >>> print(output)
        [[  63.82713   -13.823125 -116.28614 ]
         [ -53.659264  -28.157839  -70.42702 ]
         [ -79.54292    24.00183   -41.34253 ]]
    """

    ormqr_ = _get_cache_prim(Ormqr)(left, transpose)
    return ormqr_(input, tau, other)


def hypot(input, other):
    r"""
    Given the legs of a right triangle, return its hypotenuse, element-wise.

    Support broadcasting and type promotion.

    .. math::
        out_i = \sqrt{input_i^2 + other_i^2}

    Args:
        input (Tensor): The first input tensor.
        other (Tensor): The second input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([3., 5., 7.])
        >>> other = mindspore.tensor([4., 12., 24.])
        >>> output = mindspore.ops.hypot(input, other)
        >>> print(output)
        [ 5. 13. 25.]
    """

    hypot_ = Hypot()
    return hypot_(input, other)


def heaviside(input, values):
    r"""
    Perform Heaviside step function element-wise.

    Support broadcasting.

    .. math::
        \text { heaviside }(\text { input, values })=\left\{\begin{array}{ll}
        0, & \text { if input }<0 \\
        \text { values, } & \text { if input }=0 \\
        1, & \text { if input }>0
        \end{array}\right.

    Args:
        input (Tensor): The input tensor.
        values (Tensor): The value to fill when the element in `input` is 0.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[-2., 0, 3],
        ...                           [5, -1, 0],
        ...                           [0, 7, -3]])
        >>> values = mindspore.tensor([2, 0.5, 1])
        >>> output = mindspore.ops.heaviside(input, values)
        >>> print(output)
        [[0.  0.5 1. ]
         [1.  0.  1. ]
         [2.  1.  0. ]]
        >>> output = mindspore.ops.heaviside(input, mindspore.tensor(0.5))
        >>> print(output)
        [[0.  0.5 1. ]
         [1.  0.  0.5]
         [0.5 1.  0. ]]
        >>> output = mindspore.ops.heaviside(mindspore.tensor(-3.), values)
        >>> print(output)
        [0. 0. 0.]
    """

    heaviside_ = Heaviside()
    return heaviside_(input, values)


def histc(input, bins=100, min=0., max=0.):
    r"""
    Computes the histogram of a tensor.

    The elements are sorted into equal width bins between `min` and `max`.
    If `min` and `max` are both zero, the minimum and maximum values of the data are used.

    Elements lower than min or higher than max are ignored.

    Args:
        input (Tensor): The input tensor.
        bins (int, optional): Number of histogram bins. Default ``100`` .
        min (int, float, optional): Minimum value of the histogram data range. Default ``0.`` .
        max (int, float, optional): Maximum value of the histogram data range. Default ``0.`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([1., 1, 2, 2, 2, 5, 8])
        >>> output = mindspore.ops.histc(input, bins=5, min=0, max=9)
        >>> print(output)
        [2. 3. 1. 0. 1.]
    """
    if not isinstance(min, (int, float)):
        raise TypeError(f"For 'histc', parameter 'min' must be an int or float, but got {type(min)}.")
    if not isinstance(max, (int, float)):
        raise TypeError(f"For 'histc', parameter 'max' must be an int or float, but got {type(max)}.")

    histogram_op = _get_cache_prim(P.Histogram)(bins, float(min), float(max))
    return histogram_op(input)


def logspace(start, end, steps, base=10, *, dtype=mstype.float32):
    r"""
    Return a tensor with `steps` elements, evenly distributed in the interval
    [ :math:`base^{start}` , :math:`base^{end}`].

    .. math::
        \begin{aligned}
        &step = (end - start)/(steps - 1)\\
        &output = [base^{start}, base^{start + 1 * step}, ... , base^{start + (steps-2) * step}, base^{end}]
        \end{aligned}

    Args:
        start (Union[float, Tensor]): Start value of interval.
        end (Union[float, Tensor]): Last value of interval.
        steps (int): Number of elements.
        base (int, optional): Base of the logarithm function. Default ``10`` .

    Keyword Args:
        dtype (mindspore.dtype, optional): The data type specified. Default ``mstype.float32`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> output = mindspore.ops.logspace(1., 10., steps = 10, base = 10)
        >>> print(output)
        [1.e+01 1.e+02 1.e+03 1.e+04 1.e+05 1.e+06 1.e+07 1.e+08 1.e+09 1.e+10]
    """
    if isinstance(start, float):
        start = ops.cast(start, mstype.float32)
    if isinstance(end, float):
        end = ops.cast(end, mstype.float32)
    logspace_ = _get_cache_prim(P.LogSpace)(steps, base, dtype)
    return logspace_(start, end)


def logaddexp(input, other):
    r"""
    Computes the logarithm of the sum of exponentiations of the inputs.
    This function is useful in statistics where the calculated probabilities of events may be
    too small (exceed the range of normal floating point numbers).

    .. math::

        out_i = \log(exp(input_i) + \exp(other_i))

    Args:
        input (Tensor): The input tensor.
        other (Tensor): The input tensor.

    Returns:
        Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x1 = mindspore.tensor([1, 2, 3], mindspore.float16)
        >>> x2 = mindspore.tensor(2, mindspore.float16)
        >>> output = mindspore.ops.logaddexp(x1, x2)
        >>> print(output)
        [2.312 2.693 3.312]
    """

    if not isinstance(input, (Tensor, Tensor_)):
        raise TypeError(f"For logaddexp, the input must be a Tensor, but got {type(input)}.")
    if not isinstance(other, (Tensor, Tensor_)):
        raise TypeError(f"For logaddexp, the other must be a Tensor, but got {type(other)}.")
    if not ops.is_floating_point(input) or not ops.is_floating_point(other):
        raise TypeError(f"For logaddexp, the dtype of 'input' and 'other' must be float,"
                        f"but got {input.dtype} and {other.dtype}.")
    m = maximum(input, other)
    abs_val = abs(input - other)
    exp_val = tensor_exp(neg(abs_val))
    y = m + log1p(exp_val)
    return y


def logaddexp2(input, other):
    r"""
    Computes the logarithm of the sum of exponentiations in base of 2 of the inputs.

    .. math::

        out_i = \log_2(2^{input_i} + 2^{other_i})

    Args:
        input (Tensor): The input tensor.
        other (Tensor): The input tensor.

    Returns:
        Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x1 = mindspore.tensor([1, 2, 3], mindspore.float16)
        >>> x2 = mindspore.tensor(2, mindspore.float16)
        >>> output = mindspore.ops.logaddexp2(x1, x2)
        >>> print(output)
        [2.586 3.    3.586]
    """
    _check_is_tensor("input", input, "logaddexp2")
    _check_is_tensor("other", other, "logaddexp2")
    if not ops.is_floating_point(input) or not ops.is_floating_point(other):
        raise TypeError(f"For logaddexp2, the dtype of 'input' and 'other' must be float,"
                        f"but got {input.dtype} and {other.dtype}.")

    m = maximum(input, other)
    abs_val = abs(input - other)
    exp2_val = pows(2., neg(abs_val))
    y = m + log2(1. + exp2_val)
    return y


@_primexpr
def _check_and_canonicalize_axes(axes, ndim):
    """Check whether the types and values of input axes are valid."""
    return validator.check_and_canonicalize_axes(axes, ndim)


def _check_var_std_input(input, ddof, keepdims, axis, cls_name):
    _check_is_tensor("input", input, cls_name)
    _check_attr_dtype("ddof", ddof, [int, bool], cls_name)
    _check_attr_dtype("keepdims", keepdims, [bool], cls_name)
    if axis is None:
        axis = ()
    else:
        axis = _check_and_canonicalize_axes(axis, input.ndim)
    return axis


def vander(x, N=None):
    """
    Generates a Vandermonde matrix. The columns of the output matrix are powers of the input vector.
    The i-th output column is the input vector raised element-wise to the power of :math:`N - i - 1`.

    Args:
        x (Tensor): The 1-D input tensor.
        N (int, optional): Number of columns in the output. Default ``None``,
            `N` will be assigned as :math:`len(x)`.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([1., 2., 3., 5.])
        >>> output = mindspore.ops.vander(input)
        >>> print(output)
        [[  1.   1.   1.   1.]
         [  8.   4.   2.   1.]
         [ 27.   9.   3.   1.]
         [125.  25.   5.   1.]]
        >>> output = mindspore.ops.vander(input, N=3)
        >>> print(output)
        [[ 1.  1.  1.]
         [ 4.  2.  1.]
         [ 9.  3.  1.]
         [25.  5.  1.]]
    """
    if not isinstance(x, Tensor):
        raise TypeError(
            f"For vander, x must be Tensor, but got {type(x)}")
    if x.ndim != 1:
        raise ValueError(
            f"For vander, x must be 1-D, but got dimension = {x.ndim}")
    if N is None:
        N = len(x)
    if not isinstance(N, int):
        raise TypeError(
            f"For vander, N must be an integer but got {type(N)}.")
    if N <= 0:
        raise ValueError(
            f"For vander, N must be greater than 0, but got {N}.")
    exponent = ops.range(Tensor(N - 1), Tensor(-1), Tensor(-1))
    x = F.expand_dims(x, 1)
    exponent = F.expand_dims(exponent, 0)
    return F.tensor_pow(x, exponent)


def var_ext(input, dim=None, *, correction=1, keepdim=False):
    r"""
    Calculates the variance over the dimensions specified by `dim`. `dim` can be a single dimension, list of
    dimensions, or None to reduce over all dimensions.

    The variance (:math:`\delta ^2`) is calculated as:

    .. math::
        \delta ^2 = \frac{1}{\max(0, N - \delta N)}\sum^{N - 1}_{i = 0}(x_i - \bar{x})^2

    where :math:`x` is the sample set of elements, :math:`\bar{x}` is the sample mean, :math:`N` is the number
    of samples and :math:`\delta N` is the `correction`.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): The tensor used to calculate the variance.
        dim (None, int, tuple(int), optional): The dimension or dimensions to reduce. Defaults to ``None``.
            If ``None``, all dimensions are reduced.

    Keyword Args:
        correction (int, optional): The difference between the sample size and sample degrees of freedom. Defaults
            to Bessel’s correction. Defaults to ``1``.
        keepdim (bool, optional): Whether the output tensor has dim retained or not. If ``True`` , keep these
            reduced dimensions and the length is 1. If ``False``, don't keep these dimensions. Defaults to ``False``.

    Returns:
        Tensor, the variance.
        Suppose the shape of `input` is :math:`(x_0, x_1, ..., x_R)`:

        - If `dim` is () and `keepdim` is set to ``False`` , returns a 0-D Tensor, indicating the variance of all
          elements in `input`.
        - If `dim` is int, e.g. ``1`` and `keepdim` is set to ``False`` , then the returned Tensor has shape
          :math:`(x_0, x_2, ..., x_R)`.
        - If `dim` is tuple(int) or list(int), e.g. ``(1, 2)`` and `keepdim` is set to ``False`` , then the returned
          Tensor has shape :math:`(x_0, x_3, ..., x_R)`.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If `input` is not in bfloat16, float16, flaot32.
        TypeError: If `dim` is not one of the followings: None, int, list, tuple.
        TypeError: If `correction` is not an int.
        TypeError: If `keepdim` is not a bool.
        ValueError: If `dim` is out of range :math:`[-input.ndim, input.ndim)`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, ops
        >>> input = Tensor([[8, 2, 1], [5, 9, 3], [4, 6, 7]], mindspore.float32)
        >>> output = ops.var_ext(input, dim=0, correction=1, keepdim=True)
        >>> print(output)
        [[ 4.333333, 12.333333, 9.333333]]
    """
    return var_op(input, dim, correction, keepdim)


def std_ext(input, dim=None, *, correction=1, keepdim=False):
    r"""
    Calculate the standard deviation over specified dimension(s).

    The standard deviation (:math:`\sigma`) is calculated as:

    .. math::
        \sigma =\sqrt{\frac{1}{N-\delta N}\sum_{j-1}^{N-1}\left(s e l f_{i j}-\overline{x_{i}}\right)^{2}}

    where :math:`x` is the sample set of elements, :math:`\bar{x}` is the sample mean, :math:`N` is the number
    of samples and :math:`\delta N` is the `correction`.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): The input tensor.
        dim (None, int, tuple(int), optional): Specify the dimension. If ``None`` , calculate all elements. 
            Default ``None``.

    Keyword Args:
        correction (int, optional): The difference between the sample size and sample degrees of freedom. Default
            to Bessel’s correction. Default ``1``.
        keepdim (bool, optional): Whether the output tensor has dim retained. Default ``False``.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[1, 2, 3], [-1, 1, 4]], mindspore.float32)
        >>> mindspore.mint.std(input, dim=1, correction=1, keepdim=False)
        Tensor(shape=[2], dtype=Float32, value= [ 1.00000000e+00,  2.51661134e+00])
        >>> mindspore.mint.std(input, dim=1, correction=1, keepdim=True)
        Tensor(shape=[2, 1], dtype=Float32, value=
        [[ 1.00000000e+00],
        [ 2.51661134e+00]])
        >>> mindspore.mint.std(input, dim=[0, 1], correction=1, keepdim=False)
        Tensor(shape=[], dtype=Float32, value= 1.75119)
        >>> mindspore.mint.std(input, dim=[0, 1], correction=1, keepdim=True)
        Tensor(shape=[1, 1], dtype=Float32, value=
        [[ 1.75119019e+00]])
        >>> mindspore.mint.std(input, dim=[0, 1], correction=2, keepdim=True)
        Tensor(shape=[1, 1], dtype=Float32, value=
        [[ 1.95789003e+00]])
    """
    return std_op(input, dim, correction, keepdim)


def var(input, axis=None, ddof=0, keepdims=False):
    r"""
    Compute the variance of the tensor along a specified axis.

    Args:
        input (Tensor[Number]): The input tensor.
        axis (Union[int, tuple(int)], optional): Specify the axis for computation. If ``None`` , compute all elements
            in the `input` . Default ``None`` .
        ddof (Union[int, bool], optional): Means Delta Degrees of Freedom. Default ``0`` .

          - If ddof is an integer, the divisor used in calculations is :math:`N - ddof`, where :math:`N` represents
            the number of elements.
          - If ddof is a boolean, ``True`` and ``False`` correspond to when ddof is an integer ``1`` and ``0``
            respectively.
          - If ddof is 0, 1, True or False, the supported device is only Ascend and CPU. In other cases,
            the supported device is Ascend, GPU and CPU.
        keepdims (bool, optional): Whether the output tensor has dim retained. Default ``False`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[1., 3, 4, 2],
        ...                           [4, 2, 5, 3],
        ...                           [5, 4, 2, 3]])
        >>> # case 1: By default, compute the variance of all elements.
        >>> output = mindspore.ops.var(input)
        >>> print(output)
        1.4722221
        >>>
        >>> # case 2: Compute the variance along axis 0.
        >>> output = mindspore.ops.var(input, axis=0)
        >>> print(output)
        [2.8888888 0.6666667 1.5555557 0.2222222]
        >>>
        >>> # case 3: If keepdims=True, the output shape will be same of that of the input.
        >>> output = mindspore.ops.var(input, axis=0, keepdims=True)
        >>> print(output)
        [[2.8888888 0.6666667 1.5555557 0.2222222]]
        >>>
        >>> # case 4: If ddof=1:
        >>> output = mindspore.ops.var(input, axis=0, keepdims=True, ddof=1)
        >>> print(output)
        [[4.3333335 1.        2.3333335 0.3333333]]
        >>>
        >>> # case 5: If ddof=True, same as ddof=1:
        >>> output = mindspore.ops.var(input, axis=0, keepdims=True, ddof=True)
        >>> print(output)
        [[4.3333335 1.        2.3333335 0.3333333]]
        >>>
        >>> # case 6: If ddof=False, same as ddof=0:
        >>> output = mindspore.ops.var(input, axis=0, keepdims=True, ddof=False)
        >>> print(output)
        [[2.8888888 0.6666667 1.5555557 0.2222222]]
    """
    axis = _check_var_std_input(input, ddof, keepdims, axis, "var")
    output = var_mean(input, axis, ddof, keepdims)
    return output[0]


def var_mean(input, axis=None, ddof=0, keepdims=False):
    r"""
    Compute the variance and the mean of the tensor along a specified axis.

    Args:
        input (Tensor[Number]): The input tensor.
        axis (Union[int, tuple(int)], optional): Specify the axis for computation. If ``None`` , compute all
            elements in the `input` . Default ``None`` .
        ddof (Union[int, bool], optional): Means Delta Degrees of Freedom. Default ``0`` .

          - If ddof is an integer, the divisor used in calculations is :math:`N - ddof`, where :math:`N` represents
            the number of elements.
          - If ddof is a boolean, ``True`` and ``False`` correspond to when ddof is an integer ``1`` and ``0``
            respectively.
          - If ddof is 0, 1, True or False, the supported device is only Ascend and CPU. In other cases,
            the supported device is Ascend, GPU and CPU.
        keepdims (bool, optional): Whether the output tensor has dim retained. Default ``False`` .

    Returns:
        Tuple(var, mean) of 2 tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[1., 3, 4, 2],
        ...                           [4, 2, 5, 3],
        ...                           [5, 4, 2, 3]])
        >>> # case 1: By default, compute the variance and mean of all elements.
        >>> mindspore.ops.var_mean(input)
        (Tensor(shape=[], dtype=Float32, value= 1.47222),
         Tensor(shape=[], dtype=Float32, value= 3.16667))
        >>>
        >>> # case 2: Compute the variance and mean along axis 0.
        >>> mindspore.ops.var_mean(input, axis=0)
        (Tensor(shape=[4], dtype=Float32, value= [ 2.88888884e+00,  6.66666687e-01,  1.55555570e+00,  2.22222194e-01]),
         Tensor(shape=[4], dtype=Float32, value= [ 3.33333325e+00,  3.00000000e+00,  3.66666675e+00,  2.66666675e+00]))
        >>>
        >>> # case 3: If keepdims=True, the output shape will be same of that of the input.
        >>> mindspore.ops.var_mean(input, axis=0, keepdims=True)
        (Tensor(shape=[1, 4], dtype=Float32, value=
         [[ 2.88888884e+00,  6.66666687e-01,  1.55555570e+00,  2.22222194e-01]]),
         Tensor(shape=[1, 4], dtype=Float32, value=
         [[ 3.33333325e+00,  3.00000000e+00,  3.66666675e+00,  2.66666675e+00]]))
        >>>
        >>> # case 4: If ddof=1:
        >>> mindspore.ops.var_mean(input, axis=0, keepdims=True, ddof=1)
        (Tensor(shape=[1, 4], dtype=Float32, value=
         [[ 4.33333349e+00,  1.00000000e+00,  2.33333349e+00,  3.33333313e-01]]),
         Tensor(shape=[1, 4], dtype=Float32, value=
         [[ 3.33333325e+00,  3.00000000e+00,  3.66666675e+00,  2.66666675e+00]]))
        >>>
        >>> # case 5: If ddof=True, same as ddof=1:
        >>> mindspore.ops.var_mean(input, axis=0, keepdims=True, ddof=True)
        (Tensor(shape=[1, 4], dtype=Float32, value=
         [[ 4.33333349e+00,  1.00000000e+00,  2.33333349e+00,  3.33333313e-01]]),
         Tensor(shape=[1, 4], dtype=Float32, value=
         [[ 3.33333325e+00,  3.00000000e+00,  3.66666675e+00,  2.66666675e+00]]))
        >>>
        >>> # case 6: If ddof=False, same as ddof=0:
        >>> mindspore.ops.var_mean(input, axis=0, keepdims=True, ddof=False)
        (Tensor(shape=[1, 4], dtype=Float32, value=
         [[ 2.88888884e+00,  6.66666687e-01,  1.55555570e+00,  2.22222194e-01]]),
         Tensor(shape=[1, 4], dtype=Float32, value=
         [[ 3.33333325e+00,  3.00000000e+00,  3.66666675e+00,  2.66666675e+00]]))
    """
    axis = _check_var_std_input(input, ddof, keepdims, axis, "var_mean")
    if ddof in (0, 1):
        output = _get_cache_prim(P.ReduceStd)(axis=axis, unbiased=bool(ddof), keep_dims=keepdims)(input)
        return tensor_pow(output[0], 2), output[1]
    x_mean = mean(input, axis, True)
    x_sub = tensor_sub(input, x_mean)
    x_pow = tensor_pow(x_sub, 2)
    x_sum = sum(x_pow, axis, keepdims)
    res_mean = mean(input, axis, keepdims)
    nums = 1
    if axis == ():
        nums = input.size
    else:
        for ax in axis:
            nums *= input.shape[ax]
    return true_divide(x_sum, nums - ddof), res_mean


def std(input, axis=None, ddof=0, keepdims=False):
    r"""
    Compute the standard deviation of the tensor along a specified axis.

    Args:
        input (Tensor[Number]): The input tensor.
        axis (Union[int, tuple(int)], optional): Specify the axis for computation. If ``None`` , compute all elements
            in the `input` . Default ``None`` .
        ddof (Union[int, bool], optional): Means Delta Degrees of Freedom. Default ``0`` .

          - If ddof is an integer, the divisor used in calculations is :math:`N - ddof`, where :math:`N` represents
            the number of elements.
          - If ddof is a boolean, ``True`` and ``False`` correspond to when ddof is an integer ``1`` and ``0``
            respectively.
          - If ddof is 0, 1, True or False, the supported device is only Ascend and CPU. In other cases,
            the supported device is Ascend, GPU and CPU.
        keepdims (bool, optional): Whether the output tensor has dim retained. Default ``False`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[1., 3, 4, 2],
        ...                           [4, 2, 5, 3],
        ...                           [5, 4, 2, 3]])
        >>> # case 1: By default, compute the standard deviation of all elements.
        >>> output = mindspore.ops.std(input)
        >>> print(output)
        1.2133516
        >>>
        >>> # case 2: Compute the standard deviation along axis 0.
        >>> output = mindspore.ops.std(input, axis=0)
        >>> print(output)
        [1.6996732 0.8164966 1.2472192 0.4714045]
        >>>
        >>> # case 3: If keepdims=True, the output shape will be same of that of the input.
        >>> output = mindspore.ops.std(input, axis=0, keepdims=True)
        >>> print(output)
        [[1.6996732 0.8164966 1.2472192 0.4714045]]
        >>>
        >>> # case 4: If ddof=1:
        >>> output = mindspore.ops.std(input, axis=0, keepdims=True, ddof=1)
        >>> print(output)
        [[2.081666   1.         1.5275253  0.57735026]]
        >>>
        >>> # case 5: If ddof=True, same as ddof=1:
        >>> output = mindspore.ops.std(input, axis=0, keepdims=True, ddof=True)
        >>> print(output)
        [[2.081666   1.         1.5275253  0.57735026]]
        >>>
        >>> # case 6: If ddof=False, same as ddof=0:
        >>> output = mindspore.ops.std(input, axis=0, keepdims=True, ddof=False)
        >>> print(output)
        [[1.6996732 0.8164966 1.2472192 0.4714045]]
    """
    axis = _check_var_std_input(input, ddof, keepdims, axis, "std")
    output = std_mean(input, axis, ddof, keepdims)
    return output[0]


def std_mean(input, axis=None, ddof=0, keepdims=False):
    r"""
    Compute the standard deviation and the mean of the tensor along a specified axis.

    Args:
        input (Tensor[Number]): The input tensor.
        axis (Union[int, tuple(int)], optional): Specify the axis for computation. If ``None`` , compute all
            elements in the `input` . Default ``None`` .
        ddof (Union[int, bool], optional): Means Delta Degrees of Freedom. Default ``0`` .

          - If ddof is an integer, the divisor used in calculations is :math:`N - ddof`, where :math:`N` represents
            the number of elements.
          - If ddof is a boolean, ``True`` and ``False`` correspond to when ddof is an integer ``1`` and ``0``
            respectively.
          - If ddof is 0, 1, True or False, the supported device is only Ascend and CPU. In other cases,
            the supported device is Ascend, GPU and CPU.
        keepdims (bool, optional): Whether the output tensor has dim retained. Default ``False`` .

    Returns:
        Tuple(std, mean) of 2 tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[1., 3, 4, 2],
        ...                           [4, 2, 5, 3],
        ...                           [5, 4, 2, 3]])
        >>> # case 1: By default, compute the standard deviation and mean of all elements.
        >>> mindspore.ops.std_mean(input)
        (Tensor(shape=[], dtype=Float32, value= 1.21335),
         Tensor(shape=[], dtype=Float32, value= 3.16667))
        >>>
        >>> # case 2: Compute the standard deviation and mean along axis 0.
        >>> mindspore.ops.std_mean(input, axis=0)
        (Tensor(shape=[4], dtype=Float32, value= [ 1.69967318e+00,  8.16496611e-01,  1.24721920e+00,  4.71404493e-01]),
         Tensor(shape=[4], dtype=Float32, value= [ 3.33333325e+00,  3.00000000e+00,  3.66666675e+00,  2.66666675e+00]))
        >>>
        >>> # case 3: If keepdims=True, the output shape will be same of that of the input.
        >>> mindspore.ops.std_mean(input, axis=0, keepdims=True)
        (Tensor(shape=[1, 4], dtype=Float32, value=
         [[ 1.69967318e+00,  8.16496611e-01,  1.24721920e+00,  4.71404493e-01]]),
         Tensor(shape=[1, 4], dtype=Float32, value=
         [[ 3.33333325e+00,  3.00000000e+00,  3.66666675e+00,  2.66666675e+00]]))
        >>>
        >>> # case 4: If ddof=1:
        >>> mindspore.ops.std_mean(input, axis=0, keepdims=True, ddof=1)
        (Tensor(shape=[1, 4], dtype=Float32, value=
         [[ 2.08166599e+00,  1.00000000e+00,  1.52752531e+00,  5.77350259e-01]]),
         Tensor(shape=[1, 4], dtype=Float32, value=
         [[ 3.33333325e+00,  3.00000000e+00,  3.66666675e+00,  2.66666675e+00]]))
        >>>
        >>> # case 5: If ddof=True, same as ddof=1:
        >>> mindspore.ops.std_mean(input, axis=0, keepdims=True, ddof=True)
        (Tensor(shape=[1, 4], dtype=Float32, value=
         [[ 2.08166599e+00,  1.00000000e+00,  1.52752531e+00,  5.77350259e-01]]),
         Tensor(shape=[1, 4], dtype=Float32, value=
         [[ 3.33333325e+00,  3.00000000e+00,  3.66666675e+00,  2.66666675e+00]]))
        >>>
        >>> # case 6: If ddof=False, same as ddof=0:
        >>> mindspore.ops.std_mean(input, axis=0, keepdims=True, ddof=False)
        (Tensor(shape=[1, 4], dtype=Float32, value=
         [[ 1.69967318e+00,  8.16496611e-01,  1.24721920e+00,  4.71404493e-01]]),
         Tensor(shape=[1, 4], dtype=Float32, value=
         [[ 3.33333325e+00,  3.00000000e+00,  3.66666675e+00,  2.66666675e+00]]))
    """
    axis = _check_var_std_input(input, ddof, keepdims, axis, "std_mean")
    if ddof in (0, 1):
        return _get_cache_prim(P.ReduceStd)(axis=axis, unbiased=bool(ddof), keep_dims=keepdims)(input)
    output = var_mean(input, axis, ddof, keepdims)
    return tensor_pow(output[0], 0.5), output[1]


def std_mean_ext(input, dim=None, *, correction=1, keepdim=False):
    r"""
    By default, return the standard deviation and mean of each dimension in Tensor.
    If dim is a dimension list, calculate the standard deviation and mean of the corresponding dimension.

    The standard deviation (:math:`\sigma`) is calculated as:

    .. math::

        \sigma = \sqrt{\frac{1}{N - \delta N} \sum_{j=0}^{N-1} \left(self_{ij} - \overline{x_{i}}\right)^{2}}

    where is :math:`x` the sample set of elements, :math:`\bar{x}` is the sample mean,
    :math:`N` is the number of samples and :math:`\delta N` is the `correction` .

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): The input tensor. Supported dtypes: float16, float32.
        dim (Union[int, tuple(int), list(int)], optional):
            Specify the dimensions for calculating standard deviation and mean. Default value: ``None``.

    Keyword Args:
        correction (int, optional): Difference between the sample size and sample degrees of freedom.
            Defaults to Bessel's correction. Default: ``1``.
        keepdim (bool, optional): Whether to preserve the dimensions of the output Tensor.
            If True, retain the reduced dimension with a size of 1. Otherwise, remove the dimensions.
            Default value: ``False``.

    Returns:
        A tuple of standard deviation and mean.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If `dim` is not one of the following data types: int, tuple, list, or Tensor.
        TypeError: If `keepdim` is not a bool.
        ValueError: If `dim` is out of range.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> input = ms.Tensor([[1, 2, 3, 4], [-1, 1, 4, -10]], ms.float32)
        >>> output_std, output_mean = ms.ops.function.math_func.std_mean_ext(input, 1, correction=2, keepdim=True)
        >>> print(output_std)
        [[1.5811388]
         [7.3824115]]
        >>> print(output_mean)
        [[ 2.5]
         [-1.5]]
    """
    return std_mean_op(input, dim, correction, keepdim)


def var_mean_ext(input, dim=None, *, correction=1, keepdim=False):
    r"""
    By default, return the variance and mean of each dimension in Tensor.
    If dim is a dimension list, calculate the variance and mean of the corresponding dimension.

    The variance (:math:`\sigma ^2`) is calculated as:

    .. math::

        \sigma ^2 = \frac{1}{N - \delta N} \sum_{j=0}^{N-1} \left(self_{ij} - \overline{x_{i}}\right)^{2}

    where is :math:`x` the sample set of elements, :math:`\bar{x}` is the sample mean,
    :math:`N` is the number of samples and :math:`\delta N` is the `correction` .

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): The input tensor. Supported dtypes: float16, float32.
        dim (Union[int, tuple(int), list(int)], optional):
            Specify the dimensions for calculating variance and mean. Default value: ``None``.

    Keyword Args:
        correction (int, optional): Difference between the sample size and sample degrees of freedom.
            Defaults to Bessel's correction. Default: ``1``.
        keepdim (bool, optional): Whether to preserve the dimensions of the output Tensor.
            If True, retain the reduced dimension with a size of 1. Otherwise, remove the dimensions.
            Default value: ``False``.

    Returns:
        A tuple of variance and mean.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If `dim` is not one of the following data types: int, tuple, list, or Tensor.
        TypeError: If `keepdim` is not a bool.
        ValueError: If `dim` is out of range.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> input = ms.Tensor([[1, 2, 3, 4], [-1, 1, 4, -10]], ms.float32)
        >>> output_var, output_mean = ms.ops.function.math_func.var_mean_ext(input, 1, correction=2, keepdim=True)
        >>> print(output_var)
        [[ 2.5]
         [54.5]]
        >>> print(output_mean)
        [[ 2.5]
         [-1.5]]
    """
    return var_mean_op(input, dim, correction, keepdim)


def reciprocal(input):
    r"""
    Returns reciprocal of a tensor element-wise.

    .. math::

        out_{i} =  \frac{1}{x_{i}}

    Args:
        input (Tensor): The input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([1.0, 2.0, 4.0], mindspore.float32)
        >>> output = mindspore.ops.reciprocal(input)
        >>> print(output)
        [1.   0.5  0.25]
    """
    return reciprocal_(input)


def outer(input, vec2):
    """
    Compute outer product of two tensors.

    If `input` ’s length is :math:`n` and `vec2` ’s length is :math:`m` , then output must be a matrix of shape
    :math:`(n, m)` .

    Note:
        This function does not broadcast.

    Args:
        input (Tensor): The 1-D input tensor.
        vec2 (Tensor): The 1-D input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([1, 2, 3])
        >>> vec2 = mindspore.tensor([4, 5, 6])
        >>> mindspore.ops.outer(input, vec2)
        Tensor(shape=[3, 3], dtype=Int64, value=
        [[ 4,  5,  6],
         [ 8, 10, 12],
         [12, 15, 18]])
    """

    if not isinstance(input, (Tensor, Tensor_)):
        raise TypeError("the input input must be Tensor!")
    if not isinstance(vec2, (Tensor, Tensor_)):
        raise TypeError("the input vec2 must be Tensor!")
    input = input.reshape(-1, 1)
    y = tensor_mul(input, vec2)
    return y


def mv(mat, vec):
    """
    Multiplies matrix `mat` and vector `vec`.

    If `mat` is a :math:`(N, M)` tensor, `vec` is a 1-D :math:`M` tensor,
    out will be a 1-D :math:`N` tensor.

    Args:
        mat (Tensor): Input matrix.
        vec (Tensor): Input vector.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> output = mindspore.ops.mv(mindspore.tensor([[3., 4.], [1., 6.], [1., 3.]]), mindspore.tensor([1., 2.]))
        >>> print(output)
        [11. 13. 7.]
    """
    if not isinstance(mat, (Tensor, Tensor_)):
        raise TypeError("The input mat must be Tensor.")
    if not isinstance(vec, (Tensor, Tensor_)):
        raise TypeError("The input vec must be Tensor.")

    length_vec = get_x_shape(vec.shape)
    vec = reshape_(vec, (length_vec[0], 1))

    out = matmul_(mat, vec)
    out = out.T
    out = out[0]
    return out


def addbmm(input, batch1, batch2, *, beta=1, alpha=1):
    r"""
    Apply batch matrix multiplication to `batch1` and `batch2`, with a reduced add step and add `input` to the result.

    .. note::
        - `batch1` and `batch2` must be 3-D tensors each containing the same number of matrices.
        - When batch1 is a :math:`(C, W, T)` tensor and batch2 is a :math:`(C, T, H)` tensor, input must be
          broadcastable with :math:`(W, H)` tensor, and out will be a  :math:`(W, H)` tensor.
        - If `beta` is 0, then `input` will be ignored.

    .. math::
        output = \beta input + \alpha (\sum_{i=0}^{b-1} {batch1_i @ batch2_i})

    Args:
        input (Tensor): The input tensor.
        batch1 (Tensor): The first batch of tensor to be multiplied.
        batch2 (Tensor): The second batch of tensor to be multiplied.

    Keyword Args:
        beta (Union[int, float], optional): Scale factor for `input`. Default ``1`` .
        alpha (Union[int, float], optional): Scale factor for ( `batch1` @ `batch2` ). Default ``1`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> m = mindspore.ops.ones((3, 3))
        >>> arr1 = mindspore.tensor([[8., 7., 6.], [5., 4., 3.], [2., 1., 0.]])
        >>> arr2 = mindspore.tensor([[5., 4., 3.], [2., 1., 0.], [8., 7., 6.]])
        >>> output = mindspore.ops.addbmm(m, arr1, arr2)
        >>> print(output)
        [[172. 136. 100.]
         [172. 136. 100.]
         [172. 136. 100.]]
    """
    if not isinstance(alpha, (int, float)):
        raise TypeError(f"For 'addbmm', parameter 'alpha' must be an int or float, but got {type(alpha)}.")
    if not isinstance(beta, (int, float)):
        raise TypeError(f"For 'addbmm', parameter 'beta' must be an int or float, but got {type(beta)}.")
    bmm_res = batch_matmul_(batch1, batch2)
    return beta * input + alpha * (bmm_res.sum(axis=0))


def addbmm_ext(input, batch1, batch2, *, beta=1, alpha=1):
    r"""
    Applies batch matrix multiplication to `batch1` and `batch2`, with a reduced add step and add `input` to the result.

    The optional value `alpha` is the matrix-matrix product between `batch1` and `batch2`, and `beta` is the scale
    factor for the added tensor `input`. If `beta` is 0, then `input` will be ignored.

    .. math::
        output = \beta input + \alpha (\sum_{i=0}^{b-1} {batch1_i @ batch2_i})

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): Tensor to be added.
        batch1 (Tensor): The first batch of tensor to be multiplied.
        batch2 (Tensor): The second batch of tensor to be multiplied.

    Keyword Args:
        beta (Union[int, float], optional): Multiplier for `input`. Default: ``1`` .
        alpha (Union[int, float], optional): Multiplier for `batch1` @ `batch2`. Default: ``1`` .

    Returns:
        Tensor, has the same dtype as `input`.

    Raises:
        TypeError: If `alpha` or `beta` is not an int or float.
        ValueError: If `batch1`, `batch2` cannot apply batch matrix multiplication.
        ValueError: If `batch1` and `batch2` are not 3-D tensors.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> m = np.ones((3, 3)).astype(np.float32)
        >>> arr1 = np.arange(24).astype(np.float32).reshape((2, 3, 4))
        >>> arr2 = np.arange(24).astype(np.float32).reshape((2, 4, 3))
        >>> a = Tensor(arr1)
        >>> b = Tensor(arr2)
        >>> c = Tensor(m)
        >>> output = ops.addbmm_ext(c, a, b)
        >>> print(output)
        [[ 949. 1009. 1069.]
         [1285. 1377. 1469.]
         [1621. 1745. 1869.]]
    """
    return addbmm_op(input, batch1, batch2, beta, alpha)


def addmm_ext(input, mat1, mat2, *, beta=1, alpha=1):
    r"""
    Performs a matrix multiplication of the 2-D matrices mat1 and mat2. The matrix input is added to the final result.
    The formula is defined as follows:

    .. math::
        output = \beta input + \alpha (mat1 @ mat2)

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): matrix to be added, the shape must be broadcastable with :math:`mat1 @ mat2`.
        mat1 (Tensor): the first matrix to be matrix multiplied, must be 2-D Tensor, with the same shape of the input.
        mat2 (Tensor): the second matrix to be matrix multiplied, must be 2-D Tensor, with the same shape of the input.

    Keyword Args:
        beta (Union[float, int], optional): multiplier for input. Default: ``1`` .
        alpha (Union[float, int], optional): multiplier for :math:`mat1 @ mat2`. Default: ``1`` .

    Returns:
        Tensor, with the same dtype as `input` and the same shape as :math:`mat1 @ mat2`.

    Raises:
        TypeError: If the type of `input`, `mat1` or `mat2` is not Tensor.
        TypeError: If the types of `input`, `mat1`, `mat2` are different.
        ValueError: If `mat1` and `mat2` are not 2-D tensors.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input = Tensor(np.ones([3, 3]).astype(np.float32))
        >>> mat1 = Tensor(np.ones([3, 4]).astype(np.float32))
        >>> mat2 = Tensor(np.ones([4, 3]).astype(np.float32))
        >>> output =  ops.function.math_func.addmm_ext(input, mat1, mat2)
        >>> print(output)
        [[5. 5. 5.]
         [5. 5. 5.]
         [5. 5. 5.]]
    """
    return addmm_op(input, mat1, mat2, beta, alpha)


def addmm(input, mat1, mat2, *, beta=1, alpha=1):
    r"""
    Multiply matrix `mat1` and matrix `mat2`. The matrix `input` is added to the final result.

    .. note::
        - If `beta` is 0, then `input` will be ignored.

    .. math::
        output = \beta input + \alpha (mat1 @ mat2)

    Args:
        input (Tensor): The input tensor.
        mat1 (Tensor): The first tensor to be multiplied.
        mat2 (Tensor): The second tensor to be multiplied.

    Keyword Args:
        beta (Union[int, float], optional): Scale factor for `input`. Default ``1`` .
        alpha (Union[int, float], optional): Scale factor for ( `mat1` @ `mat2` ) . Default ``1`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> m = mindspore.ops.ones((3, 3))
        >>> arr1 = mindspore.tensor([[8., 7., 6.], [5., 4., 3.], [2., 1., 0.]])
        >>> arr2 = mindspore.tensor([[5., 4., 3.], [2., 1., 0.], [8., 7., 6.]])
        >>> output = mindspore.ops.addmm(m, arr1, arr2)
        >>> print(output)
        [[103.  82.  61.]
         [ 58.  46.  34.]
         [ 13.  10.   7.]]
    """
    if not isinstance(alpha, (int, float)):
        raise TypeError(f"For 'addmm', parameter 'alpha' must be an int or float, but got {type(alpha)}.")
    if not isinstance(beta, (int, float)):
        raise TypeError(f"For 'addmm', parameter 'beta' must be an int or float, but got {type(beta)}.")
    return beta * input + alpha * (matmul_(mat1, mat2))


def addmv(input, mat, vec, *, beta=1, alpha=1):
    """
    Multiply the matrix `mat` and vector `vec` , and then add the result to the `input` .

    .. note::
        - If mat is a :math:`(N, M)` tensor, vec is a 1-D tensor of size :math:`M`, then `input` must
          be broadcastable with a 1-D tensor of size :math:`N`, `out` will be a 1-D tensor of size :math:`N`.
        - If `beta` is 0, `input` will be ignored.

    .. math::
        output = β input + α (mat @ vec)

    Args:
        input (Tensor): The input tensor.
        mat (Tensor): The matrix tensor to be multiplied.
        vec (Tensor): The vector tensor to be multiplied.

    Keyword Args:
        beta (scalar[int, float, bool], optional): Scale factor for `input` . Default ``1`` .
        alpha (scalar[int, float, bool], optional): Scale factor for ( `mat` @ `vec` ). Default ``1`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([2., 3.])
        >>> mat = mindspore.tensor([[2., 5., 3.], [4., 2., 2.]])
        >>> vec = mindspore.tensor([3., 2., 4.])
        >>> output = mindspore.ops.addmv(input, mat, vec)
        >>> print(output)
        [30. 27.]
    """

    input_dtype = dtype_(input)
    if not (isinstance(input, Tensor) and isinstance(mat, Tensor) and isinstance(vec, Tensor)):
        raise TypeError("For Addmv, inputs must be all tensors.")
    if dtype_(mat) != dtype_(vec):
        raise TypeError("For Addmv, the mat and vec should be the same dtype.")
    _check_input_dtype("input", input_dtype,
                       [mstype.float16, mstype.float32, mstype.float64,
                        mstype.int16, mstype.int32, mstype.int64], "Addmv")
    _check_attr_dtype("alpha", alpha, [int, float, bool], "Addmv")
    _check_attr_dtype("beta", beta, [int, float, bool], "Addmv")
    if input_dtype in (mstype.int16, mstype.int32, mstype.int64):
        alpha = ops.scalar_cast(alpha, mstype.int64)
        beta = ops.scalar_cast(beta, mstype.int64)
    out = beta * input + alpha * mv(mat, vec)
    return out


def addmv_ext(input, mat, vec, *, beta=1, alpha=1):
    r"""
    Performs a matrix-vector product of `mat` and `vec`, and add the input vector `input` to the final result.

    If `mat` is a tensor of size :math:`(N, M)` , `vec` is a 1-D tensor of size :math:`M` , then `input` must be
    broadcastable with a 1-D tensor of size :math:`N` . In this case, `output` is a 1-D Tensor of size :math:`N` .

    .. math::
        output = \beta input + \alpha (mat @ vec)

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): Vector to be added.
        mat (Tensor): The first tensor needs to be multiplied.
        vec (Tensor): The second tensor needs to be multiplied.

    Keyword Args:
        beta (Union[float, int], optional): Coefficient of `input`. Default: ``1``.
        alpha (Union[float, int], optional): Coefficient of :math:`mat @ vec` . Default: ``1``.

    Returns:
        Tensor, with a shape of :math:`(N,)` , and its dtype is the same as `input`.

    Raises:
        TypeError: If dtype of `input`, `mat` or `vec` is not tensor.
        TypeError: If dtypes of `mat` and `vec` are not the same.
        ValueError: If `mat` is not a 2-D tensor.
        ValueError: If `vec` is not a 1-D tensor.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input = Tensor(np.array([2., 3.]).astype(np.float32))
        >>> mat = Tensor(np.array([[2., 5., 3.], [4., 2., 2.]]).astype(np.float32))
        >>> vec = Tensor(np.array([3., 2., 4.]).astype(np.float32))
        >>> output = ops.function.math_func.addmv_ext(input, mat, vec)
        >>> print(output)
        [30. 27.]
    """
    return addmv_op(input, mat, vec, beta, alpha)


def adjoint(x):
    r"""
    Calculate the conjugation of tensor element-wise, and transpose the last two dimensions.

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor(([[0. + 0.j, 1. + 1.j], [2. + 2.j, 3. + 3.j]]), mindspore.complex128)
        >>> mindspore.ops.adjoint(x)
        Tensor(shape=[2, 2], dtype=Complex128, value=
        [[0-0j, 2-2j],
         [1-1j, 3-3j]])
    """
    _dtype = x.dtype
    _t = x.swapaxes(-1, -2)
    if _dtype in mstype.complex_type:
        return _t.conj()
    return _t


def addr(x, vec1, vec2, *, beta=1, alpha=1):
    """
    Compute the outer product of two vector `vec1` and `vec2`, and add the resulting matrix to `x`.

    .. note::
        - Given `vec1` and `vec2` of sizes :math:`N` and :math:`M`, `x` must be able to broadcast
          to a matrix of shape :math:`(N, M)`, and out will be a matrix of shape :math:`(N, M)` .
        - Setting `beta` to 0 will exclude `x` from the computation.

    .. math::
        output = β x + α (vec1 ⊗ vec2)

    Args:
        x (Tensor): Vector to be added.
        vec1 (Tensor): The first tensor to be multiplied.
        vec2 (Tensor): The second tensor to be multiplied.

    Keyword Args:
        beta (scalar[int, float, bool], optional): Scale factor for `x`  Default ``1`` .
        alpha (scalar[int, float, bool], optional): Scale factor for ( `vec1` ⊗ `vec2` ). Default ``1`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([[2., 2.], [3., 2.], [3., 4.]])
        >>> vec1 = mindspore.tensor([2., 3., 2.])
        >>> vec2 = mindspore.tensor([3., 4.])
        >>> output = mindspore.ops.addr(x, vec1, vec2)
        >>> print(output)
        [[ 8. 10.]
         [12. 14.]
         [ 9. 12.]]
    """

    input_dtype = dtype_(x)
    if not (isinstance(x, Tensor) and isinstance(vec1, Tensor) and isinstance(vec2, Tensor)):
        raise TypeError("For Addr, inputs must be all tensors.")
    if dtype_(vec1) != dtype_(vec2):
        raise TypeError("For Addr, the vec1 and vec2 should be the same dtype.")
    _check_input_dtype("x", input_dtype,
                       [mstype.float16, mstype.float32, mstype.float64,
                        mstype.int16, mstype.int32, mstype.int64], "Addr")
    _check_attr_dtype("alpha", alpha, [int, float, bool], "Addr")
    _check_attr_dtype("beta", beta, [int, float, bool], "Addr")
    if input_dtype in (mstype.int16, mstype.int32, mstype.int64):
        alpha = ops.scalar_cast(alpha, mstype.int64)
        beta = ops.scalar_cast(beta, mstype.int64)

    length_vec1 = get_x_shape(vec1.shape)
    vec1 = reshape_(vec1, (length_vec1[0], 1))
    length_vec2 = get_x_shape(vec2.shape)
    vec2 = reshape_(vec2, (1, length_vec2[0]))

    out = beta * x + alpha * matmul_(vec1, vec2)
    return out


def lcm(input, other):
    """
    Computes least common multiplier of input tensors element-wise.

    Support broadcast, support implicit type conversion and type promotion.

    Args:
        input (Tensor): The first input tensor.
        other (Tensor): The second input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> mindspore.ops.lcm(mindspore.tensor([7, 8, 9]), mindspore.tensor([14, 6, 12]))
        Tensor(shape=[3], dtype=Int64, value= [14, 24, 36])
    """
    return lcm_(input, other)


def lerp(input, end, weight):
    """
    Does a linear interpolation of two tensors input and end based on a float or tensor weight.

    .. math::
        output_{i} = input_{i} + weight_{i} * (end_{i} - input_{i})

    .. note::
        - The shapes of `input` and `end` must be broadcastable.
        - If weight is a tensor, then the shapes of `weight` , `start` , and `end` must be broadcastable.
        - On the Ascend platform, if `weight` dtype is float, the type of `input` and `end` need to be float32.

    Args:
        input (Tensor): The tensor with the starting points.
        end (Tensor): The tensor with the ending points.
        weight (Union[float, Tensor]): The weight for the interpolation formula.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> start = mindspore.tensor([1., 2., 3., 4.], mindspore.float32)
        >>> end = mindspore.tensor([10., 10., 10., 10.], mindspore.float32)
        >>> output = mindspore.ops.lerp(start, end, 0.5)
        >>> print(output)
        [5.5 6.  6.5 7. ]
        >>> output = mindspore.ops.lerp(start, end, mindspore.tensor([0.5, 0.5, 0.5, 0.5], mindspore.float32))
        >>> print(output)
        [5.5 6.  6.5 7. ]
    """
    return lerp_(input, end, weight)


def bernoulli(input, p=0.5, seed=None):
    r"""
    Generates Bernoulli random values (0 or 1).

    .. math::
        out_{i} \sim Bernoulli(p_{i})

    Args:
        input (Tensor): The input Tensor.
        p (Union[Tensor, float], optional): The probability of setting 1 for the
            corresponding position of the returned tensor. The value of `p` must be in the range `[0, 1]`.
            Default ``0.5`` .
        seed (Union[int, None], optional): The random seed. Default ``None`` means using the current timestamp.

    Returns:
        Tensor

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([1, 2, 3])
        >>> mindspore.ops.bernoulli(input, p=1.0)
        Tensor(shape=[3], dtype=Int64, value= [1, 1, 1])
        >>> p = mindspore.tensor([0.0, 1.0, 1.0])
        >>> mindspore.ops.bernoulli(input, p)
        Tensor(shape=[3], dtype=Int64, value= [0, 1, 1])
    """
    if seed is None:
        seed = -1
    validator.check_is_int(seed, 'seed', 'bernoulli')
    bernoulli_ = _get_cache_prim(Bernoulli)(seed)
    if not isinstance(p, Tensor):
        p = Tensor([p])
    return bernoulli_(input, p)


def bernoulli_ext(input, *, generator=None):
    r"""
    Sample from the Bernoulli distribution and randomly set the i^{th} element of the `output` to (0 or 1) according to
    the i^{th} probability value given in the `input`.

    .. math::
        output_{i} \sim Bernoulli(p=input_{i})

    Args:
        input (Tensor): The input tensor of Bernoulli distribution, where the i^{th} element 'input_{i}' represents the
            probability that the corresponding output element 'output_{i}' is set to '1', therefore each element in
            'input' have to be in the range '[0,1]'. Supported dtype: float16, float32, float64, bfloat16
            (only supported by Atlas A2 training series products).

    Keyword Args:
        generator (:class:`mindspore.Generator`, optional): a pseudorandom number generator.
            Default: ``None``, uses the default pseudorandom number generator.

    Returns:
        output (Tensor), The output tensor, with the same shape and dtype as `input`.

    Raises:
        TypeError: If dtype of `input` is not one of: float16, float32, float64, bfloat16.
        ValueError: If any element of the `input` is not in the range [0, 1].

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore import ops
        >>> input_x = Tensor(np.ones((3, 3)), mindspore.float32)
        >>> output = ops.bernoulli_ext(input_x)
        >>> print(output)
        [[ 1. 1. 1.]
         [ 1. 1. 1.]
         [ 1. 1. 1.]]
        >>> input_x = Tensor(np.zeros((3, 3)), mindspore.float32)
        >>> output = ops.bernoulli_ext(input_x)
        >>> print(output)
        [[ 0. 0. 0.]
         [ 0. 0. 0.]
         [ 0. 0. 0.]]
    """
    if generator is None:
        generator = default_generator
    seed, offset = generator._step(generator_step_)  # pylint: disable=protected-access
    return bernoulli_ext_(input, seed, offset)


def bernoulli_(input, p=0.5, *, generator=None):
    r"""
    bernoulli_(input, p=0.5, *, generator=None) -> Tensor

    In-place version of :func:`mindspore.ops.bernoulli_ext`.
    """
    if generator is None:
        generator = default_generator
    seed, offset = generator._step(generator_step_)  # pylint: disable=protected-access
    return ops.functional_overload.bernoulli_(input, p, seed, offset)


def bessel_i1(x):
    r"""
    Computes the first order modified Bessel function of the first kind for each element input.

    .. math::
        \begin{array}{ll} \\
            I_{1}(x)=\mathrm{i}^{-1} J_{1}(\mathrm{i} x)=\sum_{m=0}^
            {\infty} \frac{x^{2m+1}}{2^{2m+1} m ! (m+1) !}
        \end{array}

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([-1., -0.5, 0.5, 1.])
        >>> output = mindspore.ops.bessel_i1(x)
        >>> print(output)
        [-0.5651591  -0.25789431  0.25789431  0.5651591]
    """
    return bessel_i1_(x)


def bessel_i1e(x):
    r"""
    Computes the exponentially scaled first order modified Bessel function of the
    first kind for each element input.

    .. math::
        \begin{array}{ll} \\
            \text I_{1}e(x)=e^{(-|x|)} * I_{1}(x)=e^{(-|x|)} * \sum_{m=0}^
            {\infty} \frac{x^{2m+1}}{2^{2m+1} m ! (m+1) !}
        \end{array}

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([-1., -0.5, 0.5, 1.])
        >>> output = mindspore.ops.bessel_i1e(x)
        >>> print(output)
        [-0.20791042  -0.15642083  0.15642083  0.20791042]
    """
    return bessel_i1e_(x)


def bessel_k1(x):
    r"""
    Computes the first order modified Bessel function of the second kind for each element input.

    .. math::
        \begin{array}{ll} \\
            K_{1}(x)=\lim_{\nu \to 1} \left(\frac{\pi}{2}\right) \frac{I_{-\nu}(x)-
            I_{\nu}(x)}{\sin (\nu \pi)} = \int_{0}^{\infty} e^{-x \cosh t} \cosh (t) d t
        \end{array}

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([0.5, 1., 2., 4.])
        >>> output = mindspore.ops.bessel_k1(x)
        >>> print(output)
        [1.65644112  0.60190723  0.13986588  0.0124835]
    """
    return bessel_k1_(x)


def bessel_k1e(x):
    r"""
    Computes the exponentially scaled first order modified Bessel function of the
    second kind for each element input.

    .. math::
        \begin{array}{ll} \\
            K_{1}e(x)= e^{(-|x|)} * K_{1}(x) = e^{(-|x|)} * \int_{0}
            ^{\infty} e^{-x \cosh t} \cosh (t) d t
        \end{array}

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([0.5, 1., 2., 4.])
        >>> output = mindspore.ops.bessel_k1e(x)
        >>> print(output)
        [2.73100971  1.63615349  1.03347685  0.68157595]
    """
    return bessel_k1e_(x)


@constexpr
def _check_input_dtype(param_name, input_dtype, allow_dtypes, cls_name):
    validator.check_type_name(param_name, input_dtype, allow_dtypes, cls_name)


def deg2rad(x):
    """
    Convert angles from degrees to radians element-wise.

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[90.0, -90.0], [180.0, -180.0], [270.0, -270.0]])
        >>> output = mindspore.ops.deg2rad(input)
        >>> print(output)
        [[ 1.5707964 -1.5707964]
         [ 3.1415927 -3.1415927]
         [ 4.712389  -4.712389 ]]
    """
    if not isinstance(x, (Tensor, Tensor_)):
        raise TypeError("The input x must be tensor")
    x_dtype = dtype_(x)
    _check_input_dtype("x", x_dtype, [mstype.float16, mstype.float32, mstype.float64], "")
    if x_dtype == mstype.float16:
        out = x * (Tensor(math.pi / 180.0).astype(mstype.float16))
    else:
        out = x * math.pi / 180.0
    return out


def rad2deg(x):
    """
    Converts angles in radians to angles in degrees element-wise.

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([[6.283, -3.142],[1.570, -6.283],[3.142, -1.570]], mindspore.float32)
        >>> output = mindspore.ops.rad2deg(x)
        >>> print(output)
        [[ 359.98935 -180.02333]
         [  89.95438 -359.98935]
         [ 180.02333  -89.95438]]

    """
    if not isinstance(x, (Tensor, Tensor_)):
        raise TypeError("The input x must be tensor")
    x_dtype = dtype_(x)
    _check_input_dtype("x", x_dtype, [mstype.float16, mstype.float32, mstype.float64], "")
    if x_dtype == mstype.float16:
        out = x * (Tensor(180.0 / math.pi).astype(mstype.float16))
    else:
        out = x * 180.0 / math.pi
    return out


def frac(x):
    """
    Return the fractional part of each element in the input tensor.

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([2, 4.2, -2.5])
        >>> output = mindspore.ops.frac(input)
        >>> print(output)
        [ 0.          0.19999981 -0.5       ]
    """
    return mod_(x, 1)


#####################################
# Reduction Operation Functions.
#####################################


@_primexpr
def _create_cummin_perm(axis, x_shape):
    """Insure axis is in [-len(x_shape),len(s_shape)-1]"""

    def _check(axis, len_axis):
        if not isinstance(axis, int):
            raise TypeError(f"The date type of 'axis' must be Int, but got {axis}.")
        if axis < -len_axis or axis > len_axis:
            raise ValueError(f"The value of axis must be in [{-len_axis}, {len_axis}], but got {axis}.")

    len_axis = len(x_shape)
    _check(axis, len_axis)
    prem = list(range(len_axis))
    if axis < 0:
        axis = axis + len_axis
    prem[0], prem[axis] = axis, 0
    prem = tuple(prem)
    return prem


def cummin(input, axis):
    r"""
    Return the cumulative minimum values and their indices along the given axis of the tensor.

    .. math::
        \begin{array}{ll} \\
            y_{i} = \min(x_{1}, x_{2}, ... , x_{i})
        \end{array}

    Args:
        input (Tensor): The input tensor.
        axis (int): Specify the axis for computation.

    Returns:
        Tuple(min, min_indices) of 2 tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor, ops
        >>> import mindspore
        >>> a = Tensor([-0.2284, -0.6628,  0.0975,  0.2680, -1.3298, -0.4220], mindspore.float32)
        >>> output = ops.cummin(a, axis=0)
        >>> print(output[0])
        [-0.2284 -0.6628 -0.6628 -0.6628 -1.3298 -1.3298]
        >>> print(output[1])
        [0 1 1 1 4 4]
    """
    if isinstance(axis, bool):
        raise TypeError(f"For 'cummin', the date type of 'axis' must be Int, but got {axis}.")
    cummin_op = _get_cache_prim(Cummin)(axis=0)
    if axis == 0:
        out1, out2 = cummin_op(input)
    else:
        x_shape = shape_(input)
        prem = _create_cummin_perm(axis, x_shape)
        input = transpose_op(input, prem)
        out1, out2 = cummin_op(input)
        out1 = transpose_op(out1, prem)
        out2 = transpose_op(out2, prem)
    return (out1, out2)


def cumsum(x, axis, dtype=None):
    """
    Return the cumulative sum along the given axis of the tensor.

    .. math::

        y_i = x_1 + x_2 + x_3 + ... + x_i

    Args:
        x (Tensor): The input tensor.
        axis (int): Specify the axis for computation.
        dtype (:class:`mindspore.dtype`, optional): The data type returned. Default ``None`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[1, 2, 3],
        ...                           [4, 5, 6]], mindspore.int32)
        >>> mindspore.ops.cumsum(input, axis=0)
        Tensor(shape=[2, 3], dtype=Int32, value=
        [[1, 2, 3],
         [5, 7, 9]])
        >>> mindspore.ops.cumsum(input, axis=1)
        Tensor(shape=[2, 3], dtype=Int32, value=
        [[ 1,  3,  6],
         [ 4,  9, 15]])
    """
    if dtype is not None and x.dtype != dtype:
        x = x.astype(dtype, copy=False)
    return cumsum_(x, axis)


@deprecated("2.8.0", None, False, "ops.")
def sparse_segment_mean(x, indices, segment_ids):
    r"""
    `ops.sparse_segment_mean` is deprecated from version 2.8.0 and will be
    removed in a future version.

    Computes the mean of sparse segments in the input tensor.

    .. math::
        output_i = \frac{\sum_j x_{indices[j]}}{N}

    where `N` is the number of elements where :math:`segment\_ids[j] == i` .
    If `segment_ids` doesn't contain `i`, then :math:`output[i] = 0` .

    Note:
        - On CPU, values in `segment_ids` must be sorted and `indices` must be within range[0, x.shape[0]).
        - On GPU, unsorted `segment_ids` may result in undefined but safe behavior.Out-of-range `indices` will
          be ignored.

    Args:
        x (Tensor): The input tensor with at least one dimension.
        indices (Tensor): The specified indices, a 1-D tensor.
        segment_ids (Tensor): A 1-D tensor, must be sorted and can contain duplicates.

    Returns:
        Tensor

    Supported Platforms:
        Deprecated

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([[0, 1, 2], [1, 2, 3], [3, 6, 7]], dtype=mindspore.float32)
        >>> indices = mindspore.tensor([0, 1, 2], dtype=mindspore.int32)
        >>> segment_ids = mindspore.tensor([1,2,2], dtype=mindspore.int32)
        >>> out = mindspore.ops.sparse_segment_mean(x, indices, segment_ids)
        >>> print(out)
        [[0. 0. 0.]
         [0. 1. 2.]
         [2. 4. 5.]]
    """
    return _get_sparse_segment_mean_prim()(x, indices, segment_ids)


def block_diag(*inputs):
    r"""
    Creates a block diagonal matrix from the provided tensor.

    Args:
        inputs (Tensor): One or more tensors, the dimension of tensor should be 0, 1 or 2.

    Returns:
        2-D Tensor, with all input tensors arranged in
        order so that their top left and bottom right corners are
        diagonally adjacent. All other elements are set to 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x1 = mindspore.tensor([[4], [3], [2]], mindspore.int32)
        >>> x2 = mindspore.tensor([7, 6, 5], mindspore.int32)
        >>> x3 = mindspore.tensor(1, mindspore.int32)
        >>> x4 = mindspore.tensor([[5, 4, 3], [2, 1, 0]], mindspore.int32)
        >>> x5 = mindspore.tensor([[8, 7], [7, 8]], mindspore.int32)
        >>> out = mindspore.ops.block_diag(x1, x2, x3, x4, x5)
        >>> print(out.asnumpy())
        [[4 0 0 0 0 0 0 0 0 0]
         [3 0 0 0 0 0 0 0 0 0]
         [2 0 0 0 0 0 0 0 0 0]
         [0 7 6 5 0 0 0 0 0 0]
         [0 0 0 0 1 0 0 0 0 0]
         [0 0 0 0 0 5 4 3 0 0]
         [0 0 0 0 0 2 1 0 0 0]
         [0 0 0 0 0 0 0 0 8 7]
         [0 0 0 0 0 0 0 0 7 8]]
    """

    def to_col_block(arys, i, a):
        return [
            a if idx == i else ops.zeros((ary.shape[0], a.shape[1]), ary.dtype)
            for idx, ary in enumerate(arys)
        ]

    def to_2d(ary):
        if not isinstance(ary, Tensor):
            raise TypeError(
                f"For 'block_diag', each element of 'inputs' must be a tensor, but got {type(ary)}"
            )
        if ary.ndim == 0:
            return ops.expand_dims(ops.expand_dims(ary, 0), 0)
        if ary.ndim == 1:
            return ops.expand_dims(ary, 0)
        if ary.ndim == 2:
            return ary
        raise ValueError(
            "For 'block_diag', the dimension of each elements in 'inputs' must be 0, 1, or 2, but got "
            f"{ary.ndim}"
        )

    if not inputs:
        raise RuntimeError("For 'block_diag', the input is empty.")
    arys = [to_2d(ary) for ary in inputs]
    matrix = [ops.concat(to_col_block(arys, idx, ary)) for idx, ary in enumerate(arys)]
    return ops.concat(matrix, 1)


def atleast_1d(inputs):
    r"""
    Returns a one-dimensional tensor of each zero-dimensional tensor,
    while tensors with one or more dimensions remain unchanged.

    Args:
        inputs (Union[Tensor, list[Tensor]]): The input tensor or list of tensors.

    Returns:
        Tensor or list of tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> # case 1: Input is a zero-dimensional tensor.
        >>> x = mindspore.tensor(1)
        >>> mindspore.ops.atleast_1d(x)
        Tensor(shape=[1], dtype=Int64, value= [1])
        >>>
        >>> # case 2: Input is a one-dimensional tensor.
        >>> y = mindspore.tensor([0, 1])
        >>> mindspore.ops.atleast_1d(y)
        Tensor(shape=[2], dtype=Int64, value= [0, 1])
        >>>
        >>> # case 3: Input is a list containing tensors of various dimensions.
        >>> mindspore.ops.atleast_1d([x, y])
        (Tensor(shape=[1], dtype=Int64, value= [1]), Tensor(shape=[2], dtype=Int64, value= [0, 1]))
    """
    if isinstance(inputs, Tensor):
        return _expand(inputs, 1)
    for tensor in inputs:
        if not isinstance(tensor, Tensor):
            raise TypeError(f"For 'atleast_1d', each element of 'inputs' must be a tensor, but got {type(tensor)}")
    return tuple(_expand(arr, 1) for arr in inputs)


def dstack(tensors):
    r"""
    Stacks tensors along the third axis.

    .. note::
        - 1-D tensors :math:`(N,)` should be reshaped to :math:`(1,N,1)`. 2-D tensors :math:`(M,N)` should
          be reshaped to :math:`(M,N,1)` before concatenation.
        - The tensors must have the same shape along all but the third axis.
          1-D or 2-D tensors must have the same shape.

    Args:
        tensors (Union(List[Tensor], tuple[Tensor])): The list of tensors or tuple of tensors.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x1 = mindspore.tensor(mindspore.ops.arange(1, 7).reshape(2, 3))
        >>> x2 = mindspore.tensor(mindspore.ops.arange(7, 13).reshape(2, 3))
        >>> out = mindspore.ops.dstack([x1, x2])
        >>> print(out.asnumpy())
        [[[ 1.  7.]
          [ 2.  8.]
          [ 3.  9.]]
         [[ 4. 10.]
          [ 5. 11.]
          [ 6. 12.]]]
    """
    if not isinstance(tensors, (tuple, list)):
        raise TypeError(f"For 'dstack', 'tensors' must be list or tuple of tensors, but got {type(tensors)}")
    if not tensors:
        raise TypeError("For 'dstack', 'tensors' can not be empty.")
    trans_tensors = ()
    for tensor in tensors:
        if not isinstance(tensor, Tensor):
            raise TypeError(f"For 'dstack', each elements of 'tensors' must be Tensor, but got {type(tensor)}")
        if tensor.ndim == 0:
            tensor = reshape_(tensor, (1, 1, 1))
        elif tensor.ndim == 1:
            tensor = expand_dims_(expand_dims_(tensor, 0), 2)
        elif tensor.ndim == 2:
            tensor = expand_dims_(tensor, 2)
        trans_tensors += (tensor,)
    if not trans_tensors:
        raise ValueError("For 'dstack', at least one tensor is needed to concatenate.")
    return _get_cache_prim(P.Concat)(2)(trans_tensors)


@_primexpr
def _check_is_int(arg_value, arg_name, cls_name):
    validator.check_is_int(arg_value, arg_name, cls_name)


def diff(x, n=1, axis=-1, prepend=None, append=None):
    r"""
    Computes the n-th forward difference along the given axis.

    The first-order differences is calculated as :math:`out[i] = x[i+1] - x[i]` .
    Higher-order differences are calculated by using :func:`mindspore.ops.diff` recursively.

    Note:
        Zero-shaped Tensor is not supported, a value error is raised if
        an empty Tensor is encountered. Any dimension of a Tensor is 0, which is considered
        an empty Tensor. Tensor with shape of :math:`(0,)`, :math:`(1, 2, 0, 4)` are all
        empty Tensor.

    Args:
        x (Tensor): The input tensor.
        n (int, optional): The number of times to compute the difference. Currently only 1 is supported. Default
            ``1`` .
        axis (int, optional): The axis to compute the difference along. Default ``-1`` .
        prepend (Tensor, optional): Values to prepend to `x` along
            `axis` before performing the difference. Their dimensions must be equivalent to that of `x`, and their
            shapes must match input's shape except on dim. Default ``None`` .
        append (Tensor, optional): Values to append to `x` along
            `axis` before performing the difference. Their dimensions must be equivalent to that of `x`, and their
            shapes must match input's shape except on dim. Default ``None`` .

    Returns:
        Tensor, the n-th differences of input. The shape of the output is the same as `x`
        except along `axis` where the size is reduced by `n`. The type of the output
        is the same as `x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([1, 3, 2])
        >>> # case 1: By default, compute the first-order differences along axis -1.
        >>> mindspore.ops.diff(x)
        Tensor(shape=[2], dtype=Int64, value= [ 2, -1])
        >>>
        >>> # case 2: When argument prepend is setting:
        >>> n = mindspore.tensor([4, 5])
        >>> mindspore.ops.diff(x, prepend=n)
        Tensor(shape=[4], dtype=Int64, value= [ 1, -4,  2, -1])
        >>>
        >>> # case 3: When argument append is setting:
        >>> mindspore.ops.diff(x, append=n)
        Tensor(shape=[4], dtype=Int64, value= [ 2, -1,  2,  1])
        >>>
        >>> # case 4: When input is 2-D dimensional tensor, compute forward difference along different axis.
        >>> x = mindspore.tensor([[1, 2, 3], [3, 4, 5]])
        >>> mindspore.ops.diff(x, axis=0)
        Tensor(shape=[1, 3], dtype=Int64, value=
        [[2, 2, 2]])
        >>> mindspore.ops.diff(x, axis=1)
        Tensor(shape=[2, 2], dtype=Int64, value=
        [[1, 1],
         [1, 1]])
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"For 'diff', 'x' must be a tensor, but got {type(x)}")
    if x.ndim < 1:
        raise TypeError(f"For 'diff', the dimension 'x' must be at least 1, but got {x.ndim}")
    if 0 in x.shape:
        raise ValueError("For 'diff', 'x' can not be an empty Tensor.")
    _check_is_int(n, 'n', 'diff')
    if n != 1:
        raise RuntimeError(f"For 'diff', 'n' must be 1, but got {n}")
    if x.dtype in (mstype.uint16, mstype.uint32, mstype.uint64):
        msg = f"For 'diff', the data type of the elements in 'x' cannot be uint16, uint32, uint64, but got {x.dtype}"
        raise TypeError(msg)
    if prepend is not None and append is not None:
        x = ops.Concat(axis)((prepend, x, append))
    elif append is not None:
        x = ops.Concat(axis)((x, append))
    elif prepend is not None:
        x = ops.Concat(axis)((prepend, x))
    a = ops.make_range(x.shape[axis])
    a1 = x.gather(TupleToTensor()(a[:-1], mstype.int64), axis)
    a2 = x.gather(TupleToTensor()(a[1:], mstype.int64), axis)
    return a2 - a1


def _diff_is_scalar_or_scalar_tensor(value):
    """judge the value"""
    if isinstance(value, int):
        return True

    if isinstance(value, ms.Tensor) and value.shape == ():
        return True

    return False


def _diff_check(input, n, dim):
    """judge the input n and dim"""
    if not isinstance(input, Tensor):
        raise TypeError("For 'diff', 'input' must be a tensor")

    if not _diff_is_scalar_or_scalar_tensor(n):
        raise TypeError("For 'diff', 'n' must be a int scalar or int scalar tensor")

    if not _diff_is_scalar_or_scalar_tensor(dim):
        raise TypeError("For 'diff', 'dim' must be a scalar or scalar tensor")

    if input.dtype in (mstype.complex64, mstype.complex128, mstype.float64, mstype.int16):
        raise TypeError("For 'diff', 'input' do not support complex64/complex128/float64/int16")


def _diff_helper(input, n, dim):
    """calculate the forward difference"""
    out_len = input.shape[dim] - 1
    is_bool = input.dtype == mstype.bool_
    result = input

    for _ in range(n):  # pylint: disable=unused-variable
        if is_bool:
            result = logical_xor(narrow(result, dim, 1, out_len), narrow(result, dim, 0, out_len))
        else:
            result = sub_ext(narrow(result, dim, 1, out_len), narrow(result, dim, 0, out_len))

        if out_len == 0:
            break
        out_len -= 1

    return result


def _diff_prepend_append_on_dim(input, prepend, append, dim):
    """append tensor on dim"""
    if prepend is not None and append is None:
        return cat((prepend, input), dim)

    if prepend is None and append is not None:
        return cat((input, append), dim)

    return cat((prepend, input, append), dim)


def diff_ext(input, n=1, dim=-1, prepend=None, append=None):
    r"""
    Computes the n-th forward difference along the given dimension.

    The first-order differences are given by :math:`out[i] = input[i+1] - input[i]`. Higher-order differences are
    calculated by using `diff` recursively.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): the tensor to compute the differences on.
        n (int, optional): the number of times to recursively compute the difference.
            Default: ``1`` .
        dim (int, optional): the dimension to compute the difference along.
            Default is the last dimension. Default: ``0`` .
        prepend (Tensor, optional): values to prepend or append to `input` along `dim`
            before computing the difference. Their dimensions must be equivalent to that of input,
            and their shapes must match input's shape except on `dim`. Default: ``None`` .
        append (Tensor, optional): values to prepend or append to `input` along `dim`
            before computing the difference. Their dimensions must be equivalent to that of input,
            and their shapes must match input's shape except on `dim`. Default: ``None`` .

    Returns:
        Tensor, the result of n-th forward difference computation.

    Raises:
        TypeError: If `input` is not a tensor.
        TypeError: If `n` is not a scalar or scalar tensor.
        TypeError: If `dim` is not a scalar or scalar tensor.
        TypeError: If `input` type is complex64, complex128, float64, int16.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindspore import Tensor, ops
        >>> x = Tensor([1, 3, -1, 0, 4])
        >>> out = ops.diff_ext(x)
        >>> print(out.asnumpy())
        [ 2 -4  1  4]
    """
    _diff_check(input, n, dim)

    if (prepend is None and append is None) or n == 0:
        return _diff_helper(input, n, dim)

    input = _diff_prepend_append_on_dim(input, prepend, append, dim)
    return _diff_helper(input, n, dim)


def tril_indices(row, col, offset=0, *, dtype=mstype.int64):
    r"""
    Return a 2-by-N tensor containing the indices of the lower triangular elements of a `row` * `col` matrix.
    The first row of the Tensor contains row coordinates, and the second row contains column coordinates.
    The coordinates are sorted by rows and then columns.

    Note:
        When running on CUDA, row * col must be less than 2^59 to prevent overflow during calculation.

    Args:
        row (int): number of rows in the 2-D matrix.
        col (int): number of columns in the 2-D matrix.
        offset (int, optional): Diagonal offset. Default ``0`` .

            - When `offset` is a positive integer, the diagonal is shifted upward.
            - When `offset` is a negative integer, the diagonal is shifted downward.

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): The specified type of output tensor.
            An optional data type of `mindspore.int32` and `mindspore.int64`. Default ``mstype.int64`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> # case 1: By default, offset=0, all elements on and below the main diagonal are retained.
        >>> mindspore.ops.tril_indices(3, 2, 0)
        Tensor(shape=[2, 5], dtype=Int64, value=
        [[0, 1, 1, 2, 2],
         [0, 0, 1, 0, 1]])
        >>>
        >>> # case 2: Offset=1, the indices on and below the first sub-diagonal above the main diagonal are returned.
        >>> mindspore.ops.tril_indices(3, 2, 1)
        Tensor(shape=[2, 6], dtype=Int64, value=
        [[0, 0, 1, 1, 2, 2],
         [0, 1, 0, 1, 0, 1]])
        >>>
        >>> # case 3: Offset=-1, the indices on and below the first sub-diagonal below the main diagonal are returned.
        >>> mindspore.ops.tril_indices(3, 2, -1)
        Tensor(shape=[2, 3], dtype=Int64, value=
        [[1, 2, 2],
         [0, 0, 1]])
    """

    tril_indices_ = TrilIndices(row=row, col=col, offset=offset, dtype=dtype)
    return tril_indices_()


def triu_indices(row, col, offset=0, *, dtype=mstype.int64):
    r"""
    Return a 2-by-N tensor containing the indices of the upper triangular elements of a `row` * `col` matrix.
    The first row of the Tensor contains row coordinates, and the second row contains column coordinates.
    The coordinates are sorted by rows and then columns.

    Note:
        When running on CUDA, row * col must be less than 2^59 to prevent overflow during calculation.

    Args:
        row (int): number of rows in the 2-D matrix.
        col (int): number of columns in the 2-D matrix.
        offset (int, optional): Diagonal offset. Default ``0`` .

            - When `offset` is a positive integer, the diagonal is shifted upward.
            - When `offset` is a negative integer, the diagonal is shifted downward.

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): The specified type of output tensor.
            An optional data type of `mindspore.int32` and `mindspore.int64`. Default ``mstype.int64`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> # case 1: By default, offset=0, all elements on and below the main diagonal are retained.
        >>> mindspore.ops.triu_indices(3, 2, 0)
        Tensor(shape=[2, 3], dtype=Int64, value=
        [[0, 0, 1],
         [0, 1, 1]])
        >>>
        >>> # case 2: If offset=-1, the indices on and above the first sub-diagonal below the main diagonal are returned.
        >>> mindspore.ops.triu_indices(3, 2, -1)
        Tensor(shape=[2, 5], dtype=Int64, value=
        [[0, 0, 1, 1, 2],
         [0, 1, 0, 1, 1]])
    """

    triu_indices_ = TriuIndices(row=row, col=col, offset=offset, dtype=dtype)
    return triu_indices_()


def atleast_2d(inputs):
    r"""
    Returns a 2-dimensional tensor of each tensor, while tensors with two or more dimensions remain unchanged.

    Args:
        inputs (Union[Tensor, list[Tensor]]): The input tensor or list of tensors.

    Returns:
        Tensor or list of tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> # case 1: Input is a zero-dimensional tensor.
        >>> x = mindspore.tensor(1)
        >>> mindspore.ops.atleast_2d(x)
        Tensor(shape=[1, 1], dtype=Int64, value= [[1]])
        >>>
        >>> # case 2: Input is a 2-dimensional tensor.
        >>> y = mindspore.tensor([[0, 1], [2, 3]])
        >>> mindspore.ops.atleast_2d(y)
        Tensor(shape=[2, 2], dtype=Int64, value=
        [[0, 1],
         [2, 3]])
        >>>
        >>> # case 3: Input is a list containing tensors of various dimensions.
        >>> mindspore.ops.atleast_2d([x, y])
        (Tensor(shape=[1, 1], dtype=Int64, value=
         [[1]]),
         Tensor(shape=[2, 2], dtype=Int64, value=
         [[0, 1],
          [2, 3]]))
    """
    if isinstance(inputs, Tensor):
        return _expand(inputs, 2)
    for tensor in inputs:
        if not isinstance(tensor, Tensor):
            msg = "expect Tensor or list of tensors, but got " + f"{type(tensor)}"
            raise TypeError(msg)
    return tuple(_expand(arr, 2) for arr in inputs)


def cartesian_prod(*inputs):
    r"""
    Performs a Cartesian product for a given tensor sequence.
    The behavior is similar to Python's `itertools.product`.

    Args:
        inputs (List[Tensor]): Tensor sequence.

    Returns:
        Tensor, a Cartesian product for a given tensor sequence.

    Raises:
        TypeError: If the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor, ops
        >>> x1 = Tensor([1, 2])
        >>> x2 = Tensor([5])
        >>> out = ops.cartesian_prod(x1, x2)
        >>> print(out.asnumpy())
        [[1 5]
         [2 5]]
        >>> x1 = Tensor([1, 2, 3, 4])
        >>> x2 = Tensor([5, 6, 7])
        >>> x3 = Tensor([8, 9, 0, 1, 2])
        >>> out = ops.cartesian_prod(x1, x2, x3)
        >>> print(len(out))
        60
    """
    meshgrid = _get_cache_prim(P.Meshgrid)(indexing="ij")
    meshgrid_output = meshgrid(inputs)
    stack = _get_cache_prim(P.Stack)(axis=-1)
    stack_output = stack(meshgrid_output)
    return reshape_(stack_output, (-1, len(inputs)))


def atleast_3d(inputs):
    r"""
    Returns a 3-dimensional tensor of each tensor, while tensors with three or more dimensions remain unchanged.

    Args:
        inputs (Union[Tensor, list[Tensor]]): The input tensor or list of tensors.

    Returns:
        Tensor or list of tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> # case 1: Input is a zero-dimensional tensor.
        >>> x = mindspore.tensor(1)
        >>> mindspore.ops.atleast_3d(x)
        Tensor(shape=[1, 1, 1], dtype=Int64, value= [[[1]]])
        >>>
        >>> # case 2: Input is a 3-dimensional tensor.
        >>> y = mindspore.tensor([[[0, 1], [2, 3]]])
        >>> mindspore.ops.atleast_3d(y)
        Tensor(shape=[1, 2, 2], dtype=Int64, value=
        [[[0, 1],
          [2, 3]]])
        >>>
        >>> # case 3: Input is a list containing tensors of various dimensions.
        >>> mindspore.ops.atleast_3d([x, y])
        (Tensor(shape=[1, 1, 1], dtype=Int64, value=
         [[[1]]]),
         Tensor(shape=[1, 2, 2], dtype=Int64, value=
         [[[0, 1],
           [2, 3]]])
    """

    def _expand3(arr):
        ndim = rank_(arr)
        if ndim == 0:
            return reshape_(arr, (1, 1, 1))
        if ndim == 1:
            return reshape_(arr, (1, size_(arr), 1))
        if ndim == 2:
            return reshape_(arr, shape_(arr) + (1,))
        return arr

    if isinstance(inputs, Tensor):
        return _expand3(inputs)
    for tensor in inputs:
        if not isinstance(tensor, Tensor):
            raise TypeError(f"For 'atleast_3d', each element of 'inputs' must be a tensor, but got {type(tensor)}")
    return tuple(_expand3(arr) for arr in inputs)


def view_as_real(input):
    r"""
    Return a real tensor with the last dimension of size 2, composed of the real and imaginary parts of the complex
    elements in the input tensor.

    Args:
        input (Tensor): The complex input tensor.

    Returns:
        A real tensor.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([2+1j,2+3j,2-1j,2])
        >>> output = mindspore.ops.view_as_real(input)
        >>> print(output)
        [[ 2.  1.]
         [ 2.  3.]
         [ 2. -1.]
         [ 2.  0.]]
    """
    if not is_complex(input):
        raise TypeError("For view_as_real, the dtype of input Tensor must be complex.")
    real_part = input.real().expand_dims(-1)
    imag_part = input.imag().expand_dims(-1)
    con = _get_cache_prim(ops.Concat)(-1)
    return con((real_part, imag_part))


def vstack(inputs):
    r"""
    Stacks tensors in sequence vertically.

    This is equivalent to concatenation along the first axis.
    1-D tensors :math:`(N,)` should firstly be reshaped to :math:`(1, N)`,
    and then be concatenated along the first axis.

    Args:
        inputs (Union(List[tensor], Tuple[tensor])): The 1-D or 2-D input.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x1 = mindspore.tensor([1, 2, 3])
        >>> x2 = mindspore.tensor([4, 5, 6])
        >>> mindspore.ops.vstack((x1, x2))
        Tensor(shape=[2, 3], dtype=Int64, value=
        [[1, 2, 3],
         [4, 5, 6]])
        >>> x1 = mindspore.tensor([[1],[2],[3]])
        >>> x2 = mindspore.tensor([[4],[5],[6]])
        >>> mindspore.ops.vstack([x1, x2])
        Tensor(shape=[6, 1], dtype=Int64, value=
        [[1],
         [2],
         [3],
         [4],
         [5],
         [6]])
    """
    if not isinstance(inputs, (tuple, list)):
        msg = f"For 'vstack', list or tuple of tensors are required, but got {type(inputs)}"
        raise TypeError(msg)
    if not inputs:
        msg = "For 'vstack', inputs can not be empty"
        raise TypeError(msg)
    trans_tup = ()
    for tensor in inputs:
        if not isinstance(tensor, Tensor):
            msg = f"For 'vstack', Tensor is required, but got {type(tensor)}"
            raise TypeError(msg)
        if tensor.ndim <= 1:
            shape = shape_(tensor)
            if isinstance(shape, int):
                shape = (shape,)
            ndim_diff = 2 - len(shape)
            if ndim_diff > 0:
                shape = [1] * ndim_diff + list(shape)
            tensor = reshape_(tensor, tuple(shape))
        trans_tup += (tensor,)
    if not trans_tup:
        raise ValueError("For 'vstack', need at least one tensor to concatenate.")
    out = _get_cache_prim(P.Concat)(0)(trans_tup)
    return out


def row_stack(tensors):
    """
    Alias for :func:`mindspore.ops.vstack` .

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    return vstack(tensors)


def combinations(input, r=2, with_replacement=False):
    r"""
    Return all r-length subsequences of input tensor.

    When `with_replacement` is set to ``False``, it works similar to Python's
    `itertools.combinations`, and when `with_replacement` is set to ``True``,
    it behaves like `itertools.combinations_with_replacement`.

    Args:
        input (Tensor): One-dimensional input tensor.
        r (int, optional): Number of elements to perform combination. Default ``2`` .
        with_replacement (bool, optional): Allow duplication or not. Default ``False`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([1, 2, 3])
        >>> mindspore.ops.combinations(input)
        Tensor(shape=[3, 2], dtype=Int64, value=
        [[1, 2],
         [1, 3],
         [2, 3]])
        >>> mindspore.ops.combinations(input, r=3)
        Tensor(shape=[1, 3], dtype=Int64, value=
        [[1, 2, 3]])
        >>> mindspore.ops.combinations(input, with_replacement=True)
        Tensor(shape=[6, 2], dtype=Int64, value=
        [[1, 1],
         [1, 2],
         [1, 3],
         [2, 2],
         [2, 3],
         [3, 3]])
        >>> # It has the same results as using itertools.combinations.
        >>> import itertools
        >>> input = [1, 2, 3]
        >>> list(itertools.combinations(input, r=2))
        [(1, 2), (1, 3), (2, 3)]
        >>> list(itertools.combinations(input, r=3))
        [(1, 2, 3)]
        >>> list(itertools.combinations_with_replacement(input, r=2))
        [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
    """

    def _combinations(iterable, r):
        lst = ops.StridedSlice()(ops.zeros(r), (0,), (0,), (1,))
        pool = tuple(iterable)
        n = len(pool)
        if r > n:
            return lst
        indices = list(range(r))
        lst = ops.concat([ops.reshape(pool[i], (1,)) for i in indices])
        while True:
            stop = True
            i = 0
            for index in range(r)[::-1]:
                if indices[index] != index + n - r:
                    stop = False
                    i = index
                    break
            if stop:
                return lst
            indices[i] += 1
            for j in range(i + 1, r):
                indices[j] = indices[j - 1] + 1
            item = ops.concat([ops.reshape(pool[i], (1,)) for i in indices])
            lst = ops.concat((lst, item), -1)
        return None

    def _combinations_with_replacement(iterable, r):
        lst = Tensor_([])
        pool = tuple(iterable)
        n = len(pool)
        if not n and r:
            return lst
        indices = [0] * r
        lst = ops.concat([ops.reshape(pool[i], (1,)) for i in indices])
        while True:
            stop = True
            i = 0
            for index in range(r)[::-1]:
                if indices[index] != n - 1:
                    stop = False
                    i = index
                    break
            if stop:
                return lst
            indices[i:] = [indices[i] + 1] * (r - i)
            item = ops.concat([ops.reshape(pool[i], (1,)) for i in indices])
            lst = ops.concat((lst, item), -1)
        return None

    if not isinstance(input, Tensor):
        raise TypeError(f"For 'combinations', 'x' must be a tensor, but got {type(input)}")
    if input.ndim != 1:
        raise ValueError(f"For 'combinations', the dimension 'x' must be 1, but got {input.ndim}")
    if not isinstance(r, int):
        raise TypeError(f"For 'combinations', 'r' must be an integer, but got {type(r)}")
    comb_func = _combinations_with_replacement if with_replacement else _combinations
    ret = comb_func(input, r)
    if ret.size == 0:
        return ret
    return ops.reshape(ret, (-1, r))


def dist(input, other, p=2):
    r"""
    Computes batched the :math:`p`-norm distance between each pair of the two collections of row vectors.

    Note:
        Since only normalization for integer :math:`p`-normal form is supported in MindSpore,
        a type error will be raised if :math:`p` is not an integer.

    Args:
        input (Tensor): The first input tensor. The dtype must be float16 or float32.
        other (Tensor): The second input tensor. The dtype must be float16 or float32.
        p (int, optional): The order of norm. `p` is greater than or equal to 0. Default: ``2`` .

    Returns:
        Tensor, has the same dtype as `input`, which shape is :math:`(1)`.

    Raises:
        TypeError: If `input` or `other` is not a Tensor.
        TypeError: If dtype of `input` or `other` is neither float16 nor float32.
        TypeError: If `p` is not a non-negative integer.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor, ops
        >>> input_x = Tensor([[[1.0, 1.0], [2.0, 2.0]]])
        >>> input_y = Tensor([[[3.0, 3.0], [3.0, 3.0]]])
        >>> out = ops.dist(input_x, input_y)
        >>> print(out.asnumpy())
        3.1622777
    """
    if not isinstance(input, Tensor):
        raise TypeError(f"For 'dist', 'input' must be a tensor, but got {type(input)}")
    if not isinstance(other, Tensor):
        raise TypeError(f"For 'dist', 'other' must be a tensor, but got {type(other)}")
    z = input - other
    if z.ndim == 0:
        return ops.abs(z)

    # the types of p will expend once ops.LpNorm supports float
    return ops.LpNorm(axis=0, p=p)(ops.reshape(z, (-1,)))


def copysign(x, other):
    r"""
    Create a float tensor composed of the absolute values of `x` and the signs of `other` .
    Support broadcasting.

    Args:
        x (Union[Tensor]): The input tensor.
        other (Union[int, float, Tensor]): A tensor that determines the sign of the return value.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> # case 1: When other is a tensor
        >>> x = mindspore.tensor([[0.3, -0.7],
        ...                       [0.5, 0.5]])
        >>> other = mindspore.tensor([-0.4, 0.6])
        >>> output = mindspore.ops.copysign(x, other)
        >>> print(output)
        [[-0.3  0.7]
         [-0.5  0.5]]
        >>> # case 2: When other is a scalar
        >>> output = mindspore.ops.copysign(x, 2)
        >>> print(output)
        [[0.3 0.7]
         [0.5 0.5]]
    """

    def _broadcast_to_shape(x, shape):
        """Broadcasts x from current shape to shape"""
        ndim_to = len(shape)
        x = _expand(x, ndim_to)
        return _broadcast_to(x, shape_(x), shape, ndim_to)

    if not isinstance(x, Tensor):
        raise TypeError("Tensor is expected, but got " + f"{type(x)}")
    if not isinstance(other, (int, float, Tensor)):
        raise TypeError(
            "integer, float or Tensor is expected, but got " + f"{type(other)}"
        )

    if not isinstance(other, Tensor):
        other = _type_convert(Tensor, other)
    other = _broadcast_to_shape(other, shape_(x))

    if _check_same_type(dtype_(x), mstype.bool_):
        raise TypeError("copysign does not accept dtype bool.")

    if _check_same_type(dtype_(x), mstype.complex64):
        raise TypeError("copysign does not accept dtype complex64.")
    if _check_same_type(dtype_(other), mstype.complex64):
        raise TypeError("copysign does not accept dtype complex64.")

    if _check_same_type(dtype_(x), mstype.complex128):
        raise TypeError("copysign does not accept dtype complex128.")
    if _check_same_type(dtype_(other), mstype.complex128):
        raise TypeError("copysign does not accept dtype complex128.")

    x_float = (
        x
        if x.dtype in (mstype.float16, mstype.float32, mstype.float64)
        else x.astype("float32")
    )
    pos_tensor = absolute_(x_float)
    less_zero = tensor_lt(other, 0)
    return select_(less_zero, neg(pos_tensor), pos_tensor)


def hann_window(window_length, periodic=True, *, dtype=None):
    r"""
    Hann window function.

    .. math::
        w(n) = \frac{1}{2} - \frac{1}{2} \cos\left(\frac{2\pi{n}}{M-1}\right),\qquad 0 \leq n \leq M-1

    Args:
        window_length (int): The size of window.
        periodic (bool, optional): If ``True`` , return a periodic window. If ``False``, return a symmetric window.
            Default ``True`` .

    Keyword Args:
        dtype (mindspore.dtype, optional): The data type specified. Default ``None`` .

    Returns:
        A 1-D tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> output = mindspore.ops.hann_window(5)
        >>> print(output)
        [0.        0.3454915 0.9045085 0.9045085 0.3454915]
        >>> output = mindspore.ops.hann_window(5, periodic=False)
        >>> print(output)
        [0.  0.5 1.  0.5 0. ]
    """
    if not isinstance(window_length, int):
        raise TypeError(
            f"For 'hann_window', 'window_length' must be a non-negative integer, but got {type(window_length)}"
        )
    if window_length < 0:
        raise ValueError(
            f"For 'hann_window', 'window_length' must be a non-negative integer, but got {window_length}"
        )
    if window_length <= 1:
        return Tensor(np.ones(window_length))
    if not isinstance(periodic, (bool, np.bool_)):
        raise TypeError(
            f"For 'hann_window', 'periodic' must be a variable of Boolean type, but got {type(periodic)}"
        )
    if dtype is not None and dtype not in mstype.float_type:
        raise TypeError(f"For 'hann_window', 'dtype' must be floating point dtypes, but got {dtype}.")
    if periodic:
        window_length = window_length + 1
    n = np.arange(0, window_length)
    w = 0.5 - 0.5 * np.cos(2 * math.pi / (window_length - 1) * n)

    if dtype is not None:
        w = cast_(ms.tensor(w), dtype)
    return Tensor(w[:-1]) if periodic else Tensor(w)


@constexpr
def _type_convert(force, obj):
    """
    Convert type of `obj` to `force`.
    """
    return force(obj)


def logcumsumexp(input, axis):
    """
    Calculate the log of cumulative summed exponentials of the tensor along a specified dimension.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): The input tensor.
        axis (int): Specify the axis for computation.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``CPU`` ``GPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[9., 3, 4, 5],
        ...                           [5, 2, 7, 4],
        ...                           [8, 1, 3, 6]])
        >>> # case 1: Compute the log of cumulative summed exponential along dim 0.
        >>> output = mindspore.ops.logcumsumexp(input, 0)
        >>> print(output)
        [[9.        3.        4.        5.       ]
         [9.01815   3.3132617 7.0485873 5.3132615]
         [9.326563  3.407606  7.065884  6.407606 ]]
        >>>
        >>> # case 2: Compute the log of cumulative summed exponential along dim 1.
        >>> output = mindspore.ops.logcumsumexp(input, 1)
        >>> print(output)
        [[9.        9.002476  9.009174  9.02716  ]
         [5.        5.0485873 7.1328454 7.175515 ]
         [8.        8.000912  8.007621  8.133643 ]]
    """
    if not isinstance(axis, int):
        raise TypeError(
            f"For 'logcumsumexp', 'axis' must be int type, but got {type(axis)}"
        )
    return cumulative_logsumexp_(input, Tensor(axis))


def logsumexp(input, dim, keepdim=False):
    r"""
    Calculate the log of summed exponentials of the tensor along a specified dimension.

    .. math::

        logsumexp(input) = \log(\sum(e^{input-input_{max}})) + input_{max}

    Args:
        input (Tensor): The input tensor.
        dim (Union[int, tuple(int), list(int)]): Specify the dimension for computation. If ``()`` , compute all
            elements in the `input` .
        keepdim (bool, optional): Whether the output tensor has dim retained. Default ``False`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[9., 3, 4, 5],
        ...                           [5, 2, 7, 4],
        ...                           [8, 1, 3, 6]])
        >>> # case 1: By default, compute the log of summed exponential of all elements.
        >>> output = mindspore.ops.logsumexp(input, ())
        >>> print(output)
        9.475807
        >>>
        >>> # case 2: Compute the log of summed exponential along dim 0.
        >>> output = mindspore.ops.logsumexp(input, 0)
        >>> print(output)
        [9.326562  3.4076054 7.065884  6.4076056]
        >>>
        >>> # case 3: If keepdim=True, the output shape will be same of that of the input.
        >>> output = mindspore.ops.logsumexp(input, 1, True)
        >>> print(output)
        [[9.02716 ]
         [7.175515]
         [8.133643]]
    """
    input_max = ops.ReduceMax(keep_dims=True)(input, dim)
    input_exp = tensor_exp(input - input_max)
    input_sumexp = ops.sum(input_exp, dim, keepdim)
    input_logsumexp = log_(input_sumexp)
    if not keepdim:
        input_max = input_max.squeeze(dim)
    return input_logsumexp + input_max


def amin(input, axis=None, keepdims=False, *, initial=None, where=None):
    r"""
    Return the minimum values along the given axis of the tensor.

    Args:
        input (Tensor[Number]): The input tensor.
        axis (Union[int, tuple(int), list(int), Tensor], optional): Specify the axis for computation. If ``None`` ,
            compute all elements in the `input` . Default ``None`` .
        keepdims (bool, optional): Whether the output tensor has dim retained. Default ``False`` .

    Keyword Args:
        initial (scalar, optional): Initial value for the minimum. Default ``None`` .
        where (Tensor[bool], optional): Specifies the range over which to compute the minimum values. The shape of this
            tensor must bebroadcastable to the shape of `input` . An `initial` value must be specified. Default
            ``None`` , indicating that all elements are to be computed.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[2, 5, 1, 6],
        ...                           [3, -7, -2, 4],
        ...                           [8, -4, 1, -3]])
        >>> # case 1: By default, compute the minimum of all elements.
        >>> mindspore.ops.amin(input)
        Tensor(shape=[], dtype=Int64, value= -7)
        >>>
        >>> # case 2: Compute minimum along axis 1.
        >>> mindspore.ops.amin(input, axis=1)
        Tensor(shape=[3], dtype=Int64, value= [ 1, -7, -4])
        >>>
        >>> # case 3: If keepdims=True, the output shape will be same of that of the input.
        >>> mindspore.ops.amin(input, axis=1, keepdims=True)
        Tensor(shape=[3, 1], dtype=Int64, value=
        [[ 1],
         [-7],
         [-4]])
        >>>
        >>> # case 4: Use "where" to include only specific elements in computing the minimum.
        >>> where = mindspore.tensor([[1, 0, 1, 0],
        ...                           [0, 0, 1, 1],
        ...                           [1, 1, 1, 0]], dtype=mindspore.bool)
        >>> mindspore.ops.amin(input, axis=1, keepdims=True, initial=0, where=where)
        Tensor(shape=[3, 1], dtype=Int64, value=
         [[ 0],
          [-2],
          [-4]])
        >>>
        >>> # case 5: The shape of "where" must be broadcast compatible with input.
        >>> where = mindspore.tensor([[False],
        ...                           [False],
        ...                           [False]])
        >>> mindspore.ops.amin(input, axis=0, keepdims=True, initial=0, where=where)
        Tensor(shape=[1, 4], dtype=Int64, value=
         [[0, 0, 0, 0]])
    """
    if axis is None:
        axis = ()
    input = _init_and_select_elem(input, initial, where, ops.minimum)
    return _get_cache_prim(P.ReduceMin)(keepdims)(input, axis)


def _init_and_select_elem(input, initial, where, cmp_fn):
    """Initialize the input according to Initial, and select the element according to where."""
    if initial is not None:
        initial = ops.fill(input.dtype, input.shape, initial)
        input = cmp_fn(input, initial)

    if isinstance(where, Tensor):
        if initial is None:
            raise ValueError('initial value must be provided for where masks')
        where = where.broadcast_to(input.shape)
        initial = initial.broadcast_to(input.shape)
        input = ops.select(where, input, initial)
    return input


def amax(input, axis=None, keepdims=False, *, initial=None, where=None):
    r"""
    Return the maximum values along the given axis of the tensor.

    Args:
        input (Tensor[Number]): The input tensor.
        axis (Union[int, tuple(int), list(int), Tensor], optional): Specify the axis for computation. If ``None`` ,
            compute all elements in the `input` . Default ``None`` .
        keepdims (bool, optional): Whether the output tensor has dim retained. Default ``False`` .

    Keyword Args:
        initial (scalar, optional): Initial value for the maximum. Default ``None`` .
        where (Tensor[bool], optional): Specifies the range over which to compute the maximum values. The shape of this
            tensor must be broadcastable to the shape of `input` . An `initial` value must be specified. Default
            ``None`` , indicating that all elements are to be computed.


    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[9, 3, 4, 5],
        ...                           [5, 2, 7, 4],
        ...                           [8, 1, 3, 6]])
        >>> # case 1: By default, compute the maximum of all elements.
        >>> mindspore.ops.amax(input)
        Tensor(shape=[], dtype=Int64, value= 9)
        >>>
        >>> # case 2: Compute maximum along axis 1.
        >>> mindspore.ops.amax(input, axis=1)
        Tensor(shape=[3], dtype=Int64, value= [9, 7, 8])
        >>>
        >>> # case 3: If keepdims=True, the output shape will be same of that of the input.
        >>> mindspore.ops.amax(input, axis=1, keepdims=True)
        Tensor(shape=[3, 1], dtype=Int64, value=
        [[9],
         [7],
         [8]])
        >>>
        >>> # case 4: Use "where" to include only specific elements in computing the maximum.
        >>> where = mindspore.tensor([[0, 0, 1, 0],
        ...                           [0, 0, 1, 1],
        ...                           [1, 1, 1, 0]], dtype=mindspore.bool)
        >>> mindspore.ops.amax(input, axis=1, keepdims=True, initial=0, where=where)
        Tensor(shape=[3, 1], dtype=Int64, value=
        [[4],
         [7],
         [8]])
        >>>
        >>> # case 5: The shape of "where" must be broadcast compatible with input.
        >>> where = mindspore.tensor([[False],
        ...                           [False],
        ...                           [False]])
        >>> mindspore.ops.amax(input, axis=0, keepdims=True, initial=0, where=where)
        Tensor(shape=[1, 4], dtype=Int64, value=
        [[0, 0, 0, 0]])
    """
    if axis is None:
        axis = ()
    input = _init_and_select_elem(input, initial, where, ops.maximum)
    return _get_cache_prim(P.ReduceMax)(keepdims)(input, axis)


def amax_ext(input, dim=(), keepdim=False):
    r"""
    Compute the maximum value of all elements along the specified dimension.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): Input tensor.
        dim (Union[int, tuple(int), list(int)], optional): The dimension to be reduced,
            when the `dim` is `()`, all dimensions are reduced. Default ``()``.
        keepdim (bool, optional): Whether the output tensor retains the dimension `dim`. Default ``False``.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> x = mindspore.tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
        >>> output = mindspore.ops.function.math_func.amax_ext(x, dim=1, keepdim=True)
        >>> print(output.shape)
        (3, 1, 5, 6)
        >>> # case 1: Reduce a dimension by the maximum value of all elements in the dimension.
        >>> x = mindspore.tensor(np.array([[[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]],
        ... [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
        ... [[7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9]]]),
        ... mindspore.float32)
        >>> output = mindspore.ops.function.math_func.amax_ext(x)
        >>> print(output)
        9.0
        >>> print(output.shape)
        ()
        >>> # case 2: Reduce a dimension along axis 0.
        >>> output = mindspore.ops.function.math_func.amax_ext(x, dim=0, keepdim=True)
        >>> print(output)
        [[[7. 7. 7. 7. 7. 7.]
          [8. 8. 8. 8. 8. 8.]
          [9. 9. 9. 9. 9. 9.]]]
        >>> # case 3: Reduce a dimension along axis 1.
        >>> output = mindspore.ops.function.math_func.amax_ext(x, dim=1, keepdim=True)
        >>> print(output)
        [[[3. 3. 3. 3. 3. 3.]]
        <BLANKLINE>
         [[6. 6. 6. 6. 6. 6.]]
        <BLANKLINE>
         [[9. 9. 9. 9. 9. 9.]]]
        >>> # case 4: Reduce a dimension along axis 2.
        >>> output = mindspore.ops.function.math_func.amax_ext(x, dim=2, keepdim=True)
        >>> print(output)
        [[[1.]
          [2.]
          [3.]]
        <BLANKLINE>
         [[4.]
          [5.]
          [6.]]
        <BLANKLINE>
         [[7.]
          [8.]
          [9.]]]
    """
    return reduce_max_impl(input, dim, keepdim)


def amin_ext(input, dim=(), keepdim=False):
    r"""
    Compute the minimum value of all elements along the specified dimension.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): The input tensor.
        dim (Union[int, tuple(int), list(int)], optional): Specify the dimension for computation,
            when the `dim` is `()`, all dimensions are reduced. Default ``()``.
        keepdim (bool, optional): Whether the output tensor retains the dimension `dim`. Default ``False``.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> x = mindspore.tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
        >>> output = mindspore.ops.function.math_func.amin_ext(x, 1, keepdim=True)
        >>> result = output.shape
        >>> print(result)
        (3, 1, 5, 6)
        >>> # case 1: Reduce a dimension by the minimum value of all elements in the dimension.
        >>> x = mindspore.tensor(np.array([[[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]],
        ... [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
        ... [[7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9]]]),
        ... mindspore.float32)
        >>> output = mindspore.ops.function.math_func.amin_ext(x)
        >>> print(output)
        1.0
        >>> print(output.shape)
        ()
        >>> # case 2: Reduce a dimension along axis 0.
        >>> output = mindspore.ops.function.math_func.amin_ext(x, 0, True)
        >>> print(output)
        [[[1. 1. 1. 1. 1. 1.]
          [2. 2. 2. 2. 2. 2.]
          [3. 3. 3. 3. 3. 3.]]]
        >>> # case 3: Reduce a dimension along axis 1.
        >>> output = mindspore.ops.function.math_func.amin_ext(x, 1, True)
        >>> print(output)
        [[[1. 1. 1. 1. 1. 1.]]
        <BLANKLINE>
         [[4. 4. 4. 4. 4. 4.]]
        <BLANKLINE>
         [[7. 7. 7. 7. 7. 7.]]]
        >>> # case 4: Reduce a dimension along axis 2.
        >>> output = mindspore.ops.function.math_func.amin_ext(x, 2, True)
        >>> print(output)
        [[[1.]
          [2.]
          [3.]]
        <BLANKLINE>
         [[4.]
          [5.]
          [6.]]
        <BLANKLINE>
         [[7.]
          [8.]
          [9.]]]
    """
    return reduce_min_impl(input, dim, keepdim)


def mean(x, axis=None, keep_dims=False):
    r"""
    Compute the mean(s) of the tensor along the specified axis(axes).

    Args:
        x (Tensor[Number]): The input tensor.
        axis (Union[int, tuple(int), list(int), Tensor]): Specify the axis(axes) for computation. If ``None`` , compute
            all elements in the `input` .
        keep_dims (bool): Whether the output tensor has dim retained. Default ``False`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[9, 3, 4, 5],
        ...                           [5, 2, 7, 4],
        ...                           [8, 1, 3, 6]])
        >>> # case 1: By default, compute the mean of all elements.
        >>> mindspore.ops.mean(input)
        Tensor(shape=[], dtype=Int64, value= 4)
        >>>
        >>> # case 2: Compute the mean along axis 1.
        >>> mindspore.ops.mean(input, axis=1)
        Tensor(shape=[3], dtype=Int64, value= [5, 4, 4])
        >>>
        >>> # case 3: If keep_dims=True, the output shape will be same of that of the input.
        >>> mindspore.ops.mean(input, axis=1, keep_dims=True)
        Tensor(shape=[3, 1], dtype=Int64, value=
        [[5],
         [4],
         [4]])
    """
    if axis is None:
        axis = ()
    return _get_cache_prim(P.ReduceMean)(keep_dims)(x, axis)


def mean_ext(input, axis=None, keep_dims=False, dtype=None):
    r"""
    Reduces all dimension of a tensor by averaging all elements in the dimension, by default.
    And reduce a dimension of `input` along the specified `axis`. `keep_dims`
    determines whether the dimensions of the output and input are the same.

    Note:
        The `axis` with tensor type is only used for compatibility with older versions and is not recommended.

    Args:
        input (Tensor[Number]): The input tensor. The dtype of the tensor to be reduced is number.
            :math:`(N, *)` where :math:`*` means, any number of additional dimensions.
        axis (Union[int, tuple(int), list(int), Tensor]): The dimensions to reduce. Default: ``None`` ,
            reduce all dimensions. Only constant value is allowed. Assume the rank of `input` is r,
            and the value range is [-r,r).
        keep_dims (bool): If ``True`` , keep these reduced dimensions and the length is 1.
            If ``False`` , don't keep these dimensions. Default: ``False`` .
        dtype (:class:`mindspore.dtype`): The desired data type of returned Tensor. Default: ``None`` .

    Returns:
        Tensor, has the same data type as the `input`.

        - If `axis` is ``None`` , and `keep_dims` is ``False`` ,
          the output is a 0-D tensor representing the product of all elements in the input tensor.
        - If `axis` is int, set as 1, and `keep_dims` is ``False`` ,
          the shape of output is :math:`(input_0, input_2, ..., input_R)`.
        - If `axis` is tuple(int), set as (1, 2), and `keep_dims` is ``False`` ,
          the shape of output is :math:`(input_0, input_3, ..., input_R)`.
        - If `axis` is 1-D Tensor, set as [1, 2], and `keep_dims` is ``False`` ,
          the shape of output is :math:`(input_0, input_3, ..., input_R)`.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If `axis` is not one of the following: int, tuple, list or Tensor.
        TypeError: If `keep_dims` is not a bool.
        ValueError: If `axis` is out of range.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> x = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
        >>> output = ops.function.math_func.mean_ext(x, 1, keep_dims=True)
        >>> result = output.shape
        >>> print(result)
        (3, 1, 5, 6)
        >>> # case 1: Reduces a dimension by averaging all elements in the dimension.
        >>> x = Tensor(np.array([[[2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2]],
        ... [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
        ... [[6, 6, 6, 6, 6, 6], [8, 8, 8, 8, 8, 8], [10, 10, 10, 10, 10, 10]]]),
        ... mindspore.float32)
        >>> output = ops.function.math_func.mean_ext(x)
        >>> print(output)
        5.0
        >>> print(output.shape)
        ()
        >>> # case 2: Reduces a dimension along the axis 0
        >>> output = ops.function.math_func.mean_ext(x, 0, True)
        >>> print(output)
        [[[4. 4. 4. 4. 4. 4.]
        [5. 5. 5. 5. 5. 5.]
        [6. 6. 6. 6. 6. 6.]]]
        >>> # case 3: Reduces a dimension along the axis 1
        >>> output = ops.function.math_func.mean_ext(x, 1, True)
        >>> print(output)
        [[[2. 2. 2. 2. 2. 2.]]
        [[5. 5. 5. 5. 5. 5.]]
        [[8. 8. 8. 8. 8. 8.]]]
        >>> # case 4: Reduces a dimension along the axis 2
        >>> output = ops.function.math_func.mean_ext(x, 2, True)
        >>> print(output)
        [[[ 2.]
        [ 2.]
        [ 2.]]
        [[ 4.]
        [ 5.]
        [ 6.]]
        [[ 6.]
        [ 8.]
        [10.]]]
        """
    return mean_ext_op(input, axis, keep_dims, dtype)


def prod(input, axis=None, keep_dims=False, dtype=None):
    r"""
    Return the product(s) of the tensor along the specified axis(axes).

    Args:
        input (Tensor[Number]): The input tensor.
        axis (Union[int, tuple(int), list(int), Tensor]): Specify the axis(axes) for computation. If ``None`` , compute
            all elements in the `input` . Default ``None`` .
        keep_dims (bool): Whether the output tensor has dim retained. Default ``False`` .
        dtype (:class:`mindspore.dtype`): The data type returned. Default ``None`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[9, 3, 4, 5],
        ...                           [5, 2, 7, 4],
        ...                           [8, 1, 3, 6]])
        >>> # case 1: By default, compute the product of all elements.
        >>> mindspore.ops.prod(input)
        Tensor(shape=[], dtype=Int64, value= 21772800)
        >>>
        >>> # case 2: Compute the product along axis 1.
        >>> mindspore.ops.prod(input, axis=1)
        Tensor(shape=[3], dtype=Int64, value= [540, 280, 144])
        >>>
        >>> # case 3: If keep_dims=True, the output shape will be same of that of the input.
        >>> mindspore.ops.prod(input, axis=1, keep_dims=True)
        Tensor(shape=[3, 1], dtype=Int64, value=
        [[540],
         [280],
         [144]])
    """
    if not isinstance(axis, (tuple, list, Tensor)):
        return prod_ext_op(input, axis, keep_dims, dtype)
    if dtype is not None:
        input = input.astype(dtype)
    return _get_cache_prim(P.ReduceProd)(keep_dims)(input, axis)


def _multi_svd_norm(x, row_axis, col_axis, op):
    """_multi_svd_norm for norm."""
    y = _moveaxis(x.astype(mstype.float32), (row_axis, col_axis), (-2, -1))
    svd_res = ops.svd(y, compute_uv=False)
    if op == 'amax':
        return ops.amax(svd_res, axis=-1)
    if op == 'amin':
        return ops.amin(svd_res, axis=-1)
    if op == 'sum':
        return ops.sum(svd_res, dim=-1)
    raise ValueError(f"For svd_norm, the op input must be one of ['amax', 'amin', 'sum'], but got f{op}")


def _reshape_matrix_norm(input, res, dim, keepdims):
    """reshape res of matrix_norm if keepdims is True."""
    if keepdims:
        res_shape = list(input.shape)
        res_shape[dim[0]] = 1
        res_shape[dim[1]] = 1
        res = res.reshape(res_shape)
    return res


def _normalize_axis_index(axis, ndim):
    """normalize_axis_index for norm."""
    # pylint: disable=chained-comparison
    if axis >= 0 and axis < ndim:
        return axis
    # pylint: disable=chained-comparison
    if axis < 0 and axis >= -ndim:
        return ndim + axis
    raise ValueError('For norm, the dim is out of range.')


@_primexpr
def _get_perm_for_norm(x_ndim, source, destination):
    destination = tuple(_normalize_axis_index(ax, x_ndim) for ax in destination)
    source = tuple(_normalize_axis_index(ax, x_ndim) for ax in source)
    perm = [n for n in range(x_ndim) if n not in source]
    for dest, src in sorted(zip(destination, source)):
        perm.insert(dest, src)
    perm = tuple(perm)
    return perm


def _moveaxis(x, source, destination):
    perm = _get_perm_for_norm(x.ndim, source, destination)
    return ops.transpose(x, perm)


@_primexpr
def _check_axis(axis, ord, ndim):
    """axis check"""
    if axis is None:
        axis = tuple(range(ndim))
        if (ord is None) or (ord == 'fro' and ndim == 2) or (ord == 2 and ndim == 1):
            return axis, True
        return axis, False
    if isinstance(axis, int):
        axis = (axis,)
    elif isinstance(axis, tuple):
        if len(axis) > 2:
            raise ValueError("For norm, the dimensions is out of range.")
    else:
        raise TypeError(f'For norm, the dim should be int or tuple of int, but got {type(axis)}')
    return axis, False


@_primexpr
def _check_ord(ord, axis):
    if len(axis) == 1:
        if isinstance(ord, str):
            raise TypeError(f"For norm, ord mode can not be str for vectors, but got {ord}.")
    elif len(axis) == 2:
        if ord not in [2, -2, 1, -1, float('inf'), -float('inf'), 'fro', 'nuc', None]:
            raise ValueError(f"For norm, the ord mode must be in "
                             f"[2, -2, 1, -1, float('inf'), -float('inf'), 'fro', 'nuc', None] for matrices, "
                             f"but got {ord}.")


def _check_dtype(d1, d2):
    if mstype.float32 in (d1, d2):
        return mstype.float32
    if d1 == d2:
        return d1
    raise ValueError('the dtype is not supported.')


@_primexpr
def _check_last_dim_shape_eq(a, b):
    if a.shape[-1] != b.shape[-1]:
        raise ValueError('shapes are not aligned')


def _complex_square(A):
    """calculate square with complex or not"""
    if ops.is_complex(A):
        return ops.conj(A) * A
    return ops.square(A)


def _reshape_keepdim_matrix_norm(ret, in_shape, dim):
    """Reshape matrix norm result to keep reduced dims."""
    ret_shape = list(in_shape)
    ret_shape[dim[0]] = 1
    ret_shape[dim[1]] = 1
    return ret.reshape(ret_shape)


def _norm_immediate(A, dim, keepdim, ndim):
    ret = ops.sqrt(ops.reduce_sum(_complex_square(A), dim))
    if keepdim:
        ret = ret.reshape(ndim * [1])
    return ret


def _norm_matrix_int_ord(A, ord, dim, keepdim, ndim):
    """Compute matrix norm for integer `ord` (e.g. 1, -1, 2, -2) along two axes."""
    row_axis, col_axis = dim
    row_axis = _normalize_axis_index(row_axis, ndim)
    col_axis = _normalize_axis_index(col_axis, ndim)

    if ord in (1, -1):
        # After summing along row_axis, the rank is reduced by 1, so col_axis may shift.
        if col_axis > row_axis:
            col_axis -= 1
        sum_abs = A.abs().sum(row_axis)
        ret = (ops.max(sum_abs, axis=col_axis)[0] if ord == 1 else ops.min(sum_abs, axis=col_axis)[0])
    elif ord == 2:
        ret = _multi_svd_norm(A, row_axis, col_axis, 'amax')
    elif ord == -2:
        ret = _multi_svd_norm(A, row_axis, col_axis, 'amin')
    else:
        raise ValueError(f"For norm, the ord {ord} are not support for matrices.")

    if keepdim:
        ret = _reshape_keepdim_matrix_norm(ret, A.shape, dim)
    return ret


def _norm_vector_int_ord(A, ord, dim, keepdim):
    if ord == 0:
        return (A != 0).astype(A.dtype).sum(axis=dim, keepdims=keepdim)
    if ord > 0:
        _lp_norm = _get_cache_prim(ops.LpNorm)(dim, ord, keepdim)
        return _lp_norm(A)
    return ops.sum(ops.abs(A).pow(ord), dim=dim, keepdim=keepdim).pow(1.0 / ord)


def _norm_vector_other_ord(A, ord, dim, keepdim):
    """Compute vector norm for special/other `ord` values (inf, -inf, 0, None, tensor ord)."""
    if ord == float('inf'):
        return ops.max(ops.abs(A), axis=dim[0], keepdims=keepdim)[0]
    if ord == -float('inf'):
        return ops.min(ops.abs(A), axis=dim[0], keepdims=keepdim)[0]
    if ord == 0:
        # Zero norm
        return (A != 0).astype(A.dtype).sum(axis=dim, keepdims=keepdim)
    if ord is None:
        # special case for speedup
        s = _complex_square(A)
        reduce_sum = _get_cache_prim(ops.ReduceSum)(keepdim)
        return ops.sqrt(reduce_sum(s, dim))

    # None of the str-type keywords for ord ('fro', 'nuc') are valid for vectors.
    absx = ops.abs(A)
    absx **= ord
    reduce_sum = _get_cache_prim(ops.ReduceSum)(keepdim)
    ret = reduce_sum(absx, dim)
    if isinstance(ord, Tensor):
        ret **= ops.reciprocal(ord)
    else:
        ret **= 1 / ord
    return ret


def _norm_matrix_other_ord(A, ord, dim, keepdim, ndim):
    """Compute matrix norm for other `ord` values (inf, -inf, 'nuc', None/'fro')."""
    row_axis, col_axis = dim
    row_axis = _normalize_axis_index(row_axis, ndim)
    col_axis = _normalize_axis_index(col_axis, ndim)

    if row_axis == col_axis:
        raise ValueError('For norm, the elements of dim can not be duplicate.')

    if ord in (float('inf'), -float('inf')):
        # After summing along col_axis, the rank is reduced by 1, so row_axis may shift.
        if row_axis > col_axis:
            row_axis -= 1
        sum_abs = ops.reduce_sum(ops.abs(A), col_axis)
        ret = (ops.max(sum_abs, axis=row_axis)[0] if ord == float('inf') else ops.min(sum_abs, axis=row_axis)[0])
    elif ord == 'nuc':
        ret = _multi_svd_norm(A, row_axis, col_axis, 'sum')
    else:
        # ord is None or 'fro'
        ret = ops.sqrt(ops.reduce_sum(_complex_square(A), dim))

    if keepdim:
        ret = _reshape_keepdim_matrix_norm(ret, A.shape, dim)
    return ret


def norm(A, ord=None, dim=None, keepdim=False, *, dtype=None):
    r"""
    Compute the matrix norm or vector norm of the tensor along a specified dimension.

    `ord` is the calculation mode of norm. The following norm modes are supported.

    ====================== ================================ ==========================================
    `ord`                   norm for matrices               norm for vectors
    ====================== ================================ ==========================================
    `None` (default)        Frobenius norm                   `2`-norm (see below)
    `'fro'`                 Frobenius norm                   -- not supported --
    `'nuc'`                 nuclear norm                     -- not supported --
    `inf`                   :math:`max(sum(abs(x), dim=1))`  :math:`max(abs(x))`
    `-inf`                  :math:`min(sum(abs(x), dim=1))`  :math:`min(abs(x))`
    `0`                     -- not supported --              :math:`sum(x != 0)`
    `1`                     :math:`max(sum(abs(x), dim=0))`  as below
    `-1`                    :math:`min(sum(abs(x), dim=0))`  as below
    `2`                     largest singular value           as below
    `-2`                    smallest singular value          as below
    other `int` or `float`  -- not supported --              :math:`sum(abs(x)^{ord})^{(1 / ord)}`
    ====================== ================================ ==========================================

    Args:
        A (Tensor): The input tensor.
        ord (Union[int, float, inf, -inf, 'fro', 'nuc'], optional): Specify the kind of norm to take. Default
            ``None`` .
        dim (Union[int, Tuple(int)], optional): Specify the dimension for computation. Default ``None`` .

            - If `dim` is int, calculate the vector norm.

            - if `dim` is a 2-tuple, calculate the matrix norm.

            - If `dim` is None and `ord` is ``None`` , flattened `A` to 1D and calculate 2-norm of the vector.

            - If `dim` is None and `ord` is not ``None``, `A` must be 1D or 2D.

        keepdim (bool): Whether the output tensor has dim retained. Default ``False`` .

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): The data type returned. The data type returned. When set, `A` will
            be converted to the specified data type, before calculaing. Default ``None`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Note:
        - Currently, complex numbers are not supported.
        - Depending on the input range of values, the Ascend backend calculation results may have precision errors.

    Examples:
        >>> import mindspore
        >>> # Vector norms:
        >>> A = mindspore.tensor([3., 4., 12.])
        >>> mindspore.ops.norm(A)
        Tensor(shape=[], dtype=Float32, value= 13)
        >>> mindspore.ops.norm(A, ord=1)
        Tensor(shape=[], dtype=Float32, value= 19)
        >>> mindspore.ops.norm(A, ord=0)
        Tensor(shape=[], dtype=Float32, value= 3)
        >>>
        >>> # Matrix norms:
        >>> A = mindspore.tensor([[1., 2., 3.],
        ...                       [4., 5., 7.]])
        >>> mindspore.ops.norm(A)  # Frobenius norm
        Tensor(shape=[], dtype=Float32, value= 10.198)
        >>> mindspore.ops.norm(A, ord='nuc')  # nuclear norm
        Tensor(shape=[], dtype=Float32, value= 10.7625)
        >>> mindspore.ops.norm(A, ord=1)  # 1-norm
        Tensor(shape=[], dtype=Float32, value= 10)
        >>>
        >>> # Batched vector norm:
        >>> mindspore.ops.norm(A, dim=1)
        Tensor(shape=[2], dtype=Float32, value= [ 3.74165726e+00,  9.48683262e+00])
    """
    ndim = A.ndim
    dim, immediate = _check_axis(dim, ord, ndim)
    _check_ord(ord, dim)
    if dtype is not None:
        A = ops.cast(A, dtype)
    if immediate:
        # Immediately handle some default, simple, fast, and common cases.
        return _norm_immediate(A, dim, keepdim, ndim)

    if len(dim) == 2:
        if isinstance(ord, int):
            return _norm_matrix_int_ord(A, ord, dim, keepdim, ndim)
        return _norm_matrix_other_ord(A, ord, dim, keepdim, ndim)

    if len(dim) == 1:
        if isinstance(ord, int):
            return _norm_vector_int_ord(A, ord, dim, keepdim)
        return _norm_vector_other_ord(A, ord, dim, keepdim)
    return None


@_primexpr
def _check_vector_norm_axis(axis, ndim):
    """vector_norm axis check"""
    if (not isinstance(axis, int)) and (not isinstance(axis, tuple)) and (axis is not None):
        raise TypeError(f'For vector_norm , the dim must be tuple or int, but got {type(axis)}')

    if axis is None:
        axis = tuple(range(ndim))
    if isinstance(axis, int):
        axis = (axis,)

    dim = []
    for elem_dim in axis:
        elem_dim = _normalize_axis_index(elem_dim, ndim)
        if elem_dim in dim:
            raise ValueError('For vector_norm, the elements of axis can not be duplicate.')
        dim.append(elem_dim)
    tuple_dim = tuple(dim)
    return tuple_dim


@_primexpr
def _check_vector_norm_ord(ord):
    """vector_norm ord check"""
    if ord not in [0, 2, float('inf'), -float('inf')] and not isinstance(ord, (int, float)):
        raise ValueError(f"For vector_norm, the ord mode must be in [0, 2, float('inf'), -float('inf')] "
                         f"or must be int or float, but got {ord}.")


def _compute_vector_norm_inf(x, dim, keepdims, norm_func):
    """compute vector norm of `x` when ord is ``inf`` or ``-inf`` """
    if len(dim) == 1:
        ret_norm = norm_func(ops.abs(x), axis=dim[0], keepdims=keepdims)[0]
    else:
        start_dim = min(dim)
        end_dim = max(dim)
        flatten_x = ops.flatten(x, start_dim=start_dim, end_dim=end_dim)
        ret_norm = norm_func(ops.abs(flatten_x), axis=start_dim, keepdims=False)[0]
        if keepdims is True:
            ret_shape = list(x.shape)
            for i in dim:
                ret_shape[i] = 1
            ret_norm = ret_norm.reshape(ret_shape)
    return ret_norm


@_primexpr
def _check_vector_norm_inputs(x, ord):
    """vector_norm inputs check"""
    if not isinstance(x, (Tensor, Tensor_)):
        raise TypeError(f"For `vector_norm`, the `x` must be Tensor!, but get {type(x)}.")

    if not isinstance(ord, (bool, int, float)):
        raise ValueError(f"For `vector_norm`, the ord mode must be one of [bool, int, float, inf, -inf], "
                         f"but got {ord}.")


def vector_norm_ext(x, ord=2, dim=None, keepdim=False, *, dtype=None):
    r"""
    Returns the vector norm of the given tensor on the specified dimensions.

    `ord` is the calculation mode of norm. The following norm modes are supported.

    ==========================      ==========================================
    `ord`                           norm for vectors
    ==========================      ==========================================
    ``2`` (Default)                 ``2``-norm (see below)
    ``inf``                         :math:`max(abs(x))`
    ``-inf``                        :math:`min(abs(x))`
    ``0``                           :math:`sum(x!=0)`
    other ``int`` or ``float``      :math:`sum(abs(x)^{ord})^{(1 / ord)}`
    ==========================      ==========================================

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        x (Tensor): Tensor of shape :math:`(*)` where * is zero s more batch dimensions.
        ord (Union[bool, int, float, inf, -inf], optional): norm's mode. refer to the table above for
            behavior. Default: ``2`` .
        dim (Union[int, List(int), Tuple(int)], optional): The dimensions along which to perform the vector norm
            calculation. Default: ``None`` .

            - When `dim` is an integer, a list or a tuple, the norm calculation will be performed across these specified
              dimensions, while the remaining dimensions will be considered as batch dimensions.

            - When `dim` is None, the norm will be calculated after flattening the Tensor `x` .

        keepdim (bool): whether the output Tensor retains the original dimension. Default: ``False`` .

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): When set, `x` will be converted to the specified type,
            `dtype` before execution, and dtype of returned Tensor will also be `dtype`.
            When `dtype` is ``None`` , the dtype of `x` is preserved. Default: ``None`` .

    Returns:
        Tensor, the result of norm calculation on the specified dimension, `dim`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If `dim` is neither an int nor a list or tuple.
        ValueError: If `ord` is not in [bool, int, float, inf, -inf].
        ValueError: The elements of `dim` are duplicate.
        ValueError: If any elements of `dim` is out of range.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> x = ms.ops.arange(0, 12, dtype=ms.float32) - 6
        >>> print(ms.ops.vector_norm_ext(x, ord=2))
        12.083046
        >>> print(ms.ops.vector_norm_ext(x, ord=float('inf')))
        6.0
        >>> print(ms.ops.vector_norm_ext(x, ord=float('-inf')))
        0.0
        >>> print(ms.ops.vector_norm_ext(x, ord=0))
        11.0
        >>> print(ms.ops.vector_norm_ext(x, ord=4.5))
        7.2243643
    """
    _check_vector_norm_inputs(x, ord)
    if float(ord) in [0.0, 1.0, 2.0, 3.0]:
        return linalg_vector_norm_op(x, float(ord), dim, keepdim, dtype)

    if x.dtype in [mstype.bfloat16, mstype.float16, mstype.float32]:
        if dtype is None:
            return lp_norm_v2_op(x, ord, dim, keepdim, 0.0)
        return ops.cast(lp_norm_v2_op(x, ord, dim, keepdim, 0.0), dtype)

    cast_dtype = x.dtype if dtype is None else dtype
    x = ops.cast(x, mstype.float32)
    return ops.cast(lp_norm_v2_op(x, ord, dim, keepdim, 0.0), cast_dtype)


def matrix_norm_ext(A, ord='fro', dim=(-2, -1), keepdim=False, *, dtype=None):
    r"""
    Returns the matrix norm of a given tensor on the specified dimensions.

    `ord` is the calculation mode of norm. The following norm modes are supported.

    ====================== ================================
    `ord`                  norm for matrix
    ====================== ================================
    ``'fro'`` (Default)    Frobenius norm
    ``'nuc'``              nuclear norm
    ``inf``                :math:`max(sum(abs(x), dim=1))`
    ``-inf``               :math:`min(sum(abs(x), dim=1))`
    ``1``                  :math:`max(sum(abs(x), dim=0))`
    ``-1``                 :math:`min(sum(abs(x), dim=0))`
    ``2``                  largest singular value
    ``-2``                 smallest singular value
    ====================== ================================

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        A (Tensor): Tensor of shape :math:`(*, m, n)` where * is zero or more batch dimensions.
        ord (Union[int, inf, -inf, 'fro', 'nuc'], optional): norm's mode. refer to the table above for
            behavior. Default: ``'fro'`` .
        dim (Tuple(int, int), optional): calculate the dimension of the matrix norm.
            Default: ``(-2, -1)`` .
        keepdim (bool): whether the output Tensor retains the original dimension. Default: ``False`` .

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): When set, `A` will be converted to the specified type,
            `dtype`, before execution, and dtype of returned Tensor will also be `dtype`.
            When `dtype` is ``None`` , the dtype of `A` is preserved. Default: ``None`` .

    Returns:
        Tensor, the result of norm calculation on the specified dimension, `dim`.

    Raises:
        TypeError: If `dim` is not a tuple of int.
        ValueError: If the length of `dim` is not equal to 2.
        ValueError: If `ord` is not in [2, -2, 1, -1, float('inf'), float('-inf'), 'fro', 'nuc'].
        ValueError: If two elements of `dim` is same after normalize.
        ValueError: If any elements of `dim` is out of range.

    Note:
        Dynamic shape, Dynamic rank and mutable input is not supported in `graph mode (mode=mindspore.GRAPH_MODE)
        <https://www.mindspore.cn/tutorials/en/master/compile/static_graph.html>`_.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> A = ms.ops.arange(0, 12, dtype=ms.float32).reshape(3, 4)
        >>> print(ms.ops.matrix_norm_ext(A, ord='fro'))
        22.494444
        >>> print(ms.ops.matrix_norm_ext(A, ord='nuc'))
        24.364643
        >>> print(ms.ops.matrix_norm_ext(A, ord=float('inf')))
        38.0
        >>> print(ms.ops.matrix_norm_ext(A, ord=float('-inf')))
        6.0
    """
    ndim = A.ndim
    row_axis, col_axis = _check_matrix_norm_axis(dim, ndim)
    _check_matrix_norm_ord(ord)
    if ord == 'fro':
        return vector_norm_ext(A, 2, dim, keepdim, dtype=dtype)
    if ord == 'nuc':
        res = _multi_svd_norm(A, row_axis, col_axis, 'sum')
        return _reshape_matrix_norm(A, res, dim, keepdim)
    if ord == 2:
        res = _multi_svd_norm(A, row_axis, col_axis, 'amax')
        return _reshape_matrix_norm(A, res, dim, keepdim)
    if ord == -2:
        res = _multi_svd_norm(A, row_axis, col_axis, 'amin')
        return _reshape_matrix_norm(A, res, dim, keepdim)
    if ord in [float('inf'), -float('inf')]:
        row_axis, col_axis = col_axis, row_axis
    if not keepdim and col_axis > row_axis:
        col_axis -= 1
    if ord < 0:
        return ops.amin(vector_norm_ext(A, 1, row_axis, keepdim, dtype=dtype), col_axis, keepdim)
    return ops.amax(vector_norm_ext(A, 1, row_axis, keepdim, dtype=dtype), col_axis, keepdim)


@_primexpr
def _check_linalg_norm_input(dim, ord, ndim):
    """dim check"""
    if dim is None:
        if ord is not None and ndim > 2:
            raise ValueError("For `linalg.norm`, the input must be 1D or 2D when `ord` is specified but `dim` is None.")
        dim = tuple(range(ndim))
        if (ord is None) or (ord == 'fro' and ndim == 2) or (ord == 2 and ndim == 1):
            return dim, True
        return dim, False
    if isinstance(dim, int):
        dim = (dim,)
    elif isinstance(dim, (list, tuple)):
        if len(dim) > 2:
            raise ValueError("For `linalg.norm`, the length of `dim` must be 1 or 2 when dim is not None",
                             f"but got {len(dim)}.")
    else:
        raise TypeError(f'For `linalg.norm`, the dim should be int, list of int or tuple of int, but got {type(dim)}')
    return dim, False


def linalg_norm(A, ord=None, dim=None, keepdim=False, *, dtype=None):
    r"""
    Returns the matrix norm or vector norm of a given tensor.

    `ord` is the calculation mode of norm. The following norm modes are supported.

    ====================== ================================ ==========================================
    `ord`                   norm for matrices               norm for vectors
    ====================== ================================ ==========================================
    `None` (default)        Frobenius norm                   `2`-norm (see below)
    `'fro'`                 Frobenius norm                   -- not supported --
    `'nuc'`                 nuclear norm                     -- not supported --
    `inf`                   :math:`max(sum(abs(x), dim=1))`  :math:`max(abs(x))`
    `-inf`                  :math:`min(sum(abs(x), dim=1))`  :math:`min(abs(x))`
    `0`                     -- not supported --              :math:`sum(x != 0)`
    `1`                     :math:`max(sum(abs(x), dim=0))`  as below
    `-1`                    :math:`min(sum(abs(x), dim=0))`  as below
    `2`                     largest singular value           as below
    `-2`                    smallest singular value          as below
    other `int` or `float`  -- not supported --              :math:`sum(abs(x)^{ord})^{(1 / ord)}`
    ====================== ================================ ==========================================

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        A (Tensor): Tensor of shape :math:`(*, n)` or :math:`(*, m, n)` where * is zero or more batch dimensions.
        ord (Union[int, float, inf, -inf, 'fro', 'nuc'], optional): norm's mode. Refer to the table above for
            behavior. Default: ``None`` .
        dim (Union[int, Tuple(int)], optional): calculate the dimension of vector norm or matrix norm.
            Default: ``None`` .

            - When `dim` is int, it will be calculated by vector norm.

            - When `dim` is a 2-tuple, it will be calculated by matrix norm.

            - If `dim` is ``None`` and `ord` is ``None``, `A` will be flattened to 1D and the 2-norm
              of the vector will be calculated.

            - If `dim` is ``None`` and `ord` is not ``None``, `A` must be 1D or 2D.

        keepdim (bool): whether the output Tensor retains the original dimension. Default: ``False`` .

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): When set, `A` will be converted to the specified type,
            `dtype`, before execution, and dtype of returned Tensor will also be `dtype`. Default: ``None`` .

    Returns:
        Tensor, the result of norm calculation on the specified dimension, `dim`, has the same dtype as `A`.

    Raises:
        ValueError: If `dim` is out of range.
        TypeError: If `dim` is neither an int nor a tuple of int.
        TypeError: If `A` is a vector and `ord` is a str.
        ValueError: If `A` is a matrices and `ord` is not in valid mode.
        ValueError: If two elements of `dim` is same after normalize.
        ValueError: If any elements of `dim` is out of range.

    Note:
        Dynamic shape, Dynamic rank and mutable input is not supported in `graph mode (mode=mindspore.GRAPH_MODE)
        <https://www.mindspore.cn/tutorials/en/master/compile/static_graph.html>`_.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import ops
        >>> data_range = ops.arange(-13, 13, dtype=ms.float32)
        >>> x = data_range[data_range != 0]
        >>> print(ops.function.math_func.linalg_norm(x))
        38.327538
        >>> print(ops.function.math_func.linalg_norm(x, 1))
        169.0
        >>> n = ops.arange(27, dtype=ms.float32).reshape(3, 3, 3)
        >>> print(ops.function.math_func.linalg_norm(n, dim=(1, 2)))
        [14.282857 39.76179  66.45299 ]
        >>> print(ops.function.math_func.linalg_norm(n[0, :, :]))
        14.282857
        >>> print(ops.function.math_func.linalg_norm(n[1, :, :]))
        39.76179
    """
    dim, immediate = _check_linalg_norm_input(dim, ord, A.ndim)
    if immediate:
        return vector_norm_ext(A, 2, dim, keepdim, dtype=dtype)
    if ord is not None:
        if ord in ['fro', 'nuc'] or (dim is not None and len(dim) == 2) or (dim is None and A.ndim == 2):
            return matrix_norm_ext(A, ord, dim, keepdim, dtype=dtype)
        return vector_norm_ext(A, ord, dim, keepdim, dtype=dtype)
    return vector_norm_ext(A, 2, dim, keepdim, dtype=dtype)


def norm_ext(input, p='fro', dim=None, keepdim=False, *, dtype=None):
    r"""
    Compute the matrix norm or vector norm of the tensor along a specified dimension.

    `p` is the calculation mode of norm. The following norm modes are supported.

    ====================== ================================ ==========================================
    `p`                     norm for matrices               norm for vectors
    ====================== ================================ ==========================================
    `'fro'`                 Frobenius norm                   -- not supported --
    `'nuc'`                 nuclear norm                     -- not supported --
    other `int` or `float`  -- not supported --              :math:`sum(abs(x)^{p})^{(1 / p)}`
    ====================== ================================ ==========================================

    Args:
        input (Tensor): The input tensor.
        p (Union[bool, int, float, inf, -inf, 'fro', 'nuc'], optional): Specify the kind of norm
         to take. Default ``fro`` .
        dim (Union[int, List(int), Tuple(int)], optional): Specify the dimension for computation.
            Default ``None`` .
        keepdim (bool, optional): Whether the output tensor has dim retained. Default ``False`` .

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): When set, `input` will be converted to the specified type,
            `dtype`, before calculating. Default ``None`` .

    Returns:
        Tensor

    Note:
        - Dynamic shape, Dynamic rank and mutable input is not supported in `graph mode (mode=mindspore.GRAPH_MODE)
          <https://www.mindspore.cn/tutorials/en/master/compile/static_graph.html>`_.
        - Depending on the input range of values, the Ascend backend calculation results may have precision errors.


    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> data_range = mindspore.ops.arange(-13, 13, dtype=mindspore.float32)
        >>> x = data_range[data_range != 0]
        >>> y = x.reshape(5, 5)
        >>> print(mindspore.ops.function.math_func.norm_ext(x, 2.0))
        38.327534
    """
    if not isinstance(input, (Tensor, Tensor_)):
        raise TypeError(f"For `norm_ext`, the `input` must be Tensor!, but get {type(input)}.")
    if isinstance(p, (bool, int, float)):
        return vector_norm_ext(input, p, dim, keepdim, dtype=dtype)
    if p == 'fro':
        if isinstance(dim, (list, tuple)) and len(dim) > 2:
            raise ValueError("For `norm_ext`, the size of `dim` cannot be greater than 2 "
                             "when the norm mode is `fro`.")
        return linalg_vector_norm_op(input, 2.0, dim, keepdim, dtype)
    if p == 'nuc':
        dim = tuple(range(input.ndim)) if dim is None else dim
        return matrix_norm_ext(input, p, dim, keepdim, dtype=dtype)
    raise ValueError(f"For `norm_ext`, the value of `p` must be one of [int, float, inf, -inf, 'fro', 'nuc',] "
                     f"but got `{p}`.")


def vector_norm(x, ord=2, axis=None, keepdims=False, *, dtype=None):
    r"""
    Returns the vector norm of the given tensor on the specified dimensions.

    `ord` is the calculation mode of norm. The following norm modes are supported.

    ==========================      ==========================================
    `ord`                           norm for vectors
    ==========================      ==========================================
    ``2`` (Default)                 ``2``-norm (see below)
    ``inf``                         :math:`max(abs(x))`
    ``-inf``                        :math:`min(abs(x))`
    ``0``                           :math:`sum(x!=0)`
    other ``int`` or ``float``      :math:`sum(abs(x)^{ord})^{(1 / ord)}`
    ==========================      ===========================================

    Args:
        x (Tensor): Tensor of shape :math:`(*, n)` where * is zero s more batch dimensions.
        ord (Union[int, float, inf, -inf], optional): norm's mode. refer to the table above for
            behavior. Default: ``2`` .
        axis (Union[int, Tuple(int)], optional): The dimensions along which to perform the vector norm calculation.
            Default: ``None`` .

            - When `axis` is int or a tuple, the norm calculation will be performed across these specified dimensions,
              while the remaining dimensions will be considered as batch dimensions.

            - When `dim` is None, the norm will be calculated after flattening the Tensor `x` .

        keepdims (bool): whether the output Tensor retains the original dimension. Default: ``False`` .

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): When set, `x` will be converted to the specified type,
            `dtype` before execution, and dtype of returned Tensor will also be `dtype`.
            When `dtype` is ``None`` , the dtype of `A` is preserved. Default: ``None`` .

    Returns:
        Tensor, the result of norm calculation on the specified dimension, `axis`, has the same dtype as `x`.

    Raises:
        TypeError: If `axis` is not an int or tuple.
        ValueError: If `ord` is not in [int, float, inf, -inf].
        ValueError: The elements of `axis` are duplicate.
        ValueError: If any elements of `axis` is out of range.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> x = ms.ops.arange(0, 12, dtype=ms.float32) - 6
        >>> print(ms.ops.vector_norm(x, ord=2))
        12.083046
        >>> print(ms.ops.vector_norm(x, ord=float('inf')))
        6.0
        >>> print(ms.ops.vector_norm(x, ord=float('-inf')))
        0.0
        >>> print(ms.ops.vector_norm(x, ord=0))
        11.0
        >>> print(ms.ops.vector_norm(x, ord=4.5))
        7.2243643
    """
    ndim = x.ndim
    dim = _check_vector_norm_axis(axis, ndim)
    _check_vector_norm_ord(ord)

    if dtype is not None:
        x = ops.cast(x, dtype)

    if ord == 2:
        s = _complex_square(x)
        reduce_sum = _get_cache_prim(ops.ReduceSum)(keepdims)
        return ops.sqrt(reduce_sum(s, dim))
    if ord == float('inf'):
        inf_norm = _compute_vector_norm_inf(x, dim, keepdims, ops.max)
        return inf_norm
    if ord == float('-inf'):
        inf_norm = _compute_vector_norm_inf(x, dim, keepdims, ops.min)
        return inf_norm
    if ord == 0:
        return (x != 0).astype(x.dtype).sum(axis=dim, keepdims=keepdims)
    # ord is other int or float
    abs_x = ops.abs(x)
    abs_x **= ord
    reduce_sum = _get_cache_prim(ops.ReduceSum)(keepdims)
    ret = reduce_sum(abs_x, dim)
    ret **= 1 / ord
    return ret


@_primexpr
def _check_matrix_norm_axis(axis, ndim):
    """matrix_norm axis check"""
    if not isinstance(axis, (list, tuple)):
        raise TypeError(f'For matrix_norm , the axis should be tuple of int, but got {type(axis)}')
    if len(axis) != 2:
        raise ValueError(f'For matrix_norm, the length of axis should be 2, but got {len(axis)}.')

    row_axis, col_axis = axis
    row_axis = _normalize_axis_index(row_axis, ndim)
    col_axis = _normalize_axis_index(col_axis, ndim)
    if row_axis == col_axis:
        raise ValueError('For matrix_norm, the elements of axis can not be duplicate.')
    return row_axis, col_axis


@_primexpr
def _check_matrix_norm_ord(ord):
    """matrix_norm ord check"""
    if ord not in [2, -2, 1, -1, float('inf'), float('-inf'), 'fro', 'nuc']:
        raise ValueError(f"For matrix_norm, the ord mode must be in "
                         f"[2, -2, 1, -1, float('inf'), float('-inf'), 'fro', 'nuc'] "
                         f"but got {ord}.")


def matrix_norm(A, ord='fro', axis=(-2, -1), keepdims=False, *, dtype=None):
    r"""
    Returns the matrix norm of a given tensor on the specified dimensions.

    `ord` is the calculation mode of norm. The following norm modes are supported.

    ====================== ================================
    `ord`                  norm for matrix
    ====================== ================================
    ``'fro'`` (Default)    Frobenius norm
    ``'nuc'``              nuclear norm
    ``inf``                :math:`max(sum(abs(x), dim=1))`
    ``-inf``               :math:`min(sum(abs(x), dim=1))`
    ``1``                  :math:`max(sum(abs(x), dim=0))`
    ``-1``                 :math:`min(sum(abs(x), dim=0))`
    ``2``                  largest singular value
    ``-2``                 smallest singular value
    ====================== ================================

    Args:
        A (Tensor): Tensor of shape :math:`(*, m, n)` where * is zero or more batch dimensions.
        ord (Union[int, inf, -inf, 'fro', 'nuc'], optional): norm's mode. refer to the table above for
            behavior. Default: ``'fro'`` .
        axis (Tuple(int, int), optional): calculate the dimension of the matrix norm.
            Default: ``(-2, -1)`` .
        keepdims (bool): whether the output Tensor retains the original dimension. Default: ``False`` .

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): When set, `A` will be converted to the specified type,
            `dtype`, before execution, and dtype of returned Tensor will also be `dtype`.
            When `dtype` is ``None`` , the dtype of `A` is preserved. Default: ``None`` .

    Returns:
        Tensor, the result of norm calculation on the specified dimension, `axis`, has the same dtype as `A`.

    Raises:
        TypeError: If `axis` is not a tuple of int.
        ValueError: If the length of `axis` is not equal to 2.
        ValueError: If `ord` is not in [2, -2, 1, -1, float('inf'), float('-inf'), 'fro', 'nuc'].
        ValueError: If two elements of `axis` is same after normalize.
        ValueError: If any elements of `axis` is out of range.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> A = ms.ops.arange(0, 12, dtype=ms.float32).reshape(3, 4)
        >>> print(ms.ops.matrix_norm(A, ord='fro'))
        22.494444
        >>> print(ms.ops.matrix_norm(A, ord='nuc'))
        24.364643
        >>> print(ms.ops.matrix_norm(A, ord=float('inf')))
        38.0
        >>> print(ms.ops.matrix_norm(A, ord=float('-inf')))
        6.0
        >>> print(ms.ops.vector_norm(A, ord=1))
        21.0
        >>> print(ms.ops.vector_norm(A, ord=-1))
        12.0
        >>> print(ms.ops.vector_norm(A, ord=2))
        22.409302
        >>> print(ms.ops.vector_norm(A, ord=-2))
        1.672928e-07
    """
    ndim = A.ndim
    row_axis, col_axis = _check_matrix_norm_axis(axis, ndim)
    _check_matrix_norm_ord(ord)
    if dtype is not None:
        A = ops.cast(A, dtype)

    ret = None
    if ord == 'fro':
        ret = ops.sqrt(ops.reduce_sum(_complex_square(A), axis))
    if ord == 'nuc':
        ret = _multi_svd_norm(A, row_axis, col_axis, 'sum')
    if ord == float('inf'):
        if row_axis > col_axis:
            row_axis -= 1
        ret = ops.max(ops.reduce_sum(abs(A), col_axis), axis=row_axis)[0]
    if ord == float('-inf'):
        if row_axis > col_axis:
            row_axis -= 1
        ret = ops.min(ops.reduce_sum(abs(A), col_axis), axis=row_axis)[0]
    if ord == 1:
        if col_axis > row_axis:
            col_axis -= 1
        ret = ops.max(A.abs().sum(row_axis), axis=col_axis)[0]
    if ord == -1:
        if col_axis > row_axis:
            col_axis -= 1
        ret = ops.min(A.abs().sum(row_axis), axis=col_axis)[0]
    if ord == 2:
        ret = _multi_svd_norm(A, row_axis, col_axis, 'amax')
    if ord == -2:
        ret = _multi_svd_norm(A, row_axis, col_axis, 'amin')
    if keepdims:
        ret_shape = list(A.shape)
        ret_shape[axis[0]] = 1
        ret_shape[axis[1]] = 1
        ret = ret.reshape(ret_shape)
    return ret


def lu_unpack(LU_data, LU_pivots, unpack_data=True, unpack_pivots=True):
    r"""
    Unpack the LU decomposition returned by :func:`mindspore.scipy.linalg.lu_factor` into the P, L, U matrices.

    .. note::
        - `LU_data` of shape :math:`(*, M, N)` , `LU_pivots` of shape :math:`(*, min(M, N))` , where
          :math:`*` is batch dimensions.

    Args:
        LU_data (Tensor): The packed LU factorization data, the rank is greater than or equal to 2.
        LU_pivots (Tensor): The packed LU factorization pivots.
        unpack_data (bool, optional): A flag indicating if the `LU_data` should be unpacked. If ``False`` ,
            then the returned L and U are None. Default ``True`` .
        unpack_pivots (bool, optional): A flag indicating if the `LU_pivots` should be unpacked into
            a permutation matrix P. If ``False`` , then the returned P is None. Default ``True`` .

    Returns:
        Tuple of tensors(Pivots, L, U).

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> LU_data = mindspore.tensor([[[-0.3806, -0.4872,  0.5536],
        ...                             [-0.1287,  0.6508, -0.2396],
        ...                             [ 0.2583,  0.5239,  0.6902]],
        ...                            [[ 0.6706, -1.1782,  0.4574],
        ...                             [-0.6401, -0.4779,  0.6701],
        ...                             [ 0.1015, -0.5363,  0.6165]]])
        >>> LU_pivots = mindspore.tensor(([[1, 3, 3], [2, 3, 3]]), mindspore.int32)
        >>> pivots, L, U = mindspore.ops.lu_unpack(LU_data, LU_pivots)
        >>> print(pivots)
        [[[1. 0. 0.]
          [0. 0. 1.]
          [0. 1. 0.]]
         [[0. 0. 1.]
          [1. 0. 0.]
          [0. 1. 0.]]]
        >>> print(L)
        [[[ 1.       0.       0.]
          [-0.1287   1.       0.]
          [ 0.2583   0.5239   1.]]
         [[ 1.0000   0.       0.]
          [-0.6401   1.       0.]
          [ 0.1015  -0.5363   1.]]]
        >>> print(U)
        [[[-0.3806  -0.4872   0.5536]
          [ 0.       0.6508  -0.2396]
          [ 0.       0.       0.6902]]
         [[ 0.6706  -1.1782   0.4574]
          [ 0.      -0.4779   0.6701]
          [ 0.       0.       0.6165]]]
    """
    pivots, l, u = lu_unpack_(LU_data, LU_pivots)
    if unpack_data:
        if unpack_pivots:
            return pivots, l, u
        return None, l, u
    if unpack_pivots:
        return pivots, None, None
    return None, None, None


def renorm(input, p, axis, maxnorm):
    """
    Returns a tensor where each subtensor along the specified dimension is renormalized such that its `p` norm is less
    than or equal to `maxnorm`. If the `p` norm exceeds `maxnorm`, return the values that are obtained by dividing the
    original values of the subtensor by its `p` norm and then multiplying by `maxnorm`.

    Args:
        input (Tensor): The input tensor.
        p (int): The power of norm calculation.
        axis (int): Specify the axis for computation.
        maxnorm (float32): The max norm specified.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[1., 1, 1], [2, 2, 2], [3, 3, 3]])
        >>> output = mindspore.ops.norm(input, 1, 1)
        >>> print(output)
        [3. 6. 9.]
        >>> output = mindspore.ops.renorm(input, 1, 0, 6.)
        >>> print(output)
        [[1. 1. 1.]
         [2. 2. 2.]
         [2. 2. 2.]]
    """
    renorm_ = _get_cache_prim(Renorm)(p, axis, maxnorm)
    return renorm_(input)


@constexpr
def _check_attr_dtype(param_name, input_dtype, allow_dtypes, cls_name):
    validator.check_value_type(param_name, input_dtype, allow_dtypes, cls_name)


@_primexpr
def _check_positive_float(arg_value, arg_name, cls_name):
    validator.check_positive_float(arg_value, arg_name, cls_name)


@_primexpr
def _check_int_range(arg_value, lower_limit, upper_limit, arg_name=None, prim_name=None):
    validator.check_int_range(arg_value, lower_limit,
                              upper_limit, validator.INC_LEFT, arg_name, prim_name)


def _check_logits_tensor(logits):
    if not isinstance(logits, (Tensor, Tensor_)):
        raise TypeError("The input logits must be tensor")


def _check_logits_shape(logits):
    if not logits.shape:
        raise ValueError("For gumbel_softmax, the 0-D input is not supported.")


def gumbel_softmax(logits, tau=1.0, hard=False, dim=-1):
    r"""
    Returns the samples from the Gumbel-Softmax distribution and optionally discretizes.
    If `hard` is ``True``, the returned
    samples will be one-hot, otherwise it will be probability distributions that sum to 1 across `dim`.

    Args:
        logits (Tensor): Unnormalized log probabilities. The data type must be float16 or float32.
        tau (float, optional): The scalar temperature, which is a positive number. Default: ``1.0`` .
        hard (bool, optional): If ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated
            as if it is the soft sample in autograd. Default: ``False`` .
        dim (int, optional): Dim for softmax to compute. Default: ``-1`` .

    Returns:
        Tensor, has the same dtype and shape as `logits`.

    Raises:
        TypeError: If `logits` is not a Tensor.
        TypeError: If dtype of `logits` is not one of: float16, float32.
        TypeError: If `tau` is not a float.
        TypeError: If `hard` is not a bool.
        TypeError: If `dim` is not an int.
        ValueError: If If `tau` is not positive.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
        >>> output = ops.gumbel_softmax(input_x, 1.0, True, -1)
        >>> print(output.shape)
        (2, 3)
    """
    _check_logits_tensor(logits)
    _check_logits_shape(logits)
    logits_dtype = dtype_(logits)
    _check_input_dtype("logits", logits_dtype, [mstype.float16, mstype.float32], "gumbel_softmax")
    valid_types = [mstype.float16, mstype.float32]
    if logits_dtype not in valid_types:
        names = [t.__name__ if hasattr(t, "__name__") else t for t in valid_types]
        logits_dtype = logits_dtype.__name__ if hasattr(logits_dtype, '__name__') else repr(logits_dtype)
        raise TypeError(f"For 'gumbel_softmax', the 'logits' should be one of '{names}', but got type '{logits_dtype}'")
    _check_attr_dtype("tau", tau, [float], "gumbel_softmax")
    _check_attr_dtype("hard", hard, [bool], "gumbel_softmax")
    _check_attr_dtype("dim", dim, [int], "gumbel_softmax")
    _check_positive_float(tau, "tau", "gumbel_softmax")
    if hard:
        _check_int_range(dim, -1, len(logits.shape), 'dim', "gumbel_softmax")
    else:
        _check_int_range(dim, -len(logits.shape),
                         len(logits.shape), 'dim', "gumbel_softmax")

    sample_shape = shape_(logits)
    uniform = C.uniform(sample_shape, scalar_to_tensor_(
        0.0, mstype.float32), scalar_to_tensor_(1.0, mstype.float32))
    uniform = cast_(uniform, logits_dtype)
    gumbel = neg(log_(neg(log_(uniform))))
    gumbel = (logits + gumbel) / tau
    y_soft = _get_cache_prim(P.Softmax)(dim)(gumbel)
    if hard:
        index = y_soft.argmax(axis=dim)
        y_hard = _get_cache_prim(P.OneHot)(dim)(index, sample_shape[dim], Tensor(1, logits_dtype),
                                                Tensor(0, logits_dtype))
        ret = ops.stop_gradient(y_hard - y_soft) + y_soft
    else:
        ret = y_soft
    return ret


def kaiser_window(window_length, periodic=True, beta=12.0, *, dtype=None):
    r"""
    Kaiser window function.

    .. math::
        w(n) = \frac{I_{0}\left( \beta\sqrt{1 - \frac{4n^{2}}{(M - 1)^{2}}} \right)}{I_{0}(\beta)}

    with

    .. math::
        - \frac{M - 1}{2} \leq n \leq \frac{M - 1}{2}

    where :math:`I_0` is the modified zeroth-order Bessel function.

    Args:
        window_length (int): The size of window.
        periodic (bool, optional): If ``True`` , return a periodic window. If ``False``, return a symmetric window.
            Default ``True`` .
        beta (float, optional): Shape parameter, when `beta` gets large, the window narrows. Default: ``12.0`` .

    Keyword Args:
        dtype (mindspore.dtype, optional): The data type specified. Default ``None`` .

    Returns:
        A 1-D tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> output = mindspore.ops.kaiser_window(5)
        >>> print(output)
        [5.27734413e-05 1.01719688e-01 7.92939834e-01 7.92939834e-01
         1.01719688e-01]
        >>> output = mindspore.ops.kaiser_window(5, periodic=False)
        >>> print(output)
        [5.27734413e-05 2.15672745e-01 1.00000000e+00 2.15672745e-01
         5.27734413e-05]
    """
    if not isinstance(periodic, bool):
        raise TypeError(
            f"For 'kaiser_window', 'periodic' must be a variable of Boolean type, but got {type(periodic)}"
        )
    if not isinstance(window_length, int):
        raise TypeError(
            f"For 'kaiser_window', 'window_length' must be a non-negative integer, but got {type(window_length)}"
        )
    if window_length < 0:
        raise ValueError(
            f"For 'kaiser_window', 'window_length' must be a non-negative integer, but got {window_length}"
        )
    if window_length <= 1:
        return Tensor(np.ones(window_length))
    if dtype is not None and dtype not in mstype.float_type:
        raise TypeError(f"For 'kaiser_window', 'dtype' must be floating point dtypes, but got {dtype}.")
    if periodic:
        window_length = window_length + 1
    n = np.arange(0, window_length)
    alpha = (window_length - 1) / 2.0
    w = np.i0(
        beta * np.sqrt(1 - ((n - alpha) / alpha) ** 2.0)
    ) / np.i0(float(beta))
    if dtype is not None:
        w = cast_(ms.tensor(w), dtype)
    out = Tensor(w[:-1]) if periodic else Tensor(w)
    return out


def stft(x, n_fft, hop_length=None, win_length=None, window=None, center=True,
         pad_mode="REFLECT", normalized=False, onesided=None, return_complex=None):
    r"""
    STFT segments the signal into narrow time intervals and takes the Fourier transform
    of each segment to quantify the change of a nonstationary signal's frequency
    and phase content over time.

    Ignoring the optional batch dimension, this operation computes the following expression:

    .. math::

        X[\omega, m]=\sum_{k=0}^{\text {win_length-1 }}
        \text { window }[k] \text { input }[m \times \text { hop_length }+
        k] \exp \left(-j \frac{2 \pi \cdot \omega k}{\text { win_length }}\right)

    where :math:`m` is the index of the sliding window, and
    :math:`ω` is the frequency in range :math:`0 \leq \omega < \text{n\_fft}0≤ω<n\_fft`.

    Args:
        x (Tensor): Time sequences of stft, must be either a 1-D time tensor or a 2-D tensor.
        n_fft (int): The size of Fourier transform.
        hop_length (int, optional): The distance between neighboring sliding window
            frames. Default: ``None``(treated as equal to :math:`floor(n\_fft / 4)`).
        win_length (int, optional): the size of window frame and STFT filter.
            Default: ``None``(treated as equal to `n_fft`).
        window (Tensor, optional): the optional window function, 1-D tensor of size `win_length`.
            Default: ``None``(treated as window of all :math:`1` s). If `win_length` < `n_fft`,
            `window` will be padded on both sides with ones to length `n_fft` before it takes effect.
        center (bool, optional): whether to pad `x` on both sides. Default: ``True``.
        pad_mode (str, optional): controls the padding method used when
            `center` is True. Default: 'REFLECT'.
        normalized (bool, optional): controls whether to return the normalized STFT results
             Default: ``False``.
        onesided (bool, optional): controls whether to return half of results to
            avoid redundancy for real inputs.
            Default: ``None``. True for real `x` and `window`, False otherwise.
        return_complex (bool, optional): whether to return a complex tensor, or
            a real tensor with an extra last dimension for the real and
            imaginary components.
            Default: ``None``. True for complex `x` or `window`, False otherwise.

    Returns:
        - **output** (Tensor) - A tensor containing the STFT result.
            If `return_complex` is True, it returns a complex Tensor with shape :math:`(*, N, T)`.
            If `return_complex` is False, it returns a real Tensor with shape :math:`(*, N, T, 2)`.

            `N` is size of Fourier transform, it depends on parameter `onesided`:
            - If `onesided` is False, :math:`N = n\_fft`.
            - If `onesided` is True, :math:`N = n\_fft // 2 + 1`.

            `T` is the total number of frames used, calculated by this formula:
            :math:`T = 1 + (len - n\_fft) / hop\_length`, where `len` depends on parameter `center`:
            - If `center` is False, :math:`len = signal\_length`.
            - If `center` is True, :math:`len = signal\_length + (n\_fft // 2) * 2`.
            where :math:`signal\_length` is the signal length, it equals to :math:`x.shape[-1]`.

    Raises:
        TypeError: If `x` is not a 1-D or 2-D tensor.
        TypeError: If `window` is not a 1-D tensor.
        TypeError: If any one of `center` , `normalized` , `onesided`
            and `return_complex` is assigned a nonboolean value.
        TypeError: If `pad_mode` is is assigned a value that is not string.
        TypeError: If `n_fft` , `hop_length` or `win_length` is not an int.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import ops
        >>> import numpy as np
        >>> x = ms.Tensor(np.random.rand(2,7192), ms.float32)
        >>> output = ops.stft(n_fft=64, x=x)
        >>> print(output.shape)
        (2, 33, 450, 2)
    """
    hop_length = int(n_fft // 4) if hop_length is None else hop_length
    win_length = int(n_fft // 1) if win_length is None else win_length
    window = ops.ones(win_length, mstype.float32) if window is None else window

    def _is_complex(x):
        return dtype_(x) in [mstype.complex64, mstype.complex128]

    if onesided is None:
        onesided = (not _is_complex(x)) and (not _is_complex(window))
    if return_complex is None:
        return_complex = _is_complex(x) or _is_complex(window)
    if center:
        _check_attr_dtype("center", center, [bool], "stft")
        signal_dim = len(x.shape)
        pad = n_fft // 2
        if signal_dim == 1:
            x = layer.Pad(((pad, pad),), pad_mode)(x)
        elif signal_dim == 2:
            x = layer.Pad(((0, 0), (pad, pad)), pad_mode)(x)
        else:
            raise ValueError(
                f"Expected a 1-D tensor or a 2-D tensor, but got {signal_dim}")
    stft_ = STFT(n_fft, hop_length, win_length,
                 normalized, onesided, return_complex)
    return stft_(x, window)


def _check_same_type(dtype1, dtype2):
    return dtype1 == dtype2


@constexpr
def _max(*args):
    """Returns the maximum value."""
    return max(*args)


@constexpr
def _min(*args):
    """Returns the minimum value."""
    return min(*args)


@_primexpr
def _infer_shape_rem(shape1, shape2, ndim1, ndim2, transpose_b):
    """Infers the shape of the last two dimensions after performing matmul."""
    shape_rem = []
    if ndim1 >= 2:
        shape_rem.append(shape1[-2])
    if transpose_b:
        if ndim2 >= 2:
            shape_rem.append(shape2[-2])
    else:
        if ndim1 >= 1:
            shape_rem.append(shape2[-1])
    return tuple(shape_rem)


def _check_value(items, max_size, msg_prefix, shape1, shape2):
    for item in items:
        if item not in (1, max_size):
            raise ValueError(f"{msg_prefix} operands could not be broadcast together with shape1 {shape1} and "
                             f"shape2 {shape2}.")


@_primexpr
def _check_matmul_shapes(shape1, shape2, prim_name=None):
    """Checks shape1 and shape2 are valid to perform matmul, and returns output shape after broadcasting."""
    msg_prefix = f"For '{prim_name}', the" if prim_name else "The"
    shape_out = []
    r_shape1 = shape1[:-2]
    r_shape2 = shape2[:-2]
    max_len = max(len(r_shape1), len(r_shape2))
    for i in range(max_len):
        items = [it[i - max_len + len(it)] if i - max_len + len(it) >= 0 else 1 for it in (r_shape1, r_shape2)]
        max_size = max(items)
        _check_value(items, max_size, msg_prefix, shape1, shape2)
        shape_out.append(max_size)
    return tuple(shape_out)


@_primexpr
def _check_need_broadcast(shape1, shape2):
    """Returns True if broadcast is necessary for batchmatmul."""
    return shape1[:-2] != shape2[:-2]


@_primexpr
def _expand(x, ndim):
    """Expand x to ndim from axis, which can be 0 or -1."""
    while rank_(x) < ndim:
        x = expand_dims_(x, 0)
    return x


def _broadcast_to(x, shape_cur, shape_to, ndim_to):
    """Broadcasts x from shape_cur to shape_to."""
    size = tile_size_(shape_cur, shape_to, ndim_to)
    F.stop_gradient(size)
    return tile_(x, size)


def matmul(input, other):
    """
    Return the matrix product of two tensors.

    Note:
        - `input` and `other` must have same data type, and both of them must be not scalar and support broadcast.
        - On Ascend, the rank of `input` or `other` must be between 1 and 6.
        - `input` and `other` must not be empty tensor when executing the backward process for dynamic shape case in
          JIT mode.

    Args:
        input (Tensor): The first input tensor.
        other (Tensor): The second input tensor.

    Returns:
        Tensor or scalar

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> # case 1 : Reasonable application of broadcast mechanism.
        >>> input = mindspore.ops.arange(24, dtype=mindspore.float32).reshape(2, 3, 4)
        >>> other = mindspore.ops.arange(20, dtype=mindspore.float32).reshape(4, 5)
        >>> output = mindspore.ops.matmul(input, other)
        >>> print(output)
        [[[  70.   76.   82.   88.   94.]
          [ 190.  212.  234.  256.  278.]
          [ 310.  348.  386.  424.  462.]]
         [[ 430.  484.  538.  592.  646.]
          [ 550.  620.  690.  760.  830.]
          [ 670.  756.  842.  928. 1014.]]]
        >>>
        >>> # case 2 : The rank of `input` is 1.
        >>> input = mindspore.ops.ones(([1, 2]))
        >>> other = mindspore.ops.ones(([2]))
        >>> output = mindspore.ops.matmul(input, other)
        >>> print(output)
        [2.]
    """
    return auto_generate.matmul_ext(input, other)


def inner(input, other):
    r"""
    Return the dot product of two 1-D tensors.

    For higher dimensions, return a sum product over the last axis.

    Note:
        If `input` or `other` is a Tensor scalar, :func:`mindspore.ops.inner` will be the same as
        :func:`mindspore.ops.mul` .

    Args:
        input (Tensor): The first input.
        other (Tensor): The second input.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> # case 1: Two 1D tensors
        >>> input = mindspore.tensor([1., 2., 3.])
        >>> y = mindspore.tensor([4., 5., 6.])
        >>> mindspore.ops.inner(input, y)
        Tensor(shape=[], dtype=Float32, value= 32)
        >>> # case2: Tensor scalar and tensor
        >>> input = mindspore.tensor([[[1., 2., 3.], [3., 2., 1.]], [[4., 5., 6.], [4., 5., 6.]]])
        >>> y = mindspore.tensor(2.)
        >>> output = mindspore.ops.inner(input, y)
        >>> print(output)
        [[[ 2.  4.  6.]
          [ 6.  4.  2.]]
         [[ 8. 10. 12.]
          [ 8. 10. 12.]]]
        >>> # case3: Two tensors
        >>> input = mindspore.tensor([[[1., 2., 3.], [3., 2., 1.]], [[4., 5., 6.], [4., 5., 6.]]])
        >>> y = mindspore.tensor([[2., 3., 4.], [4., 3., 2.]])
        >>> output = mindspore.ops.inner(input, y)
        >>> print(output)
        [[[20. 16.]
          [16. 20.]]
         [[47. 43.]
          [47. 43.]]]
    """
    x_dim = input.ndim
    other_dim = other.ndim

    if x_dim == 0 or other_dim == 0:
        output = input * other
        return output

    x_shape = input.shape
    other_shape = other.shape
    if x_shape[-1] != other_shape[-1]:
        raise ValueError(f"For 'inner', the last dimension of 'input' and 'other' must be the same, \
                         but got input.shape: {x_shape} and other.shape: {other_shape}.")
    return ops.tensor_dot(input, other, axes=(-1, -1))


def bmm(input_x, mat2):
    r"""
    Perform a batch matrix-matrix product of matrices in input tensors.

    .. math::

        \text{output}[..., :, :] = \text{matrix}(input_x[..., :, :]) * \text{matrix}(mat2[..., :, :])

    The dim of `input_x` can not be less than `3` and the dim of `mat2` can not be less than `2`.

    Args:
        input_x (Tensor): The first tensor to be multiplied.
        mat2 (Tensor): The second tensor to be multiplied.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input_x = mindspore.ops.arange(24, dtype=mindspore.float32).reshape(2, 4, 1, 3)
        >>> mat2 = mindspore.ops.arange(72, dtype=mindspore.float32).reshape(2, 4, 3, 3)
        >>> out = mindspore.ops.bmm(input_x, mat2)
        >>> print(out)
        [[[[  15.   18.   21.]]
          [[ 150.  162.  174.]]
          [[ 447.  468.  489.]]
          [[ 906.  936.  966.]]]
         [[[1527. 1566. 1605.]]
          [[2310. 2358. 2406.]]
          [[3255. 3312. 3369.]]
          [[4362. 4428. 4494.]]]]
    """
    return batch_matmul_(input_x, mat2)


def quantile(input, q, axis=None, keepdims=False):
    r"""
    Computes the q-th quantiles of all elements in `input`, when the
    q-th quantile lies between two data points, a linear interpolation is implemented between them.

    Args:
        input (Tensor): The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
            Supported dtypes: float32, float64.
        q (Union[float, Tensor]): A scalar or 1D tensor of quantile values in the range [0, 1].
            Supported dtypes: float32, float64.
        axis (int, optional): The dimension to reduce. By default, `axis` is None resulting in the
            input tensor being flattened before computation. Default: ``None``.
        keepdims (bool, optional): Whether the output tensor has dim retained or not. Default: ``False``.

    Returns:
        Tensor, has the same dtype as the `input`.

        Suppose the shape of `input` is :math:`(m, x_0, x_1, ..., x_i, ..., X_R)`, `axis` = :math:`i` and m is
        the element count of input `q`.

        - If `q` is scalar and `keepdims` is True, the shape of output is :math:`(x_0, x_1, ..., 1, ..., X_R)`.
        - If `q` is scalar and `keepdims` is False, the shape of output is :math:`(x_0, x_1, ..., X_R)`.
        - If `q` is 1D Tensor and `keepdims` is True, the shape of output is :math:`(m, x_0, x_1, ..., 1, ..., X_R)`.
        - If `q` is 1D Tensor and `keepdims` is False, the shape of output is :math:`(m, x_0, x_1, ..., X_R)`.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If `q` is not a Tensor or float.
        TypeError: If dtype of `input` is not float32 or float64.
        TypeError: If dtype of `q` is not float32 or float64.
        TypeError: If dtype of `input` and the dtype of `q` is different.
        ValueError: If the `q` values not in the range [0, 1].
        ValueError: If the `axis` values out of range.

    Supported Platforms:


    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> x = Tensor(np.array([-0.7832, 0.8003, 0.8111]), mindspore.float32)
        >>> q = Tensor(np.array([0.1, 0.7, 0.9]), mindspore.float32)
        >>> output = ops.quantile(x, q)
        >>> print(output.asnumpy())
        [-0.4665 0.80462 0.80894]
    """

    if axis is not None:
        _check_attr_dtype("axis", axis, [int], "quantile")
    if keepdims is not None:
        _check_attr_dtype("keepdims", keepdims, [bool], "quantile")

    quantile_ = _get_cache_prim(Quantile)(dim=axis, keep_dims=keepdims)
    return quantile_(input, q)


def nanquantile(input, q, axis=None, keepdims=False):
    r"""
    This operator is derived from mindspore.ops.quantile() that 'ignores' NaN values.
    It computes quantiles as though the input has no NaN values. If all values in a
    reduced dimension are NaN then the quantiles for that reduction will be NaN.

    Refer to :func:`mindspore.ops.quantile` for more details.

    Args:
        input (Tensor): The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
            Supported dtypes: float32, float64.
        q (Union[float, Tensor]): A scalar or 1D tensor of quantile values in the range [0, 1].
            Supported dtypes: float32, float64.
        axis (int, optional): The dimension to reduce. By default, `axis` is None resulting in the
            input tensor being flattened before computation. Default: ``None``.
        keepdims (bool, optional): Whether the output tensor has dim retained or not. Default: ``False``.

    Returns:
        Tensor, has the same dtype as the `input`.

        Suppose the shape of `input` is :math:`(m, x_0, x_1, ..., x_i, ..., X_R)`, `axis` = :math:`i` and m is
        the element count of input `q`.

        - If `q` is scalar and `keepdims` is True, the shape of output is :math:`(x_0, x_1, ..., 1, ..., X_R)`.
        - If `q` is scalar and `keepdims` is False, the shape of output is :math:`(x_0, x_1, ..., X_R)`.
        - If `q` is 1D Tensor and `keepdims` is True, the shape of output is :math:`(m, x_0, x_1, ..., 1, ..., X_R)`.
        - If `q` is 1D Tensor and `keepdims` is False, the shape of output is :math:`(m, x_0, x_1, ..., X_R)`.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If `q` is not a Tensor or float.
        TypeError: If dtype of `input` is not float32 or float64.
        TypeError: If dtype of `q` is not float32 or float64.
        TypeError: If dtype of `input` and the dtype of `q` is different.
        ValueError: If the `q` values not in the range [0, 1].
        ValueError: If the `axis` values out of range.

    Supported Platforms:


    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> x = Tensor(np.array([float('nan'), 0.8003, 0.8111]), mindspore.float32)
        >>> q = Tensor(np.array([0.1, 0.7, 0.9]), mindspore.float32)
        >>> output = ops.nanquantile(x, q)
        >>> print(output.asnumpy())
        [0.80138 0.80786 0.81002]
    """

    if axis is not None:
        _check_attr_dtype("axis", axis, [int], "nanquantile")
    if keepdims is not None:
        _check_attr_dtype("keepdims", keepdims, [bool], "nanquantile")

    quantile_ = _get_cache_prim(Quantile)(dim=axis, keep_dims=keepdims, ignore_nan=True)
    return quantile_(input, q)


def baddbmm(input, batch1, batch2, beta=1, alpha=1):
    r"""
    Perform a batch matrix-matrix product of matrices in `batch1` and `batch2` , `input` is added to the final result.

    .. note::
        - `batch1` and `batch2` must be 3-D tensors each containing the same number of matrices.
        - When batch1 is a :math:`(C, W, T)` tensor and batch2 is a :math:`(C, T, H)` tensor, input must be
          broadcastable with :math:`(C, W, H)` tensor, and out will be a  :math:`(C, W, H)` tensor.
        - If `beta` is 0, then `input` will be ignored.
        - `beta` and `alpha` must be integers when inputs of type not FloatTensor.

    .. math::
        \text{out}_{i} = \beta \text{input}_{i} + \alpha (\text{batch1}_{i} \mathbin{@} \text{batch2}_{i})

    Args:
        input (Tensor): The input tensor.
        batch1 (Tensor): The first batch of matrices to be multiplied.
        batch2 (Tensor): The second batch of matrices to be multiplied.
        beta (Union[float, int], optional): Scale factor for `input`. Default ``1`` .
        alpha (Union[float, int], optional): Scale factor for ( `batch1` @ `batch2` ). Default ``1`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.ops.ones((3, 3))
        >>> batch1 = mindspore.tensor([[8., 7., 6.], [5., 4., 3.], [2., 1., 0.]])
        >>> batch2 = mindspore.tensor([[5., 4., 3.], [2., 1., 0.], [8., 7., 6.]])
        >>> output = mindspore.ops.baddbmm(input, batch1, batch2)
        >>> print(output)
        [[103.  82.  61.]
         [ 58.  46.  34.]
         [ 13.  10.   7.]]
    """
    bmmop = _get_cache_prim(BatchMatMul)(False, False)
    if not (isinstance(input, Tensor) and isinstance(batch1, Tensor) and isinstance(batch2, Tensor)):
        raise TypeError("For Baddbmm, inputs must be all tensors.")
    input_dtype = dtype_(input)
    if not (input_dtype == dtype_(batch1) and input_dtype == dtype_(batch2)):
        raise TypeError("For Baddbmm, the inputs should be the same dtype.")
    if input_dtype in (mstype.float16, mstype.float32, mstype.float64):
        if not (isinstance(alpha, (int, float)) and isinstance(beta, (int, float))):
            raise TypeError("For attributes alpha and beta should be real numbers.")
        check_is_number(alpha, (int, float))
        check_is_number(beta, (int, float))
    else:
        if not (isinstance(alpha, int) and isinstance(beta, int)):
            raise TypeError("For inputs of type not FloatTensor or DoubleTensor, "
                            "arguments beta and alpha must be integers.")
    y = beta * input + alpha * (bmmop(batch1, batch2))
    return y


def baddbmm_ext(input, batch1, batch2, *, beta=1, alpha=1):
    r"""
    The result is the sum of the input and a batch matrix-matrix product of matrices in batch1 and batch2.
    The formula is defined as follows:

    .. math::
        \text{out}_{i} = \beta \text{input}_{i} + \alpha (\text{batch1}_{i} \mathbin{@} \text{batch2}_{i})

    Args:
        input (Tensor): The input Tensor. When batch1 is a :math:`(C, W, T)` Tensor and batch2 is a
            :math:`(C, T, H)` Tensor, input must be broadcastable with :math:`(C, W, H)` Tensor.
        batch1 (Tensor): :math:`batch1` in the above formula. Must be 3-D Tensor, dtype is same as input.
        batch2 (Tensor): :math:`batch2` in the above formula. Must be 3-D Tensor, dtype is same as input.

    Keyword Args:
        beta (Union[float, int], optional): multiplier for input. Default: ``1`` .
        alpha (Union[float, int], optional): multiplier for :math:`batch1 @ batch2`. Default: ``1`` .

    Returns:
        Tensor, has the same dtype as input, shape will be :math:`(C, W, H)`.

    Raises:
        TypeError: If the type of `input`, `batch1`, `batch2` is not Tensor.
        TypeError: If the types of `input`, `batch1`, `batch2` are different.
        ValueError: If `batch1` and `batch2` are not 3-D tensors.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input = Tensor(np.ones([1, 3, 3]).astype(np.float32))
        >>> batch1 = Tensor(np.ones([1, 3, 4]).astype(np.float32))
        >>> batch2 = Tensor(np.ones([1, 4, 3]).astype(np.float32))
        >>> output = ops.baddbmm_ext(input, batch1, batch2)
        >>> print(output)
        [[[5. 5. 5.]
          [5. 5. 5.]
          [5. 5. 5.]]]
    """
    return ops.auto_generate.baddbmm(input, batch1, batch2, beta, alpha)


def log2(input): # pylint: disable=redefined-builtin
    r"""
    Compute the logarithm to the base 2 of the input tensor element-wise.

    .. math::
        y_i = \log_2(input_i)

    .. warning::
        If the input value of operator log2 is within the range (0, 0.01] or [0.95, 1.05], the output accuracy may
        be affected.

    Args:
        input (Tensor): The input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([2, 4, 8], mindspore.float16)
        >>> output = mindspore.ops.log2(x)
        >>> print(output)
        [1. 2. 3.]
    """
    if isinstance(input, Tensor) and input.dtype == mstype.bool_:
        input = input.astype(mstype.int64)
    x_dtype = dtype_(input)
    denominator = log_(_make_tensor(2, x_dtype))
    frac_log = log_(input)
    output = frac_log / denominator
    return output


def arrange(x):
    lists = []
    for i in range(0, x):
        lists.append(i)
    return lists


def rot90(input, k, dims):
    """
    Rotate a n-D tensor by 90 degrees in the plane specified by dims axis.
    Rotation direction is from the first towards the second axis if k > 0,
    and from the second towards the first for k < 0.

    Args:
        input (Tensor): The input tensor.
        k (int): Number of times to rotate.
        dims (Union[list(int), tuple(int)]): Axis to rotate.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([[1, 2, 3],
        ...                       [4, 5, 6]], mindspore.float32)
        >>> output = mindspore.ops.rot90(x, k=1, dims=(0, 1))
        >>> print(output)
        [[3. 6.]
         [2. 5.]
         [1. 4.]]
        >>> mindspore.ops.rot90(x, k=1, dims=(1, 0)) == mindspore.ops.rot90(x, k=-1, dims=(0,1))
        Tensor(shape=[3, 2], dtype=Bool, value=
        [[ True,  True],
         [ True,  True],
         [ True,  True]])
        >>> # when input array has ndim>2
        >>> x = mindspore.tensor([[[1, 2, 3],
        ...                       [4, 5, 6]],
        ...                      [[7, 8, 9],
        ...                       [10, 11, 12]]], mindspore.float32)
        >>> output = mindspore.ops.rot90(x, k=1, dims=(2, 1))
        >>> print(output)
        [[[ 4.  1.]
         [ 5.  2.]
         [ 6.  3.]]
        [[10.  7.]
         [11.  8.]
         [12.  9.]]]
    """

    if not isinstance(input, (Tensor, Tensor_)):
        raise TypeError(f"For `rot90`, the `input` must be Tensor!, but get {type(input)}.")
    if not isinstance(k, int):
        raise TypeError(f"For `rot90`, the `k` must be int!, but get {type(k)}.")
    if not isinstance(dims, (list, tuple)):
        raise TypeError(f"For `rot90`, the `dims` must be list or tuple!, but get {type(dims)}.")

    total_dims = input.ndim
    total_rot_dims = len(dims)

    if total_rot_dims != 2:
        raise ValueError(f"For `rot90`, total rotation dims must be 2, but get {total_rot_dims}.")
    if dims[0] == dims[1] or (dims[0] - dims[1]) == total_dims or (dims[1] - dims[0]) == total_dims:
        raise RuntimeError(f"For `rot90`, rotation dims must be different, but get dim0={dims[0]}, dim1={dims[1]}.")
    if dims[0] >= total_dims or dims[0] < -total_dims:
        raise ValueError(f"For `rot90`, rotation dim0 is out of range, dim0={dims[0]}.")
    if dims[1] >= total_dims or dims[1] < -total_dims:
        raise ValueError(f"For `rot90`, rotation dim1 is out of range, dim1={dims[1]}.")

    k = (4 + (k % 4)) % 4

    if k == 0:
        out = input
        return out
    if k == 2:
        op1 = P.ReverseV2(axis=[dims[0]])
        output = op1(input)
        op2 = P.ReverseV2(axis=[dims[1]])
        out = op2(output)
        return out

    axes_list = arrange(total_dims)
    (axes_list[dims[0]], axes_list[dims[1]]) = (axes_list[dims[1]],
                                                axes_list[dims[0]])

    if k == 1:
        op = P.ReverseV2(axis=[dims[1]])
        output = op(input)
        out = output.transpose(axes_list)
    else:
        output = input.transpose(axes_list)
        op = P.ReverseV2(axis=[dims[1]])
        out = op(output)
    return out


def xdivy(x, y):
    """
    Divide `x` by `y` element-wise.

    .. note::
        - Support broadcast, support implicit type conversion and type promotion.
        - When `x` and `y` are both of datatype complex, they should be both complex64 or complex128 at the same time.
        - `x` and `y` can not be both bool at the same time.

    Args:
        x (Union[Tensor, Number, bool]): Numerator tensor.
        y (Union[Tensor, Number, bool]): Denominator tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> output = mindspore.ops.xdivy(mindspore.tensor([2., 4., -1.]), mindspore.tensor([2., 2., 2.]))
        >>> print(output)
        [ 1.   2.  -0.5]
    """
    return xdivy_(x, y)


def log10(input):
    r"""
    Compute the logarithm to the base 10 of the input tensor element-wise.

    .. math::
        y_i = \log_{10}(input_i)

    .. warning::
        If the input value of operator log10 is within the range (0, 0.01] or [0.95, 1.05], the output accuracy may
        be affected.

    Args:
        input (Tensor): The input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([2, 4, 10], mindspore.float16)
        >>> output = mindspore.ops.log10(x)
        >>> print(output)
        [0.301 0.602 1.   ]
    """
    x_dtype = dtype_(input)
    denominator = log_(_make_tensor(10, x_dtype))
    frac_log = log_(input)
    output = frac_log / denominator
    return output


def kron(input, other):
    """
    Compute the Kronecker product of two tensors.

    If the shape of `input` is :math:`(a_{0}` input :math:`a_{1}` input ... input :math:`a_{n})`
    and the shape of `other` is :math:`(b_{0}` input :math:`b_{1}` input ... input :math:`b_{n})` ,
    the result will be :math:`(a_{0}*b_{0}` input :math:`a_{1}*b_{1}` input ... input :math:`a_{n}*b_{n})` .

    .. math::
        (input ⊗ other)_{k_{0},k_{1},...k_{n}} =
        input_{i_{0},i_{1},...i_{n}} * other_{j_{0},j_{1},...j_{n}},

    where :math:`k_{t} = i_{t} * b_{t} + j_{t}` for 0 ≤ `t` ≤ `n`.

    Note:
        Supports real-valued and complex-valued inputs.

    Args:
        input (Tensor): First input tensor.
        other (Tensor): Second input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[0., 1., 2.], [3., 4., 5.]])
        >>> other = mindspore.tensor([[-1., -2., -3.], [-4., -6., -8.]])
        >>> output = mindspore.ops.kron(input, other)
        >>> print(output)
        [[  0.   0.   0.  -1.  -2.  -3.  -2.  -4.  -6.]
         [  0.   0.   0.  -4.  -6.  -8.  -8. -12. -16.]
         [ -3.  -6.  -9.  -4.  -8. -12.  -5. -10. -15.]
         [-12. -18. -24. -16. -24. -32. -20. -30. -40.]]
    """

    if not isinstance(input, (Tensor, Tensor_)):
        raise TypeError("the input must be Tensor!")
    if not isinstance(other, (Tensor, Tensor_)):
        raise TypeError("the other must be Tensor!")
    if input is None or other is None:
        return None
    if input.ndim == 0 or other.ndim == 0:
        return input * other

    if input.ndim >= other.ndim:
        maxdim = input.ndim
    else:
        maxdim = other.ndim
    pad_x = maxdim - input.ndim
    pad_y = maxdim - other.ndim
    x_reshape = [0 for _ in range(2 * maxdim)]
    y_reshape = [0 for _ in range(2 * maxdim)]
    result_shape = [0 for _ in range(maxdim)]

    for i in range(maxdim):
        if i >= pad_x:
            x_reshape[2 * i] = input.shape[i - pad_x]
        else:
            x_reshape[2 * i] = 1
        x_reshape[2 * i + 1] = 1
        y_reshape[2 * i] = 1
        if i >= pad_y:
            y_reshape[2 * i + 1] = other.shape[i - pad_y]
        else:
            y_reshape[2 * i + 1] = 1
        result_shape[i] = x_reshape[2 * i] * y_reshape[2 * i + 1]

    input = input.reshape(x_reshape)
    other = other.reshape(y_reshape)
    result = (input * other).reshape(result_shape)
    return result


def _check_is_tensor(param_name, input, cls_name):
    """Returns True if input is Tensor."""
    if not isinstance(input, Tensor):
        raise TypeError(f"For {cls_name}, {param_name} must be a Tensor, but got {type(input)}.")


def any(input, axis=None, keep_dims=False):
    r"""
    Tests if any element in `input` evaluates to `True` along the given axes.

    Args:
        input (Tensor): The input tensor.
        axis (Union[int, tuple(int), list(int), Tensor], optional): The dimensions to reduce. If ``None`` ,
                all dimensions are reduced. Default ``None`` .
        keep_dims (bool, optional): Whether the output tensor has dim retained or not. Default ``False`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[True, False], [True, True]])
        >>>
        >>> # case 1:  By default, mindspore.ops.any tests along all the axes.
        >>> mindspore.ops.any(input)
        Tensor(shape=[], dtype=Bool, value= True)
        >>>
        >>> # case 2: Reduces a dimension along axis 1, with keep_dims False.
        >>> mindspore.ops.any(input, axis=1)
        Tensor(shape=[2], dtype=Bool, value= [ True,  True])
        >>>
        >>> # case 3: Reduces a dimension along axis (0, 1), with keep_dims False.
        >>> mindspore.ops.any(input, axis=(0,1))
        Tensor(shape=[], dtype=Bool, value= True)
        >>>
        >>> # case 4: Reduces a dimension along axis [0, 1], with keep_dims True.
        >>> mindspore.ops.any(input, axis=[0,1], keep_dims=True)
        Tensor(shape=[1, 1], dtype=Bool, value=
        [[ True]])
    """
    if axis is None:
        axis = ()
    return _get_cache_prim(P.ReduceAny)(keep_dims)(input, axis)


def remainder(input, other):
    r"""
    Compute the remainder of division for the input tensor element-wise.

    Support implicit type conversion and type promotion.

    .. code:: python

        remainder(input, other) == input - input.div(other, rounding_mode="floor") * other

    .. warning::
        - When the elements of input exceed 2048, there might be accuracy problems.
        - The calculation results of this operator on Ascend and CPU might be inconsistent.
        - If shape is expressed as (D1,D2... ,Dn), then D1\*D2... \*DN<=1000000,n<=8.

    Args:
        input (Union[Tensor, numbers.Number, bool]): The first input.
        other (Union[Tensor, numbers.Number, bool]): The second input.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> # case 1: The shape of two inputs are different
        >>> x = mindspore.tensor([-4.0, 5.0, 6.0], mindspore.float16)
        >>> y = mindspore.tensor([3.0], mindspore.float16)
        >>> output = mindspore.ops.remainder(x, y)
        >>> print(output)
        [2. 2. 0.]
        >>> # case 2: The shape of two inputs are the same
        >>> x = mindspore.tensor([-4.0, 5.0, 6.0], mindspore.float16)
        >>> y = mindspore.tensor([3.0, 2.0, 3.0], mindspore.float16)
        >>> output = mindspore.ops.remainder(x, y)
        >>> print(output)
        [2. 1. 0.]
    """

    out = input - tensor_floordiv(input, other) * other
    return out


def remainder_ext(input, other):
    r"""
    Computes the remainder of `input` divided by `other` element-wise. The result has the same sign as the divisor and
    its absolute value is less than that of `other`.

    Supports broadcasting to a common shape and implicit type promotion.

    .. code:: python

        remainder(input, other) == input - input.div(other, rounding_mode="floor") * other


    Note:
        Complex inputs are not supported. At least one input need to be tensor, but not both are bool tensors.

    Args:
        input (Union[Tensor, numbers.Number, bool]): The dividend is a numbers.Number or
            a bool or a tensor whose data type is
            `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.dtype.html>`_ or
            `bool <https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.dtype.html>`_.
        other (Union[Tensor, numbers.Number, bool]): The divisor is a numbers.Number or
            a bool or a tensor whose data type is number or bool when the dividend is a tensor.
            When the dividend is Scalar, the divisor must be a Tensor whose data type is number or bool.

    Returns:
        Tensor, with dtype promoted and shape broadcasted.

    Raises:
        TypeError: If `input` and `other` are not of types: (tensor, tensor), (tensor, number), (tensor, bool),
            (number, tensor) or (bool, tensor).
        ValueError: If `input` and `other` are not broadcastable.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> x = Tensor(np.array([-4.0, 5.0, 6.0]).astype(np.float32))
        >>> y = Tensor(np.array([3.0, 2.0, 3.0]).astype(np.float64))
        >>> output = ops.remainder_ext(x, y)
        >>> print(output)
        [2.  1.  0.]
    """

    if isinstance(input, Tensor) and isinstance(other, Tensor):
        return remainder_tensor_tensor_(input, other)
    if isinstance(input, Tensor) and isinstance(other, (float, int, bool)):
        return remainder_tensor_scalar_(input, other)
    if isinstance(input, (float, int, bool)) and isinstance(other, Tensor):
        return remainder_scalar_tensor_(input, other)
    raise TypeError("For 'remainder', inputs should either be two tensors, or a tensor and a scalar.")


@deprecated("2.8.0", "ops.addn", False, "ops.")
def accumulate_n(x):
    r"""
    `ops.accumulate_n` is deprecated from version 2.8.0 and will be removed in a future version,
    please use :func:`mindspore.ops.addn` instead.

    Return the element-wise sum of all input tensors.

    :func:`mindspore.ops.accumulate_n` is similar to :func:`mindspore.ops.addn`,
    but accumulate_n will not wait for all of its inputs to be ready before summing, which is able
    to reduce peak memory.

    Args:
        x (Union(tuple[Tensor], list[Tensor])): List of tensors or tuple of tensors.

    Returns:
        Tensor

    Supported Platforms:
        Deprecated

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([1, 2, 3], mindspore.float32)
        >>> y = mindspore.tensor([4, 5, 6], mindspore.float32)
        >>> output = mindspore.ops.accumulate_n([x, y, x, y])
        >>> print(output)
        [10. 14. 18.]
    """
    return _get_accumulate_prim()(x)


def iou(anchor_boxes, gt_boxes, mode='iou'):
    r"""
    Calculates intersection over union for boxes.

    Computes the intersection over union (IOU) or the intersection over foreground (IOF) based on the ground-truth and
    predicted regions.

    .. math::
        \text{IOU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}

        \text{IOF} = \frac{\text{Area of Overlap}}{\text{Area of Ground Truth}}

    .. warning::
        In Ascend, only computation of float16 data is supported. To avoid overflow, the input length
        and width are scaled by 0.2 internally.

    Args:
        anchor_boxes (Tensor): Anchor boxes, tensor of shape :math:`(N, 4)` . :math:`N` indicates the number of
            anchor boxes, and the value :math:`4` refers to four boundary coordinates of the predicted area
            "x0", "y0", "x1", and "y1". Data type must be either float16, float32 or float64.
        gt_boxes (Tensor): Ground truth boxes, tensor of shape :math:`(M, 4)` . :math:`M` indicates the number
            of ground truth boxes, and the value :math:`4` refers to four boundary coordinates of the truth
            area "x0", "y0", "x1", and "y1". Data type must be either float16, float32 or float64.
        mode (str): The mode is used to specify the calculation method,
            now supporting 'iou' (intersection over union) or 'iof' (intersection over foreground) mode.
            Default: ``'iou'`` .

    Returns:
        Tensor, the IOU/IOF values, tensor of shape :math:`(M, N)` , with the same data type as `anchor_boxes`.

    Raises:
        KeyError: When `mode` is not ``'iou'`` or ``'iof'``.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> anchor_boxes = Tensor(np.random.randint(1.0, 5.0, [3, 4]), mindspore.float16)
        >>> gt_boxes = Tensor(np.random.randint(1.0, 5.0, [3, 4]), mindspore.float16)
        >>> mode = 'iou'
        >>> output = ops.iou(anchor_boxes, gt_boxes, mode)
        >>> print(output.shape)
        (3, 3)
    """
    return _get_cache_prim(P.IOU)(mode)(anchor_boxes, gt_boxes)


def _check_is_float(dtype):
    return dtype in (mstype.float16, mstype.float32)


def _list_comprehensions(obj, item):
    return tuple(item for _ in range(obj))


def _tuple_setitem(tup, idx, value):
    tup = list(tup)
    tup[idx] = value
    return tuple(tup)


@_primexpr
def _check_dim_in_range(dim, ndim):
    def _check(dim, ndim):
        if not isinstance(dim, int):
            raise TypeError(f'axes should be integers, not {type(dim)}')
        if -ndim > dim or dim >= ndim:
            raise ValueError(f'dim {dim} is out of bounds for array of dimension {ndim}')

    _check(dim, ndim)
    return dim % ndim


def dotrapezoid(y, dx, dim):
    y_left = _select(y, dim, 0)
    y_right = _select(y, dim, -1)
    y_sum = y.sum(dim)
    return (y_sum - (y_left + y_right) * 0.5) * dx


def dotrapezoid_tensor(y, dx, dim):
    y_start_dim_left = [0 for _ in range(dim)]
    y_start_dim_left = tuple(y_start_dim_left)
    y_start_dim_right = [0 for _ in range(y.ndim - dim - 1)]
    y_start_dim_right = tuple(y_start_dim_right)
    y_slice_size = _tuple_setitem(shape_(y), dim, shape_(y)[dim] - 1)
    y_slice_left = slice_(y, y_start_dim_left + (0,) + y_start_dim_right, y_slice_size)
    y_slice_right = slice_(y, y_start_dim_left + (1,) + y_start_dim_right, y_slice_size)
    return (tensor_add(y_slice_left, y_slice_right) * dx).sum(dim) / 2.


def add_padding_to_shape(curr_shape, target_n_dim):
    curr_size = len(curr_shape)
    target_n_dim = max(target_n_dim, curr_size)
    new_shape = [1 for _ in range(target_n_dim)]
    for i in range(curr_size):
        new_shape[target_n_dim - i - 1] = curr_shape[curr_size - i - 1]
    return new_shape


def zeros_like_except(y, dim):
    _check_dim_in_range(dim, y.ndim)
    dim = dim + y.ndim if dim < 0 else dim
    sizes = y.shape[:dim] + y.shape[dim + 1:]
    zeros = F.zeros(sizes, y.dtype)
    return zeros


def trapezoid_tensor(y, x, dim):
    r"""
    add trapezoid implementation when x is not None.
    """
    if y.shape[dim] == 0:
        return zeros_like_except(y, dim)
    if x.ndim < y.ndim and x.ndim != 1:
        x_start_dim_left = [0 for _ in range(dim)]
        x_start_dim_left = tuple(x_start_dim_left)
        x_start_dim_right = [0 for _ in range(x.ndim - dim - 1)]
        x_start_dim_right = tuple(x_start_dim_right)
        x_slice_size = _tuple_setitem(x.shape, dim, x.shape[dim] - 1)
        x_left = slice_(x, x_start_dim_left + (0,) + x_start_dim_right, x_slice_size)
        x_right = slice_(x, x_start_dim_left + (1,) + x_start_dim_right, x_slice_size)
        dx = x_right - x_left
        new_sizes = add_padding_to_shape(dx.shape, y.ndim)
        dx = dx.view(tuple(new_sizes))
        return dotrapezoid_tensor(y, dx, dim)
    if x.ndim == 1:
        if x.shape[0] != y.shape[dim]:
            raise RuntimeError("There must be one `x` value for each sample point")
        new_sizes = [1 for _ in range(y.ndim)]
        new_sizes[dim] = x.shape[0]
        x_viewed = x.view(tuple(new_sizes))
    else:
        x_viewed = x
    x_start_dim_left = [0 for _ in range(dim)]
    x_start_dim_left = tuple(x_start_dim_left)
    x_start_dim_right = [0 for _ in range(x_viewed.ndim - dim - 1)]
    x_start_dim_right = tuple(x_start_dim_right)
    x_slice_size = _tuple_setitem(x_viewed.shape, dim, x_viewed.shape[dim] - 1)
    x_left = slice_(x_viewed, x_start_dim_left + (0,) + x_start_dim_right, x_slice_size)
    x_right = slice_(x_viewed, x_start_dim_left + (1,) + x_start_dim_right, x_slice_size)
    dx = x_right - x_left
    return dotrapezoid_tensor(y, dx, dim)


def trapezoid(y, dx, dim):
    if y.shape[dim] == 0:
        return zeros_like_except(y, dim)
    return dotrapezoid(y, dx, dim)


def get(ts, depth, dim, index, r):
    if depth == dim:
        r.append(ts[index])
        return 0
    for item in ts:
        return get(item, depth + 1, dim, index, r)


def _select(feat, dim, index):
    select_shape = feat.shape
    select_shape = list(select_shape)
    select_shape[dim] = 1
    new_shape = feat.shape[:dim] + feat.shape[dim + 1:]
    indexes = ones_(tuple(select_shape), mstype.int32) * (index)
    return feat.gather_elements(dim, indexes).reshape(new_shape)


def trapz(y, x=None, *, dx=1.0, dim=-1):
    r"""
    Compute the trapezoidal rule along dim.

    The spacing between elements is specified by the tensor `x` or the scalar `dx` ,
    default ``1`` .

    .. math::

        \mathop{ \int }\nolimits_{{}}^{{}}{y}{ \left( {x} \right) } \text{d} x

    Args:
        y (Tensor): The input tensor to integrate.
        x (Tensor, optional): If specified, defines spacing between values.

    Keyword Args:
        dx (float, optional): The spacing between elements. Default ``1.0`` .
                              If `x` is specified, `dx` does not take effect.
        dim (int, optional): The dimension along which to compute the trapezoidal rule. Default ``-1`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> y = mindspore.tensor([[1., 2., 3.], [2., 3., 4.], [3., 2., 1.]])
        >>> # case 1: Integrate over a regular grid, with spacing 1.
        >>> output = mindspore.ops.trapz(y, dx=1.)
        >>> print(output)
        [4. 6. 4.]
        >>>
        >>> # case 2: Integrate over an irregular grid.
        >>> x = mindspore.tensor([[1, 2, 3], [1, 3, 5], [1, 4, 7]])
        >>> output = mindspore.ops.trapz(y, x)
        >>> print(output)
        [ 4. 12. 12.]
    """

    if not isinstance(y, (Tensor, Tensor_)):
        raise TypeError(f"For `trapz`, the input `y` must be Tensor, but get {type(y)}.")
    if not isinstance(dx, float):
        raise TypeError(f"For `trapz`, the input `dx` must be float, but get f{type(dx)}.")
    if not isinstance(dim, int):
        raise TypeError(f"For `trapz`, the input `dim` must be int, but get {type(dim)}.")
    if not _check_is_float(y.dtype):
        y = cast_(y, mstype.float32)
    _check_dim_in_range(dim, y.ndim)
    dim = dim + y.ndim if dim < 0 else dim
    if x is None:
        return trapezoid(y, dx, dim)
    if not isinstance(x, (Tensor, Tensor_)):
        raise TypeError(f"For `trapz`, the input `x` must be Tensor, but get {type(x)}.")
    x = cast_(x, mstype.float32)
    return trapezoid_tensor(y, x, dim)


def cholesky(input_x, upper=False):
    r"""
    Return the Cholesky decomposition of zero or more batch dimensions consisting of symmetric positive-definite
    matrices.

    If `upper` is `True`, return an upper-triangular matrix, :math:`U`, and the decomposition has the form:

    .. math::
        A = U^TU

    If `upper` is `False`, return a lower-triangular matrix, :math:`L`, and the decomposition has the form:

    .. math::
        A = LL^T

    where `A` is the symmetric positive-definite matrix.

    Args:
        input_x (Tensor): The input tensor of shape :math:`(*, N, N)`, where :math:`*` is batch dimensions.
            In the above formula, :math:`A` .
        upper (bool, optional): Return an upper-triangular matrix or not. Default: ``False`` .

    Returns:
        Tensor

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input_x = mindspore.tensor([[1.0, 1.0], [1.0, 2.0]])
        >>> output = mindspore.ops.cholesky(input_x, upper=False)
        >>> print(output)
        [[1. 0.]
         [1. 1.]]
    """
    cholesky_op = _get_cache_prim(P.Cholesky)(upper=upper)
    return cholesky_op(input_x)


def cholesky_inverse(input_x, upper=False):
    r"""
    Returns the inverse of the positive definite matrix using cholesky matrix factorization by its Cholesky factor.

    If `upper` is `True`, :math:`U` is an upper triangular such that the output tensor is

    .. math::

        inv = (U^{T}U)^{-1}

    If `upper` is `False`, :math:`U` is a lower triangular such that the output tensor is

    .. math::

        inv = (UU^{T})^{-1}

    Note:
        The input must be either an upper-triangular matrix or a lower-triangular matrix from Cholesky decomposition.

    Args:
        input_x (Tensor): The input tensor with a rank of 2. Supported dtypes: float32, float64.
        upper (bool): If `upper` is `True`, return an upper triangular matrix. If `upper` is `False`, return
            a lower-triangular matrix. Default: ``False``.

    Returns:
        Tensor, has the same shape and dtype as `input_x`.

    Raises:
        TypeError: If `input_x` is not a Tensor.
        TypeError: If dtype of `input_x` is not one of: float32, float64.
        ValueError: If the dimension of `input_x` is not equal to 2.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input_x = Tensor(np.array([[2,0,0], [4,1,0], [-1,1,2]]), mindspore.float32)
        >>> output = ops.cholesky_inverse(input_x)
        >>> print(output)
        [[ 5.8125 -2.625   0.625 ]
         [-2.625   1.25   -0.25  ]
         [ 0.625  -0.25    0.25  ]]
    """
    cholesky_inv_op = _get_cache_prim(P.CholeskyInverse)(upper=upper)
    return cholesky_inv_op(input_x)


def cholesky_solve(input, input2, upper=False):
    r"""
    Computes the solution of a set of linear equations with a positive definite matrix,
    according to its Cholesky decomposition factor `input2` .

    If `upper` is set to ``True`` and `input2` is upper triangular, the output tensor is that:

    .. math::
        output = (input2^{T} * input2)^{{-1}}input

    If `upper` is set to ``False`` and `input2` is lower triangular, the output is that:

    .. math::
        output = (input2 * input2^{T})^{{-1}}input

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): The input tensor of shape :math:`(*, N, M)`.
        input2 (Tensor): The tensor of shape :math:`(*, N, N)` , composed of
            upper or lower triangular Cholesky factor.
        upper (bool, optional): Whether to treat the Cholesky factor as an upper triangular matrix. Default ``False`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input1 = mindspore.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        >>> input2 = mindspore.tensor([[2., 0., 0.], [4., 1., 0.], [-1., 1., 2.]])
        >>> out = mindspore.ops.cholesky_solve(input1, input2, upper=False)
        >>> print(out)
        [[ 5.8125 -2.625   0.625 ]
         [-2.625   1.25   -0.25  ]
         [ 0.625  -0.25    0.25  ]]
    """
    return _get_cache_prim(P.CholeskySolve)(upper)(input, input2)


def cross(input, other, dim=None):
    r"""
    Compute the cross product of two input tensors along the specified dimension.

    Note:
        `input` and `other` must have the same shape, and the size of their `dim` dimension should be `3`.
        If `dim` is not specified, it is set to be the first dimension found with the size `3`.

    Args:
        input (Tensor): The first input tensor.
        other (Tensor):  The second input tensor.
        dim (int, optional): Specify the dimension for computation. Default ``None``.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        >>> other = mindspore.tensor([[4, 5, 6], [4, 5, 6], [4, 5, 6]])
        >>> mindspore.ops.cross(input, other)
        Tensor(shape=[3, 3], dtype=Int64, value=
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]])
        >>> mindspore.ops.cross(input, other, dim=1)
        Tensor(shape=[3, 3], dtype=Int64, value=
        [[-3,  6, -3],
         [-3,  6, -3],
         [-3,  6, -3]])
    """
    if dim is None:
        dim = -65530
    return cross_impl(input, other, dim)


def _einsum_convert_num_to_char(num):
    """For einsum, convert number into char."""
    if [num] == [Ellipsis]:
        return '...'
    if 0 <= num < 26:
        return chr(num + 65)
    if 26 <= num < 52:
        return chr(num + 71)
    raise ValueError(f"For Einsum, the number in sublist should be in range [0, 52), but got {num}")


def einsum(equation, *operands):
    r"""
    According to the Einstein summation Convention (Einsum),
    the product of the input tensor elements is summed along the specified dimension.

    Note:
        - The sublist format is also supported. For example, ops.einsum(op1, sublist1, op2, sublist2, ..., sublist_out).
          In this format, equation can be derived by the sublists which are made up of Python's Ellipsis and list of
          integers in [0, 52). Each operand is followed by a sublist and an output sublist is at the end.
        - The value can contain only letters, commas, ellipsis and arrow. The letters represent input tensor dimension,
          commas represent separate tensors, ellipsis indicates the tensor dimension that you do not care about, the
          left of the arrow indicates the input tensors, and the right of it indicates the desired output dimension.

    Args:
        equation (str): Notation based on the Einstein summation convention.
        operands (Tensor): The input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([1.0, 2.0, 4.0], mindspore.float32)
        >>> equation = "i->"
        >>> output = mindspore.ops.einsum(equation, x)
        >>> print(output)
        [7.]
        >>> x = mindspore.tensor([1.0, 2.0, 4.0], mindspore.float32)
        >>> y = mindspore.tensor([2.0, 4.0, 3.0], mindspore.float32)
        >>> equation = "i,i->i"
        >>> output = mindspore.ops.einsum(equation, x, y)
        >>> print(output)
        [ 2. 8. 12.]
        >>> x = mindspore.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], mindspore.float32)
        >>> y = mindspore.tensor([[2.0, 3.0], [1.0, 2.0], [4.0, 5.0]], mindspore.float32)
        >>> equation = "ij,jk->ik"
        >>> output = mindspore.ops.einsum(equation, x, y)
        >>> print(output)
        [[16. 22.]
         [37. 52.]]
        >>> x = mindspore.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], mindspore.float32)
        >>> equation = "ij->ji"
        >>> output = mindspore.ops.einsum(equation, x)
        >>> print(output)
        [[1. 4.]
         [2. 5.]
         [3. 6.]]
        >>> x = mindspore.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], mindspore.float32)
        >>> equation = "ij->j"
        >>> output = mindspore.ops.einsum(equation, x)
        >>> print(output)
        [5. 7. 9.]
        >>> x = mindspore.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], mindspore.float32)
        >>> equation = "...->"
        >>> output = mindspore.ops.einsum(equation, x)
        >>> print(output)
        [21.]
        >>> x = mindspore.tensor([1.0, 2.0, 3.0], mindspore.float32)
        >>> y = mindspore.tensor([2.0, 4.0, 1.0], mindspore.float32)
        >>> equation = "j,i->ji"
        >>> output = mindspore.ops.einsum(equation, x, y)
        >>> print(output)
        [[ 2. 4. 1.]
         [ 4. 8. 2.]
         [ 6. 12. 3.]]
        >>> x = mindspore.tensor([1, 2, 3, 4], mindspore.float32)
        >>> y = mindspore.tensor([1, 2], mindspore.float32)
        >>> output = mindspore.ops.einsum(x, [..., 1], y, [..., 2], [..., 1, 2])
        >>> print(output)
        [[1. 2.]
         [2. 4.]
         [3. 6.]
         [4. 8.]]
    """
    if isinstance(equation, Tensor):
        equ_tmp = ''
        for i, lst in enumerate(operands):
            if i % 2 == 0:
                for _, num in enumerate(lst):
                    equ_tmp += _einsum_convert_num_to_char(num)
                if i in (len(operands) - 1, len(operands) - 2):
                    continue
                equ_tmp += ','
        if len(operands) % 2 == 0:
            equ_tmp += '->'
            for _, num in enumerate(operands[-1]):
                equ_tmp += _einsum_convert_num_to_char(num)
            operands_tmp = list([equation]) + list(operands[1:-1:2])
        else:
            operands_tmp = list([equation]) + list(operands[1::2])
        equation = equ_tmp
        operands = tuple(operands_tmp)
    return _get_cache_prim(P.Einsum)(equation)(operands)


def _einsum_convert_sublist(equation, operands):
    """Convert the sublist to an equation operand if the received input is a sublist format."""
    def _einsum_convert_sublist_to_label(num, ell_num=False):
        """Convert sublist to label."""
        if num == Ellipsis or ell_num and num == 52:
            return '...'
        if 0 <= num < 26:
            return chr(num + ord('A'))
        if 26 <= num < 52:
            return chr(num + ord('a') - 26)
        raise ValueError(
            f'For einsum, the number in sublist must be in range [0, 52), but got {num}')

    if isinstance(equation, Tensor):
        equation_tmp = ''
        for i, lst in enumerate(operands):
            if i % 2 == 0:
                for _, num in enumerate(lst):
                    equation_tmp += _einsum_convert_sublist_to_label(num)
                if i in (len(operands) - 1, len(operands) - 2):
                    continue
                equation_tmp += ','
        if len(operands) % 2 == 0:
            equation_tmp += '->'
            for _, num in enumerate(operands[-1]):
                equation_tmp += _einsum_convert_sublist_to_label(num)
            operands_tmp = list([equation]) + list(operands[1:-1:2])
        else:
            operands_tmp = list([equation]) + list(operands[1::2])
        equation = equation_tmp
        operands = tuple(operands_tmp)
    if len(operands) == 0:  # pylint: disable=len-as-condition
        raise ValueError(
            "For einsum, the 'operands' must have at least one operand.")
    return equation, operands


def einsum_ext(equation, *operands):
    r"""
    According to the Einstein summation Convention (Einsum),
    the product of the input tensor elements is summed along the specified dimension.
    You can use this operator to perform diagonal, reducesum, transpose, matmul, mul, inner product operations, etc.

    Note:
        The sublist format is also supported. For example, einsum_ext(op1, sublist1, op2, sublist2, ..., sublist_out).
        In this format, equation can be derived by the sublists which are made up of Python's Ellipsis and list of
        integers in [0, 52). Each operand is followed by a sublist and an output sublist is at the end.
        Dynamic shape, dynamic rank input is not supported in `graph mode (mode=mindspore.GRAPH_MODE)
        <https://www.mindspore.cn/tutorials/en/master/compile/static_graph.html>`_.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        equation (str): Notation based on the Einstein summation convention, represent the operation you want to do.
            the value can contain only letters, commas, ellipsis and arrow. The letters(must be in [a-zA-Z]) represent
            input tensor dimension, commas(,) represent separate tensors, ellipsis indicates the tensor dimension that
            you do not care about, the left of the arrow indicates the input tensors, and the right of it indicates the
            desired output dimension. If there are no arrows in the equation, the letters that appear exactly once in
            the equation will be part of the output, sorted in increasing alphabetical order. The output is computed by
            multiplying the input operands element-wise, with their dimensions aligned based on the letters, and then
            summing out the dimensions whose letters are not part of the output. If there is one arrow in the equation,
            the output letters must appear at least once for some input operand and at most once for the output.
        operands (Tensor): Input tensor used for calculation. The dtype of the tensor must be the same.

    Returns:
        Tensor, the shape of it can be obtained from the `equation` , and the dtype is the same as input tensors.

    Raises:
        TypeError: If `equation` is invalid, or the `equation` does not match the input tensor.
        ValueError: If the number in sublist is not in [0, 52) in sublist format.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
        >>> equation = "i->"
        >>> output = ops.einsum_ext(equation, x)
        >>> print(output)
        7.0
        >>> x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
        >>> y = Tensor(np.array([2.0, 4.0, 3.0]), mindspore.float32)
        >>> equation = "i,i->i"
        >>> output = ops.einsum_ext(equation, x, y)
        >>> print(output)
        [ 2. 8. 12.]
        >>> x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mindspore.float32)
        >>> y = Tensor(np.array([[2.0, 3.0], [1.0, 2.0], [4.0, 5.0]]), mindspore.float32)
        >>> equation = "ij,jk->ik"
        >>> output = ops.einsum_ext(equation, x, y)
        >>> print(output)
        [[16. 22.]
         [37. 52.]]
        >>> x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mindspore.float32)
        >>> equation = "ij->ji"
        >>> output = ops.einsum_ext(equation, x)
        >>> print(output)
        [[1. 4.]
         [2. 5.]
         [3. 6.]]
        >>> x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mindspore.float32)
        >>> equation = "ij->j"
        >>> output = ops.einsum_ext(equation, x)
        >>> print(output)
        [5. 7. 9.]
        >>> x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mindspore.float32)
        >>> equation = "...->"
        >>> output = ops.einsum_ext(equation, x)
        >>> print(output)
        21.0
        >>> x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
        >>> y = Tensor(np.array([2.0, 4.0, 1.0]), mindspore.float32)
        >>> equation = "j,i->ji"
        >>> output = ops.einsum_ext(equation, x, y)
        >>> print(output)
        [[ 2. 4. 1.]
         [ 4. 8. 2.]
         [ 6. 12. 3.]]
        >>> x = mindspore.Tensor([1, 2, 3, 4], mindspore.float32)
        >>> y = mindspore.Tensor([1, 2], mindspore.float32)
        >>> output = ops.einsum_ext(x, [..., 1], y, [..., 2], [..., 1, 2])
        >>> print(output)
        [[1. 2.]
         [2. 4.]
         [3. 6.]
         [4. 8.]]
    """
    _equation, _operands = _einsum_convert_sublist(equation, operands)

    return ops.functional_overload.einsum(_equation, _operands)


def cumprod(input, dim, dtype=None):
    r"""
    Return the cumulative product along the given dimension of the tensor.

    .. math::

        y_i = x_1 * x_2 * x_3 * ... * x_i

    Args:
        input (Tensor): The input tensor.
        dim (int): Specify the dimension for computation.
        dtype (:class:`mindspore.dtype`, optional): The data type returned. Default ``None`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[1, 2, 3],
        ...                           [4, 5, 6]])
        >>> mindspore.ops.cumprod(input, dim=0)
        Tensor(shape=[2, 3], dtype=Int64, value=
        [[ 1,  2,  3],
         [ 4, 10, 18]])
        >>> mindspore.ops.cumprod(input, dim=1)
        Tensor(shape=[2, 3], dtype=Int64, value=
        [[  1,   2,   6],
         [  4,  20, 120]])
    """
    output = cumprod_(input, dim)
    if dtype:
        output = cast_(output, dtype)
    return output


def igamma(input, other):
    r"""
    Calculates lower regularized incomplete Gamma function.

    If we define `input` as `a` and `other` as `x`, the lower regularized incomplete Gamma function is defined as:

    .. math::
        P(a, x) = Gamma(a, x) / Gamma(a) = 1 - Q(a, x)

    where

    .. math::
        Gamma(a, x) = \int_0^x t^{a-1} \exp^{-t} dt

    is the lower incomplete Gamma function.

    Above :math:`Q(a, x)` is the upper regularized complete Gamma function.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): The first input tensor.
        other (Tensor): The second input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([2.0, 4.0, 6.0, 8.0])
        >>> other = mindspore.tensor([2.0, 3.0, 4.0, 5.0])
        >>> output = mindspore.ops.igamma(input, other)
        >>> print(output)
        [0.5939941  0.35276785 0.21486954 0.13337176]
    """
    return igamma_(input, other)


def igammac(input, other):
    r"""
    Calculates upper regularized incomplete Gamma function.

    If we define `input` as `a` and `other` as `x`, the upper regularized incomplete Gamma function is defined as:

    .. math::
        Q(a, x) = Gamma(a, x) / Gamma(a) = 1 - P(a, x)

    where

    .. math::
        Gamma(a, x) = \int_{x}^{\infty} t^{a-1} exp(-t) dt

    is the upper incomplete Gama function.

    Above :math:`P(a, x)` is the lower regularized incomplete Gamma function.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): The first input tensor.
        other (Tensor): The second input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([2.0, 4.0, 6.0, 8.0])
        >>> other = mindspore.tensor([2.0, 3.0, 4.0, 5.0])
        >>> output = mindspore.ops.igammac(input, other)
        >>> print(output)
        [0.40600574 0.6472322  0.78513044 0.8666282 ]
    """
    return igammac_(input, other)


def lgamma(input):
    r"""
    Computes the natural logarithm of the absolute value of the gamma function on input.

    .. math::
        \text{out}_{i} = \ln \Gamma(|\text{input}_{i}|)

    Args:
        input (Tensor): The input tensor. With type of float16 or float32 or float64.

    Returns:
        Tensor, has the same dtype as `input`.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If dtype of `input` is not float16 or float32 or float64.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> x = Tensor(np.array([0.5, 3.2, 8.5]), mindspore.float32)
        >>> output = ops.lgamma(x)
        >>> print(output)
        [0.5723649 0.8854049 9.549267 ]
        >>> x = Tensor(2.1, mindspore.float32)
        >>> output = ops.lgamma(x)
        >>> print(output)
        0.045437694
    """
    return lgamma_(input)


def digamma(input):
    r"""
    Computes the logarithmic derivative of the gamma function on input tensor.

    .. math::
        P(x) = \frac{d}{dx}(\ln (\Gamma(x)))

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): The input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([1.5, 0.5, 9])
        >>> output = mindspore.ops.digamma(input)
        >>> print(output)
        [ 0.03648992 -1.9635109   2.1406415 ]
    """
    return digamma_(input)


def polygamma(n, input):
    r"""
    Computes the :math:`n`-th derivative of the polygamma function on `input`.

    .. math::
        \psi^{(a)}(x) = \frac{d^{(a)}}{dx^{(a)}} \psi(x)

    where :math:`\psi(x)` is the digamma function.

    Args:
        n (Tensor): The order of the polygamma function.
        input (Tensor): The input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([3.14, -2.71], mindspore.float64)
        >>> a = mindspore.tensor(1, mindspore.int64)
        >>> output = mindspore.ops.polygamma(a, x)
        >>> print(output)
        [ 0.37446456 15.49884838]
    """
    return poly_gamma_(n, input)


def _is_sign_inf(x, fn):
    """Tests element-wise for infinity with sign."""
    shape = x.shape
    zeros_tensor = zeros_(shape, mstype.float32)
    ones_tensor = ones_(shape, mstype.float32)
    is_inf = isinf(x)
    is_sign = fn(x, zeros_tensor)
    res = ops.select(is_inf, ones_tensor, zeros_tensor)
    res = ops.select(is_sign, res, zeros_tensor)
    return cast_(res, mstype.bool_)


def isposinf(input):
    """
    Return a boolean tensor indicating which elements are positive infinity.

    .. warning::
        For Ascend, it is only supported on platforms above Atlas A2.

    Args:
        input (Tensor): The input tensor.

    Returns:
       Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([-1, 3, float("inf"), float("-inf"), float("nan")])
        >>> mindspore.ops.isposinf(input)
        Tensor(shape=[5], dtype=Bool, value= [False, False,  True, False, False])

    """
    _check_is_tensor("input", input, "isposinf")
    return _is_sign_inf(input, tensor_gt)


def isneginf(input):
    """
    Return whether each element in the input is a negative infinity number.

    .. warning::
        For Ascend, it is only supported on platforms above Atlas A2.

    Args:
        input (Tensor): The input tensor.

    Returns:
       Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([-1, 3, float("inf"), float("-inf"), float("nan")])
        >>> mindspore.ops.isneginf(input)
        Tensor(shape=[5], dtype=Bool, value= [False, False, False,  True, False])
    """
    _check_is_tensor("input", input, "isneginf")
    return _is_sign_inf(input, tensor_lt)


def logical_xor(input, other):
    r"""
    Compute the "logical XOR" of two tensors element-wise.

    .. math::

        out_{i} = input_{i} \oplus other_{i}

    .. note::
        - Broadcasting is supported.
        - Support implicit type conversion.

    Args:
        input (Tensor): The first input tensor.
        other (Tensor): The second input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([True, False, True], mindspore.bool)
        >>> y = mindspore.tensor([True, True, False], mindspore.bool)
        >>> output = mindspore.ops.logical_xor(x, y)
        >>> print(output)
        [False  True  True]
        >>> x = mindspore.tensor(1, mindspore.bool)
        >>> y = mindspore.tensor(0, mindspore.bool)
        >>> output = mindspore.ops.logical_xor(x, y)
        >>> print(output)
        True
        >>> x = True
        >>> y = mindspore.tensor(0, mindspore.bool)
        >>> output = mindspore.ops.logical_xor(x, y)
        >>> print(output)
        True
        >>> x = True
        >>> y = mindspore.tensor([True, False], mindspore.bool)
        >>> output = mindspore.ops.logical_xor(x, y)
        >>> print(output)
        [False  True]
    """
    return logical_xor_(input, other)


def imag(input):
    r"""
    Return a new tensor containing imaginary value of the input tensor, element-wise.
    If element in the input tensor is real, it will return zero.

    Args:
        input (Tensor): The input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([0.22+0.3511j, -0.55-0.6796j, -1.92-0.033j, -0.38-0.2991j])
        >>> output = mindspore.ops.imag(input)
        >>> print(output)
        [ 0.3511 -0.6796 -0.033  -0.2991]
    """
    return imag_(input)


@_primexpr
def _check_repeat_in_axis(axis, x_ndim, prim_name):
    """check repeat dim in axis"""
    if isinstance(axis, (list, tuple)):
        axis_deal = [dim + x_ndim if dim < 0 else dim for dim in axis]
        for dim in axis_deal:
            if axis_deal.count(dim) > 1:
                raise RuntimeError(f"For {prim_name}, dim {dim} appears multiple times in axis.")


def nansum(input, axis=None, keepdims=False, *, dtype=None):
    """
    Computes sum of `input` over a given dimension, ignoring NaN.

    Args:
        input (Tensor): The input tensor.
        axis (Union[int, tuple(int)], optional): The dimensions to reduce. Supposed the rank of `input` is r,
            axis must be in the range [-rank(input), rank(input)). Default ``None``, all dimensions are reduced.
        keepdims (bool, optional): Whether the output tensor keeps has dim retained. Default ``False``.

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): The dtype of output tensor. Default ``None``.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:

        >>> import mindspore
        >>> x = mindspore.tensor([[float("nan"), 2, 3], [1, float("nan"), 3], [1, 2, float("nan")]], mindspore.float32)
        >>> # case1: axis is None, keepdims is False,
        >>> output1 = mindspore.ops.nansum(x, axis=None, dtype=mindspore.float32)
        >>> print(output1)
        12.0
        >>> # case2: axis is int, set as 0, and keepdims is False
        >>> output2 = mindspore.ops.nansum(x, axis=0, dtype=mindspore.float32)
        >>> print(output2)
        [2. 4. 6.]
        >>> # case3: axis is int, set as 0, and keepdims is False
        >>> output3 = mindspore.ops.nansum(x, axis=0, keepdims=True, dtype=mindspore.float32)
        >>> print(output3)
        [[2. 4. 6.]]
        >>> # case4: axis is tuple(int) or list(int), set as (0, 1), and keepdims is False
        >>> output4 = mindspore.ops.nansum(x, axis=(0, 1), dtype=mindspore.float32)
        >>> print(output4)
        12.0
    """
    _check_is_tensor("input", input, "nansum")
    _check_repeat_in_axis(axis, input.ndim, "nansum")
    if input.is_complex():
        raise TypeError(f'For nansum, input are not supported complex type, but got {input.dtype}.')
    if dtype is not None and dtype in mstype.complex_type:
        raise TypeError(f'For nansum, dtype not supported complex type, but got {dtype}.')
    if axis is None:
        axis = ()
    if input.dtype == mstype.bool_:
        input = input.astype(mstype.int64)
    is_nan = isnan_(input)
    input = ops.masked_fill(input, is_nan, ops.cast(0, input.dtype))
    input = _get_cache_prim(P.ReduceSum)(keepdims)(input, axis)
    if dtype is not None and input.dtype != dtype:
        input = input.astype(dtype)
    return input


def diag_embed(input, offset=0, dim1=-2, dim2=-1):
    r"""
    Create a tensor whose diagonals of certain 2D planes (specified by `dim1` and `dim2`) are filled by `input`, and
    all other positions are set to ``0``. The 2D planes formed by the last two dimensions of the returned tensor are
    chosen by default.

    Args:
        input (Tensor): Values to fill diagonal.
        offset (int, optional): Diagonal offset. Default ``0`` .

            - When `offset` is a positive integer, shift the diagonal upward.
            - When `offset` is a negative integer, shift the diagonal downward.
        dim1 (int, optional): The first dimension for diagonal filling. Default ``-2`` .
        dim2 (int, optional): The second dimension for diagonal filling. Default ``-1`` .

    Returns:
        A tensor with the same dtype as `input`, and with shape that has one dimension higher than the `input`.

    Raises:
        ValueError: If the dimension of `input` is not 1D-6D.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.tensor([[1, 2, 3],
        ...                           [4, 5, 6],
        ...                           [7, 8, 9]])
        >>> mindspore.ops.diag_embed(input)
        Tensor(shape=[3, 3, 3], dtype=Int64, value=
        [[[1, 0, 0], [0, 2, 0], [0, 0, 3]],
         [[4, 0, 0], [0, 5, 0], [0, 0, 6]],
         [[7, 0, 0], [0, 8, 0], [0, 0, 9]]])
        >>> mindspore.ops.diag_embed(input, offset=1, dim1=0, dim2=1)
        Tensor(shape=[4, 4, 3], dtype=Int64, value=
        [[[0, 0, 0], [1, 4, 7], [0, 0, 0], [0, 0, 0]],
         [[0, 0, 0], [0, 0, 0], [2, 5, 8], [0, 0, 0]],
         [[0, 0, 0], [0, 0, 0], [0, 0, 0], [3, 6, 9]],
         [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    """
    matrix_set_diag_op = MatrixSetDiagV3(align="LEFT_RIGHT")
    zeros = ops.Zeros()
    if not isinstance(input, (Tensor, Tensor_)):
        raise TypeError("For 'diag_embed', 'input' must be Tensor.")

    input_dtype = dtype_(input)
    if not (input_dtype in (mstype.int8, mstype.int16, mstype.int32, mstype.int64, mstype.uint8, mstype.uint16,
                            mstype.uint32, mstype.uint64, mstype.float16, mstype.float32, mstype.float64)):
        raise TypeError("For 'diag_embed', the dtype of 'input' must be int8, int16, int32, int64, "
                        f"uint8, uint16, uint32, uint64, float16, float32 or float64, but got '{input_dtype}'.")
    _check_attr_dtype("offset", offset, [int], "diag_embed")
    _check_attr_dtype("dim1", dim1, [int], "diag_embed")
    _check_attr_dtype("dim2", dim2, [int], "diag_embed")
    if len(input.shape) > 6:
        raise ValueError("For 'diag_embed', the dimension of 'input' must be 1-6D.")
    x_shape = input.shape
    output_dim = len(x_shape) + 1
    if dim1 < -output_dim or dim1 > (output_dim - 1):
        raise ValueError(f"For 'diag_embed', 'dim1' must be in range of [{-output_dim}, {output_dim - 1}], "
                         f"but got {dim1}.")
    if dim2 < -output_dim or dim2 > (output_dim - 1):
        raise ValueError(f"For 'diag_embed', 'dim2' must be in range of [{-output_dim}, {output_dim - 1}], "
                         f"but got {dim2}.")
    if dim1 < 0:
        dim1_ = dim1 + output_dim
    else:
        dim1_ = dim1
    if dim2 < 0:
        dim2_ = dim2 + output_dim
    else:
        dim2_ = dim2
    if dim1_ == dim2_:
        raise ValueError("For 'diag_embed', 'dim1' must not be identical to 'dim2'.")
    batch_shape = x_shape[:-1]
    if offset > 0:
        dsize = x_shape[-1] + offset
    else:
        dsize = x_shape[-1] - offset
    diag_plane = (dsize, dsize)
    output_shape_trans = batch_shape + diag_plane
    output = zeros(output_shape_trans, input.dtype)
    k = cast_(offset, mstype.int32)
    output = matrix_set_diag_op(output, input, k)
    dim = 0
    perm = ()
    for i in range(output_dim):
        if i == dim1_:
            perm = perm + (output_dim - 2,)
        elif i == dim2_:
            perm = perm + (output_dim - 1,)
        else:
            perm = perm + (dim,)
            dim = dim + 1
    return transpose_op(output, perm)


def sum(input, dim=None, keepdim=False, *, dtype=None):
    """
    Calculate sum of tensor elements over a given dim.

    Note:
        The `dim` with tensor type is only used for compatibility with older versions and is not recommended.

    Args:
        input (Tensor): The input tensor.
        dim (Union[None, int, tuple(int), list(int), Tensor]): Dimensions along which the sum is calculated.
            Default ``None`` .
        keepdim (bool): Whether the output tensor has dim retained or not.
            If ``True`` , keep these reduced dimensions and the length is 1.
            If ``False`` , will not keep these dimensions. Default ``False`` .

    Note:
        - If `dim` is ``None`` , sum is calculated on all the elements of the input tensor.
        - If `dim` is a tuple or list of ints or tensor, sum is calculated on all dimensions specified in  `dim` .

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): The data type returned.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([[[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]],
        ...                      [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
        ...                      [[7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9]]], mindspore.float32)
        >>> out = mindspore.ops.sum(input=x)
        >>> print(out)
        270.0
        >>> out = mindspore.ops.sum(input=x, dim=1)
        >>> print(out)
        [[ 6.  6.  6.  6.  6.  6.]
         [15. 15. 15. 15. 15. 15.]
         [24. 24. 24. 24. 24. 24.]]
        >>> out = mindspore.ops.sum(input=x, dim=2)
        >>> print(out)
        [[ 6. 12. 18.]
         [24. 30. 36.]
         [42. 48. 54.]]
        >>> out = mindspore.ops.sum(input=x, dim=[1, 2])
        >>> print(out)
        [ 36.  90. 144.]
        >>> out = mindspore.ops.sum(input=x, dim=2, keepdim=True)
        >>> print(out)
        [[[ 6.]
         [12.]
         [18.]]
        [[24.]
         [30.]
         [36.]]
        [[42.]
         [48.]
         [54.]]]
        >>> print(out.ndim)
        3
    """
    return sum_ext_op(input, dim, keepdim, dtype)


def tanhshrink(input):
    '''
    Apply the element-wise Tanhshrink function.

    .. math::

        Tanhshrink(x) = x - Tanh(x)

    Args:
        input (Tensor): The input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> output = mindspore.ops.tanhshrink(mindspore.tensor([1., 2., 3., 2., 1.]))
        >>> print(output)
        [0.23840582 1.0359724  2.0049443  1.0359724  0.23840582]
    '''
    if not isinstance(input, Tensor):
        raise TypeError(f"For tanhshrink, the input must be a Tensor, but got {type(input)}.")

    if input.dtype in mstype.int_type + mstype.uint_type:
        input = input.astype(mstype.float32)
    return input - tanh(input)


def zeta(input, other):
    r"""
    Elemental-wise compute the Hurwitz zeta function values.

    .. math::

        \zeta(x, q) = \sum_{k=0}^{\infty} \frac{1}{(k + q)^x}

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Union[Tensor, int, float]): The first input tensor. Represented as :math:`x` in the formula.
        other (Union[Tensor, int, float]): The second input tensor. Represented as :math:`q` in the formula.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> z = mindspore.ops.zeta(mindspore.tensor([10.]), mindspore.tensor([1.]))
        >>> print(z)
        [1.0009946]
    """
    if isinstance(input, (int, float)):
        if not isinstance(other, Tensor):
            raise TypeError("For 'zeta', at least one of the inputs should be Tensor.")
        _dtype = other.dtype
        input = cast_(input, _dtype)
    if isinstance(other, (int, float)):
        if not isinstance(input, Tensor):
            raise TypeError("For 'zeta', at least one of the inputs should be Tensor.")
        _dtype = input.dtype
        other = cast_(other, _dtype)
    if input.size < other.size:
        input = _get_cache_prim(P.BroadcastTo)(other.shape)(input)
    elif input.size > other.size:
        other = _get_cache_prim(P.BroadcastTo)(input.shape)(other)
    output = zeta_(input, other)
    return output


def matrix_power(input, n):
    """
    Raises a square matrix to the (integer) power `n` .

    - When :math:`n=0` , returns the identity matrix, which has the same shape as `input` .
    - When :math:`n<0` and `input` is invertible, returns the inverse of `input` to the power of :math:`-n` .

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): A 3-D Tensor. Supported data types are float16 and float32.
            The shape is :math:`(b, m, m)` , represents b m-D square matrices.
        n (int): The exponent, a required int.

    Returns:
        A 3-D Tensor. Data type and shape are the same as `input` 's.

    Raises:
        TypeError: If the data type of `n` is not int.
        TypeError: If the data type of `input` is neither float32 nor float16.
        TypeError: If `input` is not a Tensor.
        ValueError: If `input` is not a 3-D tensor.
        ValueError: If shape[1] and shape[2] of `input` are not the same.
        ValueError: If `n` is negative but got input `input` has singular matrices.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import Tensor, ops
        >>> input = Tensor([[[0, 1], [-1, 0]], [[1, 0], [0, -1]]], dtype=ms.float32)
        >>> y = ops.matrix_power(input, 2)
        >>> print(y)
        [[[-1.  0.]
          [-0. -1.]]
         [[ 1.  0.]
          [ 0.  1.]]]
    """
    matrix_power_ops = _get_cache_prim(P.MatrixPower)(n=n)
    return matrix_power_ops(input)


def _maybe_wrap_dims_n(ret_dim, input_dim):
    """Check the dim"""
    if input_dim <= 0:
        input_dim = 1

    min = -input_dim
    max = input_dim - 1
    for i, value in enumerate(ret_dim):
        dim = value
        if dim < min or dim > max:
            raise ValueError(f"Dimension out of range, it must be in range of [{min}, {max}], "
                             f"but got {dim}.")

        if dim < 0:
            ret_dim[i] = dim + input_dim
    return ret_dim


def _canonicalize_fft_shape_and_dim(input, shape, dim):
    """Check the input's shape and dim"""
    input_dim = input.ndim
    input_sizes = input.shape
    ret_dim = None
    ret_shape = None

    if dim is not None:
        ret_dim = list(dim)
        ret_dim = _maybe_wrap_dims_n(ret_dim, input_dim)
        # check if dim is duplicated
        set_ret_dim = set(ret_dim)
        if len(set_ret_dim) != len(ret_dim):
            raise ValueError("FFT dims must be unique.")

    if shape is not None:
        if dim is not None and len(dim) != len(shape):
            raise ValueError(f"shape and dim must have the same length, but now they are "
                             f"{len(dim)} and {len(shape)}.")
        if len(shape) > input_dim:
            raise ValueError(f"Got shape with {len(shape)} values but input tensor only "
                             f"has {input_dim} dimensions.")

        transform_ndim = len(shape)
        if dim is None:
            ret_dim = [0] * transform_ndim
            value = input_dim - transform_ndim
            for i in range(transform_ndim):
                ret_dim[i] = value + i

        ret_shape = [0] * transform_ndim
        for i in range(transform_ndim):
            if shape[i] == -1:
                ret_shape[i] = input_sizes[ret_dim[i]]
            else:
                ret_shape[i] = shape[i]
    elif dim is None:
        ret_dim = list(range(input_dim))
        ret_shape = [0] * input_dim
        for i in range(input_dim):
            ret_shape[i] = input_sizes[i]
    else:
        ret_shape = [0] * len(ret_dim)
        for i in range(len(ret_dim)):
            value = ret_dim[i]
            ret_shape[i] = input_sizes[value]

    for value in ret_shape:
        if value <= 0:
            raise ValueError(f"The value of ret_shape must be greater than 0, "
                             f"but got '{value}'.")

    return ret_shape, ret_dim


def as_strided(x, shape=None, strides=None):
    n = np.dtype(mstype._dtype_to_nptype(x.dtype)).itemsize  # pylint:disable=protected-access
    strides = tuple(np.array(strides) * n)
    if x.dtype == mstype.bfloat16:
        return Tensor(np.lib.stride_tricks.as_strided(x.float().asnumpy(), shape, strides, False, True), dtype=x.dtype)
    return Tensor(np.lib.stride_tricks.as_strided(x.asnumpy(), shape, strides, False, True), dtype=x.dtype)


def _resize_input(input, input_dim, ret_dim, ret_shape, input_sizes):
    """Resize the input"""
    paddings = [0] * input_dim * 2
    must_copy = False
    for i in range(len(ret_dim)):
        value = ret_dim[i]
        # resize input based on n & dim
        if ret_shape[i] == -1:
            continue

        if input_sizes[value] < ret_shape[i]:
            pad_idx = len(paddings) - 2 * value - 1
            paddings[pad_idx] = ret_shape[i] - input_sizes[value]
            must_copy = True

        if input_sizes[value] > ret_shape[i]:
            start_index = [0] * input_dim
            input_sizes[value] = ret_shape[i]
            input = slice_(input, start_index, input_sizes)

    if must_copy:
        paddings = np.reshape(paddings, (input_dim, 2)).tolist()
        paddings.reverse()
        paddings = (*paddings,)
        input = _get_cache_prim(P.Pad)(paddings)(input)

    return input


def _permute_input(input, input_dim, ret_dim):
    """Permute input based on dim"""
    dim_permute = list(range(input_dim))
    # is_transformed_dim
    is_transformed_dim = [0] * input_dim
    for value in ret_dim:
        is_transformed_dim[value] = True

    # partition dim_permute
    dim_permute_a, dim_permute_b = [], []
    for i in range(len(dim_permute)):
        value = dim_permute[i]
        (dim_permute_a if not is_transformed_dim[i] else dim_permute_b).append(value)

    # strides
    type_size = np.dtype(mstype._dtype_to_nptype(input.dtype)).itemsize  # pylint:disable=protected-access
    input_strides = [int(x / type_size) for x in input.strides]

    def cmp(x, y):
        if input_strides[x] > input_strides[y]:
            return -1
        if input_strides[x] < input_strides[y]:
            return 1
        return 0

    # sort
    if dim_permute_a:
        dim_permute_a = sorted(dim_permute_a, key=cmp_to_key(cmp))

    # copy
    if dim_permute_b:
        ret_dim = sorted(ret_dim, key=cmp_to_key(cmp))
        for i in range(len(ret_dim)):
            value = ret_dim[i]
            dim_permute_b[i] = value

    # merge
    dim_permute = dim_permute_a + dim_permute_b

    # permute
    input = transpose_op(input, tuple(dim_permute))

    return input, dim_permute


def _reshape_input(input, signal_ndim, batch_dims):
    """Reshape input"""
    # Collapse batch dimensions into a single dimension
    batched_sizes = [0] * (signal_ndim + 1)
    batched_sizes[0] = -1
    i = batch_dims
    j = 1
    while i < len(input.shape):
        batched_sizes[j] = input.shape[i]
        j += 1
        i += 1
        if j >= len(batched_sizes):
            break
    input = reshape_(input, tuple(batched_sizes))
    return input


def _check_fftwithsize_input(input, s, dim, norm, fft_func_name):  # pylint: disable=redefined-outer-name
    """Check the input of fftwithsize"""
    if not isinstance(input, (Tensor, Tensor_)):
        raise TypeError("For '{fft_func_name}', 'input' must be Tensor.")

    input_dtype = dtype_(input)
    if fft_func_name in ('FFTN', 'IFFTN'):
        if not input_dtype in (mstype.complex64, mstype.complex128):
            raise TypeError("For '{fft_func_name}', the dtype of 'input' must be complex64, complex128, "
                            f"but got '{input_dtype}'.")
    else:
        raise TypeError("For '{fft_func_name}', it is not supported now.")

    if s is not None:
        if isinstance(s, int):
            s = (s,)
        elif not isinstance(s, tuple):
            raise TypeError("For '{fft_func_name}', 's' must be tuple(int).")
        for ele in s:
            if not isinstance(ele, int):
                raise TypeError(f"For '{fft_func_name}', each elements of 's' must be int, but got {type(ele)}")

    if dim is not None:
        if isinstance(dim, int):
            dim = (dim,)
        elif not isinstance(dim, tuple):
            raise TypeError("For '{fft_func_name}', 'dim' must be tuple(int).")
        for ele in dim:
            if not isinstance(ele, int):
                raise TypeError(f"For '{fft_func_name}', each elements of 'dim' must be int, but got {type(ele)}")

    ret_shape, ret_dim = _canonicalize_fft_shape_and_dim(input, s, dim)
    input_dim = input.ndim
    signal_ndim = len(ret_dim)
    batch_dims = input_dim - signal_ndim
    input_sizes = list(input.shape)
    dim_permute = None
    out_sizes = None

    if fft_func_name in ('FFTN', 'IFFTN'):
        input = _resize_input(input, input_dim, ret_dim, ret_shape, input_sizes)
        out_sizes = input.shape
        input, dim_permute = _permute_input(input, input_dim, ret_dim)

    input = _reshape_input(input, signal_ndim, batch_dims)

    if norm is None:
        norm = "backward"
    else:
        _check_attr_dtype("norm", norm, [str], fft_func_name)

    FFTInput = collections.namedtuple('FFTInput', ['input', 'signal_ndim', 'norm', 'input_dim',
                                                   'batch_dims', 'dim_permute', 'out_sizes'])
    return FFTInput(input=input, signal_ndim=signal_ndim, norm=norm, input_dim=input_dim,
                    batch_dims=batch_dims, dim_permute=dim_permute, out_sizes=out_sizes)


def _handle_fftwithsize_output(out, input_dim, batch_dims, dim_permute, out_sizes):
    """Handle the output of fftwithsize"""
    out_strides = [0] * input_dim
    batch_numel = 1
    for i in range(batch_dims - 1, -1, -1):
        out_strides[dim_permute[i]] = batch_numel * out.strides[0]
        batch_numel *= out_sizes[dim_permute[i]]

    for i in range(batch_dims, input_dim):
        out_strides[dim_permute[i]] = out.strides[1 + (i - batch_dims)]

    type_size = np.dtype(mstype._dtype_to_nptype(out.dtype)).itemsize  # pylint:disable=protected-access
    if out.shape != out_sizes or out.strides != out_strides:
        out = as_strided(out, out_sizes, [int(i / type_size) for i in out_strides])
    return out


def fft(input, n=None, dim=-1, norm=None):  # pylint: disable=redefined-outer-name
    r"""
    Calculates the one dimensional discrete Fourier transform of `input`.

    Args:
        input (Tensor): The input tensor.
        n (int, optional): Signal length.
            If given, the input will either be zero-padded or trimmed to this length before computing the FFT.
            Default: ``None``.
        dim (int, optional): The dimension along which to take the one dimensional FFT.
            Default: -1.
        norm (string, optional): Normalization mode. Three modes are defined as,
            ``"forward"`` (normalize by :math `1/n`), ``"backward"``(no normalization),
            ``"ortho"`` (normalize by :math:`1/\sqrt{n}`).
            Default: ``None`` that means ``"backward"``.

    Returns:
        Tensor, The result of `fft()` function.

    Raises:
        TypeError: If the `input` type is not Tensor.
        TypeError: If the `input` data type is not one of: complex64, complex128.
        TypeError: If `n` or `dim` type is not int32.
        ValueError: If `input` dimension is less than 1.
        ValueError: If `n` is less than 1.
        ValueError: If `dim` is not in the range of "[ `-input_dim` , `input_dim-1` ]".
        ValueError: If norm is none of "backward", "forward" or "ortho".

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor, ops
        >>> input = Tensor([ 1.6243454+0.j, -0.6117564+0.j, -0.5281718+0.j, -1.0729686+0.j])
        >>> y = ops.fft(input)
        >>> print(y)
        [-0.5885514+0.j          2.1525173-0.46121222j  2.7808986+0.j
         2.1525173+0.46121222j]
    """
    if not isinstance(input, (Tensor, Tensor_)):
        raise TypeError("For 'FFT', 'input' must be Tensor.")

    input_dtype = dtype_(input)
    if not input_dtype in (mstype.complex64, mstype.complex128):
        raise TypeError("For 'FFT', the dtype of 'input' must be complex64, complex128, "
                        f"but got '{input_dtype}'.")
    _check_attr_dtype("dim", dim, [int], "FFT")

    input_dim = input.ndim
    signal_ndim = 1
    batch_dims = input_dim - signal_ndim
    input_sizes = list(input.shape)
    dim = _maybe_wrap_dims_n([dim], input_dim)[0]
    n_opt = n
    if n is None:
        n = input.shape[dim]
    else:
        _check_attr_dtype("n", n, [int], "FFT")
    if n < 1:
        raise ValueError("For 'FFT', the value of 'n' must be greater than or equal to 1, "
                         f"but got '{n}'.")
    if n_opt is not None:
        input = _resize_input(input, input_dim, [dim], [n], input_sizes)
    out_sizes = input.shape

    input, dim_permute = _permute_input(input, input_dim, [dim])
    input = _reshape_input(input, signal_ndim, batch_dims)

    if norm is None:
        norm = "backward"
    else:
        _check_attr_dtype("norm", norm, [str], "FFT")

    fft_ = FFTWithSize(signal_ndim=1, inverse=False, real=False, norm=norm)
    out = fft_(input)
    return _handle_fftwithsize_output(out, input_dim, batch_dims, dim_permute, out_sizes)


def fft2(input, s=None, dim=(-2, -1), norm=None):  # pylint: disable=redefined-outer-name
    r"""
    Calculates the two dimensional discrete Fourier transform of `input`.

    Args:
        input (Tensor): The input tensor.
        s (Tuple[int], optional): Signal size in the transformed dimensions.
            If given, each dimension `dim[i]` will either be zero-padded or trimmed to the length `s[i]` before
            computing the FFT. If a length `-1` is specified, no padding is done in that dimension.
            Default: `s = [input.size(d) for d in dim]`
        dim (Tuple[int], optional): Dimensions to be transformed.
            Default: last two dimensions.
        norm (string, optional): Normalization mode. Three modes are defined as,
            ``"forward"``(normalize by :math `1/n`), ``"backward"``(no normalization),
            ``"ortho"``(normalize by :math:`1/\sqrt{n}`). Where :math `n = prod(s)` is the logical FFT size.
            Default: ``None`` that means ``"backward"``.

    Returns:
        Tensor, The result of `fft2()` function.

    Raises:
        TypeError: If the `input` type is not Tensor.
        TypeError: If the `input` data type is not one of: complex64, complex128.
        TypeError: If the `s` or `dim` is not tuple(int).
        ValueError: If `input` dimension is less than 2.
        ValueError: If the length of `s` and `dim` are not the same.
        ValueError: If the value in `dim` is not in the range of "[ `-input_dim` , `input_dim-1` ]".
        ValueError: If norm is none of "backward", "forward" or "ortho".

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor, ops
        >>> input = Tensor([[ 1.6243454+0.j, -0.6117564+0.j], [-0.5281718+0.j, -1.0729686+0.j]])
        >>> y = ops.fft2(input)
        >>> print(y)
        [[-0.5885514+0.j  2.7808986+0.j]
        [ 2.6137294+0.j  1.691305 +0.j]]
    """
    return fftn(input, s, dim, norm)


def fftn(input, s=None, dim=None, norm=None):  # pylint: disable=redefined-outer-name
    r"""
    Calculates the N dimensional discrete Fourier transform of `input`.

    Args:
        input (Tensor): The input tensor.
        s (Tuple[int], optional): Signal size in the transformed dimensions.
            If given, each dimension `dim[i]` will either be zero-padded or trimmed to the length `s[i]` before
            computing the FFT. If a length `-1` is specified, no padding is done in that dimension.
            Default: `s = [input.size(d) for d in dim]`
        dim (Tuple[int], optional): Dimensions to be transformed.
            Default: all dimensions, or the last `len(s)` dimensions if `s` is given.
        norm (string, optional): Normalization mode. Three modes are defined as,
            ``"forward"``(normalize by :math `1/n`), ``"backward"``(no normalization),
            ``"ortho"``(normalize by :math:`1/\sqrt{n}`). Where :math `n = prod(s)` is the logical FFT size.
            Default: ``None`` that means ``"backward"``.

    Returns:
        Tensor, The result of `fftn()` function.

    Raises:
        TypeError: If the `input` type is not Tensor.
        TypeError: If the `input` data type is not one of: complex64, complex128.
        TypeError: If the `s` or `dim` is not tuple(int).
        ValueError: If the length of `s` and `dim` are not the same.
        ValueError: If `input` dimension is less than 1.
        ValueError: If the value in `dim` is not in the range of "[ `-input_dim` , `input_dim-1` )".
        ValueError: If norm is none of "backward", "forward" or "ortho".

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor, ops
        >>> input = Tensor([[[ 1.6243454 +0.j, -0.6117564 +0.j, -0.5281718 +0.j],
        ...                  [-1.0729686 +0.j, 0.86540765+0.j, -2.3015387 +0.j]]])
        >>> y = ops.fftn(input)
        >>> print(y)
        [[[-2.02468245+0.j          1.83940642-2.6702696j
           1.83940642+2.6702696j ]
         [ 2.99351685+0.j          2.54921257+2.81504238j
           2.54921257-2.81504238j]]]
    """
    fftninput = _check_fftwithsize_input(input, s, dim, norm, "FFTN")
    fftn_ = FFTWithSize(signal_ndim=fftninput.signal_ndim, inverse=False, real=False, norm=fftninput.norm)
    out = fftn_(fftninput.input)
    return _handle_fftwithsize_output(out, fftninput.input_dim, fftninput.batch_dims,
                                      fftninput.dim_permute, fftninput.out_sizes)


def ifft(input, n=None, dim=-1, norm=None):  # pylint: disable=redefined-outer-name
    r"""
    Calculates the inverse of `fft()`.

    Args:
        input (Tensor): The input tensor.
        n (int, optional): Signal length.
            If given, the input will either be zero-padded or trimmed to this length before computing the IFFT.
            Default: ``None``.
        dim (int, optional): The dimension along which to take the one dimensional IFFT.
            Default: -1.
        norm (string, optional): Normalization mode. Three modes are defined as,
            ``"forward"``(normalize by :math `1/n`), ``"backward"``(no normalization),
            ``"ortho"``(normalize by :math:`1/\sqrt{n}`).
            Default: ``None`` that means ``"backward"``.

    Returns:
        Tensor, The result of `ifft()` function.

    Raises:
        TypeError: If the `input` type is not Tensor.
        TypeError: If the `input` data type is not one of: complex64, complex128.
        TypeError: If `n` or `dim` type is not int32.
        ValueError: If `input` dimension is less than 1.
        ValueError: If `n` is less than 1.
        ValueError: If `dim` is not in the range of "[ `-input_dim` , `input_dim-1` ]".
        ValueError: If norm is none of "backward", "forward" or "ortho".

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor, ops
        >>> input = Tensor([ 1.6243454+0.j, -0.6117564+0.j, -0.5281718+0.j, -1.0729686+0.j])
        >>> y = ops.ifft(input)
        >>> print(y)
        [-0.14713785+0.j          0.5381293 +0.11530305j  0.69522465+0.j
         0.5381293 -0.11530305j]
    """
    if not isinstance(input, (Tensor, Tensor_)):
        raise TypeError("For 'IFFT', 'input' must be Tensor.")

    input_dtype = dtype_(input)
    if not input_dtype in (mstype.complex64, mstype.complex128):
        raise TypeError("For 'IFFT', the dtype of 'input' must be complex64, complex128, "
                        f"but got '{input_dtype}'.")
    _check_attr_dtype("dim", dim, [int], "IFFT")

    input_dim = input.ndim
    signal_ndim = 1
    batch_dims = input_dim - signal_ndim
    input_sizes = list(input.shape)
    dim = _maybe_wrap_dims_n([dim], input_dim)[0]
    n_opt = n
    if n is None:
        n = input.shape[dim]
    else:
        _check_attr_dtype("n", n, [int], "IFFT")
    if n < 1:
        raise ValueError("For 'IFFT', the value of 'n' must be greater than or equal to 1, "
                         f"but got '{n}'.")
    if n_opt is not None:
        input = _resize_input(input, input_dim, [dim], [n], input_sizes)
    out_sizes = input.shape

    input, dim_permute = _permute_input(input, input_dim, [dim])
    input = _reshape_input(input, signal_ndim, batch_dims)

    if norm is None:
        norm = "backward"
    else:
        _check_attr_dtype("norm", norm, [str], "IFFT")

    fft_ = FFTWithSize(signal_ndim=1, inverse=True, real=False, norm=norm)
    out = fft_(input)

    return _handle_fftwithsize_output(out, input_dim, batch_dims, dim_permute, out_sizes)


def ifft2(input, s=None, dim=(-2, -1), norm=None):  # pylint: disable=redefined-outer-name
    r"""
    Calculates the inverse of `fft2()`.

    Args:
        input (Tensor): The input tensor.
        s (Tuple[int], optional): Signal size in the transformed dimensions.
            If given, each dimension `dim[i]` will either be zero-padded or trimmed to the length `s[i]` before
            computing the FFT. If a length `-1` is specified, no padding is done in that dimension.
            Default: `s = [input.size(d) for d in dim]`
        dim (Tuple[int], optional): Dimensions to be transformed.
            Default: (-2, -1).
        norm (string, optional): Normalization mode. Three modes are defined as,
            ``"forward"``(normalize by :math `1/n`), ``"backward"``(no normalization),
            ``"ortho"``(normalize by :math:`1/\sqrt{n}`). Where :math `n = prod(s)` is the logical IFFT size.
            Default: ``None`` that means ``"backward"``.

    Returns:
        Tensor, The result of `ifft2()` function.

    Raises:
        TypeError: If the `input` type is not Tensor.
        TypeError: If the `input` data type is not one of: complex64, complex128.
        TypeError: If the `s` or `dim` is not tuple(int).
        ValueError: If the length of `s` and `dim` are not the same.
        ValueError: If `input` dimension is less than 2.
        ValueError: If the value in `dim` is not in the range of "[ `-input_dim` , `input_dim-1` )".
        ValueError: If norm is none of "backward", "forward" or "ortho".

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor, ops
        >>> input = Tensor([[ 1.6243454+0.j, -0.6117564+0.j], [-0.5281718+0.j, -1.0729686+0.j]])
        >>> y = ops.ifft2(input)
        >>> print(y)
        [[-0.14713785+0.j  0.69522465+0.j]
        [ 0.65343235+0.j  0.42282625+0.j]]
    """
    return ifftn(input, s, dim, norm)


def ifftn(input, s=None, dim=None, norm=None):  # pylint: disable=redefined-outer-name
    r"""
    Calculates the inverse of `fftn()`.

    Args:
        input (Tensor): The input tensor.
        s (Tuple[int], optional): Signal size in the transformed dimensions.
            If given, each dimension `dim[i]` will either be zero-padded or trimmed to the length `s[i]` before
            computing the FFT. If a length `-1` is specified, no padding is done in that dimension.
            Default: `s = [input.size(d) for d in dim]`
        dim (Tuple[int], optional): Dimensions to be transformed.
            Default: all dimensions, or the last `len(s)` dimensions if `s` is given.
        norm (string, optional): Normalization mode. Three modes are defined as,
            ``"forward"``(normalize by :math `1/n`), ``"backward"``(no normalization),
            ``"ortho"``(normalize by :math:`1/\sqrt{n}`). Where :math `n = prod(s)` is the logical IFFT size.
            Default: ``None`` that means ``"backward"``.

    Returns:
        Tensor, The result of `ifftn()` function.

    Raises:
        TypeError: If the `input` type is not Tensor.
        TypeError: If the `input` data type is not one of: complex64, complex128.
        TypeError: If the `s` or `dim` is not tuple(int).
        ValueError: If the length of `s` and `dim` are not the same.
        ValueError: If `input` dimension is less than 1.
        ValueError: If the value in `dim` is not in the range of "[ `-input_dim` , `input_dim-1` )".
        ValueError: If norm is none of "backward", "forward" or "ortho".

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor, ops
        >>> input = Tensor([[[ 1.6243454 +0.j, -0.6117564 +0.j, -0.5281718 +0.j],
        ...                  [-1.0729686 +0.j, 0.86540765+0.j, -2.3015387 +0.j]]])
        >>> y = ops.ifftn(input)
        >>> print(y)
        [[[-0.33744708+0.j          0.30656774+0.44504493j
           0.30656774-0.44504493j]
         [ 0.49891948+0.j          0.42486876-0.46917373j
           0.42486876+0.46917373j]]]
    """
    ifftninput = _check_fftwithsize_input(input, s, dim, norm, "IFFTN")
    ifftn_ = FFTWithSize(signal_ndim=ifftninput.signal_ndim, inverse=True, real=False, norm=ifftninput.norm)
    out = ifftn_(ifftninput.input)
    return _handle_fftwithsize_output(out, ifftninput.input_dim, ifftninput.batch_dims,
                                      ifftninput.dim_permute, ifftninput.out_sizes)


@_primexpr
def _check_validate_axis(axis, name):
    def _check(axis):
        if isinstance(axis, (tuple, list)):
            for idx, item in enumerate(axis):
                validator.check_value_type("axis[%d]" % idx, item, [int], name)

    _check(axis)
    axis = validator.check_value_type('axis', axis, [int, tuple, list], name)
    return axis


@constexpr
def _check_validate_keepdims(keep_dims, name):
    keep_dims = validator.check_value_type('keep_dims', keep_dims, [bool], name)
    return keep_dims


def count_nonzero(x, axis=(), keep_dims=False, dtype=mstype.int32):
    r"""
    Counts the number of non-zero values in the input tensor along the given axis.
    If no axis is specified then all non-zeros in the tensor are counted.

    Args:
        x (Tensor): The input tensor.
        axis (Union[int, tuple(int), list(int)], optional): Specify the axis for computation.
            Default ``()`` , which counts all non-zero elements.
        keep_dims (bool, optional): Whether to maintain dimensions specified by `axis`.
            Default ``False`` , don't keep these dimensions.
        dtype (Union[Number, mindspore.bool], optional): The data type returned.
            Default ``mstype.int32`` .


    Returns:
          Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> # case 1: each value specified.
        >>> x = mindspore.tensor([[0, 1, 0], [1, 1, 0]], mindspore.float32)
        >>> nonzero_num = mindspore.ops.count_nonzero(x=x, axis=[0, 1], keep_dims=True, dtype=mindspore.int32)
        >>> print(nonzero_num)
        [[3]]
        >>> # case 2: all value is default.
        >>> nonzero_num = mindspore.ops.count_nonzero(x=x)
        >>> print(nonzero_num)
        3
        >>> # case 3: axis value was specified 0.
        >>> nonzero_num = mindspore.ops.count_nonzero(x=x, axis=[0,])
        >>> print(nonzero_num)
        [1 2 0]
        >>> # case 4: axis value was specified 1.
        >>> nonzero_num = mindspore.ops.count_nonzero(x=x, axis=[1,])
        >>> print(nonzero_num)
        [1 2]
        >>> # case 5: keep_dims value was specified.
        >>> nonzero_num = mindspore.ops.count_nonzero(x=x,  keep_dims=True)
        >>> print(nonzero_num)
        [[3]]
        >>> # case 6: keep_dims and axis value was specified.
        >>> nonzero_num = mindspore.ops.count_nonzero(x=x, axis=[0,], keep_dims=True)
        >>> print(nonzero_num)
        [[1 2 0]]
    """

    const_utils.check_type_valid(dtype_(x), mstype.number_type, 'input x')
    keep_dims = _check_validate_keepdims(keep_dims, "count_nonzero")
    const_utils.check_type_valid(dtype, mstype.number_type + (mstype.bool_,), 'dtype')

    reduce_sum = _get_cache_prim(P.ReduceSum)(keep_dims)

    tensor_0 = ops.zeros(x.shape, x.dtype)
    nonzero_bool = not_equal(x, tensor_0)
    # ReduceSum only support float16 or float32 tensor.
    nonzero_val = cast_(nonzero_bool, mstype.float32)
    nonzero_num = cast_(reduce_sum(nonzero_val, axis), dtype)

    return nonzero_num


@_primexpr
def _int_to_tuple_conv(axes):
    """
    Converts ints to tuples in input axes, expected by most validation checks.
    """
    for x in [0, 1]:
        if isinstance(axes[x], int):
            axes[x] = (axes[x],)
    return axes


@_primexpr
def _check_axes(axes, prim_name=None):
    """
    Check for validity and type of axes passed to function.
    """
    msg_prefix = f"For '{prim_name}', the" if prim_name else "The"
    validator.check_value_type('axes', axes, [int, tuple, list], "tensor dot")
    if not isinstance(axes, int):
        axes = list(axes)  # to avoid immutability issues
        if len(axes) != 2:
            raise ValueError(f"{msg_prefix} dimension of 'axes' must be 2, but got 'axes': {axes}.")
        axes = _int_to_tuple_conv(axes)  # convert before length checks
        if len(axes[0]) != len(axes[1]):
            raise ValueError(f"{msg_prefix} first and second dim of 'axes' have to be the same size/length, "
                             f"but got 'axes': {axes}.")
        if len(axes[0]) != len(set(axes[0])) or len(axes[1]) != len(set(axes[1])):
            raise ValueError(f"{msg_prefix} 'axes' cannot have duplicating values, but got {axes}.")
    return axes


@constexpr
def _typecheck_input(x1_type, x2_type, prim_name=None):
    """
    Check input tensor types to be valid and confirm they are the same type.
    """
    msg_prefix = f"For '{prim_name}', the" if prim_name else "The"
    const_utils.check_type_valid(x1_type, [mstype.float32, mstype.float16], 'x1')
    const_utils.check_type_valid(x2_type, [mstype.float32, mstype.float16], 'x2')
    if x1_type != x2_type:
        raise TypeError(f"{msg_prefix} inputs must be the same type, but got x1_type: {x1_type} "
                        f"and x2_type: {x2_type}.")


@_primexpr
def _axes_int_check(x1_shape, x2_shape, axes, prim_name=None):
    """
    Convert from single int axes to 2d tuple if required
    """
    msg_prefix = f"For '{prim_name}', the" if prim_name else "The"

    def _check_lt_zero(axes):
        if axes < 0:
            raise ValueError(f"{msg_prefix} 'axes' must be at least 0, but got {axes}.")

    def _check_len(axes, x1_shape, x2_shape):
        if axes > len(x1_shape) or axes > len(x2_shape):
            raise ValueError(f"{msg_prefix} 'axes' cannot be greater than the length of 'x1_shape' and 'x2_shape', "
                             f"but got 'axes': {axes}, 'x1_shape': {x1_shape}, 'x2_shape': {x2_shape}.")

    if isinstance(axes, int):
        _check_lt_zero(axes)
        if axes == 0:
            # outer product, no input validation required
            return [], []
        _check_len(axes, x1_shape, x2_shape)
        x1_ind = tuple(range(len(x1_shape))[-1 * axes:])
        x2_ind = tuple(range(len(x2_shape))[:axes])
        axes = tuple((x1_ind, x2_ind))
        axes = _int_to_tuple_conv(axes)
    return axes


@_primexpr
def _validate_axes(x1_shape, x2_shape, axes, prim_name=None):
    """
    Checks for axes having the correct length according to input, for any value in axis
    being out of range with given shape and also checking for compatible axes values
    with given inputs.
    """
    msg_prefix = f"For '{prim_name}', the" if prim_name else "The"

    def _check_len(axes_len, shape_dim_len, x_axes):
        if axes_len > shape_dim_len:
            raise ValueError(f"{msg_prefix} length of element {x_axes} in 'axes' must be less than or equal to "
                             f"{shape_dim_len}, but got {axes_len}.")

    def _check_axes_value(x_axes, min_val, max_val):
        for _, x_value in enumerate(x_axes):
            if x_value > max_val or x_value < min_val:
                raise ValueError(f"{msg_prefix} value in 'axes' must be in range: [{min_val}, {max_val}], "
                                 f"but got {x_value}.")

    shapes = [x1_shape, x2_shape]

    # axis length check
    for ix_input, x_axes in enumerate(axes):
        axes_len = len(x_axes)
        shape_dim_len = len(shapes[ix_input])
        _check_len(axes_len, shape_dim_len, x_axes)

    # axis values range check
    for ix_input, x_axes in enumerate(axes):
        comp_shape = shapes[ix_input]
        max_val = len(comp_shape) - 1
        min_val = -1 * len(comp_shape)
        _check_axes_value(x_axes, min_val, max_val)

    # check axis value with input shape - both ways for axis valid
    invalid_a = False
    invalid_b = False
    for i in range(len(axes[0])):  # sizes already validated
        if x1_shape[axes[0][i]] != x2_shape[axes[1][i]]:
            invalid_a = True
        if x1_shape[axes[0][i]] != x2_shape[axes[1][len(axes[0]) - 1 - i]]:
            invalid_b = True

    def _check(invalid_a, invalid_b, x1_shape, x2_shape, axes):
        if invalid_a and invalid_b:
            raise ValueError(f"{msg_prefix} 'i' should exist such that 'x1_shape[axes[0][i]]' is equal to "
                             f"'x2_shape[axes[1][i]]' or 'x2_shape[axes[1][len(axes[0])-1-i]]', but got "
                             f"'x1_shape': {x1_shape}, 'x2_shape': {x2_shape}, 'axes': {axes}.")

    _check(invalid_a, invalid_b, x1_shape, x2_shape, axes)


@_primexpr
def _calc_new_shape(shape, axes, position=0):
    """
    Calculate transpose and reshape parameters for input transformations,
    'position' refers to whether tensor is first or second in the op.
    """
    contraction_axes = tuple(i if i >= 0 else i + len(shape) for i in axes[position])
    prod_contraction = 1
    for i in contraction_axes:
        prod_contraction *= shape[i]
    free_axes = tuple(i for i in range(len(shape)) if i not in contraction_axes)
    free_dims = tuple(shape[i] if shape[i] is not None else -1 for i in free_axes)
    prod_free = 1
    for free_dim in free_dims:
        prod_free *= free_dim

    transpose_perm = contraction_axes + free_axes if position else free_axes + contraction_axes
    new_shape = (prod_contraction, prod_free) if position else (prod_free, prod_contraction)
    return new_shape, transpose_perm, free_dims


def tensor_dot(x1, x2, axes):
    """
    Compute the tensor dot product along the specified axes.

    Args:
        x1 (Tensor): Input tensor.
        x2 (Tensor): Input tensor.
        axes (Union[int, tuple(int), tuple(tuple(int)), list(list(int))]): The number of dimensions to sum over. If an
            integer `k` is provided, then sum over the last `k` axes of `x1` and the first `k` axes of `x2`, in order.
            If a tuple or list is provided, then `axes[0]` specifies the axes of `x1` and `axes[1]` specifies the axes
            of `x2`.

    Returns:
        Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor, ops
        >>> import mindspore
        >>> import numpy as np
        >>> input_x1 = Tensor(np.ones(shape=[1, 2, 3]), mindspore.float32)
        >>> input_x2 = Tensor(np.ones(shape=[3, 1, 2]), mindspore.float32)
        >>> output = ops.tensor_dot(input_x1, input_x2, ((0,1),(1,2)))
        >>> print(output)
        [[2. 2. 2]
         [2. 2. 2]
         [2. 2. 2]]
    """
    matmul_op = _get_cache_prim(P.MatMul)(False, False)
    # input validity checks
    x1_shape = shape_(x1)
    x2_shape = shape_(x2)
    axes = _check_axes(axes, 'tensor_dot')
    # input compatibility check & axes format update
    axes = _axes_int_check(x1_shape, x2_shape, axes, 'tensor_dot')
    _validate_axes(x1_shape, x2_shape, axes, 'tensor_dot')
    x1_reshape_fwd, x1_transpose_fwd, x1_ret = _calc_new_shape(x1_shape, axes, 0)
    x2_reshape_fwd, x2_transpose_fwd, x2_ret = _calc_new_shape(x2_shape, axes, 1)
    output_shape = x1_ret + x2_ret  # combine free axes from both inputs
    # run tensor_dot op
    x1_transposed = transpose_op(x1, x1_transpose_fwd)
    x2_transposed = transpose_op(x2, x2_transpose_fwd)
    x1_reshaped = reshape_(x1_transposed, x1_reshape_fwd)
    x2_reshaped = reshape_(x2_transposed, x2_reshape_fwd)
    mul_result = matmul_op(x1_reshaped, x2_reshaped)
    final_result = reshape_(mul_result, output_shape)
    return final_result


def vecdot(x, y, *, axis=-1):
    r"""
    Calculates the dot product of two batches of vectors along the specified dimension.

    Support broadcasting.

    The formula of calculation is as follows.
    :math:`\bar{x_{i}}` represents the conjugate for complex vectors,
    and :math:`\bar{x_{i}}` is the raw value for real vectors.

    .. math::

        \sum_{i=1}^{n} \bar{x_{i}}{y_{i}}

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        x (Tensor): The first batch of tensors.
        y (Tensor): The second batch of tensors.

    Keyword Args:
        axis (int): Specify the axis for computation. Default ``-1`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    .. note::
        Currently, complex numbers are not supported on GPU.

    Examples:
        >>> import mindspore
        >>> x = mindspore.tensor([[1, 3], [5, 7], [9, 8]])
        >>> y = mindspore.tensor([[4, 5], [6, 7], [3, 2]])
        >>> mindspore.ops.vecdot(x, y)
        Tensor(shape=[3], dtype=Int64, value= [19, 79, 43])
        >>> mindspore.ops.vecdot(x, y, axis=0)
        Tensor(shape=[2], dtype=Int64, value= [61, 80])
    """
    if (not isinstance(x, Tensor)) or (not isinstance(y, Tensor)):
        raise TypeError("For vecdot, x or y must be Tensor.")
    if not isinstance(axis, int):
        raise TypeError(f"For vecdot, the dim should be int, but got {type(axis)}.")
    ndim = x.ndim if x.ndim > y.ndim else y.ndim
    if (axis < -ndim) or (axis >= ndim):
        raise ValueError("For vecdot, the dim is out of range.")
    if x.dtype in mstype.complex_type:
        x = x.conj()
    result = x * y
    result = result.sum(axis=axis)
    return result


@_primexpr
def _check_invalid_input(x1_shape, x2_shape, prim_name=None):
    msg_prefix = f"For \\\'{prim_name}\\\', the" if prim_name else "The"
    if len(x1_shape) < 2 or len(x2_shape) < 2:
        raise ValueError(f"{msg_prefix} inputs x1, x2 should have \\\'dimension >= 2\\\',"
                         f"but got \\\'len(x1_shape)\\\': ({len(x1_shape)})"
                         f" and \\\'len(x2_shape)\\\': ({len(x2_shape)}).")


@constexpr
def _typecheck_input_dot(x1_type, x2_type, prim_name=None):
    """
    Check input tensor types to be valid and confirm they are the same type for dot and batch dot ops.
    """
    msg_prefix = f"For \\\'{prim_name}\\\', the" if prim_name else "The"
    const_utils.check_type_valid(x1_type, [mstype.float16, mstype.float32], 'x1')
    const_utils.check_type_valid(x2_type, [mstype.float16, mstype.float32], 'x2')
    if x1_type != x2_type:
        raise TypeError(f"{msg_prefix} inputs must be the same type, but got "
                        f"x1_type: {x1_type} and x2_type: {x2_type}.")


@_primexpr
def _get_transpose_shape(x2_shape):
    x2_shape_range = tuple(range(len(x2_shape)))
    x2_shape_transpose = x2_shape_range[-2:-1] + x2_shape_range[:-2] + x2_shape_range[-1:]
    return x2_shape_transpose


def dot(input, other):
    """
    Computation a dot product of two input tensors.

    .. note::
        - Datatype of the input tensors must be float16 or float32, and the rank must
          be greater than or equal to 2.

    Args:
        input (Tensor): The first input tensor.
        other (Tensor): The second input tensor.

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> input = mindspore.ops.ones([2, 3])
        >>> other = mindspore.ops.ones([1, 3, 2])
        >>> output = mindspore.ops.dot(input, other)
        >>> print(output)
        [[[3. 3.]]
         [[3. 3.]]]
        >>> print(output.shape)
        (2, 1, 2)
        >>> input = mindspore.ops.ones([1, 2, 3])
        >>> other = mindspore.ops.ones([1, 3, 2])
        >>> output = mindspore.ops.dot(input, other)
        >>> print(output)
        [[[[3. 3.]]
          [[3. 3.]]]]
        >>> print(output.shape)
        (1, 2, 1, 2)
        >>> input = mindspore.ops.ones([1, 2, 3])
        >>> other = mindspore.ops.ones([2, 3, 2])
        >>> output = mindspore.ops.dot(input, other)
        >>> print(output)
        [[[[3. 3.]
           [3. 3.]]
          [[3. 3.]
           [3. 3.]]]]
        >>> print(output.shape)
        (1, 2, 2, 2)
        >>> input = mindspore.ops.ones([3, 2, 3])
        >>> other = mindspore.ops.ones([2, 1, 3, 2])
        >>> output = mindspore.ops.dot(input, other)
        >>> print(output)
        [[[[[3. 3.]]
           [[3. 3.]]]
          [[[3. 3.]]
           [[3. 3.]]]]
         [[[[3. 3.]]
           [[3. 3.]]]
          [[[3. 3.]]
           [[3. 3.]]]]
         [[[[3. 3.]]
           [[3. 3.]]]
          [[[3. 3.]]
           [[3. 3.]]]]]
        >>> print(output.shape)
        (3, 2, 2, 1, 2)
    """
    matmul_op = _get_cache_prim(P.MatMul)(False, False)
    input_shape = shape_(input)
    other_shape = shape_(other)
    input_type = dtype_(input)
    other_type = dtype_(other)
    _typecheck_input_dot(input_type, other_type, 'dot')
    _check_invalid_input(input_shape, other_shape, 'dot')

    if len(input_shape) > 2 or len(other_shape) > 2:
        other_shape_transpose = _get_transpose_shape(other_shape)
        other_transpose = transpose_op(other, other_shape_transpose)
        input_reshape = reshape_(input, (-1, input_shape[-1]))
        other_reshape = reshape_(other_transpose, (other_shape[-2], -1))
        mul_result = matmul_op(input_reshape, other_reshape)
        reshape_shape = input_shape[:-1] + other_shape[:-2] + other_shape[-1:]
        reshape_shape = (-1,) + reshape_shape[1:]
        return reshape_(mul_result, reshape_shape)
    return matmul_op(input, other)


@_primexpr
def _get_batch_size(x1_shape, x2_shape, prim_name=None):
    """
    Get batch sizes from two inputs
    """

    def _check():
        msg_prefix = f"For '{prim_name}', the" if prim_name else "The"
        if len(x1_shape) < 2 or len(x2_shape) < 2:
            raise ValueError(f"{msg_prefix} inputs x1, x2 should have 'dimension >= 2', "
                             f"but got 'len(x1_shape)': ({len(x1_shape)}) and 'len(x2_shape)': ({len(x2_shape)}).")

    _check()
    return x1_shape[0], x2_shape[0]


@constexpr
def _typecheck_input_batch_dot(x1_type, x2_type, prim_name=None):
    """
    Check input tensor types to be valid and confirm they are the same type for batch dot ops.
    """
    msg_prefix = f"For '{prim_name}', the" if prim_name else "The"
    const_utils.check_type_valid(x1_type, [mstype.float32], 'x1')
    const_utils.check_type_valid(x2_type, [mstype.float32], 'x2')
    if x1_type != x2_type:
        raise TypeError(f"{msg_prefix} inputs must be the same type, but got x1_type: {x1_type} and "
                        f"x2_type: {x2_type}.")


@_primexpr
def _check_axes_for_batch_dot(x1_shape, x2_shape, axes, prim_name=None):
    """
    Check whether axes are valid and cast axes from tuple to list
    """
    msg_prefix = f"For '{prim_name}', the" if prim_name else "The"

    def _check_1(axes):
        if 0 in axes:
            raise ValueError(f"{msg_prefix} 'axes' cannot contain 0, but got axes: {axes}.")
        if len(axes) != 2:
            raise ValueError(f"{msg_prefix} length of 'axes' must be equal to 2, but got {len(axes)}.")

    def _check_2(axes, x1_shape, x2_shape):
        if axes[0] > len(x1_shape) or axes[1] > len(x2_shape):
            raise ValueError(f"{msg_prefix} axes[0] must be less than or equal to len(x1_shape), "
                             f"and axes[1] must be less than or equal to len(x2_shape)."
                             f"But got 'axes': {axes}, 'x1_shape': {x1_shape}, 'x2_shape': {x2_shape}.")

    def _check_3(axes, x1_shape, x2_shape):
        if axes == 0:
            raise ValueError(f"{msg_prefix} 'axes' should not be equal to 0, but got {axes}.")

        if axes > len(x1_shape) or axes > len(x2_shape):
            raise ValueError(f"{msg_prefix} 'axes' cannot be greater than the length of 'x1_shape' and 'x2_shape', "
                             f"but got 'axes': {axes}, 'x1_shape': {x1_shape}, 'x2_shape': {x2_shape}.")

    if axes is None:
        if len(x2_shape) == 2:
            axes = [len(x1_shape) - 1, len(x2_shape) - 1]
        else:
            axes = [len(x1_shape) - 1, len(x2_shape) - 2]

    if isinstance(axes, (list, tuple)):
        _check_1(axes)
        if isinstance(axes, tuple):
            axes = list(axes)
        validator.check_value_type('axes[0]', axes[0], [int], 'batch_dot')
        validator.check_value_type('axes[1]', axes[1], [int], 'batch_dot')
        # Reverse if axis < 0
        if axes[0] < 0:
            axes[0] += len(x1_shape)
        if axes[1] < 0:
            axes[1] += len(x2_shape)
        validator.check_non_negative_int(axes[0], 'reversed axes[0]', 'batch_dot')
        validator.check_non_negative_int(axes[1], 'reversed axes[1]', 'batch_dot')
        _check_2(axes, x1_shape, x2_shape)
    elif isinstance(axes, int):
        _check_3(axes, x1_shape, x2_shape)
        if axes < 0:
            axes = [axes + len(x1_shape), axes + len(x2_shape)]
            validator.check_non_negative_int(axes[0], 'reversed axes', 'batch_dot')
        else:
            axes = [axes, axes]
    else:
        raise ValueError(f"{msg_prefix} type of 'axes' must be one of those: int, tuple(int), list(int), "
                         f"but got {type(axes).__name__}.")
    return axes


@_primexpr
def _calc_new_shape_batchdot(shape, axes, position=0):
    """
    Calculate transpose and reshape parameters for input transformations,
    'position' refers to whether tensor is first or second in the op.
    """
    axis = axes[position]
    contraction_axes = tuple([axis])
    prod_contraction = 1
    for i in contraction_axes:
        prod_contraction *= shape[i]
    free_axes = tuple(i for i in range(1, len(shape)) if i not in contraction_axes)
    free_dims = tuple(shape[i] for i in free_axes)
    prod_free = 1
    for free_dim in free_dims:
        prod_free *= free_dim

    transpose_perm = contraction_axes + free_axes if position else free_axes + contraction_axes
    transpose_perm = tuple([0]) + transpose_perm
    new_shape = (prod_contraction, prod_free) if position else (prod_free, prod_contraction)
    new_shape = tuple([shape[0]]) + new_shape
    return new_shape, transpose_perm, free_dims


@_primexpr
def _check_batch_size(x1_batch_size, x2_batch_size, prim_name=None):
    """
    Check whether batch size of two inputs are the same
    """
    msg_prefix = f"For '{prim_name}', the" if prim_name else "The"
    if x1_batch_size != x2_batch_size:
        raise ValueError(f"{msg_prefix} inputs 'x1', 'x2' should have the same batch sizes, but got "
                         f"'x1_batch_size': {x1_batch_size} and 'x2_batch_size': {x2_batch_size}.")


@_primexpr
def _get_output_shape(batch_size, x1_ret, x2_ret):
    """
    Compute output shape for batch dot
    """
    output_shape = tuple([batch_size]) + x1_ret + x2_ret
    return output_shape


@deprecated("2.8.0", None, False, "ops.")
def batch_dot(x1, x2, axes=None):
    """
    `ops.batch_dot` is deprecated from version 2.8.0 and will be removed in a
    future version.

    Computation of batch dot product between samples in two tensors containing batch dims.

    .. note::
        - `x1` or `x2` first dimension is batch size. Datatype must be float32 and the rank must
          be greater than or equal to 2.

    .. math::
        output = x1[batch, :] · x2[batch, :]

    Args:
        x1 (Tensor): The first input tensor.
        x2 (Tensor): The second input tensor.
        axes (Union[int, tuple(int), list(int)]): Specify the axes for computation. Default ``None`` .

    Returns:
        Tensor

    Supported Platforms:
        Deprecated

    Examples:
        >>> import mindspore
        >>> # case 1: axes is a tuple(axes of `x1` , axes of `x2` )
        >>> x1 = mindspore.ops.ones([2, 2, 3])
        >>> x2 = mindspore.ops.ones([2, 3, 2])
        >>> axes = (-1, -2)
        >>> output = mindspore.ops.batch_dot(x1, x2, axes)
        >>> print(output)
        [[[3. 3.]
          [3. 3.]]
         [[3. 3.]
          [3. 3.]]]
        >>> print(output.shape)
        (2, 2, 2)
        >>> x1 = mindspore.ops.ones([2, 2], mindspore.float32)
        >>> x2 = mindspore.ops.ones([2, 3, 2], mindspore.float32)
        >>> axes = (1, 2)
        >>> output = mindspore.ops.batch_dot(x1, x2, axes)
        >>> print(output)
        [[2. 2. 2.]
         [2. 2. 2.]]
        >>> print(output.shape)
        (2, 3)
        >>>
        >>> # case 2: axes is None
        >>> x1 = mindspore.ops.ones([6, 2, 3, 4], mindspore.float32)
        >>> x2 = mindspore.ops.ones([6, 5, 4, 8], mindspore.float32)
        >>> output = mindspore.ops.batch_dot(x1, x2)
        >>> print(output.shape)
        (6, 2, 3, 5, 8)
        >>>
        >>> # case 3: axes is a int data.
        >>> x1 = mindspore.ops.ones([2, 2, 4])
        >>> x2 = mindspore.ops.ones([2, 5, 4, 5])
        >>> output = mindspore.ops.batch_dot(x1, x2, 2)
        >>> print(output.shape)
        (2, 2, 5, 5)

    """
    squeeze_one_op = _get_cache_prim(P.Squeeze)(1)
    squeeze_minus_one_op = _get_cache_prim(P.Squeeze)(-1)
    # input validity checks
    x1_shape = shape_(x1)
    x2_shape = shape_(x2)
    x1_dim_num = len(x1_shape)
    x2_dim_num = len(x2_shape)
    x1_type = dtype_(x1)
    x2_type = dtype_(x2)

    x1_batch_size, x2_batch_size = _get_batch_size(x1_shape, x2_shape, 'batch_dot')

    _typecheck_input_batch_dot(x1_type, x2_type, 'batch_dot')
    _check_batch_size(x1_batch_size, x2_batch_size, 'batch_dot')
    axes = _check_axes_for_batch_dot(x1_shape, x2_shape, axes, 'batch_dot')

    if x1_dim_num == 2:
        x1 = F.expand_dims(x1, 1)
        axes[0] += 1
    if x2_dim_num == 2:
        x2 = F.expand_dims(x2, 2)

    x1_shape = shape_(x1)
    x2_shape = shape_(x2)

    x1_reshape_fwd, x1_transpose_fwd, x1_ret = _calc_new_shape_batchdot(x1_shape, axes, 0)
    x2_reshape_fwd, x2_transpose_fwd, x2_ret = _calc_new_shape_batchdot(x2_shape, axes, 1)
    output_shape = _get_output_shape(x1_batch_size, x1_ret, x2_ret)

    x1_transposed = transpose_op(x1, x1_transpose_fwd)
    x2_transposed = transpose_op(x2, x2_transpose_fwd)
    x1_reshaped = reshape_(x1_transposed, x1_reshape_fwd)
    x2_reshaped = reshape_(x2_transposed, x2_reshape_fwd)

    # Batch matmal op part
    mul_result = batch_matmul_(x1_reshaped, x2_reshaped)

    final_result = reshape_(mul_result, output_shape)

    # if the original dims are expanded, restore them from 3 to 2
    if x1_dim_num == 2:
        final_result = squeeze_one_op(final_result)
    elif x2_dim_num == 2:
        final_result = squeeze_minus_one_op(final_result)

    return final_result


def round(input, *, decimals=0):
    r"""
    Round elements of input to the nearest integer.

    .. math::
        out_i \approx input_i

    .. note::
        The input data types supported by the Ascend platform include
        bfloat16 (Atlas training series products are not supported), float16, float32, float64, int32, and int64.

    Args:
        input (Tensor): The input tensor.

    Keyword Args:
        decimals (int, optional): Number of decimal places to round. If decimals is negative,
            it specifies the number of positions to the left of the decimal point. Default ``0`` .

    Returns:
        Tensor

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> output = mindspore.ops.round(mindspore.tensor([4.7, -2.3, 9.1, -7.7]))
        >>> print(output)
        [ 5. -2.  9. -8.]
        >>> # Values equidistant from two integers are rounded towards the
        >>> # the nearest even value (zero is treated as even)
        >>> output = mindspore.ops.round(mindspore.tensor([-0.5, 0.5, 1.5, 2.5]))
        >>> print(output)
        [0. 0.  2.  2.]
        >>> # A positive decimals argument rounds to the to that decimal place
        >>> output = mindspore.ops.round(mindspore.tensor([0.1234567]), decimals=3)
        >>> print(output)
        [0.123]
        >>> # A negative decimals argument rounds to the left of the decimal
        >>> output = mindspore.ops.round(mindspore.tensor([1200.1234567]), decimals=-3)
        >>> print(output)
        [1000.]
    """

    return round_op(input, decimals)


def isnan_ext(tensor):
    r"""
    Returns a new tensor with boolean elements representing if each element of input is :math:`Nan` or not.
    Complex values are considered NaN when either their real and/or imaginary part is :math:`Nan`.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): The input tensor.

    Returns:
        Tensor, return a Boolean Tensor. If the input is :math:`Nan`, the value is ``True``.
        Otherwise, the value is ``False``.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> input1 = ms.Tensor([np.nan, 2, 3, 4])
        >>> output = ms.ops.function.math_func.isnan_ext(input1)
        >>> print(output)
        [ True  False  False  False]
    """
    return not_equal_op(tensor, tensor)


def multi_scale_deformable_attn_function(value, shape, offset, locations, weight):
    r"""
    The multi-scale deformable attention mechanism fuses the feature maps of multiple perspectives.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    .. note::
        Atlas training series products are not supported.

    Args:
        value (Tensor): Feature tensor, the data type supports float32 and float16.
            The shape is :math:`(bs, num\_keys, num\_heads, embed\_dims)`. Among them, bs is the batch size,
            num_keys is the size of the feature map, num_heads is the number of heads, embed_dims is the
            dimension of the fearure map, where embed_dims needs to be a multiple of ``8``.
        shape (Tensor): Shape of feature, the data type supports int32 and int64. The shape is :math:`(num\_levels, 2)`.
            Among them, num_levels is the number of feature maps, and 2 represents H and W respectively.
        offset (Tensor): Offset tensor, the data type supports int32 and int64. The shape is :math:`(num\_levels)`.
        locations (Tensor): Location tensor, the data type supports float32 and float16.
            The shape is :math:`(bs, num\_queries, num\_heads, num\_levels, num\_points, 2)`. Among them,
            bs is the batch size, num_queries is the number of queries, num_heads is the number of heads,
            num_levels is the number of feature maps, num_points is the number of sampling points,
            and ``2`` represents x and y respectively.
        weight (Tensor): Weight tensor, the data type supports float32 and float16. The shape is
            :math:`(bs, num\_queries, num\_heads, num\_levels, num\_points)`. Among them, bs is the batch size,
            num_queries is the number of queries, num_heads is the number of heads, num_levels is the number
            of feature maps, num_points is the number of sampling points.

    Returns:
        Tensor, The fused feature tensor, the data type is float32 or float16.
        The shape is :math:`(bs, num\_queries, num\_heads*embed\_dims)`.

    Raises:
        RuntimeError: If the data type of `value` is neither float32 nor float16.
        RuntimeError: If the data type of `shape` is neither int32 nor int64.
        RuntimeError: If the data type of `offset` is neither int32 nor int64.
        RuntimeError: If the data type of `locations` is neither float32 nor float16.
        RuntimeError: If the data type of `weight` is neither float32 nor float16.
        RuntimeError: `embed_dims` is not the multiples of ``8``.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> value = Tensor(np.random.randn(2, 3, 4, 8), dtype=ms.float32)
        >>> data = [[1, 2], [3, 4], [5, 6]]
        >>> shape = ms.Tensor(data, dtype=ms.int32)
        >>> data1 = [1, 2, 3]
        >>> shape = ms.Tensor(data1, dtype=ms.int32)
        >>> locations = ms.Tensor(np.random.randn(2, 3, 4, 3, 2, 2), dtype=ms.float32)
        >>> weight = ms.Tensor(np.random.randn(2, 3, 4, 3, 2), dtype=ms.float32)
        >>> out = ms.ops.multi_scale_deformable_attn_function(value, shape, offset, locations, weight)
    """
    return multi_scale_deformable_attn_op(value, shape, offset, locations, weight)


def rotated_iou(boxes, query_boxes, trans=False, mode=0, is_cross=True, v_threshold=0.0, e_threshold=0.0):
    r"""
    Calculate the overlap area between rotated rectangles.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    .. note::
        The input data types supported by the Ascend platform include
        bfloat16, float16, float32.

    Args:
        boxes (Tensor): The first set of rectangles which has a
            shape of :math:`(B, N, 5)`.
        query_boxes (Tensor): The second set of rectangles which
            has a shape of :math:`(B, K, 5)`.
        trans (bool, optional): Distinguish the rectangles representations
            of boxes and query_boxes. If ``True``, the format of boxes
            and query_boxes is ``'xyxyt'``, else the format is ``'xywht'``.
            The default value is ``False``.
        mode (int, optional): Distinguish the calculation mode. If the value
            is ``1``, the calculation mode is ``'iof'``, else the
            calculation mode is ``'iou'``. The default value is ``0``.
        is_cross (bool, optional): If ``True``, use cross-calculation, else use
            one-to-one calculation. The default value is ``True``.
        v_threshold (float, optional): Tolerance threshold for vertex determination.
            The default value is ``0.0``.
        e_threshold (float, optional): Tolerance threshold for edge intersection
            determination. The default value is ``0.0``.

    Returns:
        Tensor, the shape is :math:`(B, N, K)`.

    Raises:
        TypeError: If `boxes` is not a Tensor.
        TypeError: If `query_boxes` is not a Tensor.
        ValueError: If `boxes` and `query_boxes` do not has same first dim.
        ValueError: If the third dimension of `boxes` or `query_boxes` is not ``5``.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> a = np.random.uniform(0,1,(2,2,5)).astype(np.float16)
        >>> b = np.random.uniform(0,1,(2,3,5)).astype(np.float16)
        >>> box1 = Tensor(a)
        >>> box2 = Tensor(b)
        >>> output = ops.rotated_iou(box1, box2, trans=False, mode=0, is_cross=True)
    """
    origin_dtype = boxes.dtype
    if origin_dtype not in {mstype.float16, mstype.float32, mstype.bfloat16}:
        raise ValueError("input boxes type is illegal.")

    if query_boxes.dtype not in {mstype.float16, mstype.float32, mstype.bfloat16}:
        raise ValueError("input query_boxes type is illegal.")

    boxes_perm = (0, 2, 1)
    boxes_cp = permute(boxes, boxes_perm)
    if boxes_cp.dtype in {mstype.float16, mstype.bfloat16}:
        boxes_cp = cast_(boxes_cp, mstype.float32)

    query_boxes_perm = (0, 2, 1)
    query_boxes_cp = permute(query_boxes, query_boxes_perm)
    if query_boxes_cp.dtype in {mstype.float16, mstype.bfloat16}:
        query_boxes_cp = cast_(query_boxes_cp, mstype.float32)

    iou = rotated_iou_op(boxes_cp, query_boxes_cp, trans, mode, is_cross, v_threshold, e_threshold)
    return cast_(iou, origin_dtype)


def mul_ext(input, other):
    r"""
    Multiply other value by input Tensor.

    .. math::

        out_{i} = input_{i} \times other_{i}

    Note:
        - When the two inputs have different shapes, they must be able to broadcast to a common shape.
        - The two inputs comply with the implicit type conversion rules to make the data types
          consistent.

    Args:
        input (Union[Tensor, number.Number, bool]): The first input is a number.Number or
            a bool or a tensor whose data type is
            `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.dtype.html>`_ or
            `bool <https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.dtype.html>`_.
        other (Union[Tensor, number.Number, bool]): The second input, is a number.Number or
            a bool or a tensor whose data type is
            `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.dtype.html>`_ or
            `bool <https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.dtype.html>`_.

    Returns:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If the type of `input`, `other` is not one of the following: Tensor, number.Number, bool.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindspore import ops
        >>> x = Tensor(np.array([2, 6, 9]).astype(np.int32))
        >>> y = Tensor(np.array([4, 5, 6]).astype(np.float32))
        >>> output = ops.mul_ext(x, y)
        >>> print(output)
        [8. 30. 54.]
        >>> # the data type of x is int32, the data type of y is float32,
        >>> # and the output is the data format of higher precision float32.
        >>> print(output.dtype)
        Float32
    """
    if isinstance(other, (float, int, bool)) and isinstance(input, Tensor):
        return muls(input, other)
    return mul(input, other)


def gmm_backward(grad, x, weight, group_list=None, group_list_type=0):
    r"""
    the grad of ops.function.math_func.gmm
    """
    gradients = ops.functional_overload.gmm_backward(grad, x, weight, group_list, group_list_type)
    dx = gradients[:len(x)]
    dw = gradients[-len(weight):]
    db = []
    return dx, dw, db


def gmm_backward_fusion(grad, weight, group_list=None, group_list_type=0):
    r"""
    the grad of ops.function.math_func.gmm, only dx
    """
    dx = ops.functional_overload.gmm_backward_fusion(grad, weight, group_list, group_list_type)
    dw = []
    db = []
    return dx, dw, db


__all__ = [
    'addn',
    'absolute',
    'abs',
    'bucketize',
    'tensor_add',
    'add',
    'addbmm',
    'addcdiv',
    'addcmul',
    'angle',
    'argmin',
    'arccosh',
    'arcsin',
    'arctan',
    'arctan2',
    'bincount',
    'neg',
    'negative',
    'tensor_lt',
    'less',
    'lt',
    'logaddexp2',
    'tensor_le',
    'lcm',
    'le',
    'lerp',
    'norm',
    'vector_norm',
    'matrix_norm',
    'tensor_gt',
    'logaddexp',
    'mv',
    'addmm',
    'addmv',
    'adjoint',
    'outer',
    'gt',
    'tensor_ge',
    'ge',
    'addr',
    'tensor_sub',
    'sub',
    'subtract',
    'tensor_mul',
    'tensor_muls',
    'mul',
    'multiply',
    'nan_to_num',
    'nansum',
    'nanmean',
    'nanmedian',
    'digamma',
    'lgamma',
    'tensor_div',
    'div',
    'divide',
    'true_divide',
    'tensor_floordiv',
    'floor_div',
    'floor_divide',
    'floordiv',
    'float_power',
    'fmod',
    'xdivy',
    'tensor_pow',
    'pow',
    'pows',
    'renorm',
    'tensor_mod',
    'floor_mod',
    'floormod',
    'tensor_exp',
    'exp',
    'tensor_expm1',
    'expm1',
    'eq',
    'equal',
    'not_equal',
    'ne',
    'numel',
    'permute',
    'inplace_update',
    'inplace_add',
    'inplace_sub',
    'isfinite',
    'isnan',
    'isnan_ext',
    'isclose',
    'isreal',
    'isneginf',
    'isposinf',
    'is_complex',
    'log',
    'logdet',
    'log_matrix_determinant',
    'matrix_determinant',
    'det',
    'linspace',
    'logspace',
    'lu_solve',
    'matrix_solve',
    'std',
    'maximum',
    'minimum',
    'median',
    'positive',
    'floor',
    'logical_not',
    'logical_or',
    'logical_and',
    'logit',
    'gcd',
    'logcumsumexp',
    'logsumexp',
    'ldexp',
    'rsqrt',
    'reciprocal',
    'real',
    'sqrt',
    'square',
    't',
    'sin',
    'cos',
    'tan',
    'asin',
    'acos',
    'arccos',
    'atan',
    'sinc',
    'sinh',
    'cosh',
    'tanh',
    'tanh_',
    'tanhshrink',
    'asinh',
    'arcsinh',
    'acosh',
    'atanh',
    'arctanh',
    'atan2',
    'round',
    'bitwise_and',
    'bitwise_or',
    'bitwise_xor',
    'bitwise_left_shift',
    'bitwise_right_shift',
    'inv',
    'inverse',
    'invert',
    'erf',
    'erfc',
    'cdist',
    'ceil',
    'bernoulli',
    'heaviside',
    'hypot',
    'i0',
    'bessel_j0',
    'bessel_j1',
    'bessel_i0',
    'bessel_i0e',
    'bessel_k0',
    'bessel_k0e',
    'bessel_y0',
    'bessel_y1',
    'bessel_i1',
    'bessel_i1e',
    'bessel_k1',
    'bessel_k1e',
    'exp2',
    'deg2rad',
    'stft',
    'rad2deg',
    'truncate_div',
    'truncate_mod',
    'trunc',
    'gumbel_softmax',
    'kaiser_window',
    'matmul',
    'inner',
    'cummin',
    'cummax',
    'cumsum',
    'amin',
    'amax',
    'mean',
    'prod',
    'all',
    'any',
    'sparse_segment_mean',
    'block_diag',
    'atleast_1d',
    'dstack',
    'diff',
    'diff_ext',
    'atleast_2d',
    'cartesian_prod',
    'atleast_3d',
    'view_as_real',
    'vstack',
    'vander',
    'row_stack',
    'var',
    'var_mean',
    'std_mean',
    'combinations',
    'dist',
    'copysign',
    'hann_window',
    'log2',
    'slogdet',
    'trace',
    'xlogy',
    'log10',
    'log1p',
    'approximate_equal',
    'frac',
    'kron',
    'rot90',
    'remainder',
    'sgn',
    'sign',
    'signbit',
    'accumulate_n',
    'iou',
    'rotated_iou',
    'baddbmm',
    'baddbmm_ext',
    'bmm',
    'trapz',
    'cholesky',
    'cholesky_inverse',
    'cholesky_solve',
    'conj',
    'cosine_similarity',
    'cov',
    'cross',
    'einsum',
    'einsum_ext',
    'erfinv',
    'less_equal',
    'cumprod',
    'greater',
    'greater_equal',
    'igamma',
    'igammac',
    'isinf',
    'logical_xor',
    'imag',
    'roll',
    'sum',
    'matrix_exp',
    'matrix_power',
    'orgqr',
    'ormqr',
    'diag_embed',
    'fmax',
    'fmin',
    'inplace_index_add',
    'lu_unpack',
    'nanquantile',
    'polar',
    'polygamma',
    'quantile',
    'tril_indices',
    'histc',
    'nextafter',
    'triu_indices',
    'zeta',
    'fft',
    'fft2',
    'fftn',
    'ifft',
    'ifft2',
    'ifftn',
    'count_nonzero',
    'tensor_dot',
    'vecdot',
    'dot',
    'batch_dot',
    'eps',
    'multi_scale_deformable_attn_function',
]
__all__.sort()
