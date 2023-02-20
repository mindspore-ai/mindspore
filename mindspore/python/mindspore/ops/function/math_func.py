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

"""Defines math operators with functional form."""

import math
import numbers
import numpy as np
import mindspore.ops as ops
from mindspore.common import dtype as mstype
from mindspore.ops.primitive import constexpr
from mindspore.ops.primitive import _primexpr
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops.operations._inner_ops import Cummin
from mindspore.ops.operations.math_ops import STFT
from mindspore.ops.operations.math_ops import ReduceStd
from mindspore.ops.operations.math_ops import Logit
from mindspore.ops.operations.math_ops import LuUnpack
from mindspore.ops.operations.math_ops import Roll
from mindspore.ops.operations.array_ops import MatrixSetDiagV3, Transpose
from mindspore.nn import layer
from mindspore._checkparam import check_is_number
from mindspore._checkparam import Rel
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
    Sinc,
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
)
from mindspore.common.tensor import Tensor
from mindspore._checkparam import Validator as validator
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore._c_expression import Tensor as Tensor_
import mindspore.ops.function as F


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
tensor_ceil = P.Ceil()
tensor_add = P.Add()
neg_tensor = P.Neg()
tensor_sub = P.Sub()
tensor_mul = P.Mul()
tensor_div = P.RealDiv()
tensor_floordiv = P.FloorDiv()
floordiv = tensor_floordiv
xdivy_ = P.Xdivy()
tensor_pow = P.Pow()
pows = tensor_pow
tensor_mod = P.FloorMod()
floormod = tensor_mod
tensor_exp = P.Exp()
tensor_expm1 = P.Expm1()
tensor_lt = P.Less()
tensor_le = P.LessEqual()
tensor_gt = P.Greater()
tensor_ge = P.GreaterEqual()
not_equal_ = P.NotEqual()
transpose_ = P.Transpose()
cast_ = P.Cast()

#####################################
# Private Operation Functions.
#####################################
addn_ = P.AddN()
angle_ = Angle()
log_ = P.Log()
floor_ = P.Floor()
logical_not_ = P.LogicalNot()
logical_or_ = P.LogicalOr()
logical_and_ = P.LogicalAnd()
sin_ = P.Sin()
sinc_ = Sinc()
cos_ = P.Cos()
tan_ = P.Tan()
asin_ = P.Asin()
polar_ = Polar()
acos_ = P.ACos()
atan_ = P.Atan()
sinh_ = P.Sinh()
cosh_ = P.Cosh()
tanh_ = P.Tanh()
asinh_ = P.Asinh()
acosh_ = P.Acosh()
atanh_ = P.Atanh()
bitwise_and_ = P.BitwiseAnd()
bitwise_or_ = P.BitwiseOr()
bitwise_xor_ = P.BitwiseXor()
inv_ = P.math_ops.Inv()
invert_ = P.Invert()
erf_ = P.Erf()
erfc_ = P.Erfc()
bessel_j1_ = BesselJ1()
bessel_j0_ = BesselJ0()
bessel_i0_ = BesselI0()
bessel_i0e_ = P.BesselI0e()
bessel_k0_ = BesselK0()
bessel_k0e_ = BesselK0e()
bessel_y0_ = BesselY0()
bessel_y1_ = BesselY1()
bessel_i1_ = BesselI1()
bessel_i1e_ = P.BesselI1e()
bessel_k1_ = BesselK1()
bessel_k1e_ = BesselK1e()
equal_ = P.Equal()
isfinite_ = P.IsFinite()
isnan_ = P.IsNan()
maximum_ = P.Maximum()
minimum_ = P.Minimum()
lerp_ = P.Lerp()
tensor_round_ = P.Round()
linspace_ = P.LinSpace()
matrix_determinant_ = P.MatrixDeterminant()
log_matrix_determinant_ = P.LogMatrixDeterminant()
matrix_exp_ = MatrixExp()
exp2_ = P.Pow()
truncate_div_ = P.TruncateDiv()
truncate_mod_ = P.TruncateMod()
sparse_segment_mean_ = SparseSegmentMean()
lu_unpack_ = LuUnpack()
xlogy_ = P.Xlogy()
square_ = P.Square()
sqrt_ = P.Sqrt()
cumsum_ = P.CumSum()


#####################################
# Element-wise Operation Functions.
#####################################

def addn(x):
    """
    Computes addition of all input tensors element-wise.

    All input tensors must have the same shape.

    Args:
        x (Union(tuple[Tensor], list[Tensor])): A tuple or list composed of Tensor.

    Returns:
        Tensor, has the same shape and dtype as each Tensor of `x`.

    Raises:
        TypeError: If `x` is neither tuple nor list.
        ValueError: If there are Tensors with different shapes in `x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> y = Tensor(np.array([4, 5, 6]), mindspore.float32)
        >>> output = ops.addn([x, y, x, y])
        >>> print(output)
        [10. 14. 18.]
    """
    return addn_(x)


def abs(x):
    r"""
    Returns absolute value of a tensor element-wise.

    .. math::

        out_i = |x_i|

    Args:
        x (Tensor): The input tensor. The shape of tensor is
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Returns:
        Tensor, has the same shape as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([-1.0, 1.0, 0.0]), mindspore.float32)
        >>> output = ops.abs(x)
        >>> print(output)
        [1. 1. 0.]
    """
    return absolute_(x)


def absolute(x):
    """
    Alias for :func:`mindspore.ops.abs` .

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    return abs(x)


def add(x, y):
    r"""
    Adds two input tensors element-wise.

    .. math::

        out_{i} = x_{i} + y_{i}

    .. note::
        - Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
        - The inputs must be two tensors or one tensor and one scalar.
        - When the inputs are two tensors,
          dtypes of them cannot be bool at the same time, and the shapes of them can be broadcast.
        - When the inputs are one tensor and one scalar, the scalar could only be a constant.

    Args:
        x (Union[Tensor, number.Number, bool]): The first input is a number.Number or
            a bool or a tensor whose data type is
            `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_ or
            `bool_ <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_.
        y (Union[Tensor, number.Number, bool]): The second input, when the first input is a Tensor,
            the second input should be a number.Number or bool value, or a Tensor whose data type is number or bool\_.
            When the first input is Scalar, the second input must be a Tensor whose data type is number or bool\_.

    Returns:
        Tensor, the shape is the same as the one of the input `x` , `y` after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `x` and `y` is not one of the following: Tensor, number.Number, bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> # case 1: x and y are both Tensor.
        >>> x = Tensor(np.array([1, 2, 3]).astype(np.float32))
        >>> y = Tensor(np.array([4, 5, 6]).astype(np.float32))
        >>> output = ops.add(x, y)
        >>> print(output)
        [5. 7. 9.]
        >>> # case 2: x is a scalar and y is a Tensor
        >>> x = Tensor(1, mindspore.int32)
        >>> y = Tensor(np.array([4, 5, 6]).astype(np.float32))
        >>> output = ops.add(x, y)
        >>> print(output)
        [5. 6. 7.]
        >>> # the data type of x is int32, the data type of y is float32,
        >>> # and the output is the data format of higher precision float32.
        >>> print(output.dtype)
        Float32
    """
    return tensor_add(x, y)


def addcdiv(input_data, x1, x2, value):
    r"""
    Performs the element-wise division of tensor x1 by tensor x2,
    multiply the result by the scalar value and add it to input_data.

    .. math::
        y[i] = input\_data[i] + value[i] * (x1[i] / x2[i])

    Args:
        input_data (Tensor): The tensor to be added.
        x1 (Tensor): The numerator tensor.
        x2 (Tensor): The denominator tensor.
        value (Tensor): The multiplier for tensor x1/x2.

    Returns:
        Tensor, has the same shape and dtype as x1/x2.

    Raises:
        TypeError: If dtype of `x1`, `x2`, `value`, `input_data` is not tensor.
        ValueError: If `x1` could not be broadcast to a tensor with shape of `x2`.
        ValueError: If `value` could not be broadcast to tensors with shapes of `x1/x2`.
        ValueError: If `input_data` could not be broadcast to tensors with shapes of `value*(x1/x2)`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_data = Tensor(np.array([1, 1, 1, 1]), mindspore.float32)
        >>> x1 = Tensor(np.array([1, 2, 3, 4]), mindspore.float32)
        >>> x2 = Tensor(np.array([4, 3, 2, 1]), mindspore.float32)
        >>> value = Tensor([1], mindspore.float32)
        >>> y = ops.addcdiv(input_data, x1, x2, value)
        >>> print(y)
        [1.25      1.6666667 2.5       5.       ]
    """
    return _get_cache_prim(P.Addcdiv)()(input_data, x1, x2, value)


def addcmul(input_data, x1, x2, value):
    r"""
    Performs the element-wise product of tensor x1 and tensor x2,
    multiply the result by the scalar value and add it to input_data.

    .. math::
        output[i] = input\_data[i] + value[i] * (x1[i] * x2[i])

    Args:
        input_data (Tensor): The tensor to be added.
        x1 (Tensor): The tensor to be multiplied.
        x2 (Tensor): The tensor to be multiplied.
        value (Tensor): The multiplier for tensor x1*x2.

    Returns:
        Tensor, has the same shape and dtype as x1*x2.

    Raises:
        TypeError: If dtype of `x1`, `x2`, `value`, `input_data` is not tensor.
        TypeError: If dtype of `input_data` is not one of: float32, float16, int32.
        TypeError: If dtype of `x1` or `x2` is not one of: float32, float16, int32.
        TypeError: If dtype of `value` is not one of: float32, float16, int32.
        ValueError: If `x1` could not be broadcast to a tensor with shape of `x2`.
        ValueError: If `value` could not be broadcast to tensors with shapes of `x1` * `x2`.
        ValueError: If `input_data` could not be broadcast to tensors with shapes of `value*(x1*x2)`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_data = Tensor(np.array([1, 1, 1]), mindspore.float32)
        >>> x1 = Tensor(np.array([[1], [2], [3]]), mindspore.float32)
        >>> x2 = Tensor(np.array([[1, 2, 3]]), mindspore.float32)
        >>> value = Tensor([1], mindspore.float32)
        >>> y = ops.addcmul(input_data, x1, x2, value)
        >>> print(y)
        [[ 2.  3.  4.]
         [ 3.  5.  7.]
         [ 4.  7. 10.]]
    """
    return _get_cache_prim(P.Addcmul)()(input_data, x1, x2, value)


def angle(x):
    """
    Returns the element-wise argument of a complex tensor.
    The elements in input are considered to be complex numbers of the form a+bj, where a is the real part and b
    is the imaginary part. The argument returned by this function is of the form atan2(b,a).

    Args:
        x (Tensor): The input tensor. types: complex64, complex128.

    Returns:
        Tensor, has the float32 or float64 type and the same shape as input.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If the dtype of input is not one of: complex64, complex128.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> input = Tensor([-1.5 + 7.8j, 3 + 5.75j], mindspore.complex64)
        >>> output = ops.angle(input)
        >>> print(output)
        [1.7607845 1.0899091]
    """
    return angle_(x)


def bincount(x, weights=None, minlength=0):
    """
    Counts the number of occurrences of each value in `x`.

    If you don't specify 'minlength', the length of output Tensor will be
    the maximum value of the input 'x' plus one.

    If `minlength` is specified, the length of output Tensor is the value of maximum of `x` plus 1 and `minlength`.

    Each value in the output Tensor marks the number of occurrences of that index in 'x'.
    If 'weights' is specified, the output results are weighted, i.e ``out[n] += weight[i]`` instead of ``out[n] += 1``.

    Args:
        x (Tensor): 1-d input tensor.
        weights (Tensor, optional): Weights, a tensor of the same shape as `x`. Defaults to None.
        minlength (int, optional): A minimum number of bins for the output tensor. Defaults to 0.

    Returns:
        Tensor, a tensor of shape Size([max(input) + 1]) if input is non-empty, else Size(0).

    Raises:
        TypeError: if `x` or `weights` is not a tensor.
        ValueError: If `x` is not one-dimensional, or if `x` and `weights` do not have the same shape.
        ValueError: If `minlength` is a negative integer.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor([2, 4, 1, 0, 0], dtype=mstype.int64)
        >>> print(ops.bincount(x, minlength=7))
        [2. 1. 1. 0. 1. 0. 0.]
        >>> weights = Tensor([0, 0.25, 0.5, 0.75, 1], dtype=mstype.float32)
        >>> print(ops.bincount(x, weights=weights))
        [1.75 0.5  0.   0.   0.25]
    """
    if not isinstance(x, Tensor):
        raise TypeError("For math function 'bincount', 'x' must be Tensor.")
    if x.dtype not in (mstype.Int, mstype.int16, mstype.int32, mstype.int64):
        raise TypeError(f"For math function 'bincount', the type of 'x' must be in [Int, int16, int32, int64], but" \
                        f" got {x.dtype}.")
    if weights is not None and not isinstance(weights, Tensor):
        raise TypeError(f"For math function 'bincount', 'weights' must be Tensor, but got {type(weights)}.")
    if not isinstance(minlength, int):
        raise TypeError(f"For math function 'bincount', 'minlength' must be int but got {type(minlength)}.")
    rank_op = _get_cache_prim(P.Rank)()
    if rank_op(x) != 1:
        raise ValueError("For math function 'bincount', `x` should be one-dimensional tensor.")
    if x.shape[0] == 0:
        return Tensor([])
    if minlength < 0:
        raise ValueError("For bincount minlength should be >= 0.")
    if max(x.astype(mstype.float32)) > minlength - 1:
        length = (max(x.astype(mstype.float32)) + 1).astype(mstype.int32)
    else:
        length = P.Cast()(minlength, mstype.int32)
    idx = F.arange(length).expand_dims(-1)
    idx_mapping = equal(x, idx)
    if weights is not None:
        if x.shape != weights.shape:
            raise ValueError('for bincount `x` and `weights` must have the same length')
        idx_mapping *= weights
    return P.ReduceSum()(idx_mapping.astype(mstype.float32), 1).ravel()


def exp2(x):
    """
    Computes base two exponential of Tensor `x` element-wise.

    .. math::
        out_i = 2^{x_i}

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([2, 3, 4]), mindspore.float32)
        >>> output = ops.exp2(x)
        >>> print(output)
        [ 4.  8. 16.]
    """

    tensor_2 = Tensor(np.array(2.0).astype(np.float32))
    if x.dtype == mstype.float16:
        tensor_2 = Tensor(np.array(2.0).astype(np.float16))
    return exp2_(tensor_2, x)


def argmin(x, axis=None, keepdims=False):
    """
    Returns the indices of the minimum value of a tensor across the axis.

    If the shape of input tensor is :math:`(x_1, ..., x_N)`, the shape of the output tensor is
    :math:`(x_1, ..., x_{axis-1}, x_{axis+1}, ..., x_N)`.

    Args:
        x (Tensor): Input tensor. The shape is :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
        axis (Union[int, None], optional): Axis where the Argmin operation applies to. Default: None.
        keepdims (bool, optional): Whether the output tensor retains the specified
            dimension. Ignored if `axis` is None. Default: False.

    Returns:
        Tensor, indices of the min value of input tensor across the axis.

    Raises:
        TypeError: If `axis` is not an int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([2.0, 3.1, 1.2]), mindspore.float32)
        >>> index = ops.argmin(input_x)
        >>> print(index)
        2
    """
    if x.shape == ():
        return Tensor(0)
    is_axis_none = False
    if axis is None:
        x = P.Reshape()(x, (-1,))
        axis = 0
        is_axis_none = True
    out = _get_cache_prim(P.Argmin)(axis)(x)
    if keepdims and not is_axis_none:
        out = P.ExpandDims()(out, axis)
    return out


neg_tensor = P.Neg()


def neg(x):
    """
    Returns a tensor with negative values of the input tensor element-wise.

    .. math::

        out_{i} = - x_{i}

    Args:
        x (Tensor): The input tensor with a dtype of Number, its rank must be in [0, 7] inclusive.

    Returns:
        Tensor, has the same shape and dtype as input.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1, 2, -1, 2, 0, -3.5]), mindspore.float32)
        >>> output = ops.neg(x)
        >>> print(output)
        [-1.  -2.   1.  -2.   0.   3.5]
    """
    return neg_tensor(x)


def negative(x):
    r"""
    Alias for :func:`mindspore.ops.neg` .

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    return neg_tensor(x)


def positive(x):
    r"""
    Return self Tensor.

    Args:
        x(Tensor): Input Tensor.

    Returns:
        Tensor, self input.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If the dtype of self Tensor is bool type.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> from mindspore import dtype as mstype
        >>> x = Tensor(np.array([-5.0, 1.5, 3.0, 100.0]), mstype.float32)
        >>> print(ops.positive(x))
        [ -5.    1.5   3.  100. ]
    """
    if not isinstance(x, (Tensor, Tensor_)):
        raise TypeError(f"For positive, the input must be a Tensor, but got {type(x)}")
    if x.dtype == mstype.bool_:
        raise TypeError("For positive, the type of tensor can not be bool.")
    return x


def numel(x):
    r"""
    Returns a Scalar of type int that represents the total number of elements in the Tensor.

    Args:
        x (Tensor): Input Tensor.

    Returns:
        int. A scalar representing the total of elements in the Tensor.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32)
        >>> print(ops.numel(input_x))
        4
    """
    if not isinstance(x, (Tensor, Tensor_)):
        raise TypeError(f"For numel, the input must be a Tensor, but got {type(x)}")
    return x.size


def permute(x, dims):
    """
    Permutes the dimensions of the input tensor according to input `dims` .

    Args:
        x(Tensor): Input Tensor.
        dims(Union[tuple(int), list(int), int]): Permute will permute the tensor to the input `dims` order.

    Returns:
        Tensor, has the same dimension as input tensor, with `dims` suitably permuted.

    Raises:
        ValueError: If `dims` is None.
        ValueError: If the number of elements of `dims` is not equal to `x` ndim.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), mindspore.float32)
        >>> input_perm = (0, 2, 1)
        >>> print(ops.permute(input_x, input_perm))
        [[[ 1.  4.]
          [ 2.  5.]
          [ 3.  6.]]
         [[ 7. 10.]
          [ 8. 11.]
          [ 9. 12.]]]
    """
    return transpose_(x, dims)


def ceil(x):
    r"""
    Rounds a tensor up to the closest integer element-wise.

    .. math::

        out_i = \lceil x_i \rceil = \lfloor x_i \rfloor + 1

    Args:
        x (Tensor): The input tensor with a dtype of float16 or float32, its rank must be in [0, 7] inclusive.

    Returns:
        Tensor, has the same shape as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16 or float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1.1, 2.5, -1.5]), mindspore.float32)
        >>> output = ops.ceil(x)
        >>> print(output)
        [ 2.  3. -1.]
    """
    return tensor_ceil(x)


def round(x):
    r"""
    Returns half to even of a tensor element-wise.

    .. math::

        out_i \approx x_i

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor, has the same shape and type as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.8, 1.5, 2.3, 2.5, -4.5]), mindspore.float32)
        >>> output = ops.round(x)
        >>> print(output)
        [ 1.  2.  2.  2. -4.]
    """
    return tensor_round_(x)


def sub(x, y):
    r"""
    Subtracts the second input tensor from the first input tensor element-wise.

    .. math::

        out_{i} = x_{i} - y_{i}

    .. note::
        - Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
        - The inputs must be two tensors or one tensor and one scalar.
        - When the inputs are two tensors,
          dtypes of them cannot be bool at the same time, and the shapes of them can be broadcast.
        - When the inputs are one tensor and one scalar, the scalar could only be a constant.

    Args:
        x (Union[Tensor, number.Number, bool]): The first input is a number.Number or
            a bool or a tensor whose data type is
            `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_ or
            `bool_ <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_.
        y (Union[Tensor, number.Number, bool]): The second input, when the first input is a Tensor,
            the second input should be a number.Number or bool value, or a Tensor whose data type is number or bool\_.
            When the first input is Scalar, the second input must be a Tensor whose data type is number or bool\_.

    Returns:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `x` and `y` is not a number.Number or a bool or a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1, 2, 3]), mindspore.int32)
        >>> y = Tensor(np.array([4, 5, 6]), mindspore.int32)
        >>> output = ops.sub(x, y)
        >>> print(output)
        [-3 -3 -3]
    """
    return tensor_sub(x, y)


def subtract(x, other, *, alpha=1):
    r"""
    Performs the element-wise subtraction of input tensors.

    .. math::
        output[i] = x[i] - alpha * y[i]

    Args:
        x (Union[Tensor, number.Number]): Tensor or Number involved in subtraction.
        other (Union[Tensor, number.Number]): Tensor or Number involved in subtraction.

    Keyword Args:
        alpha (Number): The multiplier for `other`. Default: 1.

    Returns:
        Tensor, has the same shape and dtype as input tensors.

    Raises:
        TypeError: `x` or `other` is neither Tensor nor number.Number.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([4, 5, 6]), mindspore.float32)
        >>> y = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> z = ops.subtract(x, y, alpha=1)
        >>> print(z)
        [3. 3. 3.]
    """
    return tensor_sub(x, alpha * other)


def true_divide(dividend, divisor):
    r"""
    Alias for :func:`mindspore.ops.div` with :math:`rounding\_mode=None`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    return div(dividend, divisor, rounding_mode=None)


def mul(x, y):
    r"""
    Multiplies two tensors element-wise.

    .. math::

        out_{i} = x_{i} * y_{i}
    .. note::
        - Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
        - The inputs must be two tensors or one tensor and one scalar.
        - When the inputs are two tensors,
          dtypes of them cannot be bool at the same time, and the shapes of them can be broadcast.
        - When the inputs are one tensor and one scalar, the scalar could only be a constant.

    Args:
        x (Union[Tensor, number.Number, bool]): The first input is a number.Number or
            a bool or a tensor whose data type is
            `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_ or
            `bool_ <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_.
        y (Union[Tensor, number.Number, bool]): The second input, when the first input is a Tensor,
            the second input should be a number.Number or bool value, or a Tensor whose data type is number or bool\_.
            When the first input is Scalar, the second input must be a Tensor whose data type is number or bool\_.

    Returns:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `x` and `y` is not one of the following: Tensor, number.Number, bool.
        ValueError: If `x` and `y` are not the same shape.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
        >>> y = Tensor(np.array([4.0, 5.0, 6.0]), mindspore.float32)
        >>> output = ops.mul(x, y)
        >>> print(output)
        [ 4. 10. 18.]
    """
    return tensor_mul(x, y)


def multiply(input, other):
    r"""
    Refer to :func:`mindspore.ops.mul` for more details.
    """
    return tensor_mul(input, other)


def div(input, other, rounding_mode=None):
    """
    Divides the first input tensor by the second input tensor in floating-point type element-wise.

    Inputs of `input` and `other` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be bool at the same time, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar, the scalar could only be a constant.

    .. math::

        out_{i} = input_{i} / other_{i}

    Args:
        input (Union[Tensor, Number, bool]): The first input is a number or
            a bool or a tensor whose data type is number or bool.
        other (Union[Tensor, Number, bool]): The second input is a number or
            a bool when the first input is a tensor or a tensor whose data type is number or bool.
        rounding_mode (str, optional): Type of rounding applied to the result. Three types are defined as,

            - None: Default behavior. Equivalent to true division in Python or `true_divide` in NumPy.

            - "floor": Rounds the results of the division down. Equivalent to floor division in Python
              or `floor_divide` in NumPy.

            - "trunc": Rounds the results of the division towards zero. Equivalent to C-style integer division.
              Default: None.

    Returns:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `input` and `other` is not one of the following: Tensor, Number, bool.
        ValueError: If `rounding_mode` value is not None, "floor" or "trunc".

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
        >>> y = Tensor(np.array([4.0, 5.0, 6.0]), mindspore.float32)
        >>> output = ops.div(x, y)
        >>> print(output)
        [0.25 0.4 0.5]
    """
    if rounding_mode is not None and rounding_mode not in ['floor', 'trunc']:
        raise ValueError("For ops.div, rounding_mode value should be None, 'floor' or 'trunc'.")

    if rounding_mode == 'floor':
        return _get_cache_prim(P.FloorDiv)()(input, other)
    output = _get_cache_prim(P.Div)()(input, other)
    if rounding_mode == 'trunc':
        output = _get_cache_prim(P.Trunc)()(output)
    return output


def divide(x, other, *, rounding_mode=None):
    """
    Alias for :func:`mindspore.ops.div` .

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    return div(x, other, rounding_mode)


def float_power(x, exponent):
    """
    Computes `x` to the power of the exponent.
    For the real number type, cast `x` and `exponent` to mindspore.float64 to calculate.
    Currently, complex type calculation is not supported.

    Args:
        x (Union[Tensor, Number]): The first input is a tensor or a number.
        exponent (Union[Tensor, Number]): The second input, if the first input is Tensor,
            the second input can be Number or Tensor. Otherwise, it must be a Tensor.

    Returns:
        Tensor, the shape is the same as the one after broadcasting. For the complex type,
        the return value type is the same as the input type. For the real number type,
        the return value type is mindspore.float64.

    Raises:
        TypeError: If neither `x` nor `exponent` is a Tensor.
        TypeError: If the data type of `x` or `exponent` is not in Tensor and Number.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([-1.5, 0., 2.]))
        >>> output = ops.float_power(x, 2)
        >>> print(output)
        [2.25 0.   4.  ]
    """
    if not (isinstance(x, Tensor) or isinstance(exponent, Tensor)):
        raise TypeError("At least one of the types of inputs must be tensor, " + \
                        f"but the type of 'x' got is {type(x)}, " + \
                        f"and the type of 'exponent' is {type(exponent)}.")
    if not isinstance(x, (Tensor, numbers.Number)):
        raise TypeError(f"The type of 'x' must be Tensor or Number, but got {type(x)}.")
    if not isinstance(exponent, (Tensor, numbers.Number)):
        raise TypeError(f"The type of 'exponent' must be Tensor or Number, but got {type(exponent)}.")

    if (isinstance(x, Tensor) and is_complex(x)) or \
            (isinstance(exponent, Tensor) and is_complex(exponent)) or \
            isinstance(x, complex) or isinstance(exponent, complex):
        x = cast_(x, mstype.complex128)
        exponent = cast_(exponent, mstype.complex128)
    else:
        x = cast_(x, mstype.float64)
        exponent = cast_(exponent, mstype.float64)
    return pow(x, exponent)


def floor_div(x, y):
    """
    Divides the first input tensor by the second input tensor element-wise and round down to the closest integer.

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be bool at the same time, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    .. math::

        out_{i} = \\text{floor}( \\frac{x_i}{y_i})

    where the :math:`floor` indicates the Floor operator, for more details,
    please refer to the :class:`mindspore.ops.Floor` operator.

    Args:
        x (Union[Tensor, Number, bool]): The first input is a number or
            a bool or a tensor whose data type is number or bool.
        y (Union[Tensor, Number, bool]): The second input is a number or
            a bool when the first input is a tensor, or it can be a tensor whose data type is number or bool.
    Returns:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If neither `x` nor `y` is a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([2, 4, -1]), mindspore.int32)
        >>> y = Tensor(np.array([3, 3, 3]), mindspore.int32)
        >>> output = ops.floor_div(x, y)
        >>> print(output)
        [ 0  1 -1]
    """
    return tensor_floordiv(x, y)


def fmod(x, other):
    """
    Computes the floating-point remainder of the division operation x/other.

    .. math::

        out = x - n * other

    Where :math:`n` is :math:`x/other` with its fractional part truncated.
    The returned value has the same sign as `x` and is less than `other` in magnitude.

    Args:
        x (Union[Tensor, Number]): the dividend.
        other (Union[Tensor, Number]): the divisor.

    Returns:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If neither `x` nor `other` is a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([-4., -3.5, 0, 3.5, 4]), mindspore.float32)
        >>> output = ops.fmod(x, 2.5)
        >>> print(output)
        [-1.5 -1.   0.   1.   1.5]
    """
    if not (isinstance(x, (Tensor, Tensor_)) or isinstance(other, (Tensor, Tensor_))):
        raise TypeError("At least one of the types of inputs must be tensor, " + \
                        f"but the type of 'x' got is {type(x)}, " + \
                        f"and the type of 'other' is {type(other)}.")
    return x - div(x, other, rounding_mode="trunc") * other


def pow(x, y):
    r"""
    Calculates the `y` power of each element in `x`.

    .. math::

        out_{i} = x_{i} ^{ y_{i}}

    .. note::
        - Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
        - The inputs must be two tensors or one tensor and one scalar.
        - When the inputs are two tensors,
          dtypes of them cannot be bool at the same time, and the shapes of them can be broadcast.

    Args:
        x (Union[Tensor, number.Number, bool]): The first input is a number.Number or
            a bool or a tensor whose data type is
            `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_ or
            `bool_ <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_.
        y (Union[Tensor, number.Number, bool]): The second input, when the first input is a Tensor,
            the second input should be a number.Number or bool value, or a Tensor whose data type is number or bool\_.
            When the first input is Scalar, the second input must be a Tensor whose data type is number or bool\_.

    Returns:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `x` and `y` is not one of the following: Tensor, number.Number or bool.
        ValueError: If the shape of `x` and `y` are different.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
        >>> y = 3.0
        >>> output = ops.pow(x, y)
        >>> print(output)
        [ 1.  8. 64.]
        >>>
        >>> x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
        >>> y = Tensor(np.array([2.0, 4.0, 3.0]), mindspore.float32)
        >>> output = ops.pow(x, y)
        >>> print(output)
        [ 1. 16. 64.]
    """
    return tensor_pow(x, y)


def floor_mod(x, y):
    r"""
    Computes the remainder of division element-wise. It's a flooring divide.
    E.g. :math:`floor(x / y) * y + mod(x, y) = x`.

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be both bool, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    .. math::

        out_{i} =\text{floor}(x_{i} // y_{i})

    where the :math:`floor` indicates the Floor operator, for more details,
    please refer to the :class:`mindspore.ops.Floor` operator.

    .. warning::
        - Data of input `y` should not be 0, or the maximum value of its dtype will be returned.
        - When the elements of input exceeds 2048 , the accuracy of operator cannot guarantee the requirement of
          double thousandths in the mini form.
        - Due to different architectures, the calculation results of this operator on NPU and CPU may be inconsistent.
        - If shape is expressed as (D1,D2... ,Dn), then D1\*D2... \*DN<=1000000,n<=8.

    Args:
        x (Union[Tensor, Number, bool]): The first input is a number or
            a bool or a tensor whose data type is number or bool.
        y (Union[Tensor, Number, bool]): The second input is a number or
            a bool when the first input is a tensor, or it can be a tensor whose data type is number or bool.

    Returns:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision of the two inputs.

    Raises:
        TypeError: If neither `x` nor `y` is a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([2, 4, -1]), mindspore.int32)
        >>> y = Tensor(np.array([3, 3, 3]), mindspore.int32)
        >>> output = ops.floor_mod(x, y)
        >>> print(output)
        [2 1 2]
    """
    return tensor_mod(x, y)


def exp(x):
    r"""
    Returns exponential of a tensor element-wise.

    .. math::

        out_i = e^{x_i}

    Args:
        x (Tensor): The input tensor, its rank must be in [0, 7] inclusive.

    Returns:
        Tensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
        >>> output = ops.exp(x)
        >>> print(output)
        [ 2.718282  7.389056 54.598152]
    """
    return tensor_exp(x)


def expm1(x):
    r"""
    Returns exponential then minus 1 of a tensor element-wise.

    .. math::

        out_i = e^{x_i} - 1

    Args:
        x (Tensor): The input tensor with a dtype of float16 or float32, its rank must be in [0, 7] inclusive.

    Returns:
        Tensor, has the same shape as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.0, 1.0, 2.0, 4.0]), mindspore.float32)
        >>> output = ops.expm1(x)
        >>> print(output)
        [ 0.        1.718282  6.389056 53.598152]
    """
    return tensor_expm1(x)


def log(x):
    """
    Returns the natural logarithm of a tensor element-wise.

    .. math::
        y_i = log_e(x_i)

    .. note::
        The dimension of the input Tensor on Ascend should be less than or equal to 8, and the dimension of the
        input Tensor on the CPU should be less than 8.

    .. warning::
        If the input value of operator Log is within the range (0, 0.01] or [0.95, 1.05], the output accuracy may
        be affacted.

    Args:
        x (Tensor): Input Tensor of any dimension. The value must be greater than 0.

    Returns:
        Tensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16, float32 or float64 on CPU.
        TypeError: If dtype of `x` is not float16 or float32 on Ascend.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
        >>> output = ops.log(x)
        >>> print(output)
        [0.        0.6931472 1.3862944]
    """
    return log_(x)


def logdet(x):
    r"""
    Calculates log determinant of a square matrix or batches of square matrices.

    Args:
        x (Tensor): Input Tensor of any dimension.

    Returns:
        Tensor, the log determinant of `x`. If the matrix determinant is smaller than 0, nan will be returned. If the
        matrix determinant is 0, -inf will be returned.

    Raises:
        TypeError: If dtype of `x` is not float32, float64, Complex64 or Complex128.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> a = Tensor([[[8, 9], [1, 2]], [[5, 6], [3, 4]]], mindspore.float32)
        >>> output = ops.logdet(a)
        >>> print(output)
        [1.9459091 0.6931454]
    """
    det_x = matrix_determinant(x)
    return log_(det_x)


def floor(x):
    r"""
    Rounds a tensor down to the closest integer element-wise.

    .. math::

        out_i = \lfloor x_i \rfloor

    Args:
        x (Tensor): The input tensor, its rank must be in [0, 7] inclusive
            and data type must be float16, float32 or float64.

    Returns:
        Tensor, has the same shape as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not in [float16, float32, float64].

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1.1, 2.5, -1.5]), mindspore.float32)
        >>> output = ops.floor(x)
        >>> print(output)
        [ 1.  2. -2.]
    """
    _floor = _get_cache_prim(P.Floor)()
    return _floor(x)


def i0(x):
    r"""
    Alias for :func:`mindspore.ops.bessel_i0` .

    Supported Platforms:
        ``GPU`` ``CPU``
    """
    return bessel_i0(x)


def inplace_update(x, v, indices):
    """
    Updates specified rows with values in `v`.

    Note:
            `indices` refers to the left-most dimension.

    Args:
        indices (Union[int, tuple], Tensor): Indices into the left-most dimension of `x`, and determines which rows of x
            to update with v. It is an int or tuple, whose value is in [0, the first dimension size of x). If the type
            is Tensor, it supports dynamic shape. Otherwise, it only supports static shape.
        x (Tensor): A tensor which to be inplace updated. It can be one of the following data types:
            float32, float16 and int32.
        v (Tensor): A tensor with the same type as `x` and the same dimension size as `x` except
            the first dimension, which must be the same as the size of `indices`.

    Returns:
        Tensor, with the same type and shape as the input `x`.

    Raises:
        TypeError: If `indices` is neither int nor tuple.
        TypeError: If `indices` is a tuple and its element is not an int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> indices = (0, 1)
        >>> x = Tensor(np.array([[1, 2], [3, 4], [5, 6]]), mindspore.float32)
        >>> v = Tensor(np.array([[0.5, 1.0], [1.0, 1.5]]), mindspore.float32)
        >>> output = ops.inplace_update(x, v, indices)
        >>> print(output)
        [[0.5 1. ]
         [1.  1.5]
         [5.  6. ]]
    """
    if not isinstance(indices, (Tensor, Tensor_)):
        inplace_update_inner = _get_cache_prim(P.InplaceUpdate)(indices)
        output = inplace_update_inner(x, v)
    else:
        inplace_update_inner = InplaceUpdateV2()
        output = inplace_update_inner(x, indices, v)
    return output


def inplace_add(x, v, indices):
    """
    Adds `v` into specified rows of `x`. Computes `y` = `x`; y[i,] += `v`.

    Note:
            `indices` refers to the left-most dimension.

    Args:
        x (Tensor): The first input is a tensor whose data type is float16, float32, float64 or int32.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.
        v (Tensor): The second input is a tensor that has the same dimension sizes as `x` except
            the first dimension, which must be the same as indices' size. It has the same data type with `x`.
        indices (Union[int, tuple]): Indices into the left-most dimension of `x`, and determines which rows of `x`
            to add with `v`. It is an integer or a tuple, whose value is in [0, the first dimension size of `x`).

    Returns:
        Tensor, has the same shape and dtype as `x`.

    Raises:
        TypeError: If `indices` is neither int nor tuple.
        TypeError: If `indices` is a tuple whose elements are not all int.
        ValueError: If the rank of `x` is not equal to the rank of `v`.
        ValueError: If the length of `indices` is not equal to `v.shape[0]`.
        ValueError: If the values of `indices` are not in range of `[0, x.shape[0])`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor, ops
        >>> indices = (0, 1)
        >>> x = Tensor(np.array([[1, 2], [3, 4], [5, 6]]), mindspore.float32)
        >>> input_v = Tensor(np.array([[0.5, 1.0], [1.0, 1.5]]), mindspore.float32)
        >>> output = ops.inplace_add(x, input_v, indices)
        >>> print(output)
        [[1.5 3. ]
         [4.  5.5]
         [5.  6. ]]
    """
    inplace_add_inner = _get_cache_prim(P.InplaceAdd)(indices)
    return inplace_add_inner(x, v)


def inplace_index_add(var, indices, updates, axis):
    """
    Adds tensor `updates` to specified axis and indices of tensor `var`. The axis should be in [0,  len(var.dim) - 1],
    and indices should be in [0, the size of `var` - 1] at the axis dimension.

    Args:
        var (Parameter): The input Parameter to add to, with data type uint8, int8, int16, int32,
            float16, float32, float64.
        indices (Tensor): Add the value of `var` and `updates` along the dimension of the `axis` according to the
            specified index value, with data type int32.
            The `indices` must be 1D with the same size as the size of `updates` in the `axis` dimension. The values
            of `indices` should be in [0, b), where the b is the size of `var` in the `axis` dimension.
        updates (Tensor): The input tensor with the value to add. Must have same data type as `var`.
            The shape must be the same as `var` except the `axis` th dimension.
        axis (int): The dimension along which to index.

    Returns:
        Tensor, has the same shape and dtype as `var`.

    Raises:
        TypeError: If `var` is not a Parameter.
        TypeError: If neither `indices` nor `updates` is a Tensor.
        ValueError: If axis is out of `var` rank's range.
        ValueError: If `var` rank is not the same as `updates` rank.
        ValueError: If shape of `indices` is not 1D or size of `indices` is not equal to dimension of updates[axis].
        ValueError: If `updates`'s shape is not the same as `var` except the `axis` th dimension.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> var = Parameter(np.array([[1, 2], [3, 4], [5, 6]]), mindspore.float32)
        >>> indices = Tensor(np.array([0, 1]), mindspore.int32)
        >>> updates = Tensor(np.array([[0.5, 1.0], [1.0, 1.5]]), mindspore.float32)
        >>> var = ops.inplace_index_add(var, indices, updates, axis=0)
        >>> print(var)
        [[1.5 3. ]
         [4.  5.5]
         [5.  6. ]]
    """

    inplace_index_add_ = InplaceIndexAdd(axis)
    return inplace_index_add_(var, indices, updates)


def inplace_sub(x, v, indices):
    r"""
    Subtracts `v` into specified rows of `x`. Computes :math:`y = x`; :math:`y[i,] -= input\_v`.

    Note:
            `indices` refers to the left-most dimension.

    Args:
        indices (Union[int, tuple]): Indices into the left-most dimension of `x`, and determines which rows of `x`
            to subtract with `v`. It is an int or tuple, whose value is in [0, the first dimension size of `x`).
        x (Tensor): The first input is a tensor whose data type is float16, float32, float64 or int32.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.
        v (Tensor): The second input is a tensor who has the same dimension sizes as `x` except
            the first dimension, which must be the same as indices' size. It has the same data type with `x`.

    Returns:
        Tensor, has the same shape and dtype as `x`.

    Raises:
        TypeError: If `indices` is neither int nor tuple.
        TypeError: If `indices` is a tuple whose elements are not all int.
        ValueError: If the rank of `x` is not equal to the rank of `v`.
        ValueError: If the length of `indices` is not equal to `v.shape[0]`.
        ValueError: If the values of `indices` are not in range of `[0, x.shape[0])`.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor, ops
        >>> indices = (0, 1)
        >>> x = Tensor(np.array([[1, 2], [3, 4], [5, 6]]), mindspore.float32)
        >>> input_v = Tensor(np.array([[0.5, 1.0], [1.0, 1.5]]), mindspore.float32)
        >>> output = ops.inplace_sub(x, input_v, indices)
        >>> print(output)
        [[0.5 1. ]
         [2.  2.5]
         [5.  6. ]]
    """
    inplace_sub_inner = _get_cache_prim(P.InplaceSub)(indices)
    return inplace_sub_inner(x, v)


def logical_not(x):
    """
    Computes the "logical NOT" of a tensor element-wise.

    .. math::

        out_{i} = \\neg x_{i}

    Args:
        x (Tensor): The input tensor.
            :math:`(N,*)` where :math:`*` means,any number of additional dimensions.

    Returns:
        Tensor, the shape is the same as the `x`, and the dtype is bool.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([True, False, True]), mindspore.bool_)
        >>> output = ops.logical_not(x)
        >>> print(output)
        [False  True False]
    """
    if isinstance(x, Tensor) and x.dtype != mstype.bool_:
        x = x.astype(mstype.bool_)
    return logical_not_(x)


def logical_or(x, y):
    """
    Computes the "logical OR" of two tensors element-wise.

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one bool.
    When the inputs are two tensors, the shapes of them could be broadcast,
    and the data types of them must be bool.
    When the inputs are one tensor and one bool, the bool object could only be a constant,
    and the data type of the tensor must be bool.

    .. math::

        out_{i} = x_{i} \\vee y_{i}

    Note:
        LogicalOr supports broadcasting.

    Args:
        x (Union[Tensor, bool]): The first input is a bool or a tensor whose data type can be implicitly
            converted to bool.
        y (Union[Tensor, bool]): The second input is a bool when the first input is a tensor or
            a tensor whose data type can be implicitly converted to bool.

    Returns:
        Tensor, the shape is the same as the one after broadcasting, and the data type is bool.

    Raises:
        TypeError: If neither `x` nor `y` is a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([True, False, True]), mindspore.bool_)
        >>> y = Tensor(np.array([True, True, False]), mindspore.bool_)
        >>> output = ops.logical_or(x, y)
        >>> print(output)
        [ True  True  True]
    """
    if isinstance(x, Tensor) and x.dtype != mstype.bool_:
        x = x.astype(mstype.bool_)
    if isinstance(y, Tensor) and y.dtype != mstype.bool_:
        y = y.astype(mstype.bool_)
    return logical_or_(x, y)


def logical_and(x, y):
    r"""
    Computes the "logical AND" of two tensors element-wise.

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one bool.
    When the inputs are two tensors, the shapes of them could be broadcast,
    and the data types of them must be bool.
    When the inputs are one tensor and one bool, the bool object could only be a constant,
    and the data type of the tensor must be bool.

    .. math::

        out_{i} = x_{i} \wedge y_{i}

    Note:
        LogicalAnd supports broadcasting.

    Args:
        x (Union[Tensor, bool]): The first input is a bool or a tensor whose data type can be implicitly
            converted to bool.
        y (Union[Tensor, bool]): The second input is a bool when the first input is a tensor or
            a tensor whose data type can be implicitly converted to bool.

    Returns:
        Tensor, the shape is the same as the one after broadcasting, and the data type is bool.

    Raises:
        TypeError: If neither `x` nor `y` is a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([True, False, True]), mindspore.bool_)
        >>> y = Tensor(np.array([True, True, False]), mindspore.bool_)
        >>> output = ops.logical_and(x, y)
        >>> print(output)
        [ True False False]
    """
    if isinstance(x, Tensor) and x.dtype != mstype.bool_:
        x = x.astype(mstype.bool_)
    if isinstance(y, Tensor) and y.dtype != mstype.bool_:
        y = y.astype(mstype.bool_)
    return logical_and_(x, y)


def sign(x):
    r"""
    Returns an element-wise indication of the sign of a number.

    .. math::
        \text{out}_{i} = \begin{cases}
                        0 & |\text{x}_i| == 0 \\
                        \frac{{\text{x}_i}}{|{\text{x}_i}|} & \text{otherwise}
                        \end{cases}

    Args:
        x (Tensor): Input Tensor.

    Returns:
        Tensor, the sign of x.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import mindspore.ops as ops
        >>> x = ms.Tensor([[-1, 0, 2, 4, 6], [2, 3, 5, -6, 0]])
        >>> output = ops.sign(x)
        >>> print(output)
        [[-1  0  1  1  1]
         [ 1  1  1 -1  0]]
        >>> x = ms.Tensor([[-1, 0, float('inf'), 4, float('nan')], [2, 3, float('-inf'), -6, 0]])
        >>> output = ops.sign(x)
        >>> print(output)
        [[-1.  0.  1.  1.  0.]
         [ 1.  1. -1. -1.  0.]]
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"For sign, the input must be a Tensor, but got {type(x)}")
    return _get_cache_prim(ops.Sign)()(x)


def signbit(x):
    r"""
    Returns element-wise True where signbit is set (less than zero).

    Args:
        x (Tensor): The input value.

    Returns:
        Tensor, the signbit of x.

    Raises:
        TypeError: If input is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import mindspore.ops as ops
        >>> x = ms.Tensor([0.3, 1.2, 0., -2.5])
        >>> output = ops.signbit(x)
        >>> print(output)
        [False False False  True]
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"For signbit, the input must be a Tensor, but got {type(x)}")
    res = ops.less(x, 0)
    return res


def sgn(x):
    r"""
    Extension of :func:`mindspore.ops.sign` in complex domain.
    For real number input, this function is the same as :func:`mindspore.ops.sign`.
    For complex input, this function is calculated according to the following formula.

    .. math::
        \text{out}_{i} = \begin{cases}
                        0 & |\text{x}_i| == 0 \\
                        \frac{{\text{x}_i}}{|{\text{x}_i}|} & \text{otherwise}
                        \end{cases}

    Args:
        x (Tensor): The input value.

    Returns:
        Tensor, the sgn of x.

    Raises:
        TypeError: If input is not a Tensor.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import mindspore.ops as ops
        >>> x = ms.Tensor([[3 + 4j, 7 - 24j, 0, 6 + 8j, 8], [15 + 20j, 7 - 24j, 0, 3 + 4j, 20]], dtype=ms.complex64)
        >>> output = ops.sgn(x)
        >>> print(output)
        [[0.6 +0.8j  0.28-0.96j 0.  +0.j   0.6 +0.8j  1.  +0.j  ]
         [0.6 +0.8j  0.28-0.96j 0.  +0.j   0.6 +0.8j  1.  +0.j  ]]
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"For sgn, the input must be a Tensor, but got {type(x)}")
    if not ops.is_complex(x):
        return ops.sign(x)
    modulus = ops.ComplexAbs()(x)
    zeros_mask = modulus.equal(0)
    non_zero_modulus = ops.masked_fill(modulus, zeros_mask, 1)
    zeros_modulus = ops.zeros_like(non_zero_modulus)
    complex_modulus = ops.Complex()(non_zero_modulus, zeros_modulus)
    res = x / complex_modulus
    return res


def sin(x):
    r"""
    Computes sine of the input element-wise.

    .. math::

        out_i = sin(x_i)

    Args:
        x (Tensor): The shape of tensor is
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Returns:
        Tensor, has the same shape and dtype as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16, float32 or float64, complex64, complex128.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.62, 0.28, 0.43, 0.62]), mindspore.float32)
        >>> output = ops.sin(x)
        >>> print(output)
        [0.5810352 0.27635565 0.41687083 0.5810352]
    """
    return sin_(x)


def sinc(x):
    r"""
    Computes the normalized sinc of input.

    .. math::

        out_i = \begin{cases} \frac{sin(\pi x_i)}{x_i} & x_i\neq 0\\
        1 & x_i=0 \end{cases}

    Args:
        x (Tensor): The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Returns:
        Tensor, has the same shape as the `x`. The dtype of output is float32 when dtype of `x` is in
        [int, bool]. Otherwise output has the same dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.62, 0.28, 0.43, 0.62]), mindspore.float32)
        >>> output = ops.sinc(x)
        >>> print(output)
        [0.47735003 0.8759357  0.7224278  0.47735003]
    """
    return sinc_(x)


def cos(x):
    r"""
    Computes cosine of input element-wise.

    .. math::
        out_i = cos(x_i)

    .. warning::
        Supported dtypes are float16 and float32, and using float64 may
        cause a problem of missing precision.

    Args:
        x (Tensor): The shape of tensor is
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Returns:
        Tensor, has the same shape and dtype as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16, float32 or float64, complex64, complex128.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.24, 0.83, 0.31, 0.09]), mindspore.float32)
        >>> output = ops.cos(x)
        >>> print(output)
        [0.971338 0.6748758 0.95233357 0.9959527]
    """
    return cos_(x)


def cosine_similarity(x1, x2, dim=1, eps=1e-08):
    r"""
    Calculate cosine similarity between `x1` and `x2` along the axis, `dim`. `x1` and `x2` must be broadcastable.

    .. math::
        \text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}

    Args:
        x1 (Tensor): The first input Tensor.
        x2 (Tensor): The second input Tensor.
        dim (int, optional): Axis for calculating cosine similarity. Default: 1.
        eps (float, optional): Minimal value to avoid division by zero. Default: 1e-8.

    Returns:
        Tensor, cosine similarity between x1 and x2.

    Raises:
        TypeError: If the dtype of x1 or x2 is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import mindspore.ops as ops
        >>> x1 = ms.Tensor([[-0.0256, 0.0127, -0.2475, 0.2316, 0.8037],
        ...                 [0.5809, -1.2712, -0.7038, -0.2558, 0.7494]], dtype=ms.float32)
        >>> x2 = ms.Tensor([[-0.6115, -0.1965, -0.8484, 0.2389, 0.2409],
        ...                 [1.8940, -2.1997, 0.1915, 0.0856, 0.7542]], dtype=ms.float32)
        >>> output = ops.cosine_similarity(x1, x2)
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


def cov(x, *, correction=1, fweights=None, aweights=None):
    r"""
    Given the input `x` and weights, returns the covariance matrix (the square matrix of the covariance of each pair of
    variables) of x, where the input row is the variable and the column is the observation value.

    The diagonal contains each variable and its own covariance. If x is a scalar or 1D vector of a single variable,
    its variance will be returned.

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

    Args:
        x (Tensor): A 2D matrix, or a scalar or 1D vector of a single variable

    Keyword Args:
        correction (int, optional): The difference between sample size and sample degrees of freedom.
            Defaults to Bessel's correction, `correction = 1` which returns the unbiased estimate,
            even if both `fweights` and `aweights` are specified. `correction = 0`
            will return the simple average. Default: 1.
        fweights (Tensor, optional): Scalar or one-dimensional Tensor containing integer frequency weight, indicating
            the number of repetition of each observation vector. Its numel must equal the number of columns of `x`.
            Ignored if `None`. Default: None.
        aweights (Tensor, optional): A scalar or 1D Tensor containing float observation weights represents
            the importance of each observation vector. The higher the importance, the greater the corresponding value.
            Its numel must equal the number of columns of `x`. Must have floating point dtype. Ignored if `None`.
            Default: None.

    Returns:
        Tensor, The covariance matrix Tensor of `x`.

    Raises:
        ValueError: If the dimensions of input is greater than 2.
        ValueError: If the dimensions of fweights is greater than 1.
        ValueError: If the numel of fweights not equal the number of columns of input.
        ValueError: If the numel of aweights not equal the number of columns of input.
        ValueError: If the dimensions of aweights is greater than 1.
        TypeError: If the dtype of input is bool.
        TypeError: If the dtype of fweights is not an integer type.
        TypeError: If the dtype of aweights is not a floating point type.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import mindspore.ops as ops
        >>> x = ms.Tensor([[0, 3], [5, 5], [7, 0]]).T
        >>> print(x)
        [[0 5 7]
         [3 5 0]]
        >>> print(ops.cov(x))
        [[13.        -3.5      ]
         [-3.5        6.3333335]]
        >>> print(ops.cov(x, correction=0))
        [[ 8.666667  -2.3333333]
         [-2.3333333  4.2222223]]
        >>> fw = ms.Tensor([5, 2, 4], dtype=ms.int64)
        >>> aw = ms.Tensor([0.4588, 0.9083, 0.7616], ms.float32)
        >>> print(ops.cov(x, fweights=fw, aweights=aw))
        [[10.146146 -3.47241 ]
         [-3.47241   4.716825]]
    """
    if x.ndim > 2:
        raise ValueError(f"For cov, the input must have two or fewer dimensions, but got {x.ndim} dimensions.")
    if x.dtype == mstype.bool_:
        raise TypeError(f"For cov, the input dtype can not be bool.")

    # View input tensor as 2D
    input_x = x.view((1, -1)) if x.ndim < 2 else x
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
        w_sum = w.sum().astype(mstype.float32)
        avg = (input_x * w).sum(1) / w_sum
    else:
        w_sum = Tensor(num_observations, dtype=mstype.float32)
        avg = input_x.sum(1) / w_sum

    if w is not None and aweights is not None and correction != 0:
        norm_factor = w_sum - correction * (w * aweights).sum() / w_sum
    else:
        norm_factor = w_sum - correction

    if norm_factor <= 0:
        norm_factor = ops.zeros_like(norm_factor)

    input_x = input_x - avg.unsqueeze(1)
    c = ops.mm(input_x, (input_x * w if w is not None else input_x).T)
    norm_factor = norm_factor.astype(mstype.float32)
    return ops.true_divide(c, norm_factor).squeeze()


def t(x):
    r"""
    Transposes a 2-D Tensor. 1-D Tensor are returned as it is.

    Args:
        x (Tensor): The input Tensor.

    Returns:
        Tensor, the transpose of `x` .

    Raises:
        ValueError: If the dimension of `x` is larger than 2.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor([[1, 2, 3], [2, 3, 4]], mstype.float32)
        >>> output = ops.t(x)
        >>> print(output)
        [[1. 2.]
         [2. 3.]
         [3. 4.]]
    """
    if x.ndim > 2:
        raise ValueError(f"For t(), the dimension of tensor should be less than 3, but got {x.ndim}.")
    if x.ndim == 2:
        return _get_cache_prim(P.Transpose)()(x, (1, 0))
    return x


def tan(x):
    r"""
    Computes tangent of `x` element-wise.

    .. math::

        out_i = tan(x_i)

    Args:
        x (Tensor): The input Tensor, valid for any dimensions.

    Returns:
        Tensor, has the same shape as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([-1.0, 0.0, 1.0]), mindspore.float32)
        >>> output = ops.tan(x)
        >>> print(output)
        [-1.5574081 0. 1.5574081]
    """
    return tan_(x)


def xlogy(x, y):
    r"""
    Computes the first input tensor multiplied by the logarithm of second input tensor element-wise.
    Returns zero when `x` is zero.

    .. math::

        out_i = x_{i}\ln{y_{i}}

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors, the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    .. warning::
        - On Ascend, the data type of `x` and `y` must be float16 or float32.

    Args:
        x (Union[Tensor, number.Number, bool]): The first input is a number.Number or
            a bool or a tensor whose data type is
            `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_ or
            `bool_ <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_.
        y (Union[Tensor, number.Number, bool]): The second input is a number.Number or
            a bool when the first input is a tensor or a tensor whose data type is number or bool\_.
            When the first input is Scalar, the second input must be a Tensor whose data type is number or bool\_.

    Returns:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `x` and `y` is not a number.Number or a bool or a Tensor.
        TypeError: If dtype of `x` and 'y' is not in [float16, float32, float64, complex64, complex128].
        ValueError: If `x` could not be broadcast to a tensor with shape of `y`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([-5, 0, 4]), mindspore.float32)
        >>> y = Tensor(np.array([2, 2, 2]), mindspore.float32)
        >>> output = ops.xlogy(x, y)
        >>> print(output)
        [-3.465736   0.        2.7725887]
    """
    if isinstance(x, Tensor) and isinstance(y, Tensor) and x.dtype == mstype.bool_ and y.dtype == mstype.bool_:
        x = x.astype(mstype.float32)
        y = y.astype(mstype.float32)
    return xlogy_(x, y)


def arccosh(x):
    r"""
    For details, please refer to :func:`mindspore.ops.acosh`.
    """
    return acosh_(x)


def arcsin(x):
    r"""
    For details, please refer to :func:`mindspore.ops.asin`.
    """
    return asin_(x)


def arctan(x):
    r"""
    For details, please refer to :func:`mindspore.ops.atan`.
    """
    return atan_(x)


def arctan2(input, other):
    r"""
    For details, please refer to :func:`mindspore.ops.atan2`.
    """
    _atan2 = _get_cache_prim(P.Atan2)()
    return _atan2(input, other)


def polar(abs, angle):  # pylint: disable=redefined-outer-name
    r"""
    Returns the complex tensor at polar coordinates.

    .. math::

        y_{i} =  abs_{i} * cos(angle_{i}) + abs_{i} * sin(angle_{i}) * j

    Args:
        abs (Tensor): The shape of tensor is
            :math:`(N,*)`, where :math:`*` means additional dimensions of size less than 8.
            Must be one of the following types: float32, float64.

        angle (Tensor): The shape of tensor is
            :math:`(N,*)`, where :math:`*` means additional dimensions of size less than 8.
            Must be one of the following types: float32, float64.

    Returns:
        Tensor, has the same shape and data type as `abs`.

    Raises:
        TypeError: If neither `abs` nor `angle` is a Tensor.
        TypeError: If the dtype of input is not one of: float32, float64.
        TypeError: If the dtypes of two args are not the same.
        ValueError: If the shape of `abs` is not the same as that of `angle`.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> abs = Tensor(np.array([1, 2]), mindspore.float64)
        >>> angle = Tensor(np.array([np.pi / 2, 5 * np.pi / 4]), mindspore.float64)
        >>> output = ops.polar(abs, angle)
        >>> print(output)
        [ 6.12323400e-17+1.j         -1.41421356e+00-1.41421356j]
    """
    return polar_(abs, angle)


def asin(x):
    r"""
    Computes arcsine of input tensors element-wise.

    .. math::

        out_i = sin^{-1}(x_i)

    Args:
        x (Tensor): The shape of tensor is
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Returns:
        Tensor, has the same shape and dtype as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16, float32, float64, complex64, complex128.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.74, 0.04, 0.30, 0.56]), mindspore.float32)
        >>> output = ops.asin(x)
        >>> print(output)
        [0.8330704  0.04001067 0.30469266 0.5943858 ]
    """
    return asin_(x)


def acos(x):
    r"""
    Computes arccosine of input tensors element-wise.

    .. math::

        out_i = cos^{-1}(x_i)

    Args:
        x (Tensor): The shape of tensor is
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Returns:
        Tensor, has the same shape and dtype as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16, float32 or float64, complex64, complex128.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.74, 0.04, 0.30, 0.56]), mindspore.float32)
        >>> output = ops.acos(x)
        >>> print(output)
        [0.737726  1.5307857 1.2661036 0.9764105]
    """
    return acos_(x)


def arccos(x):
    """
    Alias for :func:`mindspore.ops.acos` .

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    return acos(x)


def atan(x):
    r"""
    Computes the trigonometric inverse tangent of the input element-wise.

    .. math::

        out_i = tan^{-1}(x_i)

    Args:
        x (Tensor): The shape of tensor is
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
            The data type should be one of the following types: float16, float32.

    Returns:
        A Tensor, has the same type as the input.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16 or float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1.0, 0.0]), mindspore.float32)
        >>> output = ops.atan(x)
        >>> print(output)
        [0.7853982 0.       ]
    """
    return atan_(x)


def sinh(x):
    r"""
    Computes hyperbolic sine of the input element-wise.

    .. math::

        out_i = \sinh(x_i)

    Args:
        x (Tensor): The input tensor of hyperbolic sine function, its rank must be in [0, 7] inclusive.

    Returns:
        Tensor, has the same shape as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.62, 0.28, 0.43, 0.62]), mindspore.float32)
        >>> output = ops.sinh(x)
        >>> print(output)
        [0.6604918  0.28367308 0.44337422 0.6604918 ]
    """
    return sinh_(x)


def cosh(x):
    r"""
    Computes hyperbolic cosine of input element-wise.

    .. math::

        out_i = \cosh(x_i)

    Args:
        x (Tensor): The input tensor of hyperbolic cosine function, its rank must be in [0, 7] inclusive
            and data type must be float16, float32, float64, complex64 or complex128.

    Returns:
        Tensor, has the same shape as `x`.

    Raises:
        TypeError: If the dtype of `x` is not one of the following types:
                   float16, float32, float64, complex64, complex128.
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.24, 0.83, 0.31, 0.09]), mindspore.float32)
        >>> output = ops.cosh(x)
        >>> print(output)
        [1.0289385 1.364684 1.048436 1.0040528]
    """
    return cosh_(x)


def tanh(input_x):
    r"""
    Computes hyperbolic tangent of input element-wise. The Tanh function is defined as:

    .. math::

        tanh(x_i) = \frac{\exp(x_i) - \exp(-x_i)}{\exp(x_i) + \exp(-x_i)} = \frac{\exp(2x_i) - 1}{\exp(2x_i) + 1},

    where :math:`x_i` is an element of the input Tensor.

    Args:
        input_x (Tensor): Tensor of shape :math:`(N, *)`, where :math:`*` means, any number of
            additional dimensions, with float16 or float32 data type.

    Returns:
        Tensor, with the same type and shape as the `input_x`.

    Raises:
        TypeError: If dtype of `input_x` is neither float16 nor float32.
        TypeError: If `input_x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU``  ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([1, 2, 3, 4, 5]), mindspore.float32)
        >>> output = ops.tanh(input_x)
        >>> print(output)
        [0.7615941 0.9640276 0.9950547 0.9993293 0.9999092]
    """
    return tanh_(input_x)


def asinh(x):
    r"""
    Computes inverse hyperbolic sine of the input element-wise.

    .. math::

        out_i = \sinh^{-1}(input_i)

    Args:
        x (Tensor): The input tensor of inverse hyperbolic sine function, its rank must be in [0, 7] inclusive.

    Returns:
        Tensor, has the same shape and type as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([-5.0, 1.5, 3.0, 100.0]), mindspore.float32)
        >>> output = ops.asinh(x)
        >>> print(output)
        [-2.3124382  1.1947632  1.8184465  5.298342 ]
    """
    return asinh_(x)


def arcsinh(x):
    r"""
    Alias for :func:`mindspore.ops.asinh`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    return asinh(x)


def arctanh(x):
    r"""
    Alias for :func:`mindspore.ops.atanh`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    return atanh(x)


def acosh(x):
    r"""
    Computes inverse hyperbolic cosine of the inputs element-wise.

    .. math::

        out_i = \cosh^{-1}(input_i)

    .. warning::
        Given an input tensor x, the function computes inverse hyperbolic cosine of every element.
        Input range is [1, inf].

    Args:
        x (Tensor): The input tensor of inverse hyperbolic cosine function, its rank must be in [0, 7] inclusive.

    Returns:
        Tensor, has the same shape and type as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1.0, 1.5, 3.0, 100.0]), mindspore.float32)
        >>> output = ops.acosh(x)
        >>> print(output)
        [0.        0.9624237 1.7627472 5.298292 ]
    """
    return acosh_(x)


def atanh(x):
    r"""
    Computes inverse hyperbolic tangent of the input element-wise.

    .. math::

        out_i = \tanh^{-1}(x_{i})

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        x (Tensor): The shape of tensor is
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
            The data type should be one of the following types: float16, float32.

    Returns:
        A Tensor, has the same type as the input.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16 or float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([0, -0.5]), mindspore.float32)
        >>> output = ops.atanh(x)
        >>> print(output)
        [ 0.         -0.54930615]
    """
    return atanh_(x)


def atan2(input, other):
    r"""
    Returns arctangent of input/other element-wise.

    It returns :math:`\theta\ \in\ [-\pi, \pi]`
    such that :math:`input = r*\sin(\theta), other = r*\cos(\theta)`, where :math:`r = \sqrt{input^2 + other^2}`.

    Args of `input` and `other` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, the lower precision data type will be converted to relatively the
    highest precision data type.

    Args:
        input (Tensor, Number.number): The input tensor or scalar.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
            The data type should be one of the following types: float16, float32, float64
        other (Tensor, Number.number): The input tensor or scalar. It has the same shape with `input`.

    Returns:
        Tensor or scalar, the shape is the same as the one after broadcasting,and the data type is same as `input`.

    Raises:
        TypeError: If `input` or `other` is not a Tensor or scalar.
        RuntimeError: If the data type of `input` and `other` conversion of Parameter is required
                      when data type conversion of Parameter is not supported.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input = Tensor(np.array([0, 1]), mindspore.float32)
        >>> other = Tensor(np.array([1, 1]), mindspore.float32)
        >>> output = ops.atan2(input, other)
        >>> print(output)
        [0.        0.7853982]
    """
    _atan2 = _get_cache_prim(P.Atan2)()
    return _atan2(input, other)


def bitwise_and(input, other):
    r"""
    Returns bitwise `and` of two tensors element-wise.

    .. math::

        out_i = input_{i} \wedge other_{i}

    Args of `input` and `other` comply with the implicit type conversion rules to
    make the data types consistent.
    If they have different data types, the lower priority data type will be converted to
    the relatively highest priority data type.

    Args:
        input (Tensor): The first input tensor with shape :math:`(N,*)` where :math:`*` means
            any number of additional dimensions.
        other (Tensor): The second input tensor with the same dtype as `input`.

    Returns:
        Tensor, has the same type as the `input`.

    Raises:
        TypeError: If `input` or `other` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input = Tensor(np.array([0, 0, 1, -1, 1, 1, 1]), mindspore.int16)
        >>> other = Tensor(np.array([0, 1, 1, -1, -1, 2, 3]), mindspore.int16)
        >>> output = ops.bitwise_and(input, other)
        >>> print(output)
        [ 0  0  1 -1  1  0  1]
    """
    return bitwise_and_(input, other)


def bitwise_or(input, other):
    r"""
    Returns bitwise `or` of two tensors element-wise.

    .. math::

        out_i = input_{i} \mid other_{i}

    Args of `input` and `other` comply with the implicit type conversion rules to
    make the data types consistent.
    If they have different data types, the lower priority data type will be converted to
    the relatively highest priority data type.

    Args:
        input (Tensor): The first input tensor with shape :math:`(N,*)` where :math:`*` means
            any number of additional dimensions.
        other (Tensor): The second input tensor with the same dtype as `input`.

    Returns:
        Tensor, has the same type as the `input`.

    Raises:
        TypeError: If `input` or `other` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input = Tensor(np.array([0, 0, 1, -1, 1, 1, 1]), mindspore.int16)
        >>> other = Tensor(np.array([0, 1, 1, -1, -1, 2, 3]), mindspore.int16)
        >>> output = ops.bitwise_or(input, other)
        >>> print(output)
        [ 0  1  1 -1 -1  3  3]
    """
    return bitwise_or_(input, other)


def bitwise_xor(input, other):
    r"""
    Returns bitwise `xor` of two tensors element-wise.

    .. math::

        out_i = input_{i} \oplus other_{i}

    Args of `input` and `other` comply with the implicit type conversion rules to
    make the data types consistent.
    If they have different data types, the lower priority data type will be converted to
    the relatively highest priority data type.

    Args:
        input (Tensor): The first input tensor with shape :math:`(N,*)` where :math:`*` means
            any number of additional dimensions.
        other (Tensor): The second input tensor with the same dtype as `input`.

    Returns:
        Tensor, has the same type as the `input`.

    Raises:
        TypeError: If `input` or `other` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input = Tensor(np.array([0, 0, 1, -1, 1, 1, 1]), mindspore.int16)
        >>> other = Tensor(np.array([0, 1, 1, -1, -1, 2, 3]), mindspore.int16)
        >>> output = ops.bitwise_xor(input, other)
        >>> print(output)
        [ 0  1  0  0 -2  3  2]
    """
    return bitwise_xor_(input, other)


def bitwise_left_shift(input, other):
    r"""
    Calculates the left arithmetic shift of `input` by `other` bits.

    .. math::

        \begin{aligned}
        &out_{i} =input_{i} << other_{i}
        \end{aligned}

    Args:
        input (Union[Tensor, Scalar]): The input to be left shifted.
        other (Union[Tensor, Scalar]): The number of bit to be applied on left arithmetic shift.

    Returns:
        Tensor, the result after bitwise left shift.

    Raises:
        TypeError: If neither `input` nor `other` is a tensor.
        TypeError: If either `input` or `other` is not an int or a tensor of dtype: int or uint.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> input = Tensor(np.array([1024, 2]), mindspore.int16)
        >>> other = Tensor(np.array([2]), mindspore.int16)
        >>> output = ops.bitwise_left_shift(input, other)
        >>> print(output)
        [4096    8]
    """
    if isinstance(input, numbers.Number) and isinstance(other, numbers.Number):
        raise TypeError(f"For 'bitwise_left_shift', at least one of the inputs should be a Tensor.")

    cast = ops.Cast()
    white_list = [mstype.int8, mstype.int16, mstype.int32, mstype.int64,
                  mstype.uint8, mstype.uint16, mstype.uint32, mstype.uint64]
    if isinstance(input, numbers.Number):
        _dtype = other.dtype
        if not isinstance(input, int):
            raise TypeError(f"For 'bitwise_left_shift', 'input' must be an integer, but got input:{type(input)}.")
        if _dtype not in white_list:
            raise TypeError(f"For 'bitwise_left_shift', 'other' must be a Tensor of int or uint, but got {_dtype}.")
        input = cast(input, other.dtype)
    elif isinstance(other, numbers.Number):
        _dtype = input.dtype
        if not isinstance(other, int):
            raise TypeError(f"For 'bitwise_left_shift', 'other' must be an integer, but got other:{type(other)}.")
        if _dtype not in white_list:
            raise TypeError(f"For 'bitwise_left_shift', 'input' must be a Tensor of int or uint, but got {_dtype}.")
    other = cast(other, input.dtype)
    ls = ops.LeftShift()
    return ls(input, other)


def bitwise_right_shift(input, other):
    r"""
    Calculates the right arithmetic shift of `input` by `other` bits.

    .. math::

        \begin{aligned}
        &out_{i} =input_{i} >> other_{i}
        \end{aligned}

    Args:
        input (Union[Tensor, Scalar]): The input to be right shifted.
        other (Union[Tensor, Scalar]): The number of bit to be applied on right arithmetic shift.

    Returns:
        Tensor, the result after bitwise right shift.

    Raises:
        TypeError: If neither `input` nor `other` is a tensor.
        TypeError: If either `input` or `other` is not an int or a tensor of dtype: int or uint.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> input = Tensor(np.array([1024, 2]), mindspore.int16)
        >>> other = Tensor(np.array([2]), mindspore.int16)
        >>> output = ops.bitwise_right_shift(input, other)
        >>> print(output)
        [256   0]
    """
    if isinstance(input, numbers.Number) and isinstance(other, numbers.Number):
        raise TypeError(f"For 'bitwise_left_shift', at least one of the inputs should be a Tensor.")
    cast = ops.Cast()
    white_list = [mstype.int8, mstype.int16, mstype.int32, mstype.int64,
                  mstype.uint8, mstype.uint16, mstype.uint32, mstype.uint64]
    if isinstance(input, numbers.Number):
        _dtype = other.dtype
        if not isinstance(input, int):
            raise TypeError(f"For 'bitwise_left_shift', 'input' must be an integer, but got input:{type(input)}.")
        if _dtype not in white_list:
            raise TypeError(f"For 'bitwise_left_shift', 'other' must be a Tensor of int or uint, but got {_dtype}.")
        input = cast(input, other.dtype)
    elif isinstance(other, numbers.Number):
        _dtype = input.dtype
        if not isinstance(other, int):
            raise TypeError(f"For 'bitwise_left_shift', 'other' must be an integer, but got other:{type(other)}.")
        if _dtype not in white_list:
            raise TypeError(f"For 'bitwise_left_shift', 'input' must be a Tensor of int or uint, but got {_dtype}.")
    other = cast(other, input.dtype)
    rs = ops.RightShift()
    return rs(input, other)


def inv(x):
    r"""
    Computes Reciprocal of input tensor element-wise.

    .. math::
        out_i = \frac{1}{x_{i} }

    Args:
        x (Tensor): Tensor of any dimension. Must be one of the following types: float16, float32 or int32.

    Returns:
        Tensor, has the same type and shape as input shape value.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not one of float16, float32, int32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.25, 0.4, 0.31, 0.52]), mindspore.float32)
        >>> output = ops.inv(x)
        >>> print(output)
        [4.        2.5       3.2258065 1.923077 ]
    """
    return inv_(x)


def inverse(x):
    """
    Compute the inverse of the input matrix.

    Args:
        x (Tensor): A matrix to be calculated. Input `x` must be at least two dimensions, and the size of
            the last two dimensions must be the same size.

    Returns:
        Tensor, has the same type and shape as input `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        ValueError: If the size of the last two dimensions of `x` is not the same.
        ValueError: If the dimension of `x` is less than 2.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor([[1., 2.], [3., 4.]], mstype.float32)
        >>> print(ops.inverse(x))
        [[-2.   1. ]
         [ 1.5 -0.5]]
    """
    if x.dtype in mstype.int_type:
        _get_cache_prim(P.Cast)()(x, mstype.float64)
    return _get_cache_prim(P.MatrixInverse)()(x)


def invert(x):
    r"""
    Flips all bits of input tensor element-wise.

    .. math::
        out_i = \sim x_{i}

    Args:
        x (Tensor): The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
            The data type should be one of the following types: int16, uint16.

    Returns:
        Tensor, has the same shape as `x`.

    Raises:
        TypeError: If dtype of `x` is neither int16 nor uint16.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([25, 4, 13, 9]), mindspore.int16)
        >>> output = ops.invert(x)
        >>> print(output)
        [-26 -5 -14 -10]
    """
    return invert_(x)


def erf(x):
    r"""
    Computes the Gauss error function of `x` element-wise.

    .. math::

        erf(x)=\frac{2} {\sqrt{\pi}} \int\limits_0^{x} e^{-t^{2}} dt

    Args:
        x (Tensor): The input tensor of Gaussian error function. Its rank must be in [0, 7] inclusive
            and data type must be float16 float32 or float64.

    Returns:
        Tensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is neither float16 float32 or float64.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([-1, 0, 1, 2, 3]), mindspore.float32)
        >>> output = ops.erf(x)
        >>> print(output)
        [-0.8427168   0.          0.8427168   0.99530876  0.99997765]
    """
    return erf_(x)


def erfc(x):
    r"""
    Computes the complementary error function of `x` element-wise.

    .. math::

        erfc(x) = 1 - \frac{2} {\sqrt{\pi}} \int\limits_0^{x} e^{-t^{2}} dt

    Args:
        x (Tensor): The input tensor with a dtype of float16, float32 or float64,
            its rank should be in [0, 7] inclusive.

    Returns:
        Tensor, has the same shape and dtype as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16, float32 or float64.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([-1, 0, 1, 2, 3]), mindspore.float32)
        >>> output = ops.erfc(x)
        >>> print(output)
        [1.8427168e+00 1.0000000e+00 1.5728319e-01 4.6912432e-03 2.2351742e-05]
    """
    return erfc_(x)


def bessel_j0(x):
    r"""
    Computes the Bessel j0 function of x element-wise.

    Args:
        x (Tensor): The input tensor. The data type must be float16, float32 or float64.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Returns:
        Tensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16, float32 or float64.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.5, 1., 2., 4.]), mindspore.float32)
        >>> output = ops.bessel_j0(x)
        >>> print(output)
        [0.93846981  0.76519769  0.22389078  -0.39714981]
    """
    return bessel_j0_(x)


def bessel_j1(x):
    r"""
    Computes the Bessel j1 function of x element-wise.

    Args:
        x (Tensor): The input tensor. The data type must be float16, float32 or float64.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Returns:
        Tensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16, float32 or float64.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.5, 1., 2., 4.]), mindspore.float32)
        >>> output = ops.bessel_j1(x)
        >>> print(output)
        [0.24226846  0.44005059  0.57672481 -0.06604333]
    """
    return bessel_j1_(x)


def bessel_i0(x):
    r"""
    Computes the Bessel i0 function of x element-wise.

    Args:
        x (Tensor): The input tensor. The data type must be float16, float32 or float64.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Returns:
        Tensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16, float32 or float64.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([-1, -0.5, 0.5, 1]), mindspore.float32)
        >>> output = ops.bessel_i0(x)
        >>> print(output)
        [1.266066  1.0634835 1.0634835 1.266066]
    """
    return bessel_i0_(x)


def bessel_i0e(x):
    r"""
    Computes the Bessel i0e function of x element-wise.

    Args:
        x (Tensor): The input tensor. The data type must be float16, float32 or float64.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Returns:
        Tensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16, float32 or float64.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([-1, -0.5, 0.5, 1]), mindspore.float32)
        >>> output = ops.bessel_i0e(x)
        >>> print(output)
        [0.46575961  0.64503527  0.64503527  0.46575961]
    """
    return bessel_i0e_(x)


def bessel_k0(x):
    r"""
    Computes the Bessel k0 function of x element-wise.

    Args:
        x (Tensor): The input tensor. The data type must be float16, float32 or float64.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Returns:
        Tensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16, float32 or float64.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.5, 1., 2., 4.]), mindspore.float32)
        >>> output = ops.bessel_k0(x)
        >>> print(output)
        [0.92441907  0.42102444  0.11389387  0.01115968]
    """
    return bessel_k0_(x)


def bessel_k0e(x):
    r"""
    Computes the Bessel k0e function of x element-wise.

    Args:
        x (Tensor): The input tensor. The data type must be float16, float32 or float64.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Returns:
        Tensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16, float32 or float64.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.5, 1., 2., 4.]), mindspore.float32)
        >>> output = ops.bessel_k0e(x)
        >>> print(output)
        [1.52410939  1.14446308  0.84156822  0.60929767]
    """
    return bessel_k0e_(x)


def bessel_y0(x):
    r"""
    Computes the Bessel y0 function of x element-wise.

    Args:
        x (Tensor): The input tensor. The data type must be float16, float32 or float64.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Returns:
        Tensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16, float32 or float64.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.5, 1., 2., 4.]), mindspore.float32)
        >>> output = ops.bessel_y0(x)
        >>> print(output)
        [-0.44451874  0.08825696  0.51037567  -0.01694074]
    """
    return bessel_y0_(x)


def bessel_y1(x):
    r"""
    Computes the Bessel y1 function of x element-wise.

    Args:
        x (Tensor): The input tensor. The data type must be float16, float32 or float64.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Returns:
        Tensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16, float32 or float64.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.5, 1., 2., 4.]), mindspore.float32)
        >>> output = ops.bessel_y1(x)
        >>> print(output)
        [-1.47147239  -0.78121282  -0.10703243  0.39792571]
    """
    return bessel_y1_(x)


def linspace(start, stop, num):
    r"""
    Returns a Tensor whose value is `num` evenly spaced in the interval `start` and `stop` (including `start` and
    `stop`), and the length of the output Tensor is `num`.

    .. math::
        \begin{aligned}
        &step = (stop - start)/(num - 1)\\
        &output = [start, start+step, start+2*step, ... , stop]
        \end{aligned}

    Args:
        start (Tensor): Start value of interval. The tensor data type must be float32 or float64 and with shape of 0-D.
        stop (Tensor): Last value of interval. The tensor data type must be float32 or float64 and with shape of 0-D.
        num (Union[Tensor, int]): Number of ticks in the interval, inclusive of start and stop.
            Must be positive int number or 0D int32/int64 Tensor.

    Returns:
        Tensor, has the same dtype as `start`, and the shape of :math:`(num)`.

    Raises:
        TypeError: If `start` or `stop` is not a Tensor.
        TypeError: If dtype of `start` or dtype of `stop` is not float32 or float64.
        ValueError: If shape of `start` or shape of `stop` is not 0-D.
        TypeError: If `num` is not int or 0D int32/int64 Tensor.
        ValueError: If `num` is not positive int number.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> start = Tensor(1, mindspore.float32)
        >>> stop = Tensor(10, mindspore.float32)
        >>> num = 5
        >>> output = ops.linspace(start, stop, num)
        >>> print(output)
        [ 1.    3.25  5.5   7.75 10.  ]
    """
    return linspace_(start, stop, num)


def matrix_determinant(x):
    r"""
    Computes the determinant of one or more square matrices.

    Args:
        x (Tensor): A matrix to be calculated, its shape should be :math:`[..., M, M]` who must
          have at least two dimensions, and the last two
          dimensions must be the same size. Data type must be float32, float64, complex64 or complex128.

    Returns:
        Tensor. The shape is :math:`x.shape[:-2]`, and the dtype is same as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` not float32, float64, complex64 or complex128.
        ValueError: If the last two dimensions of `x` is not same size.
        ValueError: If the dimension of `x` is less than 2.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[[-4.5, -1.5], [7.0, 6.0]], [[2.5, 0.5], [3.0, 9.0]]]), mindspore.float32)
        >>> output = ops.matrix_determinant(input_x)
        >>> print(output)
        [-16.5 21. ]
    """
    return _get_cache_prim(P.MatrixDeterminant)()(x)


def det(x):
    """
    Alias for :func:`mindspore.ops.matrix_determinant` .

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    return matrix_determinant(x)


def matrix_exp(x):
    r"""
    Computes the matrix exponential of a square matrix. Supports batched inputs.

    .. math::

        matrix\_exp(x) = \sum_{k=0}^{\infty} \frac{1}{k !} x^{k} \in \mathbb{K}^{n \times n}

    Args:
        x (Tensor): The shape of tensor is :math:`(*, n, n)` where * is zero or more batch dimensions.
            Must be one of the following types: float16, float32, float64, complex64, complex128.

    Returns:
        Tensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If the dtype of `x` is not one of the following dtype:
                   float16, float32, float64, complex64, complex128.
        ValueError: If the rank of `x` is less than 2.
        ValueError: If the last two dimensions of `x` are not equal.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[1, 2], [0, 1]]), mindspore.float32)
        >>> output = ops.matrix_exp(x)
        >>> print(output)
        [[2.7182817 5.436563 ]
        [0.        2.7182817]]
    """
    return matrix_exp_(x)


def log_matrix_determinant(x):
    r"""
    Computes the sign and the log of the absolute value of the determinant of one or more square matrices.

    Args:
        x (Tensor): A matrix to be calculated, its shape is :math:`[..., M, M]`.
          The matrix must be at least two dimensions, and the last two
          dimensions must be the same size. Data type must be float32, float64, complex64 or complex128.

    Returns:
        - Tensor. The signs of the log determinants. The shape is :math:`x.shape[:-2]`,
          and the dtype is same as `x`.
        - Tensor. The absolute values of the log determinants. The shape is :math:`x.shape[:-2]`, and
          the dtype is same as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` not float32, float64, complex64 or complex128.
        ValueError: If the last two dimensions of `x` is not same size.
        ValueError: If the dimension of `x` is less than 2.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[[-4.5, -1.5], [7.0, 6.0]], [[2.5, 0.5], [3.0, 9.0]]]), mindspore.float32)
        >>> sign, output = ops.log_matrix_determinant(input_x)
        >>> print(sign)
        [-1.   1.]
        >>> print(output)
        [2.80336046e+00    3.04452229e+00]
    """
    return log_matrix_determinant_(x)


def matrix_solve(matrix, rhs, adjoint=False):  # pylint: disable=redefined-outer-name
    r"""
    Solves systems of linear equations.

    .. math::
        \begin{aligned}
        &matrix[..., M, M] * x[..., M, K] = rhs[..., M, K] \\
        &adjoint(matrix[..., M, M]) * x[..., M, K] = rhs[..., M, K]
        \end{aligned}

    .. warning::
        On GPU, if the matrix is irreversible, an error may be reported or an unknown result may be returned.

    Args:
        matrix (Tensor): The shape of tensor is :math:`[..., M, M]`.
        rhs (Tensor): The shape of tensor is :math:`[..., M, K]`. `rhs` must have the same dtype as `matrix`.
        adjoint(bool): Indicating whether to solve with matrix or its (block-wise) adjoint. Default: False.

    Returns:
        x (Tensor), The dtype and shape is the same as 'rhs'.

    Raises:
        TypeError: If adjoint is not the type of bool.
        TypeError: If the type of matrix is not one of the following dtype:
                   mstype.float16, mstype.float32, mstype.float64, mstype.complex64, mstype.complex128.
        TypeError: If the type of `matrix` is not the same as that of `rhs`.
        ValueError: If the rank of `matrix` less than 2.
        ValueError: If the dimension of `matrix` is not the same as `rhs`.
        ValueError: If the inner-most 2 dimension of `matrix` is not the same.
        ValueError: If the inner-most 2 dimension of `rhs` does not match `matrix`.
        ValueError: If the `matrix` is irreversible.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> matrix = Tensor([[5, 4], [3, 1]], mindspore.float32)
        >>> rhs = Tensor([[7], [2]], mindspore.float32)
        >>> result = ops.matrix_solve(matrix, rhs)
        >>> print(result)
        [[0.14285707]
         [1.5714287 ]]
    """
    matrix_solve_ = _get_cache_prim(MatrixSolve)(adjoint=adjoint)
    return matrix_solve_(matrix, rhs)


def slogdet(x):
    r"""
    Computes the sign and the log of the absolute value of the determinant of one or more square matrices.

    Args:
        x (Tensor): A matrix to be calculated, its shape is :math:`[..., M, M]`.
          The matrix must be at least two dimensions, and the last two
          dimensions must be the same size. Data type must be float32, float64, complex64 or complex128.

    Returns:
        Tensor. The signs of the log determinants. The shape is :math:`x.shape[:-2]`,
        and the dtype is same as `x`.

        Tensor. The absolute values of the log determinants. The shape is :math:`x.shape[:-2]`, and
        the dtype is same as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` not float32, float64, complex64 or complex128.
        ValueError: If the last two dimensions of `x` is not same size.
        ValueError: If the dimension of `x` is less than 2.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[[-4.5, -1.5], [7.0, 6.0]], [[2.5, 0.5], [3.0, 9.0]]]), mindspore.float32)
        >>> sign, output = ops.slogdet(input_x)
        >>> print(sign)
        [-1.   1.]
        >>> print(output)
        [2.80336046e+00    3.04452229e+00]
    """
    return log_matrix_determinant_(x)


def truncate_div(x, y):
    """
    Divides the first input tensor by the second input tensor element-wise and rounds the results
    of division towards zero. Equivalent to C-style integer division.

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be bool at the same time, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    Note:
        Broadcasting is supported.

    Args:
        x(Union[Tensor, Number, bool]): The first input is a number, or a bool,
            or a tensor whose data type is number or bool.
        y(Union[Tensor, Number, bool]): The second input is a number, or a bool when the first input
            is a tensor, or a tensor whose data type is number or bool.

    Returns:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `x` and `y` is not one of the following: Tensor, Number, bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([2, 4, -1]), mindspore.int32)
        >>> y = Tensor(np.array([3, 3, 3]), mindspore.int32)
        >>> output = ops.truncate_div(x, y)
        >>> print(output)
        [0 1 0]
    """
    return truncate_div_(x, y)


def truncate_mod(x, y):
    r"""
    Returns the remainder of division element-wise.

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be bool at the same time, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    .. warning::
        - The input data does not support 0.
        - When the elements of input exceed 2048 , the accuracy of operator cannot guarantee the requirement of
          double thousandths in the mini form.
        - Due to different architectures, the calculation results of this operator on NPU and CPU may be inconsistent.
        - If shape is expressed as (D1,D2... ,Dn), then D1\*D2... \*DN<=1000000,n<=8.

    Args:
        x (Union[Tensor, numbers.Number, bool]): The first input is a number, or a bool,
            or a tensor whose data type is number or bool.
        y (Union[Tensor, numbers.Number, bool]): The second input is a number, or a bool when the first input
            is a tensor, or a tensor whose data type is number or bool.

    Returns:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision among the two inputs.

    Raises:
        TypeError: If neither `x` nor `y` is one of the following: Tensor, number, bool.
        TypeError: If neither `x` nor `y` is a Tensor.
        ValueError: If the shape `x` and `y` cannot be broadcasted to each other.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([2, 4, -1]), mindspore.int32)
        >>> y = Tensor(np.array([3, 3, 3]), mindspore.int32)
        >>> output = ops.truncate_mod(x, y)
        >>> print(output)
        [ 2  1 -1]
    """
    return truncate_mod_(x, y)


def trunc(input):
    r"""
    Returns a new tensor with the truncated integer values of the elements of the input tensor.

    Args:
        input (Tensor): The input tensor.

    Returns:
        Tensor, the same shape and data type as the input.

    Raises:
        TypeError: If `input` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([3.4742, 0.5466, -0.8008, -3.9079]),mindspore.float32)
        >>> output = ops.trunc(x)
        >>> print(output)
        [3. 0. 0. -3.]
    """
    return _get_cache_prim(P.Trunc)()(input)


def ldexp(x, other):
    """
    Multiplies input by :math:`2^{other}` .

    .. math::

        out_{i} = x_{i} * ( 2_{i} ^{other} )

    Note:
        Typically this function can create floating point numbers
        by multiplying mantissas in input with powers of integer 2
        from the exponents in `other`.

    Args:
        x (Tensor): The input tensor.
        other (Tensor): A tensor of exponents, typically integers.

    Returns:
        Tensor, the output tensor.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If `other` is not a Tensor.
        ValueError: If the size of tensor `x` and `other` are different at non-singleton dimension, and not equal to 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore import ops
        >>> x = Tensor(np.array([1.]), mindspore.float32)
        >>> other = Tensor(np.array([1, 2, 3, 4]), mindspore.int32)
        >>> out = ops.ldexp(x, other)
        >>> print(out)
        [ 2.  4.  8. 16.]
    """

    pow_ops = _get_cache_prim(P.Pow)()
    mul_ops = _get_cache_prim(P.Mul)()

    out = mul_ops(x, pow_ops(2.0, other))
    return out


def logit(x, eps=None):
    r"""
    Calculate the logit of a tensor element-wise. When eps is not None, element in `x` is clamped to [eps, 1-eps].
    When eps is None, input `x` is not clamped.

    .. math::
        \begin{align}
        y_{i} & = \ln(\frac{z_{i}}{1 - z_{i}}) \\
        z_{i} & = \begin{cases}
        x_{i} & \text{if eps is None} \\
        \text{eps} & \text{if } x_{i} \lt \text{eps} \\
        x_{i} & \text{if } \text{eps} \leq x_{i} \leq 1 - \text{eps} \\
        1 - \text{eps} & \text{if } x_{i} \gt 1 - \text{eps}
        \end{cases}
        \end{align}

    Args:
        x (Tensor): The input tensor.
        eps (float, optional): The epsilon. The input clamp bound is defined as [eps, 1-eps]. Default: None.

    Returns:
        Tensor, with the same shape and dtype as the `x`.

    Raises:
        TypeError: If `eps` is not a float.
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16, float32 or float64.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.1, 0.2, 0.3]).astype(np.float32))
        >>> output = ops.logit(x, eps=1e-5)
        >>> print(output)
        [-2.1972246 -1.3862944 -0.8472978]
    """
    if eps is None:
        eps = -1.0
    logit_ = _get_cache_prim(Logit)(eps)
    return logit_(x)


#####################################
# Comparison Operation Functions.
#####################################


def less(x, y):
    r"""
    Computes the boolean value of :math:`x < y` element-wise.

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be bool at the same time, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    .. math::

        out_{i} =\begin{cases}
            & \text{True,    if } x_{i}<y_{i} \\
            & \text{False,   if } x_{i}>=y_{i}
            \end{cases}

    Args:
        x (Union[Tensor, Number, bool]): The first input is a number or
            a bool or a tensor whose data type is number or bool.
        y (Union[Tensor, Number, bool]): The second input is a number or
            a bool when the first input is a tensor, or it can be a tensor whose data type is number or bool.

    Returns:
        Tensor, the shape is the same as the one after broadcasting,and the data type is bool.

    Raises:
        TypeError: If `x` and `y` is not one of the following: Tensor, Number, bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1, 2, 3]), mindspore.int32)
        >>> y = Tensor(np.array([1, 1, 4]), mindspore.int32)
        >>> output = ops.less(x, y)
        >>> print(output)
        [False False True]
    """
    return tensor_lt(x, y)


def lt(x, other):
    """
    Alias for :func:`mindspore.ops.less` .

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    return less(x, other)


def le(x, y):
    r"""
    Computes the boolean value of :math:`x <= y` element-wise.

    .. math::

        out_{i} =\begin{cases}
            & \text{True,    if } x_{i}<=y_{i} \\
            & \text{False,   if } x_{i}>y_{i}
            \end{cases}

    .. note::
        - Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
        - The inputs must be two tensors or one tensor and one scalar.
        - When the inputs are two tensors,
          dtypes of them cannot be both bool , and the shapes of them can be broadcast.
        - When the inputs are one tensor and one scalar, the scalar could only be a constant.

    Args:
        x (Union[Tensor, number.Number, bool]): The first input is a number.Number or
            a bool or a tensor whose data type is
            `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_ or
            `bool_ <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_.
        y (Union[Tensor, number.Number, bool]): The second input, when the first input is a Tensor,
            the second input should be a number.Number or bool value, or a Tensor whose data type is number or bool\_.
            When the first input is Scalar, the second input must be a Tensor whose data type is number or bool\_.

    Returns:
        Tensor, the shape is the same as the one after broadcasting, and the data type is bool.

    Raises:
        TypeError: If neither `x` nor `y` is a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1, 2, 3]), mindspore.int32)
        >>> y = Tensor(np.array([1, 1, 4]), mindspore.int32)
        >>> output = ops.le(x, y)
        >>> print(output)
        [ True False  True]
    """
    return tensor_le(x, y)


def gt(x, y):
    r"""
    Compare the value of the input parameters :math:`x,y` element-wise, and the output result is a bool value.

    .. math::

        out_{i} =\begin{cases}
            & \text{True,    if } x_{i}>y_{i} \\
            & \text{False,   if } x_{i}<=y_{i}
            \end{cases}

    Note:
        - Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
        - The inputs must be two tensors or one tensor and one scalar.
        - When the inputs are two tensors, dtypes of them cannot be bool at the same time,
          and the shapes of them can be broadcast.
        - When the inputs are one tensor and one scalar, the scalar could only be a constant.
        - Broadcasting is supported.
        - If the input Tensor can be broadcast, the low dimension will be extended to the corresponding high dimension
          in another input by copying the value of the dimension.

    Args:
        x (Union[Tensor, number.Number, bool]): The first input is a number.Number or
            a bool or a tensor whose data type is
            `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_ or
            `bool_ <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_ .
        y (Union[Tensor, number.Number, bool]): The second input, when the first input is a Tensor,
            the second input should be a number.Number or bool value, or a Tensor whose data type is number or bool\_.
            When the first input is Scalar, the second input must be a Tensor whose data type is number or bool\_.

    Returns:
        Tensor, the shape is the same as the one after broadcasting, and the data type is bool.

    Raises:
        TypeError: If neither `x` nor `y` is a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1, 2, 3]), mindspore.int32)
        >>> y = Tensor(np.array([1, 1, 4]), mindspore.int32)
        >>> output = ops.gt(x, y)
        >>> print(output)
        [False True False]
    """
    _greater = _get_cache_prim(P.Greater)()
    return _greater(x, y)


def ge(x, y):
    r"""
    Computes the boolean value of :math:`x >= y` element-wise.

    Note:
        - Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
        - The inputs must be two tensors or one tensor and one scalar.
        - When the inputs are two tensors, dtypes of them cannot be bool at the same time,
          and the shapes of them can be broadcast.
        - When the inputs are one tensor and one scalar, the scalar could only be a constant.
        - Broadcasting is supported.
        - If the input Tensor can be broadcast, the low dimension will be extended to the corresponding high dimension
          in another input by copying the value of the dimension.

    .. math::

        out_{i} =\begin{cases}
            & \text{True,    if } x_{i}>=y_{i} \\
            & \text{False,   if } x_{i}<y_{i}
            \end{cases}

    Args:
        x (Union[Tensor, Number, bool]): The first input is a number or
            a bool or a tensor whose data type is number or bool.
        y (Union[Tensor, Number, bool]): The second input is a number or
            a bool when the first input is a tensor or a tensor whose data type is number or bool.

    Returns:
        Tensor, the shape is the same as the one after broadcasting, and the data type is bool.

    Raises:
        TypeError: If neither `x` nor `y` is a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1, 2, 3]), mindspore.int32)
        >>> y = Tensor(np.array([1, 1, 4]), mindspore.int32)
        >>> output = ops.ge(x, y)
        >>> print(output)
        [True True False]
    """
    _greater_equal = _get_cache_prim(P.GreaterEqual)()
    return _greater_equal(x, y)


def equal(x, y):
    r"""
    Computes the equivalence between two tensors element-wise.

    .. math::

        out_{i} =\begin{cases}
            & \text{True,    if } x_{i} = y_{i} \\
            & \text{False,   if } x_{i} \ne y_{i}
            \end{cases}

    Note:
        - Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
        - The inputs must be two tensors or one tensor and one scalar.
        - When the inputs are two tensors, the shapes of them could be broadcast.
        - When the inputs are one tensor and one scalar, the scalar could only be a constant.

    Args:
        x (Union[Tensor, Number]): The first input is a number or
            a tensor whose data type is number.
        y (Union[Tensor, Number]): The second input is a number
            when the first input is a tensor or a tensor whose data type is number.
            The data type is the same as the first input.

    Returns:
        Tensor, the shape is the same as the one after broadcasting,and the data type is bool.

    Raises:
        TypeError: If neither `x` nor `y` is a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> # case 1: The shape of two inputs are different
        >>> x = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> output = ops.equal(x, 2.0)
        >>> print(output)
        [False True False]
        >>> # case 2: The shape of two inputs are the same
        >>> x = Tensor(np.array([1, 2, 3]), mindspore.int32)
        >>> y = Tensor(np.array([1, 2, 4]), mindspore.int32)
        >>> output = ops.equal(x, y)
        >>> print(output)
        [ True  True False]
    """
    return equal_(x, y)


def ne(x, y):
    r"""
    Computes the non-equivalence of two tensors element-wise.

    Note:
        - Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
        - The inputs must be two tensors or one tensor and one scalar.
        - When the inputs are two tensors, the shapes of them could be broadcast.
        - When the inputs are one tensor and one scalar, the scalar could only be a constant.
        - Broadcasting is supported.

    .. math::

        out_{i} =\begin{cases}
            & \text{True,    if } x_{i} \ne y_{i} \\
            & \text{False,   if } x_{i} = y_{i}
            \end{cases}

    Args:
        x (Union[Tensor, Number, bool]): The first input is a number or
            a bool or a tensor whose data type is number or bool.
        y (Union[Tensor, Number, bool]): The second input is a number or
            a bool when the first input is a tensor or a tensor whose data type is number or bool.

    Returns:
        Tensor, the shape is the same as the one after broadcasting,and the data type is bool.

    Raises:
        TypeError: If `x` and `y` is not one of the following: Tensor, Number, bool.
        TypeError: If neither `x` nor `y` is a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> output = ops.ne(x, 2.0)
        >>> print(output)
        [ True False  True]
        >>>
        >>> x = Tensor(np.array([1, 2, 3]), mindspore.int32)
        >>> y = Tensor(np.array([1, 2, 4]), mindspore.int32)
        >>> output = ops.ne(x, y)
        >>> print(output)
        [False False  True]
    """
    return not_equal_(x, y)


def not_equal(x, other):
    r"""
    Alias for ops.ne.
    For details, please refer to :func:`mindspore.ops.ne`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    return ne(x, other)


def approximate_equal(x, y, tolerance=1e-5):
    r"""
    Returns True if abs(x-y) is smaller than tolerance element-wise, otherwise False.

    .. math::

        out_i = \begin{cases}
        & \text{ if } \left | x_{i} - y_{i} \right | < \text{tolerance},\ \ True  \\
        & \text{ if } \left | x_{i} - y_{i} \right | \ge \text{tolerance},\ \  False
        \end{cases}

    where `tolerance` indicates Acceptable maximum tolerance.

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, the lower precision data type will be converted to
    the relatively highest precision data type.

    Args:
        x (Tensor): A tensor. Must be one of the following types: float32, float16.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.
        y (Tensor): A tensor of the same type and shape as `x`.
        tolerance (float): The maximum deviation that two elements can be considered equal. Default: 1e-05.

    Returns:
        Tensor, the shape is the same as the shape of `x`, and the data type is bool.

    Raises:
        TypeError: If `tolerance` is not a float.
        RuntimeError: If the data type of `x`, `y` conversion of Parameter is given
                      but data type conversion of Parameter is not supported.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> tol = 1.5
        >>> x = Tensor(np.array([1, 2, 3]), mstype.float32)
        >>> y = Tensor(np.array([2, 4, 6]), mstype.float32)
        >>> output = ops.approximate_equal(Tensor(x), Tensor(y), tol)
        >>> print(output)
        [ True  False  False]
    """
    return P.ApproximateEqual(tolerance)(x, y)


def isfinite(x):
    r"""
    Determines which elements are finite for each position.

    .. math::

        out_i = \begin{cases}
          & \text{ if } x_{i} = \text{Finite},\ \ True\  \\
          & \text{ if } x_{i} \ne \text{Finite},\ \ False
        \end{cases}

    Args:
        x (Tensor): The input tensor.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Returns:
        Tensor, has the same shape of input, and the dtype is bool.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([np.log(-1), 1, np.log(0)]), mindspore.float32)
        >>> output = ops.isfinite(x)
        >>> print(output)
        [False  True False]
    """
    return isfinite_(x)


def isnan(x):
    r"""
    Determines which elements are NaN for each position.

    .. math::

        out_i = \begin{cases}
          & \ True,\ \text{ if } x_{i} = \text{Nan} \\
          & \ False,\ \text{ if } x_{i} \ne  \text{Nan}
        \end{cases}

    where :math:`Nan` means not a number.

    Args:
        x (Tensor): The input tensor.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Returns:
        Tensor, has the same shape of input, and the dtype is bool.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([np.log(-1), 1, np.log(0)]), mindspore.float32)
        >>> output = ops.isnan(x)
        >>> print(output)
        [ True False False]
    """
    return isnan_(x)


def isclose(x1, x2, rtol=1e-05, atol=1e-08, equal_nan=False):
    """
    Returns a new Tensor with boolean elements representing if each element of `x1`
    is “close” to the corresponding element of `x2`. Closeness is defined as:

    .. math::

            ∣x1−x2∣  ≤  atol + rtol × ∣x2∣

    Args:
        x1 (Tensor): First Tensor to compare, with data type belongs to float32, float16, int32.
        x2 (Tensor): Second Tensor to compare, with data type belongs to float32, float16, int32.
        rtol (float, optional): Relative tolerance. Default: 1e-05.
        atol (float, optional): Absolute tolerance. Default: 1e-08.
        equal_nan (bool, optional): If True, then two NaNs will be considered equal. Default: False.

    Returns:
        A bool Tensor, with the shape as broadcasted result of the input `x1` and `x2`.

    Raises:
        TypeError: If either of `x1` and `x2` is not Tensor.
        TypeError: If either of `x1` and `x2` is not float16, float32 or int32.
        TypeError: If either of `atol` and `rtol` is not float.
        TypeError: If `equal_nan` is not bool.
        TypeError: If the dtype of `x1` is not same as the `x2`.
        ValueError: If `x1` and `x2` can not be broadcast.
        ValueError: If either of `atol` and `rtol` is less than zero.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input = Tensor(np.array([1.3, 2.1, 3.2, 4.1, 5.1]), mindspore.float16)
        >>> other = Tensor(np.array([1.3, 3.3, 2.3, 3.1, 5.1]), mindspore.float16)
        >>> output = ops.isclose(input, other)
        >>> print(output)
        [ True False False False  True]
    """
    is_close = _get_cache_prim(P.IsClose)(rtol=rtol, atol=atol, equal_nan=equal_nan)
    return is_close(x1, x2)


def isreal(x):
    """
    Tests element-wise for real number.
    A complex value is considered real when its imaginary part is 0.

    Args:
        x (Tensor): The input tensor.

    Returns:
       Tensor, true where `x` is real number, false otherwise.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import ops, Tensor
        >>> from mindspore import dtype as mstype
        >>> x = Tensor([1, 1+1j, 2+0j], mstype.complex64)
        >>> output = ops.isreal(x)
        >>> print(output)
        [ True False True]
    """

    if not isinstance(x, (Tensor, Tensor_)):
        raise TypeError("the input x must be Tensor!")

    # Note: Integral and Floating tensor values are always real
    ones_op = _get_cache_prim(P.Ones)()
    real_dtype = mstype.int_type + mstype.uint_type + mstype.float_type + (mstype.bool_,)
    if x.dtype in real_dtype:
        return ones_op(x.shape, mstype.bool_)

    imag_op = _get_cache_prim(P.Imag)()
    return imag_op(x) == 0


def is_complex(x):
    '''
    Return True if the data type of the tensor is complex, otherwise return False.

    Args:
        x (Tensor): The input tensor.

    Returns:
        Bool, return whether the data type of the tensor is complex.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import ops, Tensor
        >>> from mindspore import dtype as mstype
        >>> x = Tensor([1, 1+1j, 2+2j], mstype.complex64)
        >>> output = ops.is_complex(x)
        >>> print(output)
        True
    '''
    if not isinstance(x, (Tensor, Tensor_)):
        raise TypeError("The input x must be Tensor!")
    return x.dtype in mstype.complex_type


def nan_to_num(x, nan=0.0, posinf=None, neginf=None):
    """
    Replace the `NaN`, positive infinity and negative infinity values in 'x' with the
    specified values, `nan`, `posinf`, and `neginf` respectively.

    Args:
        x (Tensor): The shape of tensor is :math:`(x_1, x_2, ..., x_R)`. With float32 or float16 data type.
        nan (float): The replace value of 'NaN'. Default value is 0.0.
        posinf (float): the value to replace positive infinity values with. Default: None,
            replacing positive infinity with the maximum value supported by the data type of `x`.
          Default value is None.
        neginf (float): the value to replace negative infinity values with. Default: None,
            replacing negative infinity with the minimum value supported by the data type of `x`.
          Default value is None.

    Returns:
        Tensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16 or float32.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> x = Tensor(np.array([float('nan'), float('inf'), -float('inf'), 5.0]), mindspore.float32)
        >>> output = ops.nan_to_num(x, 1.0, 2.0, 3.0)
        >>> print(output)
        [1.  2.  3.  5.0]
    """
    if not isinstance(x, (Tensor, Tensor_)):
        raise TypeError("the input x must be Tensor!")
    if nan is not None:
        if not isinstance(nan, float):
            raise TypeError("the parameter nan's dtype must be float.")
    else:
        nan = 0.0
    if posinf is not None:
        if not isinstance(posinf, float):
            raise TypeError("the parameter posinf's dtype must be float.")
    else:
        if x.dtype == mstype.float16:
            posinf = (float)(np.finfo(np.float16).max)
        elif x.dtype == mstype.float32:
            posinf = (float)(np.finfo(np.float32).max)
    if neginf is not None:
        if not isinstance(neginf, float):
            raise TypeError("the parameter neginf's dtype must be float.")
    else:
        if x.dtype == mstype.float16:
            neginf = (float)(np.finfo(np.float16).min)
        elif x.dtype == mstype.float32:
            neginf = (float)(np.finfo(np.float32).min)
    _nan_to_num = _get_cache_prim(NanToNum)(nan=nan, posinf=posinf, neginf=neginf)
    return _nan_to_num(x)


def fmax(x1, x2):
    r"""
    Computes the maximum of input tensors element-wise.

    Note:
        - Inputs of `x1` and `x2` comply with the implicit type conversion rules to make the data types consistent.
        - The inputs must be two tensors.
        - Types of them are one of the following: float16, float32, float64, int32, int64.
        - Shapes of them are supposed to be broadcast.
        - If one of the elements to be compared is NaN, another element is returned.

    .. math::
        output_i = max(x1_i, x2_i)

    Args:
        x1 (Tensor): The first input is a tensor whose data type is number.
        x2 (Tensor): The second input is is a tensor whose data type is number.

    Returns:
        A Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `x1` and `x2` is not Tensor.
        TypeError: If dtype of `x1` and 'x2' is not one of: float16, float32, float64, int32, int64.
        ValueError: If `x1` and `x2` are not the same shape after broadcast.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> x1 = Tensor(np.array([1.0, 5.0, 3.0]), mindspore.float32)
        >>> x2 = Tensor(np.array([4.0, 2.0, 6.0]), mindspore.float32)
        >>> output = ops.fmax(x1, x2)
        >>> print(output)
        [4. 5. 6.]
    """
    fmax_ = Fmax()
    return fmax_(x1, x2)


def maximum(x, y):
    """
    Computes the maximum of input tensors element-wise.

    Note:
        - Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
        - The inputs must be two tensors or one tensor and one scalar.
        - When the inputs are two tensors,
          dtypes of them cannot be bool at the same time, and the shapes of them could be broadcast.
        - When the inputs are one tensor and one scalar,
          the scalar could only be a constant.
        - Broadcasting is supported.
        - If one of the elements being compared is a NaN, then that element is returned.

    .. math::
        output_i = max(x_i, y_i)

    Args:
        x (Union[Tensor, Number, bool]): The first input is a number or
            a bool or a tensor whose data type is number or bool.
        y (Union[Tensor, Number, bool]): The second input is a number or
            a bool when the first input is a tensor or a tensor whose data type is number or bool.

    Returns:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `x` and `y` is not one of the following: Tensor, Number, bool.
        ValueError: If `x` and `y` are not the same shape.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> # case 1 : same data type
        >>> x = Tensor(np.array([1.0, 5.0, 3.0]), mindspore.float32)
        >>> y = Tensor(np.array([4.0, 2.0, 6.0]), mindspore.float32)
        >>> output = ops.maximum(x, y)
        >>> print(output)
        [4. 5. 6.]
        >>> # case 2 : different data type
        >>> x = Tensor(np.array([1.0, 5.0, 3.0]), mindspore.int32)
        >>> y = Tensor(np.array([4.0, 2.0, 6.0]), mindspore.float32)
        >>> output = ops.maximum(x, y)
        >>> print(output.dtype)
        Float32
    """
    return maximum_(x, y)


def fmin(x1, x2):
    r"""
    Computes the minimum of input tensors element-wise.

    Note:
        - Inputs of `x1` and `x2` comply with the implicit type conversion rules to make the data types consistent.
        - The inputs must be two tensors.
        - Types of them are one of the following: float16, float32, float64, int32, int64.
        - Shapes of them are supposed to be broadcast.
        - If one of the elements to be compared is NaN, another element is returned.

    .. math::
        output_i = min(x1_i, x2_i)

    Args:
        x1 (Tensor): The first input is a tensor whose data type is number.
        x2 (Tensor): The second input is is a tensor whose data type is number.

    Returns:
        A Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `x1` and `x2` is not Tensor.
        TypeError: If dtype of `x1` and 'x2' is not one of: float16, float32, float64, int32, int64.
        ValueError: If `x1` and `x2` are not the same shape after broadcast.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> x1 = Tensor(np.array([1.0, 5.0, 3.0]), mstype.float32)
        >>> x2 = Tensor(np.array([4.0, 2.0, 6.0]), mstype.float32)
        >>> output = ops.fmin(x1, x2)
        >>> print(output)
        [1. 2. 3.]
    """
    fmin_ = Fmin()
    return fmin_(x1, x2)


def minimum(x, y):
    r"""
    Computes the minimum of input tensors element-wise.

    Note:
        - Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
        - The inputs must be two tensors or one tensor and one scalar.
        - When the inputs are two tensors, dtypes of them cannot be bool at the same time.
        - When the inputs are one tensor and one scalar, the scalar could only be a constant.
        - Shapes of them are supposed to be broadcast.
        - If one of the elements being compared is a NaN, then that element is returned.

    .. math::
        output_i = min(x_i, y_i)

    Args:
        x (Union[Tensor, Number, bool]): The first input is a number or
            a bool or a tensor whose data type is number or bool.
        y (Union[Tensor, Number, bool]): The second input is a number or
            a bool when the first input is a tensor or a tensor whose data type is number or bool.

    Returns:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `x` and `y` is not one of the following: Tensor, Number, bool.
        ValueError: If `x` and `y` are not the same shape after broadcast.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> # case 1 : same data type
        >>> x = Tensor(np.array([1.0, 5.0, 3.0]), mindspore.float32)
        >>> y = Tensor(np.array([4.0, 2.0, 6.0]), mindspore.float32)
        >>> output = ops.minimum(x, y)
        >>> print(output)
        [1. 2. 3.]
        >>> # case 2 : different data type
        >>> x = Tensor(np.array([1.0, 5.0, 3.0]), mindspore.int32)
        >>> y = Tensor(np.array([4.0, 2.0, 6.0]), mindspore.float32)
        >>> output = ops.minimum(x, y)
        >>> print(output.dtype)
        Float32
    """
    return minimum_(x, y)


def median(x, axis=-1, keepdims=False):
    r"""
    Computes the median and indices of input tensor.

    Args:
        x (Tensor): A Tensor of any dimension whose data type is int16, int32, int64, float32 or float64.
        axis (int, optional): The dimension need to reduce. Default: -1.
        keepdims (bool, optional): Whether the output tensor need to retain `axis` dimension or not. Default: False.

    Returns:
        y (Tensor), has the same dtype as the `x`. If `keepdims` is true,
        the `y` has the same shape as the `x` except the shape of `y` in dimension `axis` is size 1.
        Otherwise, the `y` lacks `axis` dimension than input.

        indices (Tensor), has the same shape as the `y`, but dtype is int64.

    Raises:
        TypeError: If dtype of `x` is not one of the following: int16, int32, int64, float32, float64.
        TypeError: If input `x` is not a Tensor.
        TypeError: If `axis` is not a int.
        TypeError: If `keepdims` is not a bool.
        ValueError: If `axis` is not in range of [-x.dim, x.dim-1].

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[0.57, 0.11, 0.21],[0.38, 0.50, 0.57], [0.36, 0.16, 0.44]]).astype(np.float32))
        >>> y = ops.median(x, axis=0, keepdims=False)
        >>> print(y)
        (Tensor(shape=[3], dtype=Float32, value=[0.38, 0.16, 0.44]), Tensor(shape=[3], dtype=Int64, value=[1, 2, 2]))
    """
    return Median(False, axis, keepdims)(x)


def orgqr(x, tau):
    r"""
    Computes the first :math:`N` columns of a product of
    `Householder <https://en.wikipedia.org/wiki/Householder_transformation#Householder_matrix>`_
    matrices. Take the case of input without batch
    as an example. Suppose input `x` is a matrix of size :math:`(M, N)` after householder transformation.
    When the diagonal of `x` is set to 1, every colunm of lower triangular in `x` is
    denoted as :math:`w_j` for :math:`j` for
    :math:`j=1, \ldots, M`, this function returns the first :math:`N` columns of the matrix

    .. math::
        H_{1} H_{2} \ldots H_{k} \quad \text { with } \quad H_{j}=\mathrm{I}_{M}-\tau_{j} w_{j} w_{j}^{\mathrm{H}}

    where :math:`\mathrm{I}_{M}` is the :math:`M`-dimensional identity matrix. And when :math:`w` is complex,
    :math:`w^{\mathrm{H}}` is the conjugate transpose, otherwise the transpose.
    The output matrix is the same size as the input matrix `x`.

    Args:
        x (Tensor): Tensor of shape :math:`(*, M, N)`, indicating 2D or 3D matrices,
            with float32, float64, complex64 and complex128 data type.
        tau (Tensor): Tensor of shape :math:`(*, K)`, where `K` is less than or equal to `N`, indicating the
            reflecting coefficient in Householder transformation, which have the same type as `x`.

    Returns:
        Tensor, has the same shape and data type as `x`.

    Raises:
        TypeError: If `x` or `tau` are not Tensors.
        TypeError: If dtype of `x` and `tau` is not one of: float64, float32, complex64, complex128.
        ValueError: If `x` and `tau` have different batch size.
        ValueError: If x.shape[-2] < x.shape[-1].
        ValueError: If x.shape[-1] < tau.shape[-1].
        ValueError: If rank(x) - rank(tau) != 1.
        ValueError: If rank(x) != 2 or 3.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[-114.6, 10.9, 1.1], [-0.304, 38.07, 69.38], [-0.45, -0.17, 62.]]), mindspore.float32)
        >>> tau = Tensor(np.array([1.55, 1.94, 0.0]), mindspore.float32)
        >>> net = ops.orgqr()
        >>> y = net(x, tau)
        >>> print(y)
        [[-0.54999995 -0.2128925   0.8137956 ]
         [ 0.47119996 -0.8752807   0.08240613]
         [ 0.69749993  0.42560163  0.57772595]]
    """

    orgqr_ = Orgqr()
    return orgqr_(x, tau)


def hypot(x, other):
    r"""
    Computes hypotenuse of input tensors element-wise as legs of a right triangle.
    The shape of two inputs should be broadcastable, and data type of them should be
    one of: float32, float64

    .. math::
        out_i = \sqrt{x_i^2 + other_i^2}

    Args:
        x (Tensor): The first input tensor.
        other (Tensor): The second input tensor.

    Returns:
        Tensor, the shape is the same as the one after broadcasting, and the data type is one
        with higher precision in the two inputs.

    Raises:
        TypeError: If data type `x` or `other` is not float32 or float64.
        ValueError: If shape of two inputs are not broadcastable.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([3., 5., 7.]))
        >>> other = Tensor(np.array([4., 12., 24.]))
        >>> y = ops.hypot(x, other)
        >>> print(y)
        [ 5. 13. 25.]
    """

    hypot_ = Hypot()
    return hypot_(x, other)


def heaviside(x, values):
    r"""
    Computes the Heaviside step function for each element in input.

    .. math::
            \text { heaviside }(\text { x, values })=\left\{\begin{array}{ll}
            0, & \text { if x }<0 \\
            \text { values, } & \text { if x }==0 \\
            1, & \text { if x }>0
            \end{array}\right.

    Args:
        x (Tensor): The input tensor. With real number data type.
        values (Tensor): The values to use where `x` is zero. Values can be broadcast with `x` .
            `x` should have the same dtype with 'values'.

    Returns:
        Tensor, has the same type as `x` and `values`.

    Raises:
        TypeError: If `x` or `values` is not Tensor.
        TypeError: If data type `x` and `values` is different.
        ValueError: If shape of two inputs are not broadcastable.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([-5., 1., 0., 2., 0.]))
        >>> values = Tensor(np.array([3.]))
        >>> y = ops.heaviside(x, values)
        >>> print(y)
        [0. 1. 3. 1. 3.]
    """

    heaviside_ = Heaviside()
    return heaviside_(x, values)


def _type_checking_for_xspace(start, end, steps):
    """utility parameter checking function for linspace, logspace, geomspace."""
    array_types = (int, float, bool, list, tuple, Tensor)
    def _get_type(x):
        """get the dtype of input"""
        if isinstance(x, Tensor):
            return x.dtype
        return type(x)
    if not isinstance(start, array_types):
        raise TypeError(f"For 'logspace', 'start' should be int, float, bool, list, tuple, Tensor, but got"
                        f" {_get_type(start)}.")
    if not isinstance(end, array_types):
        raise TypeError(f"For 'logspace', 'end' should be int, float, bool, list, tuple, Tensor, but got"
                        f" {_get_type(end)}.")
    if not isinstance(steps, int):
        raise TypeError(f"For 'logspace', steps should be an integer, but got {_get_type(steps)}.")
    return start, end, steps


def logspace(start, end, steps, base=10.0, dtype=None):
    """
    Returns numbers spaced evenly on a log scale.

    In linear space, the sequence starts at base ** start (base to the power of
    start) and ends with base ** end (see endpoint below).

    Args:
        start (Union[int, list(int), tuple(int), Tensor]): ``base ** start`` is the starting
            value of the sequence.
        end (Union[int, list(int), tuple(int), Tensor]): ``base ** end`` is the final value of
            the sequence.
        steps (int): Number of samples to generate.
        base (Union[int, float], optional): The base of the log space. The step size
            between the elements in :math:`ln(samples) / ln(base)` (or :math:`log_{base}(samples)`)
            is uniform. Default is 10.0.
        dtype (Union[:class:`mindspore.dtype`, str], optional): Designated tensor dtype.
            If `dtype` is None, infer the data type from other input arguments. Default is None.

    Returns:
        Tensor, equally spaced on a log scale.

    Raises:
        TypeError: If input arguments have types not specified above.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import ops
        >>> print(ops.logspace(0, 5, 6, base=2.0))
        [ 1.  2.  4.  8. 16. 32.]
    """
    start, end, steps = _type_checking_for_xspace(start, end, steps)
    if not isinstance(base, (int, float, bool)):
        raise TypeError(f"For 'logspace', 'base' should be a number, but got {base}.")
    if dtype is None:
        dtype = mstype.float32
    linspace_res = linspace_(start, end, steps)
    _tensor_pow = _get_cache_prim(P.Pow)()
    return _tensor_pow(base, linspace_res).astype(dtype)


def logaddexp(x1, x2):
    """
    Computes the logarithm of the sum of exponentiations of the inputs.

    .. math::

        out_i = log(exp(x1_i) + exp(x2_i))

    Args:
        x1 (Tensor): Input Tensor.
        x2 (Tensor): Input Tensor. If the shape of `x1` is not equal to the shape of `x2`, they must be broadcastable to
            a common shape (which becomes the shape of the output).

    Returns:
        Tensor.

    Raises:
        TypeError: If `x1`, `x2` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x1 = Tensor(np.array([1, 2, 3]).astype(np.float16))
        >>> x2 = Tensor(np.array(2).astype(np.float16))
        >>> output = ops.logaddexp(x1, x2)
        >>> print(output)
        [2.312 2.693 3.312]
    """

    log_op = _get_cache_prim(P.Log)()
    exp_op = _get_cache_prim(P.Exp)()
    y = log_op(exp_op(x1) + exp_op(x2))
    return y


def logaddexp2(x1, x2):
    """
    Computes the logarithm of the sum of exponentiations in base of 2 of the inputs.

    .. math::

        out_i = log_2(2^{x1_i} + 2^{x2_i})

    Args:
        x1 (Tensor): Input tensor.
        x2 (Tensor): Input tensor. If ``x1.shape != x2.shape``, they must be broadcastable to
            a common shape (which becomes the shape of the output).

    Returns:
        Tensor.

    Raises:
        TypeError: If `x1`, `x2` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x1 = Tensor(np.array([2, 4, 8]).astype(np.float16))
        >>> x2 = Tensor(np.array([2]).astype(np.float16))
        >>> output = ops.logaddexp2(x1, x2)
        >>> print(output)
        [3. 4.32 8.02]
    """

    log_op = _get_cache_prim(P.Log)()
    pow_op = _get_cache_prim(P.Pow)()
    add_op = _get_cache_prim(P.Add)()

    if not isinstance(x1, (Tensor, Tensor_)):
        raise TypeError("The input x1 must be Tensor.")
    if not isinstance(x2, (Tensor, Tensor_)):
        raise TypeError("The input x2 must be Tensor.")

    add_exp = add_op(pow_op(2, x1), pow_op(2, x2))
    tensor_2 = _make_tensor(2, add_exp.dtype)

    return log_op(add_exp) / log_op(tensor_2)


def std(input_x, axis=(), unbiased=True, keep_dims=False):
    """
    Returns the standard-deviation and mean of each row of the input tensor by default,
    or it can calculate them in specified dimension `axis`.
    If `axis` is a list of dimensions, reduce over all of them.

    Args:
        input_x (Tensor[Number]): Input tensor with a dtype of number.Number, its shape should be :math:`(N, *)`
          where :math:`*` means any number of additional dims, its rank should be less than 8.
        axis (Union[int, tuple(int), list(int)]): The dimensions to reduce. Default: (), reduce all dimensions.
                                                  Only constant value is allowed.
                                                  Must be in the range [-rank(`input_x`), rank(`input_x`)).
        unbiased (bool):  Whether to use Bessel’s correction.
                          If true, will use the Bessel correction unbiased estimation.
                          If false, will through the biased estimation to calculate the standard deviation.
        keep_dims (bool): Whether the output tensor has dim retained or not.
                          If true, keep these reduced dimensions and the length is 1.
                          If false, don't keep these dimensions.

    Returns:
        A tuple of 2 Tensors (output_std, output_mean) containing the standard deviation and mean.
        Suppose the shape of `input_x` is :math:`(x_0, x_1, ..., x_R)`:

        - If `axis` is () and `keep_dims` is set to False, returns a 0-D Tensor, indicating
          the standard deviation of all elements in `input_x`.
        - If `axis` is int 1 and `keep_dims` is set to False, then the returned Tensor
          has shape :math:`(x_0, x_2, ..., x_R)`.
        - If `axis` is tuple(int) or list(int), e.g. (1, 2) and `keep_dims` is set to False,
          then the returned Tensor has shape :math:`(x_0, x_2, ..., x_R)`.

    Raises:
        TypeError: If `input_x` is not a Tensor.
        TypeError: If `axis` is not one of the following: int, tuple or list.
        TypeError: If `keep_dims` is not a bool.
        ValueError: If `axis` is out of range.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[1, 2, 3], [-1, 1, 4]]).astype(np.float32))
        >>> output = ops.std(input_x, 1, True, False)
        >>> output_std, output_mean = output[0], output[1]
        >>> print(output_std)
        [1.        2.5166116]
        >>> print(output_mean)
        [2.        1.3333334]
    """
    reduce_std_op = ReduceStd(axis=axis, unbiased=unbiased, keep_dims=keep_dims)
    output = reduce_std_op(input_x)
    return output


def real(x):
    r"""
    Returns a Tensor that is the real part of the input.
    If input is real, it is returned unchanged.

    Args:
        x (Tensor): The input tensor to compute to.

    Returns:
        Tensor, the shape is the same as the `x`.

    Raises:
       TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import mindspore.ops as ops
        >>> import numpy as np
        >>> x = ms.Tensor(np.asarray(np.complex(1.3+0.4j)), ms.complex64)
        >>> output = ops.real(x)
        >>> print(output)
        1.3
    """
    return _get_cache_prim(ops.Real)()(x)


def reciprocal(x):
    r"""
    Returns reciprocal of a tensor element-wise.

    .. math::

        out_{i} =  \frac{1}{x_{i}}

    Args:
        x (Tensor): The input tensor.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Returns:
        Tensor, has the same shape as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import mindspore.ops as ops
        >>> import numpy as np
        >>> x = ms.Tensor(np.array([1.0, 2.0, 4.0]), ms.float32)
        >>> output = ops.reciprocal(x)
        >>> print(output)
        [1.   0.5  0.25]
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"For reciprocal, the input must be a Tensor, but got {type(x)}.")
    if not is_complex(x) and not ops.is_floating_point(x):
        x = ops.cast(x, mstype.float32)
    return _get_cache_prim(ops.Reciprocal)()(x)


def rsqrt(x):
    r"""
    Computes reciprocal of square root of input tensor element-wise.

    .. math::

        out_{i} =  \frac{1}{\sqrt{x_{i}}}

    Args:
        x (Tensor): The input of rsqrt. Its rank must be in [0, 7] inclusive and
            each element must be a non-negative number, if an element is negative, the calculation result is nan.

    Returns:
        Tensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import mindspore.ops as ops
        >>> x = ms.Tensor([-0.0370,  0.2970,  1.5420, -0.9105])
        >>> output = ops.rsqrt(x)
        >>> print(output)
        [       nan 1.8349396  0.80530024        nan]
    """
    return _get_cache_prim(ops.Rsqrt)()(x)


def sqrt(x):
    """
    Returns sqrt of a tensor element-wise.

    .. math::

        out_{i} = \\sqrt{x_{i}}

    Args:
        x (Tensor): The input tensor with a dtype of number.Number, its rank must be in [0, 7] inclusive.
    Returns:
        Tensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1.0, 4.0, 9.0]), mindspore.float32)
        >>> output = ops.sqrt(x)
        >>> print(output)
        [1. 2. 3.]
    """
    return sqrt_(x)


def square(x):
    """
    Returns square of a tensor element-wise.

    .. math::

        out_{i} = (x_{i})^2

    Args:
        x (Tensor): The input tensor with a dtype of Number, its rank must be in [0, 7] inclusive.

    Returns:
        Tensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
        >>> output = ops.square(x)
        >>> print(output)
        [1. 4. 9.]
    """
    return square_(x)


def outer(x1, x2):
    """
    Return outer product of `x1` and `x2`. If `x1` is a vector of size n and `x2` is a vector of size m,
    then output must be a matrix of size n x m.

    Note:
        This function does not broadcast.

    Args:
        x1 (Tensor): 1-D input vector.
        x2 (Tensor): 1-D input vector.

    Returns:
        out (Tensor, optional), 2-D matrix, the outer product of two vectors.

    Raises:
        TypeError: If `x1` or `x2` is not a Tensor.
        ValueError: If `x1` or `x2` is not an 1-D Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore import ops
        >>> x1 = Tensor(np.array([1, 2, 3]), mindspore.int32)
        >>> x2 = Tensor(np.array([1, 2, 3]), mindspore.int32)
        >>> out = ops.outer(x1, x2)
        >>> print(out)
        [[1 2 3]
         [2 4 6]
         [3 6 9]]
    """

    if not isinstance(x1, (Tensor, Tensor_)):
        raise TypeError("the input x1 must be Tensor!")
    if not isinstance(x2, (Tensor, Tensor_)):
        raise TypeError("the input x2 must be Tensor!")
    if len(x1.shape) != 1:
        raise ValueError("the input x1 must be a 1-D vector!")
    if len(x2.shape) != 1:
        raise ValueError("the input x2 must be a 1-D vector!")
    x1 = x1.reshape(-1, 1)
    mul_ops = _get_cache_prim(P.Mul)()
    y = mul_ops(x1, x2)
    return y


def mv(mat, vec):
    """
    Multiplies matrix `mat` and vector `vec`.

    If `mat` is a Tensor with :math:`(N, M)`, `vec` is a 1-D Tensor of size :math:`M`,
    out will be 1-D of size :math:`N`.

    Args:
        mat (Tensor): Input matrix of shape :math:`(N, M)`.
        vec (Tensor): Input vector of shape :math:`(M,)`.

    Returns:
        Tensor, the shape of the output Tensor is :math:`(N,)`.

    Raises:
        TypeError: If `mat` or `vec` is not a Tensor.
        ValueError: If `mat` is not a 2-D Tensor or `vec` is not a 1-D Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> mat = Tensor(np.array([[3., 4.], [1., 6.], [1., 3.]]).astype(np.float32))
        >>> vec = Tensor(np.array([1., 2.]).astype(np.float32))
        >>> output = ops.mv(mat, vec)
        >>> print(output)
        [11. 13. 7.]
    """

    matmul_op = _get_cache_prim(P.MatMul)()
    reshape_op = _get_cache_prim(P.Reshape)()

    if not isinstance(mat, (Tensor, Tensor_)):
        raise TypeError("The input mat must be Tensor.")
    if not isinstance(vec, (Tensor, Tensor_)):
        raise TypeError("The input vec must be Tensor.")
    if len(mat.shape) != 2:
        raise ValueError("The input mat must be 2-D Tensor.")
    if len(vec.shape) != 1:
        raise ValueError("The input vec must be 1-D Tensor.")

    length_vec = get_x_shape(vec.shape)
    vec = reshape_op(vec, (length_vec[0], 1))

    out = matmul_op(mat, vec)
    out = out.T
    out = out[0]
    return out


def addbmm(x, batch1, batch2, *, beta=1, alpha=1):
    r"""
    Applies batch matrix multiplication to `batch1` and `batch2`, with a reduced add step. The matrix `x` is add to
    final result.

    The optional values `alpha` and `beta` are the matrix-matrix product between `batch1` and `batch2` and the scale
    factor for the added tensor `x` respectively. If `beta` is 0, then `x` will be ignored.

    .. math::
        output = \beta x + \alpha (\sum_{i=0}^{b-1} {batch1 @ batch2})

    Args:
        x (Tensor): Tensor to be added.
        batch1 (Tensor): The first batch of tensor to be multiplied.
        batch2 (Tensor): The second batch of tensor to be multiplied.

    Keyword Args:
        beta (Union[int, float], optional): Multiplier for `x`. Default: 1.
        alpha (Union[int, float], optional): Multiplier for `batch1` @ `batch2`. Default: 1.

    Returns:
        Tensor, has the same dtype as `x`.

    Raises:
        TypeError: If `alpha` or beta is not an int or float.
        ValueError: If `batch1`, `batch2` cannot apply batch matrix multiplication.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> m = np.ones((3, 3)).astype(np.float32)
        >>> arr1 = np.arange(24).astype(np.float32).reshape((2, 3, 4))
        >>> arr2 = np.arange(24).astype(np.float32).reshape((2, 4, 3))
        >>> a = Tensor(arr1)
        >>> b = Tensor(arr2)
        >>> c = Tensor(m)
        >>> output = ops.addbmm(c, a, b)
        >>> print(output)
        [[ 949. 1009. 1069.]
         [1285. 1377. 1469.]
         [1621. 1745. 1869.]]
    """
    dim1 = batch1.ndim
    dim2 = batch2.ndim
    if dim1 != 3 or dim2 != 3:
        raise ValueError(f"For 'addbmm', 'batch1' and 'batch2' must be 3D, but got {dim1} and {dim2} respectively.")
    if not isinstance(alpha, (int, float)):
        raise TypeError(f"For 'addbmm', parameter 'alpha' must be an int or float, but got {type(alpha)}.")
    if not isinstance(beta, (int, float)):
        raise TypeError(f"For 'addbmm', parameter 'beta' must be an int or float, but got {type(beta)}.")
    bmm_op = _get_cache_prim(P.BatchMatMul)()
    bmm_res = bmm_op(batch1, batch2)
    return beta * x + alpha * (bmm_res.sum(axis=0))


def addmm(x, mat1, mat2, *, beta=1, alpha=1):
    r"""
    Multiplies matrix `mat1` and matrix `mat2`. The matrix `x` is added to the final result.

    Args:
        x (Tensor): Tensor to be added.
        mat1 (Tensor): The first tensor to be multiplied.
        mat2 (Tensor): The second tensor to be multiplied.

    Keyword Args:
        beta (Union[int, float], optional): Multiplier for `x`. Default: 1.
        alpha (Union[int, float], optional): Multiplier for `mat1` @ `mat2`. Default: 1.

    .. math::
        output = \beta x + \alpha (mat1 @ mat2)

    Returns:
        Tensor, has the same dtype as `x`.

    Raises:
        ValueError: If `mat1`, `mat2` cannot apply matrix multiplication.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> m = np.ones((3, 3)).astype(np.float32)
        >>> arr1 = np.arange(12).astype(np.float32).reshape((3, 4))
        >>> arr2 = np.arange(12).astype(np.float32).reshape((4, 3))
        >>> a = Tensor(arr1)
        >>> b = Tensor(arr2)
        >>> c = Tensor(m)
        >>> output = ops.addmm(c, a, b)
        >>> print(output)
        [[ 43.  49.  55.]
         [115. 137. 159.]
         [187. 225. 263.]]
    """
    matmul_op = _get_cache_prim(P.MatMul)()
    return beta * x + alpha * (matmul_op(mat1, mat2))


def addmv(x, mat, vec, beta=1, alpha=1):
    """
    Multiplies matrix `mat` and vector `vec`. The vector `x` is added to the final result.

    If mat is a :math:`(N, M)` tensor, vec is a 1-D tensor of size :math:`M`, then `x` must be broadcastable
    with a 1-D tensor of size :math:`N` and `out` will be 1-D tensor of size :math:`N`.

    The optional values `beta` and `alpha` are the matrix-vector product between `mat` and `vec` and the scale
    factor for the added Tensor `x` respectively. If `beta` is 0, then `x` will be ignored.

    .. math::
        output = β x + α (mat @ vec)

    Args:
        x (Tensor): Vector to be added. The shape of the tensor is :math:`(N,)`.
        mat (Tensor): The first tensor to be multiplied. The shape of the tensor is :math:`(N, M)`.
        vec (Tensor): The second tensor to be multiplied. The shape of the tensor is :math:`(M,)`.
        beta (scalar[int, float, bool], optional): Multiplier for `x` (β). The `beta` must be int or
            float or bool. Default: 1.
        alpha (scalar[int, float, bool], optional): Multiplier for `mat` @ `vec` (α). The `alpha` must
            be int or float or bool. Default: 1.

    Returns:
        Tensor, the shape of the output tensor is :math:`(N,)`, has the same dtype as `x`.

    Raises:
        TypeError: If `mat`, `vec`, `x` is not a Tensor.
        TypeError: If inputs `x`, `mat`, 'vec' are not the same dtype.
        ValueError: If `mat` is not a 2-D Tensor.
            If `x`, `vec` is not a 1-D Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([2., 3.]).astype(np.float32))
        >>> mat = Tensor(np.array([[2., 5., 3.], [4., 2., 2.]]).astype(np.float32))
        >>> vec = Tensor(np.array([3., 2., 4.]).astype(np.float32))
        >>> output = ops.addmv(x, mat, vec)
        >>> print(output)
        [30. 27.]
    """

    dtypeop = P.DType()
    input_dtype = dtypeop(x)
    if not (isinstance(x, Tensor) and isinstance(mat, Tensor) and isinstance(vec, Tensor)):
        raise TypeError("For Addmv, inputs must be all tensors.")
    if not (input_dtype == dtypeop(mat) and input_dtype == dtypeop(vec)):
        raise TypeError("For Addmv, the inputs should be the same dtype.")
    _check_input_dtype("x", input_dtype,
                       [mstype.float16, mstype.float32, mstype.float64,
                        mstype.int16, mstype.int32, mstype.int64], "Addmv")
    _check_attr_dtype("alpha", alpha, [int, float, bool], "Addmv")
    _check_attr_dtype("beta", beta, [int, float, bool], "Addmv")
    if input_dtype in (mstype.int16, mstype.int32, mstype.int64):
        scalar_cast = P.ScalarCast()
        alpha = scalar_cast(alpha, mstype.int32)
        beta = scalar_cast(beta, mstype.int32)
    out = beta * x + alpha * mv(mat, vec)
    return out


def adjoint(x):
    r"""
    Returns a view of the tensor conjugated and with the last two dimensions transposed.

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor, the calculated result.

    Raises:
        TypeError: If `x` is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> a = Tensor(np.array([[0. + 0.j, 1. + 1.j], [2. + 2.j, 3. + 3.j]]), mindspore.complex128)
        >>> output = ops.adjoint(a)
        >>> print(output)
        [[0.-0.j 2.-2.j]
         [1.-1.j 3.-3.j]]
    """
    return x.swapaxes(-1, -2).conj()


def addr(x, vec1, vec2, beta=1, alpha=1):
    """
    Executes the outer-product of `vec1` and `vec2` and adds it to the matrix `x`.

    If `vec1` is a vector of size :math:`N` and `vec2` is a vector of size :math:`M`, then `x` must be broadcastable
    with a matrix of size :math:`(N, M)` and `out` will be a matrix of size :math:`(N, M)`.

    The optional values `beta` and `alpha` are the scale factors on the outer product between `vec1` and `vec2`
    and the added matrix `x` respectively. If `beta` is 0, then `x` will be ignored.

    .. math::
        output = β x + α (vec1 ⊗ vec2)

    Args:
        x (Tensor): Vector to be added. The shape of the tensor is :math:`(N, M)`.
        vec1 (Tensor): The first tensor to be multiplied. The shape of the tensor is :math:`(N,)`.
        vec2 (Tensor): The second tensor to be multiplied. The shape of the tensor is :math:`(M,)`.
        beta (scalar[int, float, bool], optional): Multiplier for `x` (β). The `beta` must be int or
            float or bool, Default: 1.
        alpha (scalar[int, float, bool], optional): Multiplier for `vec1` ⊗ `vec2` (α). The `alpha` must
            be int or float or bool, Default: 1.

    Returns:
        Tensor, the shape of the output tensor is :math:`(N, M)`, has the same dtype as `x`.

    Raises:
        TypeError: If `x`, `vec1`, `vec2` is not a Tensor.
        TypeError: If inputs `x`, `vec1`, `vec2` are not the same dtype.
        ValueError: If `x` is not a 2-D Tensor.
            If `vec1`, `vec2` is not a 1-D Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[2., 2.], [3., 2.], [3., 4.]], np.float32))
        >>> vec1 = Tensor(np.array([2., 3., 2.], np.float32))
        >>> vec2 = Tensor(np.array([3, 4], np.float32))
        >>> output = ops.addr(x, vec1, vec2)
        >>> print(output)
        [[ 8. 10.]
         [12. 14.]
         [ 9. 12.]]
    """

    dtypeop = P.DType()
    input_dtype = dtypeop(x)
    if not (isinstance(x, Tensor) and isinstance(vec1, Tensor) and isinstance(vec2, Tensor)):
        raise TypeError("For Addr, inputs must be all tensors.")
    if not (input_dtype == dtypeop(vec1) and input_dtype == dtypeop(vec2)):
        raise TypeError("For Addr, the inputs should be the same dtype.")
    _check_input_dtype("x", input_dtype,
                       [mstype.float16, mstype.float32, mstype.float64,
                        mstype.int16, mstype.int32, mstype.int64], "Addmv")
    _check_attr_dtype("alpha", alpha, [int, float, bool], "Addr")
    _check_attr_dtype("beta", beta, [int, float, bool], "Addr")
    if input_dtype in (mstype.int16, mstype.int32, mstype.int64):
        scalar_cast = P.ScalarCast()
        alpha = scalar_cast(alpha, mstype.int32)
        beta = scalar_cast(beta, mstype.int32)
    matmul_op = P.MatMul()
    reshape_op = P.Reshape()

    length_vec1 = get_x_shape(vec1.shape)
    vec1 = reshape_op(vec1, (length_vec1[0], 1))
    length_vec2 = get_x_shape(vec2.shape)
    vec2 = reshape_op(vec2, (1, length_vec2[0]))

    out = beta * x + alpha * matmul_op(vec1, vec2)
    return out


def lcm(x1, x2):
    """
    Computes least common multiplier of input tensors element-wise.
    The shape of two inputs should be broadcastable, and data type of them should be
    one of: int32, int64

    Args:
        x1 (Tensor): The first input tensor.
        x2 (Tensor): The second input tensor.

    Returns:
        Tensor, the shape is the same as the one after broadcasting, and the data type is one
        with higher digits in the two inputs.

    Raises:
        TypeError: If data type `x1` or `x2` is not int32 or int64.
        ValueError: If shape of two inputs are not broadcastable.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x1 = Tensor(np.array([7, 8, 9]))
        >>> x2 = Tensor(np.array([14, 6, 12]))
        >>> y = ops.lcm(x1, x2)
        >>> print(y)
        [14 24 36]
    """

    lcm_ = _get_cache_prim(Lcm)()
    return lcm_(x1, x2)


def cdist(x, y, p=2.0):
    """
    Computes p-norm distance between each pair of row vectors of two input Tensors.

    Args:
        x (Tensor): Input tensor of shape :math:`(B, P, M)`.
          Letter :math:`B` represents 0 or positive int number.
          When :math:`B` is equal to 0, it means this dimension can be ignored,
          i.e. shape of the tensor is :math:`(P, M)`. The supported dtype is
          [float32, float64] on GPU, or [float32] on CPU.
        y (Tensor): Input tensor of shape :math:`(B, R, M)`, has the same dtype as `x`.
        p (float, optional): P value for the p-norm distance to calculate between each
          vector pair, P ∈ [0,∞]. Default: 2.0.

    Returns:
        Tensor, p-norm distance, has the same dtype as `x`, its shape is :math:`(B, P, R)`.

    Raises:
        TypeError: If `x` or `y` is not Tensor.
        TypeError: If dtype of x or y is not in [float32, float64] on GPU, or is not in [float32] on CPU.
        TypeError: If `p` is not float32.
        ValueError: If `p` is negative.
        ValueError: If dimension of `x` is not the same as `y`.
        ValueError: If dimension of `x` or `y` is neither 2 nor 3.
        ValueError: If the batch shape of `x` is not the same as the shape of `y`.
        ValueError: If the number of columns of `x` is not the same as the number of `y`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[[1.0, 1.0], [2.0, 2.0]]]).astype(np.float32))
        >>> y = Tensor(np.array([[[3.0, 3.0], [3.0, 3.0]]]).astype(np.float32))
        >>> output = ops.cdist(x, y, 2.0)
        >>> print(output)
        [[[2.8284273 2.8284273]
          [1.4142137 1.4142137]]]
    """
    cdist_ = _get_cache_prim(P.Cdist)(p)
    return cdist_(x, y)


def gcd(x1, x2):
    """
    Computes greatest common divisor of input tensors element-wise.
    The shape of two inputs should be broadcastable, and data type of them should be
    one of: int32, int64

    Args:
        x1 (Tensor): The first input tensor.
        x2 (Tensor): The second input tensor.

    Returns:
        Tensor, the shape is the same as the one after broadcasting, and the data type is one
        with higher digits in the two inputs.

    Raises:
        TypeError: If data type `x1` or `x2` is not int32 or int64.
        ValueError: If shape of two inputs are not broadcastable.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x1 = Tensor(np.array([7, 8, 9]))
        >>> x2 = Tensor(np.array([14, 6, 12]))
        >>> y = ops.gcd(x1, x2)
        >>> print(y)
        [7 2 3]
    """

    gcd_ = _get_cache_prim(Gcd)()
    return gcd_(x1, x2)


def lerp(start, end, weight):
    """
    Does a linear interpolation of two tensors start and end based on a float or tensor weight.

    If `weight` is a tensor, the shapes of three inputs need to be broadcast;
    If `weight` is a float, the shapes of `start` and `end` need to be broadcast.

    .. math::

        output_{i} = start_{i} + weight_{i} * (end_{i} - start_{i})

    Args:
        start (Tensor): The tensor with the starting points. Data type must be float16 or float32.
        end (Tensor): The tensor with the ending points. Data type must be the same as `start`.
        weight (Union[float, Tensor]): The weight for the interpolation formula. Must be a float
            or a scalar tensor with float16 or float32 data type.

    Returns:
        Tensor, has the same type and shape as input `start`.

    Raises:
        TypeError: If `start` or `end` is not a tensor.
        TypeError: If `weight` is neither scalar(float) nor tensor.
        TypeError: If dtype of `start` or `end` is neither float16 nor float32.
        TypeError: If dtype of `weight` is neither float16 nor float32 when it is a tensor.
        TypeError: If `start` and `end` have different data types.
        TypeError: If `start`, `end` and `weight` have different data types when `weight` is a tensor.
        ValueError: If `end` could not be broadcast to a tensor with shape of `start`.
        ValueError: If `weight` could not be broadcast to tensors with shapes of `start` and `end` when it is a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> start = Tensor(np.array([1., 2., 3., 4.]), mindspore.float32)
        >>> end = Tensor(np.array([10., 10., 10., 10.]), mindspore.float32)
        >>> output = ops.lerp(start, end, 0.5)
        >>> print(output)
        [5.5 6. 6.5 7. ]
    """
    return lerp_(start, end, weight)


def bernoulli(x, p=0.5, seed=-1):
    r"""
    Randomly set the elements of output to 0 or 1 with the probability of P which follows the Bernoulli distribution.

    .. math::
        out_{i} \sim Bernoulli(p_{i})

    Args:
        x (Tensor): Tensor of shape :math:`(N,*)` where :math:`*` means, any number of additional dimensions. Data
                    type must be int8, uint8, int16, int32, int64, bool, float32 or float64.
        p (Union[Tensor, float], optional): The shape of p need to be broadcast. Data type must be float32 or float64.
                                            The elements of p represent the probability of setting 1 for the
                                            corresponding broadcast position of the current Tensor. The value of `p`
                                            must be in the range `[0, 1]`. Default: 0.5.
        seed (int, optional): The seed value for random generating. The value of `seed` must be -1 or a positive
                              integer. Default: -1, which means using the current timestamp.

    Returns:
        output (Tensor), with the same shape and type as x.

    Raises:
        TypeError: If dtype of `x` is not one of: int8, uint8, int16, int32, int64, bool, float32, float64.
        TypeError: If dtype of `p` is not one of: float32, float64.
        TypeError: If dtype of `seed` is not int.
        ValueError: If `p` is not in range [0, 1].
        ValueError: If `seed` is less than 0 and not -1.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> import mindspore.ops as ops
        >>> input_x = Tensor(np.array([1, 2, 3]), mindspore.int8)
        >>> output = ops.bernoulli(input_x, p=1.0)
        >>> print(output)
        [1 1 1]
        >>> input_p = Tensor(np.array([0.0, 1.0, 1.0]), mindspore.float32)
        >>> output = ops.bernoulli(input_x, input_p)
        >>> print(output)
        [0 1 1]
    """
    bernoulli_ = _get_cache_prim(Bernoulli)(seed)
    if not isinstance(p, Tensor):
        p = Tensor([p])
    return bernoulli_(x, p)


def bessel_i1(x):
    r"""
    Computes the Bessel i1 function of x element-wise.

    Args:
        x (Tensor): The input tensor. The data type must be float16, float32 or float64.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Returns:
        Tensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16, float32 or float64.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([-1, -0.5, 0.5, 1]), mindspore.float32)
        >>> output = ops.bessel_i1(x)
        >>> print(output)
        [-0.5651591  -0.25789431  0.25789431  0.5651591]
    """
    return bessel_i1_(x)


def bessel_i1e(x):
    r"""
    Computes the Bessel i1e function of x element-wise.

    Args:
        x (Tensor): The input tensor. The data type must be float16, float32 or float64.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Returns:
        Tensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16, float32 or float64.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([-1, -0.5, 0.5, 1]), mindspore.float32)
        >>> output = ops.bessel_i1e(x)
        >>> print(output)
        [-0.20791042  -0.15642083  0.15642083  0.20791042]
    """
    return bessel_i1e_(x)


def bessel_k1(x):
    r"""
    Computes the Bessel k1 function of x element-wise.

    Args:
        x (Tensor): The input tensor. The data type must be float16, float32 or float64.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Returns:
        Tensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16, float32 or float64.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.5, 1., 2., 4.]), mindspore.float32)
        >>> output = ops.bessel_k1(x)
        >>> print(output)
        [1.65644112  0.60190723  0.13986588  0.0124835]
    """
    return bessel_k1_(x)


def bessel_k1e(x):
    r"""
    Computes the Bessel k1e function of x element-wise.

    Args:
        x (Tensor): The input tensor. The data type must be float16, float32 or float64.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Returns:
        Tensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16, float32 or float64.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.5, 1., 2., 4.]), mindspore.float32)
        >>> output = ops.bessel_k1e(x)
        >>> print(output)
        [2.73100971  1.63615349  1.03347685  0.68157595]
    """
    return bessel_k1e_(x)


# TODO: remove comment
@constexpr
def _check_input_dtype(param_name, input_dtype, allow_dtypes, cls_name):
    validator.check_type_name(param_name, input_dtype, allow_dtypes, cls_name)


def deg2rad(x):
    """
    Calculates a new tensor with each of the elements of `x` converted from angles in degrees to radians.

    Args:
        x (Tensor[Number]): The input tensor. It must be a positive-definite matrix.
            With float16, float32 or float64 data type.

    Returns:
        Tensor, has the same dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` isn't float16, float32 or float64.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[90.0, -90.0], [180.0, -180.0], [270.0, -270.0]]).astype(np.float32))
        >>> output = ops.deg2rad(x)
        >>> print(output)
        [[ 1.5707964 -1.5707964]
         [ 3.1415927 -3.1415927]
         [ 4.712389  -4.712389 ]]
    """
    if not isinstance(x, (Tensor, Tensor_)):
        raise TypeError("The input x must be tensor")
    dtype_op = _get_cache_prim(P.DType)()
    x_dtype = dtype_op(x)
    _check_input_dtype("x", x_dtype, [mstype.float16, mstype.float32, mstype.float64], "")
    if x_dtype == mstype.float16:
        out = x * (Tensor(math.pi / 180.0).astype(mstype.float16))
    else:
        out = x * math.pi / 180.0
    return out


def rad2deg(x):
    """
    Returns a new tensor with each of the elements of `x` converted from angles in radians to degrees.

    Args:
        x (Tensor): The input tensor.

    Returns:
        Tensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` isn't float16, float32 or float64.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> import mindspore.ops as ops
        >>> x = Tensor([[6.283, -3.142],[1.570, -6.283],[3.142, -1.570]], mindspore.float32)
        >>> output = ops.rad2deg(x)
        >>> print(output)
        [[ 359.98935 -180.02333]
         [  89.95438 -359.98935]
         [ 180.02333  -89.95438]]

    """
    if not isinstance(x, (Tensor, Tensor_)):
        raise TypeError("The input x must be tensor")
    dtype_op = _get_cache_prim(P.DType)()
    x_dtype = dtype_op(x)
    _check_input_dtype("x", x_dtype, [mstype.float16, mstype.float32, mstype.float64], "")
    if x_dtype == mstype.float16:
        out = x * (Tensor(180.0 / math.pi).astype(mstype.float16))
    else:
        out = x * 180.0 / math.pi
    return out


def frac(x):
    """
    Calculates the fractional part of each element in the input

    Args:
        x (Tensor): x is a tensor.

    Returns:
        Tensor, has the same shape and type as input.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore.common import dtype as mstype
        >>> import mindspore.ops as ops
        >>> x = Tensor([2, 4.2, -2.5], mstype.float16)
        >>> output = frac(x)
        >>> print(output)
        [ 0.      0.1992 -0.5   ]
    """
    frac_op = P.Mod()
    return frac_op(x, 1)


#####################################
# Reduction Operation Functions.
#####################################


@constexpr
def _create_cummin_perm(axis, x_shape):
    """Insure axis is in [-len(x_shape),len(s_shape)-1]"""
    len_axis = len(x_shape)
    prem = [i for i in range(len_axis)]
    if axis < 0:
        axis = axis + len_axis
    prem[0], prem[axis] = axis, 0
    prem = tuple(prem)
    return prem


def cummin(x, axis):
    r"""
    Returns a tuple (values,indices) where 'values' is the cumulative minimum value of input Tensor `x`
    along the dimension `axis`, and `indices` is the index location of each minimum value.

    .. math::
        \begin{array}{ll} \\
            y_{i} = min(x_{1}, x_{2}, ... , x_{i})
        \end{array}

    Args:
        x (Tensor): The input Tensor, rank of `x` > 0.
        axis (int): The dimension to do the operation over. The value of `axis` must be in the range
            `[-x.ndim, x.ndim - 1]`.

    Returns:
        tuple [Tensor], tuple of 2 Tensors, containing the cumulative minimum of elements and the index,
        The shape of each output tensor is the same as input `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If `axis` is not an int.
        ValueError: If `axis` is out the range of `[-x.ndim, x.ndim - 1]`.

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
    cummin_op = _get_cache_prim(Cummin)(axis=0)
    if axis == 0:
        out1, out2 = cummin_op(x)
    else:
        transpose = _get_cache_prim(P.Transpose)()
        _shape_op = _get_cache_prim(P.Shape)()
        x_shape = _shape_op(x)
        prem = _create_cummin_perm(axis, x_shape)
        x = transpose(x, prem)
        out1, out2 = cummin_op(x)
        out1 = transpose(out1, prem)
        out2 = transpose(out2, prem)
    return [out1, out2]


def cummax(x, axis):
    r"""
    Returns a tuple (values,indices) where 'values' is the cumulative maximum value of input Tensor `x`
    along the dimension `axis`, and `indices` is the index location of each maximum value.

    .. math::
        \begin{array}{ll} \\
            y_{i} = max(x_{1}, x_{2}, ... , x_{i})
        \end{array}

    Args:
        x (Tensor): The input Tensor, rank of `x` > 0.
        axis (int): The dimension to do the operation over. The value of `axis` must be in the range
            `[-x.ndim, x.ndim - 1]`.

    Returns:
        tuple [Tensor], tuple of 2 Tensors, containing the cumulative maximum of elements and the index,
        The shape of each output tensor is the same as input `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If `axis` is not an int.
        ValueError: If `axis` is out the range of `[-x.ndim, x.ndim - 1]`.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> import mindspore.ops as ops
        >>> x = Tensor(np.array([[3, 4, 6, 10], [1, 6, 7, 9], [4, 3, 8, 7], [1, 3, 7, 9]]).astype(np.float32))
        >>> output = ops.cummax(x, axis=0)
        >>> print(output[0])
        [[ 3.  4.  6. 10.]
         [ 3.  6.  7. 10.]
         [ 4.  6.  8. 10.]
         [ 4.  6.  8. 10.]]
        >>> print(output[1])
        [[0 0 0 0]
         [0 1 1 0]
         [2 1 2 0]
         [2 1 2 0]]
    """
    _cummax = _get_cache_prim(ops.Cummax)(axis=axis)
    return _cummax(x)


def cumsum(x, axis, dtype=None):
    """
    Computes the cumulative sum of input Tensor along `axis`.

    .. math::

        y_i = x_1 + x_2 + x_3 + ... + x_i

    Note:
        On Ascend, the dtype of `x` only support :int8, uint8, int32, float16 or float32 in case of static shape.
        For the case of dynamic shape, the dtype of `x` only support int32, float16 or float32.

    Args:
        x (Tensor): The input Tensor to accumulate.
        axis (int): Axis along which the cumulative sum is computed.
        dtype (:class:`mindspore.dtype`, optional): The desired dtype of returned Tensor. If specified,
            the input Tensor will be cast to `dtype` before the computation. This is useful for preventing overflows.
            If not specified, stay the same as original Tensor. Default: None.

    Returns:
        Tensor, the shape of the output Tensor is consistent with the input Tensor's.

    Raises:
        TypeError: If `x` is not a Tensor.
        ValueError: If the axis is out of range.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> import mindspore.ops as ops
        >>> x = Tensor(np.array([[3, 4, 6, 10], [1, 6, 7, 9], [4, 3, 8, 7], [1, 3, 7, 9]]).astype(np.float32))
        >>> # case 1: along the axis 0
        >>> y = ops.cumsum(x, 0)
        >>> print(y)
        [[ 3.  4.  6. 10.]
         [ 4. 10. 13. 19.]
         [ 8. 13. 21. 26.]
         [ 9. 16. 28. 35.]]
        >>> # case 2: along the axis 1
        >>> y = ops.cumsum(x, 1)
        >>> print(y)
        [[ 3.  7. 13. 23.]
         [ 1.  7. 14. 23.]
         [ 4.  7. 15. 22.]
         [ 1.  4. 11. 20.]]
    """
    if dtype is not None and x.dtype != dtype:
        x = x.astype(dtype, copy=False)
    return cumsum_(x, axis)


def sparse_segment_mean(x, indices, segment_ids):
    r"""
    Computes a Tensor such that :math:`output_i = \frac{\sum_j x_{indices[j]}}{N}` where mean is over :math:`j` such
    that :math:`segment\_ids[j] == i` and :math:`N` is the total number of values summed. If the mean is empty for
    a given segment ID :math:`i`, :math:`output[i] = 0`.

    Note:
        - On CPU, values in `segment_ids` are always validated to be sorted, and an error is thrown for indices that
          are not increasing. Moreover, values in `indices` are validated to be bounded, and an error is thrown when
          `indices` are out of range[0, x.shape[0]).
        - On GPU, this does not throw an error for unsorted `segment_ids` and out-of-bound `indices`. Out-of-order
          `segment_ids` result in safe but unspecified behavior, while out-of-range `indices` will be ignored.

    Args:
        x (Tensor): A Tensor, and its rank must be greater than or equal to 1.
        indices (Tensor): A 1-D Tensor, with int32 or int64 data type.
        segment_ids (Tensor): A 1-D Tensor, must have the same dtype as `indices`.
            Values should be sorted and can be repeated.

    Returns:
        Tensor, whose dtype and rank is the same as `x`. The first dimension is equal to the value of the last element
        of `segment_ids` plus one, and the other dimensions are the same as those of `x`.

    Raises:
        TypeError: If `x`, `indices` or `segment_ids` is not a Tensor.
        TypeError: If the dtype of `x` is not one of the following dtype: float16, float32, float64.
        TypeError: If the dtype of `indices` and `segment_ids` are not one of the following dtype: int32, int64.
        TypeError: If the dtype of `indices` and `segment_ids` are not the same.
        ValueError: If the shape of `x`, 'indices' or `segment_ids` don't meet the parameter description.
        ValueError: If the size of 'indices' and `segment_ids` are not the same.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor([[0, 1, 2], [1, 2, 3], [3, 6, 7]], dtype=mindspore.float32)
        >>> indices = Tensor([0, 1, 2], dtype=mindspore.int32)
        >>> segment_ids = Tensor([1,2,2], dtype=mindspore.int32)
        >>> out = ops.sparse_segment_mean(x, indices, segment_ids)
        >>> print(out)
        [[0. 0. 0.]
         [0. 1. 2.]
         [2. 4. 5.]]
    """
    return sparse_segment_mean_(x, indices, segment_ids)


def block_diag(*inputs):
    r"""
    Creates a block diagonal matrix from the provided tensor.

    Args:
        inputs (List[Tensor]): The dimension of Tensor should be 0, 1 or 2.

    Returns:
        Tensor, two-dimensional with all input tensors arranged in
        order so that their top left and bottom right corners are
        diagonally adjacent. All other elements are set to 0.

    Raises:
        TypeError: If the input is not a list of tensors.
        ValueError: If the dimension of Tensor is not 0, 1 or 2.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x1 = Tensor([[4], [3], [2]], mstype.int32)
        >>> x2 = Tensor([7, 6, 5], mstype.int32)
        >>> x3 = Tensor(1, mstype.int32)
        >>> x4 = Tensor([[5, 4, 3], [2, 1, 0]], mstype.int32)
        >>> x5 = Tensor([[8, 7], [7, 8]], mstype.int32)
        >>> out = ops.block_diag(x1, x2, x3, x4, x5)
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

    arys = [to_2d(ary) for ary in inputs]
    matrix = [ops.concat(to_col_block(arys, idx, ary)) for idx, ary in enumerate(arys)]
    return ops.concat(matrix, 1)


def atleast_1d(inputs):
    r"""
    Converts `inputs` to arrays with at least one dimension.

    Scalar `inputs` are converted to 1-dimensional arrays, whilst higher-dimensional `inputs` are preserved.

    Args:
        inputs (Union[Tensor, List[Tensor]]): One or more input tensors.

    Returns:
        Tensor or list of tensors, each with ``a.ndim >= 1``.

    Raises:
        TypeError: If the input is not a tensor or a list of tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x1 = Tensor(np.ones((2, 3)))
        >>> x2 = Tensor(np.ones(()))
        >>> x3 = Tensor(np.ones(5))
        >>> out = ops.atleast_1d([x1, x2, x3])
        >>> print(out[0].asnumpy())
        [[1. 1. 1.]
         [1. 1. 1.]]
        >>> print(out[1].asnumpy())
        [1.]
        >>> print(out[2].asnumpy())
        [1. 1. 1. 1. 1.]
    """
    if isinstance(inputs, Tensor):
        return _expand(inputs, 1)
    for tensor in inputs:
        if not isinstance(tensor, Tensor):
            raise TypeError(f"For 'atleast_1d', each element of 'inputs' must be a tensor, but got {type(tensor)}")
    return tuple([_expand(arr, 1) for arr in inputs])


def dstack(inputs):
    r"""
    Stacks tensors in sequence depth wise (along the third axis).

    This is equivalent to concatenation along the third axis.
    1-D tensors :math:`(N,)` should be reshaped to :math:`(1,N,1)`.
    2-D tensors :math:`(M,N)` should be reshaped to :math:`(M,N,1)` before concatenation.

    Args:
        inputs (Union(List[tensor], Tuple[tensor])): A sequence of tensors.
            The tensors must have the same shape along all but the third axis.
            1-D or 2-D tensors must have the same shape.

    Returns:
        Tensor, formed by stacking the given tensors, will be at least 3-D.
        The output shape is similar to the output of `numpy.dstack()` function.

    Raises:
        TypeError: If `inputs` is not tuple or list.
        ValueError: If `inputs` is empty.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x1 = Tensor([1., 2., 3.])
        >>> x2 = Tensor([4., 5., 6.])
        >>> out = ops.dstack([x1, x2])
        >>> print(out.asnumpy())
        [[[1. 4.]
          [2. 5.]
          [3. 6.]]]
    """
    if not isinstance(inputs, (tuple, list)):
        raise TypeError(f"For 'dstack', 'diff', 'inputs' must be list or tuple of tensors, but got {type(inputs)}")
    if not inputs:
        raise TypeError(f"For 'dstack', 'diff', 'inputs' can not be empty.")
    trans_inputs = ()
    for tensor in inputs:
        if not isinstance(tensor, Tensor):
            raise TypeError(f"For 'dstack', 'diff', each elements of 'inputs' must be Tensor, but got {type(tensor)}")
        if tensor.ndim <= 1:
            tensor = _expand(tensor, 2)
        if tensor.ndim == 2:
            tensor = P.ExpandDims()(tensor, 2)
        trans_inputs += (tensor,)
    if not trans_inputs:
        raise ValueError("For 'dstack', 'diff', at least one tensor is needed to concatenate.")
    return P.Concat(2)(trans_inputs)


@constexpr
def _check_is_int(arg_value, arg_name, cls_name):
    validator.check_is_int(arg_value, arg_name, cls_name)


def diff(x, n=1, axis=-1, prepend=None, append=None):
    r"""
    Calculates the n-th discrete difference along the given axis.

    The first difference is given by :math:`out[i] = a[i+1] - a[i]` along the given axis,
    higher differences are calculated by using `diff` iteratively.

    Note:
        Zero-shaped Tensor is not supported, a value error is raised if
        an empty Tensor is encountered.

    Args:
        x (Tensor): Input tensor.
            Full support for signed integers, partial support for floats and complex numbers
        n (int, optional): The number of times values are differenced. If zero,
            the input is returned as-is. Default: 1. Currently only 1 is supported.
        axis (int, optional): The axis along which the difference is taken, default
            is the last axis. Default: -1.
        prepend (Tensor, optional): Values to prepend or append to a along
            `axis` prior to performing the difference. Scalar values are expanded to
            arrays with length 1 in the direction of `axis` and the shape of the input
            array in along all other axis. Otherwise the dimension and shape must
            match `x` except along `axis`. Default: None.
        append (Tensor, optional): Values to prepend or append to a along
            `axis` prior to performing the difference. Scalar values are expanded to
            arrays with length 1 in the direction of `axis` and the shape of the input
            array in along all other axis. Otherwise the dimension and shape must
            match `x` except along `axis`. Default: None.

    Returns:
        Tensor, the n-th differences of input. The shape of the output is the same as `x`
        except along `axis` where the size is reduced by `n`. The type of the output
        is the same as `x`.

    Raises:
        TypeError: If the data type of the elementes in `x` is uint16, uint32 or uint64.
        TypeError: If `x` is not a tensor.
        TypeError: If the dimension 'x' is less than 1.
        RuntimeError: If `n` is not 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor([1, 3, -1, 0, 4])
        >>> out = ops.diff(x)
        >>> print(out.asnumpy())
        [ 2 -4  1  4]
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"For 'diff', 'x' must be a tensor, but got {type(x)}")
    if x.ndim < 1:
        raise TypeError(f"For 'diff', the dimension 'x' must be at least 1, but got {x.ndim}")
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
    a = [i for i in range(x.shape[axis])]
    a1 = x.gather(Tensor(a[:-1]), axis)
    a2 = x.gather(Tensor(a[1:]), axis)
    return a2 - a1


def tril_indices(row, col, offset=0, dtype=mstype.int64):
    r"""
    Returns the indices of the lower triangular part of a row-by- col matrix in a 2-by-N Tensor,
    where the first row contains row coordinates of all indices and the second row contains column coordinates.
    Indices are ordered based on rows and then columns.

    The lower triangular part of the matrix is defined as the elements on and below the diagonal.

    Note:
        When running on CUDA, row * col must be less than 2^59 to prevent overflow during calculation.

    Args:
        row (int): number of rows in the 2-D matrix.
        col (int): number of columns in the 2-D matrix.
        offset (int): diagonal offset from the main diagonal. Default: 0.
        dtype (:class:`mindspore.dtype`): The specified type of output tensor.
            An optional data type of `mindspore.int32` and `mindspore.int64`. Default: `mindspore.int32`.

    Returns:
        - **y** (Tensor) - indices of the elements in lower triangular part of matrix. The type is specified by `dtype`.
          The shape of output is :math:`(2, tril\_size)`, where :math:`tril\_size` is the number of elements in the
          lower triangular matrix.

    Raises:
        TypeError: If `row`, `col` or `offset` is not an int.
        TypeError: If `dtype` is neither int32 nor int64.
        ValueError: If `row` or `col` < 0.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> output = ops.tril_indices(4, 3, -1, mindspore.int64)
        >>> print(output)
        [[1 2 2 3 3 3]
         [0 0 1 0 1 2]]
        >>> print(output.dtype)
        Int64
    """

    tril_indices_ = TrilIndices(row=row, col=col, offset=offset, dtype=dtype)
    return tril_indices_()


def triu_indices(row, col, offset=0, dtype=mstype.int64):
    r"""
    Returns the indices of the upper triangular part of a row by col matrix in a 2-by-N Tensor,
    where the first row contains row coordinates of all indices and the second row contains column coordinates.
    Indices are ordered based on rows and then columns.

    The upper triangular part of the matrix is defined as the elements on and above the diagonal.

    Note:
        When running on CUDA, row * col must be less than 2^59 to prevent overflow during calculation.

    Args:
        row (int): number of rows in the 2-D matrix.
        col (int): number of columns in the 2-D matrix.
        offset (int): diagonal offset from the main diagonal. Default: 0.
        dtype (:class:`mindspore.dtype`): The specified type of output tensor.
            An optional data type of `mindspore.int32` and `mindspore.int64`. Default: `mindspore.int32`.

    Returns:
        - **y** (Tensor) - indices of the elements in upper triangular part of matrix. The type is specified by `dtype`.
          The shape of output is :math:`(2, triu\_size)`, where :math:`triu\_size` is the number of elements in the
          upper triangular matrix.

    Raises:
        TypeError: If `row`, `col` or `offset` is not an int.
        TypeError: If `dtype` is neither int32 nor int64.
        ValueError: If `row` or `col` < 0.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> output = ops.triu_indices(5, 4, 2, mindspore.int64)
        >>> print(output)
        [[0 0 1]
         [2 3 3]]
        >>> print(output.dtype)
        Int64
    """

    triu_indices_ = TriuIndices(row=row, col=col, offset=offset, dtype=dtype)
    return triu_indices_()


def atleast_2d(inputs):
    r"""
    Reshapes `inputs` as arrays with at least two dimensions.
    Input tensor with two or more dimensions will be returned as is.

    Args:
        inputs (Union[tensor, List[tensor]]): one or more input tensors.

    Returns:
        Tensor or list of tensors, each with ``a.ndim >= 2``.

    Raises:
        TypeError: If the input is not a tensor or a list of tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x1 = np.ones((2, 3))
        >>> x2 = np.ones(())
        >>> x3 = np.ones(5)
        >>> out = ops.atleast_2d([x1, x2, x3])
        >>> print(out)
        [Tensor(shape=[2, 3], dtype=Float32, value=
        [[1.00000000e+000, 1.00000000e+000, 1.00000000e+000],
         [1.00000000e+000, 1.00000000e+000, 1.00000000e+000]]), Tensor(shape=[1, 1], dtype=Float32, value=
        [[1.00000000e+000]]), Tensor(shape=[1, 5], dtype=Float32, value=
        [[1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000, 1.00000000e+000]])]
    """
    if isinstance(inputs, Tensor):
        return _expand(inputs, 2)
    for tensor in inputs:
        if not isinstance(tensor, Tensor):
            msg = "expect Tensor or list of tensors, but got " + f"{type(tensor)}"
            raise TypeError(msg)
    return tuple([_expand(arr, 2) for arr in inputs])


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
    meshgrid = P.Meshgrid(indexing="ij")
    meshgrid_output = meshgrid(inputs)
    stack = P.Stack(axis=-1)
    stack_output = stack(meshgrid_output)
    reshape = P.Reshape()
    return reshape(stack_output, (-1, len(inputs)))


def atleast_3d(inputs):
    r"""
    Reshapes `inputs` as arrays with at least three dimensions.
    Input tensor with three or more dimensions will be returned as is.

    Args:
        inputs (Union[Tensor, List[Tensor]]): One or more input tensors.

    Returns:
        Tensor or list of tensors, each with ``a.ndim >= 3``. For example,
        a 1-D array of shape `(N,)` becomes a tensor of shape `(1, N, 1)`, and
        a 2-D array of shape `(M, N)` becomes a tensor of shape `(M, N, 1)`.

    Raises:
        TypeError: If the input is not a tensor or a list of tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x1 = Tensor(np.ones((2, 3)))
        >>> x2 = Tensor(np.ones(()))
        >>> x3 = Tensor(np.ones(5))
        >>> out = ops.atleast_3d([x1, x2, x3])
        >>> print(out[0].asnumpy())
        [[[1.]
          [1.]
          [1.]]
        <BLANKLINE>
         [[1.]
          [1.]
          [1.]]]
        >>> print(out[1].asnumpy())
        [[[1.]]]
        >>> print(out[2].asnumpy())
        [[[1.]
          [1.]
          [1.]
          [1.]
          [1.]]]
    """

    def _expand3(arr):
        ndim = P.Rank()(arr)
        if ndim == 0:
            return P.Reshape()(arr, (1, 1, 1))
        if ndim == 1:
            return P.Reshape()(arr, (1, P.Size()(arr), 1))
        if ndim == 2:
            return P.Reshape()(arr, P.Shape()(arr) + (1,))
        return arr

    if isinstance(inputs, Tensor):
        return _expand3(inputs)
    for tensor in inputs:
        if not isinstance(tensor, Tensor):
            raise TypeError(f"For 'atleast_3d', each element of 'inputs' must be a tensor, but got {type(tensor)}")
    return tuple([_expand3(arr) for arr in inputs])


def view_as_real(x):
    r"""
    View a complex Tensor as a real Tensor.
    The size of last dimension of the returned real Tensor is 2, and the last dimension is composed of
    the real and imaginary components of complex numbers.

    Args:
        x (Tensor): the input must be a complex Tensor.

    Returns:
        A real Tensor.

    Raises:
        TypeError: If the input Tensor is not a complex Tensor.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor([2+1j,2+3j,2-1j,2], mstype.complex64)
        >>> print(ops.view_as_real(x))
        [[ 2.  1.]
         [ 2.  3.]
         [ 2. -1.]
         [ 2.  0.]]
    """
    if not is_complex(x):
        raise TypeError("For view_as_real, the dtype of input Tensor must be complex.")
    real_part = x.real().expand_dims(-1)
    imag_part = x.imag().expand_dims(-1)
    con = _get_cache_prim(ops.Concat)(-1)
    return con((real_part, imag_part))


def vstack(inputs):
    r"""
    Stacks tensors in sequence vertically.

    This is equivalent to concatenation along the first axis.
    1-D tensors should firstly be reshaped to :math:`(1, N)`, and then be concatenated along the first axis.

    Args:
        inputs (Union(List[tensor], Tuple[tensor])): A sequence of 1-D or 2-D tensors.
            The tensors must have the same shape along all but the first axis.
            1-D tensors must have the same shape.

    Returns:
        Tensor, formed by stacking the given tensors, will be at least 3-D.
        The output shape is similar to the output of `numpy.vstack()` function.

    Raises:
        TypeError: If `inputs` is not list.
        ValueError: If `inputs` is empty.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x1 = np.array([1, 2, 3])
        >>> x2 = np.array([4, 5, 6])
        >>> out = ops.vstack([x1, x2])
        >>> print(out)
        [[1 2 3]
         [4 5 6]]
    """
    if not isinstance(inputs, (tuple, list)):
        msg = f"List or tuple of tensors are required, but got {type(inputs)}"
        raise TypeError(msg)
    if not inputs:
        msg = "Inputs can not be empty"
        raise TypeError(msg)
    trans_tup = ()
    for tensor in inputs:
        if not isinstance(tensor, Tensor):
            msg = f"Tensor is required, but got {type(tensor)}"
            raise TypeError(msg)
        if tensor.ndim <= 1:
            shape = P.Shape()(tensor)
            if isinstance(shape, int):
                shape = (shape,)
            ndim_diff = 2 - len(shape)
            if ndim_diff > 0:
                shape = [1] * ndim_diff + [i for i in shape]
            tensor = P.Reshape()(tensor, tuple(shape))
        trans_tup += (tensor,)
    if not trans_tup:
        raise ValueError("Need at least one tensor to concatenate.")
    out = P.Concat(0)(trans_tup)
    return out


def combinations(x, r=2, with_replacement=False):
    r"""
    When `with_replacement` is set to `False`, the behavior is similar to python's
    `itertools.combinations`, and when `with_replacement` is set to `True`,
    it behaves like `itertools.combinations_with_replacement`.

    Args:
        x (Tensor): One-dimensional tensors.
        r (int, optional): Number of elements. Default: 2.
        with_replacement (bool, optional): Allow duplication or not. Default: False.

    Returns:
        Tensor, equivalent to resulting list.

    Raises:
        TypeError: If `x` is not a tensor.
        ValueError: If `x` is not one-dimensional.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor([1, 3, -1, 0, 4])
        >>> output = ops.combinations(x)
        >>> print(output.asnumpy())
        [[ 1  3]
         [ 1 -1]
         [ 1  0]
         [ 1  4]
         [ 3 -1]
         [ 3  0]
         [ 3  4]
         [-1  0]
         [-1  4]
         [ 0  4]]
    """

    def _combinations(iterable, r):
        lst = ops.StridedSlice()(Tensor([np.zeros(r)]), (0,), (0,), (1,))
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
        lst = Tensor([])
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

    if not isinstance(x, Tensor):
        raise TypeError(f"For 'combinations', 'x' must be a tensor, but got {type(x)}")
    if x.ndim != 1:
        raise ValueError(f"For 'combinations', the dimension 'x' must be 1, but got {x.ndim}")
    comb_func = _combinations_with_replacement if with_replacement else _combinations
    ret = comb_func(x, r)
    if ret.size == 0:
        return ret
    return ops.reshape(ret, (-1, r))


def dist(input_x, input_y, p=2):
    r"""
    Computes batched the :math:`p`-norm distance between each pair of the two collections of row vectors.

    Note:
        Since only normalization for integer :math:`p`-normal form is supported in MindSpore,
        a type error will be raised if :math:`p` is not an integer.

    Args:
        input_x (Tensor): The first input tensor. The dtype must be float16 or float32.
        input_y (Tensor): The second input tensor. The dtype must be float16 or float32.
        p (int, optional): The order of norm. `p` is greater than or equal to 0. Default: 2.

    Returns:
        Tensor, has the same dtype as `input_x`, which shape is :math:`(1)`.

    Raises:
        TypeError: If `input_x` or `input_y` is not a Tensor.
        TypeError: If dtype of `input_x` or `input_y` is neither float16 nor float32.
        TypeError: If `p` is not a non-negative integer.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor([[[1.0, 1.0], [2.0, 2.0]]])
        >>> input_y = Tensor([[[3.0, 3.0], [3.0, 3.0]]])
        >>> out = ops.dist(input_x, input_y)
        >>> print(out.asnumpy())
        3.1622777
    """
    if not isinstance(input_x, Tensor):
        raise TypeError(f"For 'dist', 'input_x' must be a tensor, but got {type(input_x)}")
    if not isinstance(input_y, Tensor):
        raise TypeError(f"For 'dist', 'input_y' must be a tensor, but got {type(input_y)}")
    z = input_x - input_y
    if z.ndim == 0:
        return ops.abs(z)

    # the types of p will expend once ops.LpNorm supports float
    return ops.LpNorm(axis=0, p=p)(ops.reshape(z, (-1,)))


def copysign(x, other):
    r"""
    Create a new floating-point tensor with the magnitude of `x` and the sign of `other`, element-wise.

    Args:
        x (Union[Tensor]): Values to change the sign of.
        other (Union[int, float, Tensor]): The sign of `other` is copied to `x`. If `x.shape != other.shape`,
            `other` must be broadcastable to the shape of `x` (which is also the shape of the output).

    Returns:
        Tensor. The dtype of the tensor is float.
        The values of `x` with the sign of `other`, the shape is the same as `x`.

    Raises:
        TypeError: If dtype of the input is not in the given types or
            the input can not be converted to tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> import mindspore.ops as ops
        >>> x = np.array([[0.3, -0.7], [0.5, 0.5]])
        >>> other = np.array([[-0.4, 0.6], [0.4, -0.6]])
        >>> out = ops.copysign(x, other)
        >>> print(out)
        [[-0.3  0.7]
         [ 0.5 -0.5]]
    """

    def _broadcast_to_shape(x, shape):
        """Broadcasts x from current shape to shape"""
        ndim_to = len(shape)
        x = _expand(x, ndim_to)
        return _broadcast_to(x, P.Shape()(x), shape, ndim_to)

    if not isinstance(x, Tensor):
        raise TypeError("Tensor is expected, but got " + f"{type(x)}")
    if not isinstance(other, (int, float, Tensor)):
        raise TypeError(
            "integer, float or Tensor is expected, but got " + f"{type(other)}"
        )

    if not isinstance(other, Tensor):
        other = _type_convert(Tensor, other)
    other = _broadcast_to_shape(other, P.Shape()(x))

    if _check_same_type(P.DType()(x), mstype.bool_):
        raise TypeError("copysign does not accept dtype bool.")

    if _check_same_type(P.DType()(x), mstype.complex64):
        raise TypeError("copysign does not accept dtype complex64.")
    if _check_same_type(P.DType()(other), mstype.complex64):
        raise TypeError("copysign does not accept dtype complex64.")

    if _check_same_type(P.DType()(x), mstype.complex128):
        raise TypeError("copysign does not accept dtype complex128.")
    if _check_same_type(P.DType()(other), mstype.complex128):
        raise TypeError("copysign does not accept dtype complex128.")

    x_float = (
        x
        if x.dtype in (mstype.float16, mstype.float32, mstype.float64)
        else x.astype("float32")
    )
    pos_tensor = P.Abs()(x_float)
    less_zero = P.Less()(other, 0)
    return P.Select()(less_zero, P.Neg()(pos_tensor), pos_tensor)


def hann_window(window_length, periodic=True):
    r"""
    Generates a Hann Window.

    The Hann window is defined as

    .. math::
        w(n) = \frac{1}{2} - \frac{1}{2} \cos\left(\frac{2\pi{n}}{M-1}\right) \qquad 0 \leq n \leq M-1

    Args:
        window_length (int): Length of window.
        periodic (bool, optional): When set to True, generates a periodic window for spectral analysis.
            When set to False, generates a symmetric window for filter design.Default: True.

    Returns:
        Tensor, a Hann window.

    Raises:
        TypeError: If `window_length` is not an integer.
        TypeError: If `periodic` is not a variable of Boolean type.
        ValueError: If `window_length` is negative.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> window_length = 5
        >>> out = ops.hann_window(window_length)
        >>> print(out.asnumpy())
        [0.        0.3454915 0.9045085 0.9045085 0.3454915]
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
            f"For 'kaiser_window', 'periodic' must be a variable of Boolean type, but got {type(periodic)}"
        )
    if periodic:
        window_length = window_length + 1
    n = np.arange(0, window_length)
    w = 0.5 - 0.5 * np.cos(2 * math.pi / (window_length - 1) * n)
    return Tensor(w[:-1]) if periodic else Tensor(w)


@constexpr
def _type_convert(force, obj):
    """
    Convert type of `obj` to `force`.
    """
    return force(obj)


def logsumexp(x, axis, keep_dims=False):
    r"""
    Reduces a dimension of a tensor by calculating exponential for all elements in the dimension,
    then calculate logarithm of the sum.

    .. math::

        logsumexp(x) = \log(\sum(e^(x-x_{max}))) + x_{max}

    Note:
        The dimension of input Tensor on Ascend should be less than or equal to 8,
        and the dimension of input Tensor on CPU should be less than 8.

    Args:
        x (Tensor): The input tensor. With float16 or float32 data type.
        axis (Union[int, tuple(int), list(int)]): The dimensions to reduce. Default: (), reduce all dimensions.
            Only constant value is allowed.
        keep_dims (bool): If True, keep these reduced dimensions and the length is 1.
            If False, don't keep these dimensions.
            Default : False.

    Returns:
        Tensor, has the same dtype as the `x`.

        - If axis is (), and keep_dims is False,
          the output is a 0-D tensor representing the sum of all elements in the input tensor.
        - If axis is int, set as 2, and keep_dims is False,
          the shape of output is :math:`(x_1, x_3, ..., x_R)`.
        - If axis is tuple(int), set as (2, 3), and keep_dims is False,
          the shape of output is :math:`(x_1, x_4, ..., x_R)`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
        >>> output = ops.logsumexp(x, 1, keep_dims=True)
        >>> print(output.shape)
        (3, 1, 5, 6)
    """
    _exp = _get_cache_prim(P.Exp)()
    _reduce_sum = _get_cache_prim(P.ReduceSum)(keep_dims)
    _log = _get_cache_prim(P.Log)()

    x_max = x.max(axis=axis, keepdims=True)
    x_exp = _exp(x - x_max)
    x_sumexp = _reduce_sum(x_exp, axis)
    x_logsumexp = _log(x_sumexp)
    if not keep_dims:
        x_max = x_max.squeeze(axis=axis)
    return x_logsumexp + x_max


def amin(input, axis=None, keep_dims=False):
    r"""
    Reduces all dimensions of a tensor by returning the minimum value in `input`, by default. And also can
    reduce a dimension of `input` along specified `axis`. `keep_dims` determines whether the dimensions of
    output and input are the same.

    Args:
        input (Tensor[Number]): The input tensor. The dtype of the tensor to be reduced is number.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.
        axis (Union[int, tuple(int), list(int)]): The dimensions to reduce. Default: None, reduce all dimensions.
            Only constant value is allowed. Assume the rank of `x` is r, and the value range is [-r,r)..
        keep_dims (bool): If true, keep these reduced dimensions and the length is 1. If false, don't keep
            these dimensions. Default: False.

    Returns:
        Tensor, has the same data type as input tensor.

        - If `axis` is None, and `keep_dims` is False,
          the output is a 0-D tensor representing the product of all elements in the input tensor.
        - If `axis` is int, set as 1, and `keep_dims` is False,
          the shape of output is :math:`(x_0, x_2, ..., x_R)`.
        - If `axis` is tuple(int), set as (1, 2), and `keep_dims` is False,
          the shape of output is :math:`(x_0, x_3, ..., x_R)`.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If `axis` is not one of the following: int, tuple or list.
        TypeError: If `keep_dims` is not a bool.
        ValueError: If `axis` is out of range.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
        >>> output = ops.amin(x, 1, keep_dims=True)
        >>> result = output.shape
        >>> print(result)
        (3, 1, 5, 6)
        >>> # case 1: Reduces a dimension by the minimum value of all elements in the dimension.
        >>> x = Tensor(np.array([[[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]],
        ...                      [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
        ...                      [[7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9]]]), mindspore.float32)
        >>> output = ops.amin(x)
        >>> print(output)
        1.0
        >>> print(output.shape)
        ()
        >>> # case 2: Reduces a dimension along axis 0.
        >>> output = ops.amin(x, 0, True)
        >>> print(output)
        [[[1. 1. 1. 1. 1. 1.]
          [2. 2. 2. 2. 2. 2.]
          [3. 3. 3. 3. 3. 3.]]]
        >>> # case 3: Reduces a dimension along axis 1.
        >>> output = ops.amin(x, 1, True)
        >>> print(output)
        [[[1. 1. 1. 1. 1. 1.]]
         [[4. 4. 4. 4. 4. 4.]]
         [[7. 7. 7. 7. 7. 7.]]]
        >>> # case 4: Reduces a dimension along axis 2.
        >>> output = ops.amin(x, 2, True)
        >>> print(output)
        [[[1.]
          [2.]
          [3.]]
         [[4.]
          [5.]
          [6.]]
         [[7.]
          [8.]
          [9.]]]
    """
    if axis is None:
        axis = ()
    return _get_cache_prim(P.ReduceMin)(keep_dims)(input, axis)


def amax(input, axis=None, keep_dims=False):
    r"""
    Reduces all dimensions of a tensor by returning the maximum value in `input`, by default. And also can
    reduce a dimension of `input` along specified `axis`.  `keep_dims` determines whether the dimensions of
    output and input are the same.

    Args:
        input (Tensor[Number]): The input tensor. The dtype of the tensor to be reduced is number.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.
        axis (Union[int, tuple(int), list(int)]): The dimensions to reduce. Default: None, reduce all dimensions.
            Only constant value is allowed. Assume the rank of `x` is r, and the value range is [-r,r).
        keep_dims (bool): If true, keep these reduced dimensions and the length is 1. If false, don't keep these
            dimensions. Default: False.

    Returns:
        Tensor, has the same data type as input tensor.

        - If `axis` is None, and `keep_dims` is False, the output is a 0-D tensor representing the product of all
            elements in the input tensor.
        - If `axis` is int, set as 1, and `keep_dims` is False, the shape of output is :math:`(x_0, x_2, ..., x_R)`.
        - If `axis` is tuple(int), set as (1, 2), and `keep_dims` is False, the shape of output is
            :math:`(x_0, x_3, ..., x_R)`.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If `axis` is not one of the following: int, tuple or list.
        TypeError: If `keep_dims` is not a bool.
        ValueError: If `axis` is out of range.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
        >>> output = ops.amax(x, 1, keep_dims=True)
        >>> result = output.shape
        >>> print(result)
        (3, 1, 5, 6)
        >>> # case 1: Reduces a dimension by the maximum value of all elements in the dimension.
        >>> x = Tensor(np.array([[[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]],
        ...                      [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
        ...                      [[7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9]]]), mindspore.float32)
        >>> output = ops.amax(x)
        >>> print(output)
        9.0
        >>> print(output.shape)
        ()
        >>> # case 2: Reduces a dimension along axis 0.
        >>> output = ops.amax(x, 0, True)
        >>> print(output)
        [[[7. 7. 7. 7. 7. 7.]
          [8. 8. 8. 8. 8. 8.]
          [9. 9. 9. 9. 9. 9.]]]
        >>> # case 3: Reduces a dimension along axis 1.
        >>> output = ops.amax(x, 1, True)
        >>> print(output)
        [[[3. 3. 3. 3. 3. 3.]]
         [[6. 6. 6. 6. 6. 6.]]
         [[9. 9. 9. 9. 9. 9.]]]
        >>> # case 4: Reduces a dimension along axis 2.
        >>> output = ops.amax(x, 2, True)
        >>> print(output)
        [[[1.]
          [2.]
          [3.]]
         [[4.]
          [5.]
          [6.]]
         [[7.]
          [8.]
          [9.]]]
    """
    if axis is None:
        axis = ()
    return _get_cache_prim(P.ReduceMax)(keep_dims)(input, axis)


def mean(x, axis=(), keep_dims=False):
    r"""
    Reduces all dimension of a tensor by averaging all elements in the dimension, by default.
    And reduce a dimension of `x` along the specified `axis`. `keep_dims`
    determines whether the dimensions of the output and input are the same.

    Args:
        x (Tensor[Number]): The input tensor. The dtype of the tensor to be reduced is number.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.
        axis (Union[int, tuple(int), list(int)]): The dimensions to reduce. Default: (), reduce all dimensions.
          Only constant value is allowed. Assume the rank of `x` is r, and the value range is [-r,r).
        keep_dims (bool): If true, keep these reduced dimensions and the length is 1.
                          If false, don't keep these dimensions. Default: False.

    Returns:
        Tensor, has the same data type as input tensor.

        - If `axis` is (), and `keep_dims` is False,
          the output is a 0-D tensor representing the product of all elements in the input tensor.
        - If `axis` is int, set as 1, and `keep_dims` is False,
          the shape of output is :math:`(x_0, x_2, ..., x_R)`.
        - If `axis` is tuple(int), set as (1, 2), and `keep_dims` is False,
          the shape of output is :math:`(x_0, x_3, ..., x_R)`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If `axis` is not one of the following: int, tuple or list.
        TypeError: If `keep_dims` is not a bool.
        ValueError: If `axis` is out of range.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
        >>> output = ops.mean(x, 1, keep_dims=True)
        >>> result = output.shape
        >>> print(result)
        (3, 1, 5, 6)
        >>> # case 1: Reduces a dimension by averaging all elements in the dimension.
        >>> x = Tensor(np.array([[[2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2]],
        ... [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
        ... [[6, 6, 6, 6, 6, 6], [8, 8, 8, 8, 8, 8], [10, 10, 10, 10, 10, 10]]]),
        ... mindspore.float32)
        >>> output = ops.mean(x)
        >>> print(output)
        5.0
        >>> print(output.shape)
        ()
        >>> # case 2: Reduces a dimension along the axis 0
        >>> output = ops.mean(x, 0, True)
        >>> print(output)
        [[[4. 4. 4. 4. 4. 4.]
          [5. 5. 5. 5. 5. 5.]
          [6. 6. 6. 6. 6. 6.]]]
        >>> # case 3: Reduces a dimension along the axis 1
        >>> output = ops.mean(x, 1, True)
        >>> print(output)
        [[[2. 2. 2. 2. 2. 2.]]
         [[5. 5. 5. 5. 5. 5.]]
         [[8. 8. 8. 8. 8. 8.]]]
        >>> # case 4: Reduces a dimension along the axis 2
        >>> output = ops.mean(x, 2, True)
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

    return _get_cache_prim(P.ReduceMean)(keep_dims)(x, axis)


def prod(x, axis=(), keep_dims=False):
    r"""
    Reduces a dimension of a tensor by multiplying all elements in the dimension, by default. And also can
    reduce a dimension of `x` along the axis. Determine whether the dimensions of the output and input are the same by
    controlling `keep_dims`.

    Args:
        x (Tensor[Number]): The input tensor. The dtype of the tensor to be reduced is number.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.
        axis (Union[int, tuple(int), list(int)]): The dimensions to reduce. Default: (), reduce all dimensions.
          Only constant value is allowed. Assume the rank of `x` is r, and the value range is [-r,r).
        keep_dims (bool): If true, keep these reduced dimensions and the length is 1.
                          If false, don't keep these dimensions. Default: False.

    Returns:
        Tensor, has the same data type as input tensor.

        - If `axis` is (), and `keep_dims` is False,
          the output is a 0-D tensor representing the product of all elements in the input tensor.
        - If `axis` is int, set as 1, and `keep_dims` is False,
          the shape of output is :math:`(x_0, x_2, ..., x_R)`.
        - If `axis` is tuple(int), set as (1, 2), and `keep_dims` is False,
          the shape of output is :math:`(x_0, x_3, ..., x_R)`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If `axis` is not one of the following: int, tuple or list.
        TypeError: If `keep_dims` is not a bool.
        ValueError: If `axis` is out of range.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
        >>> output = ops.prod(x, 1, keep_dims=True)
        >>> result = output.shape
        >>> print(result)
        (3, 1, 5, 6)
        >>> # case 1: Reduces a dimension by multiplying all elements in the dimension.
        >>> x = Tensor(np.array([[[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]],
        ...                      [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
        ...                      [[7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9]]]), mindspore.float32)
        >>> output = ops.prod(x)
        >>> print(output)
        2.2833798e+33
        >>> print(output.shape)
        ()
        >>> # case 2: Reduces a dimension along axis 0.
        >>> output = ops.prod(x, 0, True)
        >>> print(output)
        [[[ 28.  28.  28.  28.  28.  28.]
          [ 80.  80.  80.  80.  80.  80.]
          [162. 162. 162. 162. 162. 162.]]]
        >>> # case 3: Reduces a dimension along axis 1.
        >>> output = ops.prod(x, 1, True)
        >>> print(output)
        [[[  6.   6.   6.   6.   6.   6.]]
         [[120. 120. 120. 120. 120. 120.]]
         [[504. 504. 504. 504. 504. 504.]]]
        >>> # case 4: Reduces a dimension along axis 2.
        >>> output = ops.prod(x, 2, True)
        >>> print(output)
        [[[1.00000e+00]
          [6.40000e+01]
          [7.29000e+02]]
         [[4.09600e+03]
          [1.56250e+04]
          [4.66560e+04]]
         [[1.17649e+05]
          [2.62144e+05]
          [5.31441e+05]]]
    """
    return _get_cache_prim(P.ReduceProd)(keep_dims)(x, axis)


def _multi_svd_norm(x, row_axis, col_axis, op):
    y = moveaxis(x.astype(mstype.float32), (row_axis, col_axis), (-2, -1))
    if op == 'amax':
        return ops.svd(y, compute_uv=False).max(axis=-1)
    if op == 'amin':
        return ops.svd(y, compute_uv=False).min(axis=-1)
    if op == 'sum':
        return ops.svd(y, compute_uv=False).sum(axis=-1)
    raise ValueError(f"For svd_norm, the op input must be one of ['amax', 'amin', 'sum'], but got f{op}")


def normalize_axis_index(axis, ndim):
    # pylint: disable=chained-comparison
    if axis >= 0 and axis < ndim:
        return axis
    # pylint: disable=chained-comparison
    if axis < 0 and axis >= -ndim:
        return ndim + axis
    raise ValueError('For norm, the dim is out of range.')


def moveaxis(x, source, destination):
    destination = tuple([normalize_axis_index(ax, x.ndim) for ax in destination])
    source = tuple([normalize_axis_index(ax, x.ndim) for ax in source])
    perm = [n for n in range(x.ndim) if n not in source]
    for dest, src in sorted(zip(destination, source)):
        perm.insert(dest, src)
    perm = tuple(perm)
    return ops.transpose(x, perm)


@constexpr
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


@constexpr
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


def dot(a, b):
    """dot function"""
    res_dtype = _check_dtype(a.dtype, b.dtype)
    ndim_a, ndim_b = a.ndim, b.ndim
    if ndim_a == 0 or ndim_b == 0:
        return ops.tensor_mul(a, b)
    if ndim_a > 0 and ndim_b >= 2:
        perm = ops.make_range(ndim_b)
        perm = perm[:-2] + (perm[-1],) + (perm[-2],)
        b = ops.transpose(b, perm)

    if a.shape[-1] != b.shape[-1]:
        raise ValueError('shapes are not aligned')
    a_aligned = a.reshape(-1, a.shape[-1]).astype(mstype.float32)
    b_aligned = b.reshape(-1, b.shape[-1]).astype(mstype.float32)

    res = ops.matmul(a_aligned, b_aligned.T)
    res = res.reshape(a.shape[:-1] + b.shape[:-1])

    return res.astype(res_dtype)


def norm(A, ord=None, dim=None, keepdim=False, *, dtype=None):
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

    .. note::
        Currently, complex numbers are not supported.

    Args:
        A (Tensor): Tensor of shape :math:`(*, n)` or :math:`(*, m, n)` where * is zero or more batch dimensions.
        ord (Union[int, float, inf, -inf, 'fro', 'nuc'], optional): norm's mode. refer to the table above for
            behavior. Default: None.
        dim (Union[int, Tuple(int)], optional): calculate the dimension of vector norm or matrix norm. Default: None.

            - When `dim` is int, it will be calculated by vector norm.

            - When `dim` is a 2-tuple, it will be calculated by matrix norm.

            - If `dim` is None and `ord` is None, `A` will be flattened to 1D and the 2-norm
              of the vector will be calculated.

            - If `dim` is None and `ord` is not None, `A` must be 1D or 2D.

        keepdim (bool): whether the output Tensor retains the original dimension. Default: False.

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): When set, `A` will be converted to the specified type,
            `dtype`, before execution, and dtype of returned Tensor will also be `dtype`. Default: None.

    Returns:
        Tensor, the result of norm calculation on the specified dimension, `dim`, has the same dtype as `A`.

    Raises:
        ValueError: If `dim` is out of range.
        TypeError: If `dim` is neither an int nor a tuple of int.
        TypeError: If `A` is a vector and `ord` is a str.
        ValueError: If `A` is a matrices and `ord` is not in valid mode.
        ValueError: If `A` is a matrices and `ord` is an integer but not in [1, -1, 2, -2].
        ValueError: If two elements of `dim` is same after normalize.
        ValueError: If any elements of `dim` is out of range.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import mindspore.ops as ops
        >>> x = ops.arange(-12, 13, dtype=ms.float32)
        >>> y = x.reshape(5, 5)
        >>> print(ops.norm(x))
        36.05551
        >>> print(ops.norm(x, float('inf')))
        12.0
        >>> print(ops.norm(x, float('-inf')))
        0.0
        >>> print(ops.norm(x, 0))
        24.0
        >>> print(ops.norm(x, 1))
        156.0
        >>> print(ops.norm(x, -1))
        0.0
        >>> print(ops.norm(x, 2))
        36.05551
        >>> print(ops.norm(x, -2))
        0.0
        >>> print(ops.norm(x, 3))
        23.000631
        >>> print(ops.norm(x, -3))
        0.0
        >>> print(ops.norm(y))
        36.05551
        >>> print(ops.norm(y, 'fro'))
        36.05551
        >>> print(ops.norm(y, 'nuc'))
        42.42641
        >>> print(ops.norm(y, float('inf')))
        50.0
        >>> print(ops.norm(y, float('-inf')))
        6.0
        >>> print(ops.norm(y, 1))
        32.0
        >>> print(ops.norm(y, -1))
        30.0
        >>> print(ops.norm(y, 2))
        35.355343
        >>> m = ms.Tensor([[1., -1., 2.], [-2., 3., -4.]])
        >>> print(ops.norm(m, dim=0))
        [2.236068  3.1622777 4.472136 ]
        >>> print(ops.norm(m, dim=1))
        [2.4494898 5.3851647]
        >>> print(ops.norm(m, ord=1, dim=1))
        [4. 9.]
        >>> n = ops.arange(27, dtype=ms.float32).reshape(3, 3, 3)
        >>> print(ops.norm(n, dim=(1, 2)))
        [14.282857 39.76179  66.45299 ]
        >>> print(ops.norm(n[0, :, :]), ops.norm(n[1, :, :]), ops.norm(n[2, :, :]))
        14.282857 39.76179 66.45299
    """
    ndim = A.ndim
    dim, immediate = _check_axis(dim, ord, ndim)
    _check_ord(ord, dim)
    if dtype is not None:
        A = ops.cast(A, dtype)
    # Immediately handle some default, simple, fast, and common cases.
    if immediate:
        A = A.ravel()
        sqnorm = dot(A, A)
        ret = ops.sqrt(sqnorm)
        if keepdim:
            ret = ret.reshape(ndim * [1])
        return ret

    if isinstance(ord, int):
        if len(dim) == 2:
            row_axis, col_axis = dim
            row_axis = normalize_axis_index(row_axis, ndim)
            col_axis = normalize_axis_index(col_axis, ndim)
            if ord == 1:
                if col_axis > row_axis:
                    col_axis -= 1
                return A.abs().sum(row_axis).max(axis=col_axis)
            if ord == -1:
                if col_axis > row_axis:
                    col_axis -= 1
                return A.abs().sum(row_axis).min(axis=col_axis)
            if ord == 2:
                return _multi_svd_norm(A, row_axis, col_axis, 'amax')
            if ord == -2:
                return _multi_svd_norm(A, row_axis, col_axis, 'amin')
            raise ValueError(f"For norm, the ord {ord} are not support for matrices.")
        if len(dim) == 1:
            if ord == 0:
                return (A != 0).astype(A.dtype).sum(axis=dim, keepdims=keepdim)
            if ord > 0:
                _lp_norm = _get_cache_prim(ops.LpNorm)(dim, ord, keepdim)
                return _lp_norm(A)
            return ops.sum(ops.abs(A).pow(ord), dim=dim, keepdim=keepdim).pow(1.0 / ord)
    if len(dim) == 1:
        if ord == float('inf'):
            return ops.abs(A).max(dim, keepdim)
        if ord == -float('inf'):
            return ops.abs(A).min(dim, keepdim)
        if ord is None:
            # special case for speedup
            s = ops.conj(A) * A
            reduce_sum = _get_cache_prim(ops.ReduceSum)(keepdim)
            return ops.sqrt(reduce_sum(s, dim))
        # None of the str-type keywords for ord ('fro', 'nuc')
        # are valid for vectors
        absx = ops.abs(A)
        absx **= ord
        reduce_sum = _get_cache_prim(ops.ReduceSum)(keepdim)
        ret = reduce_sum(absx, dim)
        if isinstance(ord, Tensor):
            ret **= ops.reciprocal(ord)
        else:
            ret **= 1 / ord
        return ret
    if len(dim) == 2:
        row_axis, col_axis = dim
        row_axis = normalize_axis_index(row_axis, ndim)
        col_axis = normalize_axis_index(col_axis, ndim)
        if row_axis == col_axis:
            raise ValueError('For norm, the elements of dim can not be duplicate.')

        if ord == float('inf'):
            if row_axis > col_axis:
                row_axis -= 1
            ret = ops.reduce_sum(abs(A), col_axis).max(row_axis)
        elif ord == -float('inf'):
            if row_axis > col_axis:
                row_axis -= 1
            ret = ops.reduce_sum(abs(A), col_axis).min(row_axis)
        elif ord == 'fro':
            ret = ops.sqrt(ops.reduce_sum((ops.conj(A) * A), dim))
        elif ord == 'nuc':
            ret = _multi_svd_norm(A, row_axis, col_axis, 'sum')
        else:
            ret = ops.sqrt(ops.reduce_sum((ops.conj(A) * A), dim))
        if keepdim:
            ret_shape = list(A.shape)
            ret_shape[dim[0]] = 1
            ret_shape[dim[1]] = 1
            ret = ret.reshape(ret_shape)
        return ret
    return None


def lu_unpack(LU_data, LU_pivots, unpack_data=True, unpack_pivots=True):
    """
    Unpack the LU_data and LU_pivots from a LU factorization of a tensor.

    Args:
        LU_data (Tensor): The packed LU factorization data. A tensor of size [*, M, N], where * is batch
          dimensions, with data type int8, uint8, int16, int32, int64, float16, float32, float64. The dims of LU_data
          must be equal to or greater than 2.
        LU_pivots (Tensor): The packed LU factorization pivots. A tensor of size [*, min(M, N)], where * is
          batch dimensions, with data type int8, uint8, int16, int32, int64.
        unpack_data (bool): A flag indicating if the LU_data should be unpacked. If False, then the returned L and U
            are None. Default: True.
        unpack_pivots (bool): A flag indicating if the LU_pivots should be unpacked into a permutation matrix P. If
            False, then the returned P is None. Default: True.

    Returns:
        pivots (Tensor) - The permutation matrix of LU factorization. The shape is `[*, M, M]`, the dtype is
        same as `LU_data`.
        L (Tensor) - The L matrix  of LU factorization. The dtype is same as `LU_data`.
        U (Tensor) - The U matrix  of LU factorization. The dtype is same as `LU_data`.

    Raises:
        TypeError: If the dtype of `LU_data` is not one of the following: int8, uint8, int16, int32,
                   int64, float16, float32, float64.
        TypeError: If the dtype of `LU_pivots` is not one of the following: int8, uint8, int16, int32, int64.
        ValueError: If the dimension of `LU_data` is less than 2.
        ValueError: If the dimension of `LU_pivots` is less than 1.
        ValueError: If the size of the last dimension of LU_pivots is not equal to the minimum of the sizes of the last
                    two dimensions of LU_data.
        ValueError: If the batch dimensions of LU_data's does not match LU_pivots's batch dimensions.
        ValueError: On the CPU platform, if the value of `LU_pivots` are out of range[1, LU_data.shape[-2]).
        RuntimeError: On the Ascend platform, if the value of `LU_pivots` are out of range[1, LU_data.shape[-2]).

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> from mindspore.ops import functional as F
        >>> LU_data = Tensor(np.array([[[-0.3806, -0.4872,  0.5536],
        ...                             [-0.1287,  0.6508, -0.2396],
        ...                             [ 0.2583,  0.5239,  0.6902]],
        ...                            [[ 0.6706, -1.1782,  0.4574],
        ...                             [-0.6401, -0.4779,  0.6701],
        ...                             [ 0.1015, -0.5363,  0.6165]]]), mstype.float64)
        >>> LU_pivots = Tensor(np.array([[1, 3, 3],
        ...                              [2, 3, 3]]), mstype.int32)
        >>> pivots, L, U = F.lu_unpack(LU_data, LU_pivots)
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


def renorm(input_x, p, dim, maxnorm):
    """
    Renormalizes the sub-tensors along dimension `dim`, and each sub-tensor's p-norm should not exceed the
    'maxnorm'. The values of current sub-tensor don't need change if the p-norm of the sub-tensor is less than
    `maxnorm`. Otherwise the sub-tensor needs to be modified to the original value of the corresponding position
    divided by the p-norm of the substensor and then multiplied by `maxnorm`.

    Args:
        input_x (Tensor): A Tensor, types: float32 or float16.
        p (int): Power of norm calculation.
        dim (int): The dimension that expected to get the slice-tensor.
        maxnorm (float32): Max norm.

    Returns:
        Tensor, has the same dtype and shape as input_x.

    Raises:
        TypeError: If dtype of `p` is not int.
        TypeError: If dtype of `dim` is not int.
        TypeError: If dtype of `maxnorm` is not float32.
        ValueError: If the value of `p` less than 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]), mindspore.float32)
        >>> y = ops.renorm(x, p=1, dim=0, maxnorm=5.)
        >>> print(y)
        [[1.       1.        1.        ]
        [1.6666666 1.6666666 1.6666666 ]
        [1.6666667 1.6666667 1.6666667 ]]
    """
    renorm_ = _get_cache_prim(Renorm)(p, dim, maxnorm)
    return renorm_(input_x)


# TODO: remove comment
@constexpr
def _check_attr_dtype(param_name, input_dtype, allow_dtypes, cls_name):
    validator.check_value_type(param_name, input_dtype, allow_dtypes, cls_name)


# TODO: remove comment
@constexpr
def _check_positive_float(arg_value, arg_name, cls_name):
    validator.check_positive_float(arg_value, arg_name, cls_name)


# TODO: remove comment
@constexpr
def _check_int_range(arg_value, lower_limit, upper_limit, arg_name=None, prim_name=None):
    validator.check_int_range(arg_value, lower_limit,
                              upper_limit, Rel.INC_LEFT, arg_name, prim_name)


def gumbel_softmax(logits, tau=1, hard=False, dim=-1):
    r"""
    Returns the samples from the Gumbel-Softmax distribution and optionally discretizes. If `hard = True`, the returned
    samples will be one-hot, otherwise it will be probability distributions that sum to 1 across `dim`.

    Args:
        logits (Tensor): Unnormalized log probabilities. The data type must be float16 or float32.
        tau (float): The scalar temperature, which is a positive number. Default: 1.0.
        hard (bool): if `True`, the returned samples will be discretized as one-hot vectors, but will be differentiated
          as if it is the soft sample in autograd. Default: False.
        dim (int): Dim for softmax to compute. Default: -1.

    Returns:
        Tensor, has the same dtype and shape as `logits`.

    Raises:
        TypeError: If `logits` is not a Tensor.
        TypeError: If dtype of `logits` is not one of: float16, float32.
        TypeError: If `tau` is not an float.
        TypeError: If `hard` is not a bool.
        TypeError: If `dim` is not a int.
        ValueError: If If `tau` is not positive.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
        >>> output = ops.gumbel_softmax(input_x, 1.0, True, -1)
        >>> print(output.shape)
        (2, 3)
    """
    if not isinstance(logits, (Tensor, Tensor_)):
        raise TypeError("The input logits must be tensor")
    if logits.shape == ():
        raise ValueError("For gumbel_softmax, the 0-D input is not supported.")
    logits_dtype = _get_cache_prim(P.DType)()(logits)
    _check_input_dtype("logits", logits_dtype, [mstype.float16, mstype.float32], "gumbel_softmax")
    _check_attr_dtype("tau", tau, [float], "gumbel_softmax")
    _check_attr_dtype("hard", hard, [bool], "gumbel_softmax")
    _check_attr_dtype("dim", dim, [int], "gumbel_softmax")
    _check_positive_float(tau, "tau", "gumbel_softmax")
    if hard:
        _check_int_range(dim, -1, len(logits.shape), 'dim', "gumbel_softmax")
    else:
        _check_int_range(dim, -len(logits.shape),
                         len(logits.shape), 'dim', "gumbel_softmax")

    log_op = _get_cache_prim(P.Log)()
    const_op = _get_cache_prim(P.ScalarToTensor)()

    sample_shape = _get_cache_prim(P.Shape)()(logits)
    uniform = C.uniform(sample_shape, const_op(
        0.0, mstype.float32), const_op(1.0, mstype.float32))
    uniform = _get_cache_prim(P.Cast)()(uniform, logits_dtype)
    gumbel = neg_tensor(log_op(neg_tensor(log_op(uniform))))
    gumbel = (logits + gumbel) / tau
    y_soft = _get_cache_prim(P.Softmax)(dim)(gumbel)
    if hard:
        index = y_soft.argmax(axis=dim)
        y_hard = _get_cache_prim(P.OneHot)(dim)(index, sample_shape[dim], Tensor(1, logits_dtype),
                                                Tensor(0, logits_dtype))
        ret = y_hard - ops.stop_gradient(y_soft) + y_soft
    else:
        ret = y_soft
    return ret


def kaiser_window(window_length, periodic=True, beta=12.0):
    r"""
    Generates a Kaiser window, which is also known as the Kaiser-Bessel window.

    The Kaiser window is defined as

    .. math::
        w(n) = \frac{I_{0}\left( \beta\sqrt{1 - \frac{4n^{2}}{(M - 1)^{2}}} \right)}{I_{0}(\beta)}

    with

    .. math::
        - \frac{M - 1}{2} \leq n \leq \frac{M - 1}{2},

    where :math:`I_0` is the modified zeroth-order Bessel function.

    Args:
        window_length (int): Length of window.
        periodic (bool, optional): When set to True, generates a periodic window for spectral analysis.
            When set to False, generates a symmetric window for filter design. Default: True.
        beta (float, optional): Shape parameter, when `beta` gets large, the window narrows. Default: 12.0.

    Returns:
        Tensor, a Kaiser window.

    Raises:
        TypeError: If `window_length` or `beta` is not an integer.
        TypeError: If `periodic` is not a variable of Boolean type.
        ValueError: If `window_length` is negative.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> window_length = 5
        >>> out = ops.kaiser_window(window_length)
        >>> print(out.asnumpy())
        [5.27734413e-05 1.01719688e-01 7.92939834e-01 7.92939834e-01
         1.01719688e-01]
    """
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
    if not isinstance(periodic, bool):
        raise TypeError(
            f"For 'kaiser_window', 'periodic' must be a variable of Boolean type, but got {type(periodic)}"
        )
    if periodic:
        window_length = window_length + 1
    n = np.arange(0, window_length)
    alpha = (window_length - 1) / 2.0
    w = np.i0(
        beta * np.sqrt(1 - ((n - alpha) / alpha) ** 2.0)
    ) / np.i0(float(beta))
    out = Tensor(w[:-1]) if periodic else Tensor(w)
    return out


def stft(x, n_fft, hop_length=None, win_length=None, window=None, center=True,
         pad_mode="REFLECT", normalized=False, onesided=None, return_complex=None):
    r"""
    STFT segments the signal into narrow time intervals and takes the Fourier transform
    of each segment to quantify the change of a nonstationary signal’s frequency
    and phase content over time.

    Ignoring the optional batch dimension, this operation computes the following expression:

    .. math::

        X[\omega, m]=\sum_{k=0}^{\text {win_length-1 }}
        \text { window }[k] \text { input }[m \times \text { hop_length }+
        k] \exp \left(-j \frac{2 \pi \cdot \omega k}{\text { win_length }}\right)

    where :math:`m` is the index of the sliding window, and
    :math:`ω` is the frequency in range :math:`0 \leq \omega < \text{n\_fft}0≤ω<n_fft`.

    Args:
        x (Tensor): Time sequences of stft, must be either a 1-D time tensor or a 2-D tensor.
        n_fft (int): The size of Fourier transform.
        hop_length (int, optional): The distance between neighboring sliding window
            frames. Default: None(treated as equal to :math:`floor(n_fft / 4)`).
        hop_length (int, optional): the size of window frame and STFT filter.
            Default: None(treated as equal to `n_fft`).
        window (Tensor, optional): the optional window function, 1-D tensor of size `win_length`.
            Default: None(treated as window of all :math:`1` s). If `win_length` < `n_fft`,
            `window` will be padded on both sides with ones to length `n_fft` before it takes effect.
        center (bool, optional): whether to pad `x` on both sides. Default: True.
        pad_mode (str, optional): controls the padding method used when
            `center` is True. Default: 'REFLECT'.
        normalized (bool, optional): controls whether to return the normalized STFT results
             Default: False.
        onesided (bool, optional): controls whether to return half of results to
            avoid redundancy for real inputs.
            Default: None. True for real `x` and `window`, False otherwise.
        return_complex (bool, optional): whether to return a complex tensor, or
            a real tensor with an extra last dimension for the real and
            imaginary components.
            Default: None. True for complex `x` or `window`, False otherwise.

    Returns:
        - **output** (Tensor) - A tensor containing the STFT result.
            If `return_complex` is True, it returns a complex Tensor with shape :math:`(*, N, T)`.
            If `return_complex` is False, it returns a real Tensor with shape :math:`(*, N, T, 2)`.

            `N` is size of Fourier transform, it depends on parameter `onesided`:
            - If `onesided` is False, :math:`N = n_fft`.
            - If `onesided` is True, :math:`N = n_fft // 2 + 1`.

            `T` is the total number of frames used, calculated by this formula:
            :math:`T = 1 + (len - n_fft) / hop_length`, where `len` depends on parameter `center`:
            - If `center` is False, :math:`len = signal_length`.
            - If `center` is True, :math:`len = signal_length + (n_fft // 2) * 2`.
            where :math:`signal_length` is the signal length, it equals to :math:`x.shape[-1]`.

    Raises:
        TypeError: If `x` is not a 1-D or 2-D tensor.
        TypeError: If `window` is not a 1-D tensor.
        TypeError: If any one of `center` , `normalized` , `onesided`
            and `return_complex` is assigned a nonboolean value.
        TypeError: If `pad_mode` is is assigned a value that is not string.
        TypeError: If `n_fft` , `hop_length` or `hop_length` is not an int.

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
    if hop_length is None:
        hop_length = int(np.floor(n_fft / 4))
    if win_length is None:
        win_length = int(np.floor(n_fft))
    if window is None:
        window = P.Ones()((win_length,), mstype.float32)

    def _is_complex(x):
        dtype = P.DType()
        return dtype(x) in [mstype.complex64, mstype.complex128]

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


# TODO: remove comment
@constexpr
def _max(*args):
    """Returns the maximum value."""
    return max(*args)


# TODO: remove comment
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


@_primexpr
def _check_matmul_shapes(shape1, shape2, prim_name=None):
    """Checks shape1 and shape2 are valid to perform matmul, and returns output shape after broadcasting."""
    shape_out = list()
    r_shape1 = shape1[:-2]
    r_shape2 = shape2[:-2]
    max_len = max(len(r_shape1), len(r_shape2))
    for i in range(max_len):
        items = [it[i - max_len + len(it)] if i - max_len + len(it) >= 0 else 1 for it in (r_shape1, r_shape2)]
        max_size = max(items)
        shape_out.append(max_size)
    return tuple(shape_out)


@_primexpr
def _tile_size(shape, out_shape, ndim):
    """Returns tile_size such that shape*tile_size = out_shape"""
    size = [1] * ndim
    for idx, (i, j) in enumerate(zip(shape, out_shape)):
        if i != j:
            size[idx] = j
    return tuple(size)


@_primexpr
def _expand(x, ndim):
    """Expand x to ndim from axis, which can be 0 or -1."""
    rank_op = _get_cache_prim(P.Rank)()
    expand_dims_op = _get_cache_prim(P.ExpandDims)()
    while rank_op(x) < ndim:
        x = expand_dims_op(x, 0)
    return x


def _broadcast_to(x, shape_cur, shape_to, ndim_to):
    """Broadcasts x from shape_cur to shape_to."""
    tile_op = _get_cache_prim(P.Tile)()
    size = _tile_size(shape_cur, shape_to, ndim_to)
    return tile_op(x, size)


def matmul(x1, x2):
    """
    Returns the matrix product of two tensors.

    Note:
        Numpy arguments `out`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.
        On GPU, the supported dtypes are np.float16 and np.float32.
        On CPU, the supported dtypes are np.float16 and np.float32.

    Args:
        x1 (Tensor): Input tensor, scalar not allowed.
          The last dimension of `x1` must be the same size as the second last dimension of `x2`.
          And the shape of x1 and x2 could be broadcast.
        x2 (Tensor): Input tensor, scalar not allowed.
          The last dimension of `x1` must be the same size as the second last dimension of `x2`.
          And the shape of x1 and x2 could be broadcast.

    Returns:
        Tensor or scalar, the matrix product of the inputs. This is a scalar only
        when both `x1`, `x2` are 1-d vectors.

    Raises:
        ValueError: If the last dimension of `x1` is not the same size as the
            second-to-last dimension of `x2`, or if a scalar value is passed in.
        ValueError: If the shape of `x1` and `x2` could not broadcast together.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> # case 1 : Reasonable application of broadcast mechanism
        >>> x1 = Tensor(np.arange(2*3*4).reshape(2, 3, 4), mindspore.float32)
        >>> x2 = Tensor(np.arange(4*5).reshape(4, 5), mindspore.float32)
        >>> output = ops.matmul(x1, x2)
        >>> print(output)
        [[[  70.   76.   82.   88.   94.]
        [ 190.  212.  234.  256.  278.]
        [ 310.  348.  386.  424.  462.]]
        [[ 430.  484.  538.  592.  646.]
        [ 550.  620.  690.  760.  830.]
        [ 670.  756.  842.  928. 1014.]]]
        >>> print(output.shape)
        (2, 3, 5)
        >>> # case 2 : the rank of `x1` is 1
        >>> x1 = Tensor(np.ones([1, 2]), mindspore.float32)
        >>> x2 = Tensor(np.ones([2,]), mindspore.float32)
        >>> output = ops.matmul(x1, x2)
        >>> print(output)
        [2.]
        >>> print(output.shape)
        (1,)
    """
    dtype_op = _get_cache_prim(P.DType)()
    rank_op = _get_cache_prim(P.Rank)()
    shape_op = _get_cache_prim(P.Shape)()
    reshape_op = _get_cache_prim(P.Reshape)()

    dtype1 = dtype_op(x1)
    dtype2 = dtype_op(x2)
    if not _check_same_type(dtype1, dtype2):
        x1 = x1.astype(mstype.float32)
        x2 = x2.astype(mstype.float32)

    ndim1_orig, ndim2_orig = rank_op(x1), rank_op(x2)
    shape1_orig, shape2_orig = shape_op(x1), shape_op(x2)
    transpose_b = ndim2_orig == 1
    shape_backbone = _check_matmul_shapes(shape1_orig, shape2_orig, 'matmul')
    # infers the shape of the output
    shape_out = shape_backbone + _infer_shape_rem(shape1_orig, shape2_orig,
                                                  ndim1_orig, ndim2_orig, transpose_b)

    _matmul = _get_cache_prim(P.MatMul)(False, transpose_b)
    _batch_matmul = _get_cache_prim(P.BatchMatMul)(False, transpose_b)

    x1 = _expand(x1, 2)
    x2 = _expand(x2, 2)
    if rank_op(x2) == 2:
        if rank_op(x1) > 2:
            x1 = reshape_op(x1, (-1, shape1_orig[-1]))
        res = _matmul(x1, x2)
    else:
        # broadcasts x1.shape[:-2] with x2.shape[:-2]
        ndim_aligned = _max(ndim1_orig, ndim2_orig)
        x1 = _expand(x1, ndim_aligned)
        x2 = _expand(x2, ndim_aligned)
        shape1_aligned, shape2_aligned = shape_op(x1), shape_op(x2)
        x1 = _broadcast_to(x1, shape1_aligned[:-2], shape_backbone, ndim_aligned)
        x2 = _broadcast_to(x2, shape2_aligned[:-2], shape_backbone, ndim_aligned)
        res = _batch_matmul(x1, x2)

    return reshape_op(res, shape_out)


def inner(x, other):
    r"""
    Computes the dot product of 1D tensors. For higher dimensions, the result will be the summation of the elemental
    wise production along their last dimension.

    Note:
        If either `x` or `other` is a Tensor scalar, the result is equivalent to mindspore.mul(x, other).

    Args:
        x (Tensor): First input.
        other (Tensor): Second input.

    Returns:
        Tensor, the result of the inner product.

    Raises:
        ValueError: If neither `x` nor `other` is scalar, and the last dimension of the two input tensors do not match.

    Examples:
        >>> # case1: 2 1D tensors
        >>> x = ms.Tensor([1, 2, 3], mstype.float32)
        >>> y = ms.Tensor([4, 5, 6], mstype.float32)
        >>> output = ops.inner(x, y)
        >>> print(output)
        32
        >>> # case2: Tensor scalar and tensor
        >>> x = ms.Tensor([[[1, 2, 3], [3, 2, 1]], [[4, 5, 6], [4, 5, 6]]], mstype.float32)
        >>> y = ms.Tensor(2, mstype.float32)
        >>> output = ops.inner(x, y)
        >>> print(output)
        [[[ 2.  4.  6.]
          [ 6.  4.  2.]]
         [[ 8. 10. 12.]
          [ 8. 10. 12.]]]
        >>> # case3: Two tensors
        >>> x = ms.Tensor([[[1, 2, 3], [3, 2, 1]], [[4, 5, 6], [4, 5, 6]]], mstype.float32)
        >>> y = ms.Tensor([[2, 3, 4], [4, 3, 2]], mstype.float32)
        >>> output = ops.inner(x, y)
        >>> print(output)
        [[[20. 16.]
          [16. 20.]]
         [[47. 43.]
          [47. 43.]]]
    """
    x_dim = x.ndim
    other_dim = other.ndim
    x_dtype = x.dtype

    if x_dim == 0 or other_dim == 0:
        output = x * other
        if output.dtype != x.dtype:
            return output.astype(x_dtype)
        return output

    x_shape = x.shape
    other_shape = other.shape
    if x_shape[-1] != other_shape[-1]:
        raise ValueError(f"For 'inner', the last dimension of 'x' and 'other' must be the same, \
                         but got x.shape: {x_shape} and other.shape: {other_shape}.")
    return C.tensor_dot(x, other, axes=(-1, -1))


def bmm(input_x, mat2):
    """
    Computes matrix multiplication between two tensors by batch.

    .. math::

        \text{output}[..., :, :] = \text{matrix}(input_x[..., :, :]) * \text{matrix}(mat2[..., :, :])

    The dim of `input_x` can not be less than `3` and the dim of `mat2` can not be less than `2`.

    Args:
        input_x (Tensor): The first tensor to be multiplied. The shape of the tensor is :math:`(*B, N, C)`,
            where :math:`*B` represents the batch size which can be multidimensional, :math:`N` and :math:`C` are the
            size of the last two dimensions.
        mat2 (Tensor): The second tensor to be multiplied. The shape of the tensor is :math:`(*B, C, M)`.

    Returns:
        Tensor, the shape of the output tensor is :math:`(*B, N, M)`.

    Raises:
        ValueError: If dim of `input_x` is less than `3` or dim of `mat2` is less than `2`.
        ValueError: If the length of the third dim of `input_x` is not equal to
            the length of the second dim of `mat2`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.ones(shape=[2, 4, 1, 3]), mindspore.float32)
        >>> mat2 = Tensor(np.ones(shape=[2, 4, 3, 4]), mindspore.float32)
        >>> output = ops.bmm(input_x, mat2)
        >>> print(output)
        [[[[3. 3. 3. 3.]]
          [[3. 3. 3. 3.]]
          [[3. 3. 3. 3.]]
          [[3. 3. 3. 3.]]]
         [[[3. 3. 3. 3.]]
          [[3. 3. 3. 3.]]
          [[3. 3. 3. 3.]]
          [[3. 3. 3. 3.]]]]
    """
    if not (isinstance(input_x, Tensor) and isinstance(mat2, Tensor)):
        raise TypeError("For bmm op, inputs input_x and mat2 must be all tensors.")

    bmm_op = _get_cache_prim(P.BatchMatMul)()
    return bmm_op(input_x, mat2)


def quantile(x, q, axis=None, keepdims=False):
    r"""
    Computes the q-th quantiles of all elements in the input tensor, doing a linear interpolation when the
    q-th quantile lies between two data points.

    Args:
        x (Tensor): The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
                    Supported dtypes: float32, float64.
        q (float or Tensor): A scalar or 1D tensor of quantile values in the range [0, 1].
                            Supported dtypes: float32, float64.
        axis (int): The dimension to reduce. Default: None. By default,
                    axis is None resulting in the input tensor being flattened before computation.
        keepdims (bool): Whether the output tensor has dim retained or not. Default: False.

    Returns:
        Tensor, has the same dtype as the `input`.

        Assume the input shape is :math:`(m, x_0, x_1, ..., x_i, ..., X_R)`, axis = :math:`i` and m is
        the element count of input `q`.

        - If `q` is scalar and `keepdims` is True, the shape of output is :math:`(x_0, x_1, ..., 1, ..., X_R)`.
        - If `q` is scalar and `keepdims` is False, the shape of output is :math:`(x_0, x_1, ..., X_R)`.
        - If `q` is 1D Tensor and `keepdims` is True, the shape of output is :math:`(m, x_0, x_1, ..., 1, ..., X_R)`.
        - If `q` is 1D Tensor and `keepdims` is False, the shape of output is :math:`(m, x_0, x_1, ..., X_R)`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If `q` is not a Tensor or float.
        TypeError: If dtype of `input` is not float32 or float64.
        TypeError: If dtype of `q` is not float32 or float64.
        TypeError: If dtype of `input` and the dtype of `q` is different.
        ValueError: If the `q` values not in the range [0, 1].
        ValueError: If the `axis` values out of range.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.0700, -0.5446,  0.9214]), mindspore.float32)
        >>> q = Tensor(np.array([0, 0.5, 1]), mindspore.float32)
        >>> output = ops.quantile(x, q)
        >>> print(output.asnumpy())
        [-0.5446  0.07  0.9214]
    """

    if axis is not None:
        _check_attr_dtype("axis", axis, [int], "quantile")
    if keepdims is not None:
        _check_attr_dtype("keepdims", keepdims, [bool], "quantile")

    quantile_ = _get_cache_prim(Quantile)(dim=axis, keep_dims=keepdims)
    return quantile_(x, q)


def nanquantile(x, q, axis=None, keepdims=False):
    r"""
    This is a variant of mindspore.ops.quantile() that 'ignores' NaN values,
    computing the quantiles q as if NaN values in input did not exist.
    If all values in a reduced row are NaN then the quantiles for that reduction will be NaN.

    Refer to :func:`mindspore.ops.quantile` for more detail.

    Args:
        x (Tensor): The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
                    Supported dtypes: float32, float64.
        q (float or Tensor): A scalar or 1D tensor of quantile values in the range [0, 1].
                            Supported dtypes: float32, float64.
        axis (int): The dimension to reduce. Default: None. By default,
                    axis is None resulting in the input tensor being flattened before computation.
        keepdims (bool): Whether the output tensor has dim retained or not. Default: False.

    Returns:
        Tensor, has the same dtype as the `input`.

        Assume the input shape is :math:`(m, x_0, x_1, ..., x_i, ..., X_R)`, axis = :math:`i` and m is
        the element count of input `q`.

        - If `q` is scalar and `keepdims` is True, the shape of output is :math:`(x_0, x_1, ..., 1, ..., X_R)`.
        - If `q` is scalar and `keepdims` is False, the shape of output is :math:`(x_0, x_1, ..., X_R)`.
        - If `q` is 1D Tensor and `keepdims` is True, the shape of output is :math:`(m, x_0, x_1, ..., 1, ..., X_R)`.
        - If `q` is 1D Tensor and `keepdims` is False, the shape of output is :math:`(m, x_0, x_1, ..., X_R)`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If `q` is not a Tensor or float.
        TypeError: If dtype of `input` is not float32 or float64.
        TypeError: If dtype of `q` is not float32 or float64.
        TypeError: If dtype of `input` and the dtype of `q` is different.
        ValueError: If the `q` values not in the range [0, 1]
        ValueError: If the `axis` values out of range.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.0700, -0.5446,  0.9214]), mindspore.float32)
        >>> q = Tensor(np.array([0, 0.5, 1]), mindspore.float32)
        >>> output = ops.nanquantile(x, q)
        >>> print(output.asnumpy())
        [-0.5446  0.07  0.9214]
    """

    if axis is not None:
        _check_attr_dtype("axis", axis, [int], "nanquantile")
    if keepdims is not None:
        _check_attr_dtype("keepdims", keepdims, [bool], "nanquantile")

    quantile_ = _get_cache_prim(Quantile)(dim=axis, keep_dims=keepdims, ignore_nan=True)
    return quantile_(x, q)


def baddbmm(x, batch1, batch2, beta=1, alpha=1):
    r"""
    The result is the sum of the input and a batch matrix-matrix product of matrices in batch1 and batch2.
    The formula is defined as follows:

    .. math::
        \text{out}_{i} = \beta \text{input}_{i} + \alpha (\text{batch1}_{i} \mathbin{@} \text{batch2}_{i})

    Args:
        x (Tensor): The input Tensor. When batch1 is a (CxWxT) Tensor and batch2 is a (CxTxH) Tensor,
            x must be broadcastable with (CxWxH) Tensor.
        batch1 (Tensor): :math:`batch1` in the above formula. Must be 3-D Tensor, dtype is same as x.
        batch2 (Tensor): :math:`batch2` in the above formula. Must be 3-D Tensor, dtype is same as x.
        beta (Union[float, int], optional): multiplier for input. The default is 1.
        alpha (Union[float, int], optional): multiplier for `batch1 @ batch2`. The default is 1.
            Arguments beta and alpha must be integers when inputs of type not FloatTensor, otherwise they should
            be a real number.

    Returns:
        Tensor, has the same dtype as x, shape will be (CxWxH).

    Raises:
        TypeError: The type of `x`, `batch1`, `batch2` is not Tensor.
        TypeError: The types of `x`, `batch1`, `batch2` are different.
        TypeError: For inputs of type FloatTensor or DoubleTensor, \
                    arguments beta and alpha not be real numbers, otherwise not be integers.
        TypeError: For Baddbmm, attributes alpha and beta are not real numbers
        ValueError: If `batch1` and `batch2` are not 3-D tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input = Tensor(np.ones([1, 3, 3]).astype(np.float32))
        >>> batch1 = Tensor(np.ones([1, 3, 4]).astype(np.float32))
        >>> batch2 = Tensor(np.ones([1, 4, 3]).astype(np.float32))
        >>> output = ops.baddbmm(input, batch1, batch2)
        >>> print(output)
        [[[5. 5. 5.]
          [5. 5. 5.]
          [5. 5. 5.]]]
    """
    dtypeop = _get_cache_prim(P.DType)()
    bmmop = _get_cache_prim(P.BatchMatMul)(False, False)
    if not (isinstance(x, Tensor) and isinstance(batch1, Tensor) and isinstance(batch2, Tensor)):
        raise TypeError("For Baddbmm, inputs must be all tensors.")
    if len(batch1.shape) != 3 or len(batch2.shape) != 3:
        raise ValueError("For batch1 and batch2 must be 3-D tensors each containing the same number of matrices, "
                         f"but got length of batch1:'{len(batch1.shape)}', length of batch2:'{len(batch2.shape)}'.")
    input_dtype = dtypeop(x)
    if not (input_dtype == dtypeop(batch1) and input_dtype == dtypeop(batch2)):
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
    y = beta * x + alpha * (bmmop(batch1, batch2))
    return y


def log2(x):
    r"""
    Returns a new tensor with the logarithm to the base 2 of the elements of input.

    .. math::
        y_i = log_2(x_i)

    .. warning::
        If the input value of operator log2 is within the range (0, 0.01] or [0.95, 1.05], the output accuracy may
        be affected.

    .. note::
        The dimension of the input Tensor on Ascend should be less than or equal to 8, and the dimension of the
        input Tensor on the CPU or GPU should be less than 8.

    Args:
        x (Tensor): Input Tensor of any dimension. The value must be greater than 0.

    Returns:
        Tensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16 or float32 or float64 on CPU and GPU, if dtype of `x` is not float16
            or float32 on Ascend.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([2, 4, 8]).astype(np.float16))
        >>> output = ops.log2(x)
        >>> print(output)
        [1. 2. 3.]
    """

    dtype_op = _get_cache_prim(P.DType)()

    x_dtype = dtype_op(x)
    denominator = log_(_make_tensor(2, x_dtype))
    frac_log = log_(x)
    output = frac_log / denominator
    return output


def arrange(x):
    lists = []
    for i in range(0, x):
        lists.append(i)
    return lists


def rot90(x, k, dims):
    """
    Rotate a n-D tensor by 90 degrees in the plane specified by dims axis.
    Rotation direction is from the first towards the second axis if k > 0,
    and from the second towards the first for k < 0.

    Args:
        x (Tensor): Input tensor.
        k (int): Number of times to rotate. Default: 1.
        dims (Union[list(int), tuple(int)]): Axis to rotate. Default: [0,1].

    Returns:
        Tensor.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If `k` is not integer.
        TypeError: If `dims` is not tuple of integers or list of ints.
        ValueError: If the length of `dims` is not `2`.
        ValueError: If any dims is out of Tensor's range [-x.ndim, x.ndim).
        RuntimeError: If rotation dims are not different.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> x = Tensor(np.array([[0, 1], [2, 3]])).astype(np.float32)
        >>> k = 1
        >>> dims = [0, 1]
        >>> output = ops.rot90(x, k, dims)
        >>> print(output)
        [[1. 3.]
        [0. 2.]]
    """

    if not isinstance(x, (Tensor, Tensor_)):
        raise TypeError("the input x must be Tensor!")
    if not isinstance(k, int):
        raise TypeError("the input k must be int!")
    if not isinstance(dims, (list, tuple)):
        raise TypeError("the input dims must be list or tuple!")

    total_dims = x.ndim
    total_rot_dims = len(dims)

    if total_rot_dims != 2:
        raise ValueError("total rotation dims must be 2.")
    if dims[0] == dims[1] or (dims[0] - dims[1]) == total_dims or (dims[1] - dims[0]) == total_dims:
        raise RuntimeError("rotation dims must be different.")
    if dims[0] >= total_dims or dims[0] < -total_dims:
        raise ValueError("Rotation dim0 out of range, dim0 = {}".format(dims[0]))
    if dims[1] >= total_dims or dims[1] < -total_dims:
        raise ValueError("Rotation dim1 out of range, dim1 = {}".format(dims[1]))

    k = (4 + (k % 4)) % 4

    if k == 0:
        out = x
        return out
    if k == 2:
        op1 = P.ReverseV2(axis=[dims[0]])
        output = op1(x)
        op2 = P.ReverseV2(axis=[dims[1]])
        out = op2(output)
        return out

    axes_list = arrange(total_dims)
    (axes_list[dims[0]], axes_list[dims[1]]) = (axes_list[dims[1]],
                                                axes_list[dims[0]])

    if k == 1:
        op = P.ReverseV2(axis=[dims[1]])
        output = op(x)
        out = output.transpose(axes_list)
    else:
        output = x.transpose(axes_list)
        op = P.ReverseV2(axis=[dims[1]])
        out = op(output)
    return out


def roll(x, shifts, dims=None):
    """
    Rolls the elements of a tensor along an axis.

    Args:
        x (Tensor): Input tensor.
        shifts (Union[list(int), tuple(int), int]): Specifies the number of places by which elements are shifted
            positively (towards larger indices) along the specified dimension. Negative shifts will roll the elements
            in the opposite direction.
        dims (Union[list(int), tuple(int), int], optional): Specifies the dimension indexes of shape to be rolled.
            Default: None. If dims is None, the Tensor will be flattened before rolling and then restored to the
            original shape.

    Returns:
        Tensor, has the same shape and type as `x`.

    Raises:
        TypeError: If `shifts` is not an int, a tuple or a list.
        TypeError: If `dims` is not an int, a tuple or a list.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> import mindspore.ops as ops
        >>> from mindspore import Tensor
        >>> input_x = Tensor(np.array([0, 1, 2, 3, 4]).astype(np.float32))
        >>> output = ops.roll(input_x, shifts=2, dims=0)
        >>> print(output)
        [3. 4. 0. 1. 2.]
    """
    _shape = x.shape
    if dims is None:
        flatten_x = x.reshape(-1)
        return Roll(shifts, 0)(flatten_x).reshape(_shape)
    return Roll(shifts, dims)(x)


def xdivy(x, y):
    """
    Divides the first input tensor by the second input tensor element-wise. Returns zero when `x` is zero.

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be bool at the same time, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    .. note::
        When `x` and `y` are both of datatype complex, they should be both complex64 or complex128 at the same time.

    Args:
        x (Union[Tensor, Number, bool]):  Tensor of datatype number.Number或bool, or it can be a bool or number.
        y (Union[Tensor, Number, bool]): Tensor of datatype number.Number或bool, or it can be a bool or number.
            `x` and `y` can not be both bool at the same time.

    Returns:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If `x` and `y` is not one of the following: Tensor, Number, bool.
        TypeError: If dtype of `x` and 'y' is not in [float16, float32, float64, complex64, complex128, bool].
        ValueError: If `x` could not be broadcast to a tensor with shape of `y`.
        RuntimeError: If the data type of `x`, `y` conversion of Parameter is given
                      but data type conversion of Parameter is not supported.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([2, 4, -1]), mindspore.float32)
        >>> y = Tensor(np.array([2, 2, 2]), mindspore.float32)
        >>> output = ops.xdivy(x, y)
        >>> print(output)
        [ 1.   2.  -0.5]
    """
    return xdivy_(x, y)


def log10(x):
    r"""
    Returns a new tensor with the logarithm to the base 10 of the elements of input.

    .. math::
        y_i = log_{10}(x_i)

    .. warning::
        If the input value of operator log10 is within the range (0, 0.01] or [0.95, 1.05], the output accuracy may
        be affected.

    .. note::
        The dimension of the input Tensor on Ascend should be less than or equal to 8, and the dimension of the
        input Tensor on the CPU or GPU should be less than 8.

    Args:
        x (Tensor): Input Tensor of any dimension. The value must be greater than 0.

    Returns:
        Tensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16 or float32 or float64 on CPU and GPU, if dtype of `x` is not float16
            or float32 on Ascend.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([2, 4, 10]).astype(np.float16))
        >>> output = ops.log10(x)
        >>> print(output)
        [0.301 0.602 1.   ]
    """

    dtype_op = P.DType()

    x_dtype = dtype_op(x)
    denominator = log_(_make_tensor(10, x_dtype))
    frac_log = log_(x)
    output = frac_log / denominator
    return output


def log1p(x):
    r"""
    Returns the natural logarithm of one plus the input tensor element-wise.

    .. math::
        out_i = {log_e}(x_i + 1)

    Args:
        x (Tensor): The input tensor. With float16 or float32 data type.
            The value must be greater than -1.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions,
            its rank should be less than 8.

    Returns:
        Tensor, has the same shape as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
        >>> output = ops.log1p(x)
        >>> print(output)
        [0.6931472 1.0986123 1.609438 ]
    """
    _log1p = _get_cache_prim(P.Log1p)()
    return _log1p(x)


def kron(x, y):
    """
    Computes the Kronecker product, denoted by ⊗, of `x` and `y`.

    If `x` is a :math:`(a_{0}` x :math:`a_{1}` x ... x :math:`a_{n})` Tensor
    and `y` is a :math:`(b_{0}` x :math:`b_{1}` x ... x :math:`b_{n})` Tensor,
    the result will be a :math:`(a_{0}*b_{0}` x :math:`a_{1}*b_{1}` x ... x :math:`a_{n}*b_{n})`
    Tensor with the following entries:

    .. math::
            (x ⊗ y)_{k_{0},k_{1},...k_{n}} =
            x_{i_{0},i_{1},...i_{n}} * y_{j_{0},j_{1},...j_{n}},

    where :math:`k_{t} = i_{t} * b_{t} + j_{t}` for 0 ≤ `t` ≤ `n`. If one
    Tensor has fewer dimensions than the other it is unsqueezed
    until it has the same number of dimensions.

    Note:
        Supports real-valued and complex-valued inputs.

    Args:
        x (Tensor): Input Tensor, has the shape :math:`(r0, r1, ... , rN)`.
        y (Tensor): Input Tensor, has the shape :math:`(s0, s1, ... , sN)`.

    Returns:
        Tensor, has the shape :math:`(r0 * s0, r1 * s1, ... , rN * sN)`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If `y` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, nn
        >>> from mindspore import ops
        >>> x = Tensor(np.array([[0, 1, 2], [3, 4, 5]])).astype(np.float32)
        >>> y = Tensor(np.array([[-1, -2, -3], [-4, -6, -8]])).astype(np.float32)
        >>> output = ops.kron(x, y)
        >>> print(output)
        [[  0.   0.   0.  -1.  -2.  -3.  -2.  -4.  -6.]
         [  0.   0.   0.  -4.  -6.  -8.  -8. -12. -16.]
         [ -3.  -6.  -9.  -4.  -8. -12.  -5. -10. -15.]
         [-12. -18. -24. -16. -24. -32. -20. -30. -40.]]
    """

    if not isinstance(x, (Tensor, Tensor_)):
        raise TypeError("the input x must be Tensor!")
    if not isinstance(y, (Tensor, Tensor_)):
        raise TypeError("the input y must be Tensor!")
    if x is None or y is None:
        return None
    if x.ndim == 0 or y.ndim == 0:
        return x * y

    if x.ndim >= y.ndim:
        maxdim = x.ndim
    else:
        maxdim = y.ndim
    pad_x = maxdim - x.ndim
    pad_y = maxdim - y.ndim
    x_reshape = [0 for x in range(2 * maxdim)]
    y_reshape = [0 for x in range(2 * maxdim)]
    result_shape = [0 for x in range(maxdim)]

    for i in range(maxdim):
        if i >= pad_x:
            x_reshape[2 * i] = x.shape[i - pad_x]
        else:
            x_reshape[2 * i] = 1
        x_reshape[2 * i + 1] = 1
        y_reshape[2 * i] = 1
        if i >= pad_y:
            y_reshape[2 * i + 1] = y.shape[i - pad_y]
        else:
            y_reshape[2 * i + 1] = 1
        result_shape[i] = x_reshape[2 * i] * y_reshape[2 * i + 1]

    x = x.reshape(x_reshape)
    y = y.reshape(y_reshape)
    result = (x * y).reshape(result_shape)
    return result


def all(input, axis=None, keep_dims=False):
    r"""
    Reduces a dimension of `input` by the "logicalAND" of all elements in the dimension, by default. And also can
    reduce a dimension of `input` along the axis. Determine whether the dimensions of the output and input are the same
    by controlling `keep_dims`.

    Args:
        input (Tensor[bool]): The input Tensor. The dtype of the Tensor is bool.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.
        axis (Union[int, tuple(int), list(int)], optional): The dimensions to reduce.
            Only constant value is allowed. Supposed the rank of `input` is r,
            axis must be in the range [-rank(input), rank(input)). Default: None, all dimensions are reduced.
        keep_dims (bool, optional): If true, keep these reduced dimensions and the length is 1.
            If false, don't keep these dimensions. Default : False.

    Returns:
        Tensor, the dtype is bool.

        - If axis is None, and keep_dims is False,
          the output is a 0-D Tensor representing the "logical and" of all elements in the input Tensor.
        - If axis is int, set as 2, and keep_dims is False,
          the shape of output is :math:`(input_1, input_3, ..., input_R)`.
        - If axis is tuple(int), set as (2, 3), and keep_dims is False,
          the shape of output is :math:`(input_1, input_4, ..., input_R)`.

    Raises:
        TypeError: If `keep_dims` is not a bool.
        TypeError: If `input` is not a Tensor.
        TypeError: If `axis` is not one of the following: int, tuple or list.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[True, False], [True, True]]))
        >>> # case 1: Reduces a dimension by the "logicalAND" of all elements in the dimension.
        >>> output = ops.all(x, keep_dims=True)
        >>> print(output)
        [[False]]
        >>> print(output.shape)
        (1, 1)
        >>> # case 2: Reduces a dimension along axis 0.
        >>> output = ops.all(x, axis=0)
        >>> print(output)
        [True False]
        >>> # case 3: Reduces a dimension along axis 1.
        >>> output = ops.all(x, axis=1)
        >>> print(output)
        [False True]
    """
    if axis is None:
        axis = ()
    return _get_cache_prim(P.ReduceAll)(keep_dims)(input, axis)


def any(input, axis=None, keep_dims=False):
    r"""
    Reduces a dimension of `input` by the "logical OR" of all elements in the dimension, by default. And also can
    reduce a dimension of `input` along the axis. Determine whether the dimensions of the output and input are the same
    by controlling `keep_dims`.

    Args:
        input (Tensor[bool]): The input Tensor. The dtype of the Tensor is bool.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.
        axis (Union[int, tuple(int), list(int)], optional): The dimensions to reduce.
            Only constant value is allowed. Supposed the rank of `input` is r,
            axis must be in the range [-rank(input), rank(input)). Default: None, all dimensions are reduced.
        keep_dims (bool, optional): If true, keep these reduced dimensions and the length is 1.
            If false, don't keep these dimensions. Default : False.

    Returns:
        Tensor, the dtype is bool.

        - If axis is None, and keep_dims is False,
          the output is a 0-D Tensor representing the "logical or" of all elements in the input Tensor.
        - If axis is int, set as 2, and keep_dims is False,
          the shape of output is :math:`(input_1, input_3, ..., input_R)`.
        - If axis is tuple(int), set as (2, 3), and keep_dims is False,
          the shape of output is :math:`(input_1, input_4, ..., input_R)`.

    Raises:
        TypeError: If `keep_dims` is not a bool.
        TypeError: If `input` is not a Tensor.
        TypeError: If `axis` is not one of the following: int, tuple or list.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[True, False], [True, True]]))
        >>> # case 1: Reduces a dimension by the "logical OR" of all elements in the dimension.
        >>> output = ops.any(x, keep_dims=True)
        >>> print(output)
        [[ True]]
        >>> print(output.shape)
        (1, 1)
        >>> # case 2: Reduces a dimension along axis 0.
        >>> output = ops.any(x, axis=0)
        >>> print(output)
        [True True]
        >>> # case 3: Reduces a dimension along axis 1.
        >>> output = ops.any(x, axis=1)
        >>> print(output)
        [True True]
    """
    if axis is None:
        axis = ()
    return _get_cache_prim(P.ReduceAny)(keep_dims)(input, axis)


def remainder(x, y):
    r"""
    Computes the remainder of dividing the first input tensor by the second input tensor element-wise.

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar. When the inputs are two tensors,
    both dtypes cannot be bool, and the shapes of them could be broadcast. When the inputs are one tensor
    and one scalar, the scalar could only be a constant.

    .. math::

        out_{i} = input_{i} \text{ % } other_{i}

    .. warning::
        - When the elements of input exceed 2048, there might be accuracy problems.
        - The calculation results of this operator on Ascend and CPU might be inconsistent.
        - If shape is expressed as (D1,D2... ,Dn), then D1\*D2... \*DN<=1000000,n<=8.

    Args:
        x (Union[Tensor, numbers.Number, bool]): The first input is a number, a bool
            or a tensor whose data type is number.
        y (Union[Tensor, numbers.Number, bool]): When the first input is a tensor, The second input
            could be a number, a bool or a tensor whose data type is number.

    Returns:
        Tensor, the shape is the same as the one after broadcasting,
        and the data type is the one with higher precision or higher digits among the two inputs.

    Raises:
        TypeError: If neither `x` nor `y` is one of the following: Tensor, number, bool.
        ValueError: If the shape `x` and `y` cannot be broadcasted to each other.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([-4.0, 5.0, 6.0]).astype(np.float16))
        >>> y = Tensor(np.array([3.0, 2.0, 3.0]).astype(np.float16))
        >>> output = ops.remainder(x, y)
        >>> print(output)
        [2.  1.  0.]
    """

    out = x - tensor_floordiv(x, y) * y
    return out


def accumulate_n(x):
    r"""
    Computes accumulation of all input tensors element-wise.

    :func:`mindspore.ops.accumulate_n` is similar to :func:`mindspore.ops.addn`,
    but there is a significant difference between them: accumulate_n will not wait
    for all of its inputs to be ready before summing. That is to say, accumulate_n is able to save memory when inputs
    are ready at different time since the minimum temporary storage is proportional to the output size rather than the
    input size.

    Args:
        x (Union(tuple[Tensor], list[Tensor])): The input tuple or list is made up of multiple tensors whose dtype is
            number to be added together. Each element of tuple or list should have the same shape.

    Returns:
        Tensor, has the same shape and dtype as each entry of `x`.

    Raises:
        TypeError: If `x` is neither tuple nor list.
        ValueError: If there is an input element with a different shape.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> x = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> y = Tensor(np.array([4, 5, 6]), mindspore.float32)
        >>> output = ops.accumulate_n([x, y, x, y])
        >>> print(output)
        [10. 14. 18.]
    """

    accumulate_ = _get_cache_prim(P.AccumulateNV2)()
    return accumulate_(x)


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
        anchor_boxes (Tensor): Anchor boxes, tensor of shape (N, 4). "N" indicates the number of anchor boxes,
            and the value "4" refers to "x0", "y0", "x1", and "y1".
            Data type must be either float16， float32 or float64.
        gt_boxes (Tensor): Ground truth boxes, tensor of shape (M, 4). "M" indicates the number of ground
            truth boxes, and the value "4" refers to "x0", "y0", "x1", and "y1".
            Data type must be either float16, float32 or float64.
        mode (string): The mode is used to specify the calculation method,
            now supporting 'iou' (intersection over union) or 'iof' (intersection over foreground) mode.
            Default: 'iou'.

    Returns:
        Tensor, the 'iou' values, tensor of shape (M, N), with the same data type as `anchor_boxes`.

    Raises:
        KeyError: When `mode` is not 'iou' or 'iof'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
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
    return tuple([item for _ in range(obj)])


def _tuple_setitem(tup, idx, value):
    tup = list(tup)
    tup[idx] = value
    return tuple(tup)


def _check_dim_in_range(dim, ndim):
    if not isinstance(dim, int):
        raise TypeError(f'axes should be integers, not {type(dim)}')
    if -ndim > dim or dim >= ndim:
        raise ValueError(f'dim {dim} is out of bounds for array of dimension {ndim}')
    return dim % ndim


def dotrapezoid(y, dx, dim):
    y_left = select_(y, dim, 0)
    y_right = select_(y, dim, -1)
    y_sum = y.sum(dim)
    return (y_sum - (y_left + y_right) * 0.5) * dx


def dotrapezoid_tensor(y, dx, dim):
    y_start_dim_left = [0 for _ in range(dim)]
    y_start_dim_left = tuple(y_start_dim_left)
    y_start_dim_right = [0 for _ in range(y.ndim - dim - 1)]
    y_start_dim_right = tuple(y_start_dim_right)
    y_slice_size = _tuple_setitem(P.Shape()(y), dim, P.Shape()(y)[dim] - 1)
    y_slice_left = P.Slice()(y, y_start_dim_left + (0,) + y_start_dim_right, y_slice_size)
    y_slice_right = P.Slice()(y, y_start_dim_left + (1,) + y_start_dim_right, y_slice_size)
    return (P.Add()(y_slice_left, y_slice_right) * dx).sum(dim) / 2.


def add_padding_to_shape(curr_shape, target_n_dim):
    curr_size = len(curr_shape)
    if curr_size >= target_n_dim:
        target_n_dim = curr_size
    new_shape = [1 for _ in range(target_n_dim)]
    for i in range(curr_size):
        new_shape[target_n_dim - i - 1] = curr_shape[curr_size - i - 1]
    return new_shape


def zeros_like_except(y, dim):
    _check_dim_in_range(dim, y.ndim)
    dim = dim + y.ndim if dim < 0 else dim
    sizes = y.shape[:dim] + y.shape[dim + 1:]
    zeros = P.Zeros()(sizes, y.dtype)
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
        x_left = P.Slice()(x, x_start_dim_left + (0,) + x_start_dim_right, x_slice_size)
        x_right = P.Slice()(x, x_start_dim_left + (1,) + x_start_dim_right, x_slice_size)
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
    x_left = P.Slice()(x_viewed, x_start_dim_left + (0,) + x_start_dim_right, x_slice_size)
    x_right = P.Slice()(x_viewed, x_start_dim_left + (1,) + x_start_dim_right, x_slice_size)
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


def select_(feat, dim, index):
    select_shape = feat.shape
    select_shape = list(select_shape)
    select_shape[dim] = 1
    new_shape = feat.shape[:dim] + feat.shape[dim + 1:]
    indexes = P.Ones()(tuple(select_shape), mstype.int32) * (index)
    return feat.gather_elements(dim, indexes).reshape(new_shape)


def trapz(y, x=None, dx=1.0, dim=-1):
    r"""
    Integrates `y` (x) along given dim using trapezoidal rule.
    By default x-dim distances between points will be 1.0,
    alternatively they can be provided with `x` array or with `dx` scalar.

    .. math::

        \mathop{ \int }\nolimits_{{}}^{{}}{y}{ \left( {x} \right) } \text{d} x

    Args:
        y (Tensor): Input tensor to integrate.
        x (Tensor, optional): The sample points corresponding to the `y` values. If `x` is None,
            the sample points are assumed to be evenly spaced `dx` apart. Default: None.
            If `x` is not None, after subtracting 1 from the axis specified by `dim`, the shape of `x`
            should be same as `y` or can broadcast to `y`.
        dx (float, optional): The spacing between sample points when `x` is None. Default: 1.0.
        dim (int, optional): The dim along which to integrate. Default: -1.

    Returns:
        Tensor of float, definite integral as approximated by trapezoidal rule.
        If `y` is a one-dimensional array, the result is a floating-point number. If `y` is
        an n-dimensional array, the result is an N-1 dimensional array.

    Raises:
        RuntimeError: If dim of `x` is 1, and x.shape[0] is not equal to y.shape[dim].
        ValueError: If `dim` is out of range of :math:`[-y.ndim, y.ndim)`.
        TypeError: If `y` is not a Tensor.
        TypeError: If `x` is not None and is not a Tensor.
        TypeError: If `dx` is not a float number.
        TypeError: If `dim` is not a Integer.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> y = Tensor(np.array([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]).astype(np.float32))
        >>> x = Tensor(np.array([[1, 2, 3], [1, 3, 5], [1, 4, 7]]).astype(np.float32))
        >>> output = ops.trapz(y, x)
        >>> print(output)
        [2. 4. 6.]
    """

    if not isinstance(y, (Tensor, Tensor_)):
        raise TypeError("The input y must be Tensor.")
    if not isinstance(dx, float):
        raise TypeError("The input dx must be float.")
    if not isinstance(dim, int):
        raise TypeError("The input dim must be int.")
    if not _check_is_float(y.dtype):
        y = P.Cast()(y, mstype.float32)
    _check_dim_in_range(dim, y.ndim)
    dim = dim + y.ndim if dim < 0 else dim
    if x is None:
        return trapezoid(y, dx, dim)
    if not isinstance(x, (Tensor, Tensor_)):
        raise TypeError("The input x must be Tensor.")
    x = P.Cast()(x, mstype.float32)
    return trapezoid_tensor(y, x, dim)


def cholesky(input_x, upper=False):
    r"""
    Computes the Cholesky decomposition of a symmetric positive-definite matrix :math:`A`
    or for batches of symmetric positive-definite matrices.

    If `upper` is `True`, the returned matrix :math:`U` is upper-triangular, and the decomposition has the form:

    .. math::
        A = U^TU

    If `upper` is `False`, the returned matrix :math:`L` is lower-triangular, and the decomposition has the form:

    .. math::
        A = LL^T

    Args:
        input_x (Tensor): Tensor of shape :math:`(*, N, N)`, where :math:`*` is zero or more batch dimensions
            consisting of symmetric positive-definite matrices, with float32 or float64 data type.
        upper (bool): Flag that indicates whether to return a upper or lower triangular matrix.
            Default: False.

    Returns:
        Tensor, has the same shape and data type as `input_x`.

    Raises:
        TypeError: If `upper` is not a bool.
        TypeError: If dtype of `input_x` is not one of: float64, float32.
        TypeError: If `input_x` is not a Tensor.
        ValueError: If `input_x` is not a or a batch of square matrix.
        ValueError: If `input_x` is not symmetric positive definite.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[1.0, 1.0], [1.0, 2.0]]), mindspore.float32)
        >>> output = ops.cholesky(input_x, upper=False)
        >>> print(output)
        [[1. 0.]
         [1. 1.]]
    """
    cholesky_op = _get_cache_prim(P.Cholesky)(upper=upper)
    return cholesky_op(input_x)


def cholesky_inverse(input_x, upper=False):
    r"""
    Returns the inverse of the positive definite matrix using cholesky matrix factorization.

    If `upper` is `False`, :math:`U` is a lower triangular such that the output tensor is

    .. math::

        inv = (UU^{T})^{-1}

    If `upper` is `True`, :math:`U` is an upper triangular such that the output tensor is

    .. math::

        inv = (U^{T}U)^{-1}

    Note:
        The input must be either an upper triangular matrix or a lower triangular matrix.

    Args:
        input_x (Tensor): The input tensor with a rank of 2. Supported dtypes: float32, float64.
        upper(bool): Whether to return a lower or upper triangular matrix. Default: False.

    Returns:
        Tensor, has the same shape and dtype as `input_x`.

    Raises:
        TypeError: If `input_x` is not a Tensor.
        TypeError: If dtype of `input_x` is not one of: float32, float64.
        ValueError: If the dimension of `input_x` is not equal to 2.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[2,0,0], [4,1,0], [-1,1,2]]), mindspore.float32)
        >>> output = ops.cholesky_inverse(input_x)
        >>> print(output)
        [[ 5.8125 -2.625   0.625 ]
         [-2.625   1.25   -0.25  ]
         [ 0.625  -0.25    0.25  ]]
    """
    cholesky_inv_op = _get_cache_prim(P.CholeskyInverse)(upper=upper)
    return cholesky_inv_op(input_x)


def conj(input):
    r"""
    Returns a tensor of complex numbers that are the complex conjugate of each element in input.
    The complex numbers in input must be of the form a + bj, where a is the real part and b is the imaginary part.

    The complex conjugate returned by this operation is of the form a - bj.

    If `input` is real, it is returned unchanged.

    Args:
        input (Tensor): The input tensor to compute to. Must have numeric type.

    Returns:
        Tensor, has the same dtype as the `input`.

    Raises:
        TypeError: If the dtype of `input` is not a numeric type.
        TypeError: If the `input` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.asarray(np.complex(1.3+0.4j)), mindspore.complex64)
        >>> output = ops.conj(x)
        >>> print(output)
        (1.3-0.4j)
    """
    if not isinstance(input, (Tensor, Tensor_)):
        raise TypeError("For conj op, input must be Tensor.")
    return _get_cache_prim(P.Conj)()(input)


def cross(input, other, dim=None):
    r"""
    Returns the cross product of vectors in dimension `dim` of input `input` and `other`. `input` and `other` must
    have the same shape and the same type, and the size of their `dim` dimension should be `3`.
    If `dim` is not given, it defaults to the first dimension found with the size `3`.

    Args:
        input (Tensor): input is a tensor.
        other (Tensor):  The other Tensor, `other` must have the same shape and type as input `input`, and
            the size of their `dim` dimension should be `3`.
        dim (int): dimension to apply cross product in. Default: None.

    Returns:
        Tensor, has the same shape and type as input `input`.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If `other` is not a Tensor.
        TypeError: If the type of `input` is not the same as that of `other`.
        ValueError: If `input` and `other` not have the same size, and the size of their `dim` dimension not be `3`.
        ValueError: If `input` and `other` not have the same shape.
        ValueError: If `dim` is out of range, `dim` should be [-len(input.shape), len(input.shape)-1].

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> x = Tensor([1, 2, 3], mstype.int8)
        >>> other = Tensor([1, 2, 3], mstype.int8)
        >>> output = ops.cross(x, other)
        >>> print(output)
        [0 0 0]
    """
    if dim is None:
        dim = -65530
    cross_op = _get_cache_prim(P.Cross)(dim=dim)
    return cross_op(input, other)


def _einsum_convert_num_to_char(num):
    """For einsum, convert number into char."""
    if [num] == [Ellipsis]:
        return '...'
    # pylint: disable=chained-comparison
    if num >= 0 and num < 26:
        return chr(num + ord('A'))
    # pylint: disable=chained-comparison
    if num >= 26 and num < 52:
        return chr(num - 26 + ord('a'))
    raise ValueError(f"For Einsum, the number in sublist should be in range [0， 52), but got {num}")


def einsum(equation, *operands):
    r"""
    According to the Einstein summation Convention (Einsum),
    the product of the input tensor elements is summed along the specified dimension.
    You can use this operator to perform diagonal, reducesum, transpose, matmul, mul, inner product operations, etc.

    Note::
        The sublist format is alse supported. For example, ops.einsum(op1, sublist1, op2, sublist2, ..., sublist_out).
        In this format, equation can be derived by the sublists which are made up of Python's Ellipsis and list of
        integers in [0, 52). Each operand is followed by a sublist and an output sublist is at the end.

    Args:
        equation (str): Notation based on the Einstein summation convention, represent the operation you want to do.
            the value can contain only letters, commas, ellipsis and arrow.
            The letters represent input tensor dimension, commas represent separate tensors, ellipsis indicates
            the tensor dimension that you do not care about, the left of the arrow indicates the input tensors,
            and the right of it indicates the desired output dimension.
        operands (Tensor): Input tensor used for calculation. The dtype of the tensor must be the same.

    Returns:
        Tensor, the shape of it can be obtained from the `equation` , and the dtype is the same as input tensors.

    Raises:
        TypeError: If `equation` is invalid, or the `equation` does not match the input tensor.
        ValueError: If the number in sublist is not in [0, 52) in sublist format.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
        >>> equation = "i->"
        >>> output = ops.einsum(equation, x)
        >>> print(output)
        [7.]
        >>> x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
        >>> y = Tensor(np.array([2.0, 4.0, 3.0]), mindspore.float32)
        >>> equation = "i,i->i"
        >>> output = ops.einsum(equation, x, y)
        >>> print(output)
        [ 2. 8. 12.]
        >>> x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mindspore.float32)
        >>> y = Tensor(np.array([[2.0, 3.0], [1.0, 2.0], [4.0, 5.0]]), mindspore.float32)
        >>> equation = "ij,jk->ik"
        >>> output = ops.einsum(equation, x, y)
        >>> print(output)
        [[16. 22.]
         [37. 52.]]
        >>> x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mindspore.float32)
        >>> equation = "ij->ji"
        >>> output = ops.einsum(equation, x)
        >>> print(output)
        [[1. 4.]
         [2. 5.]
         [3. 6.]]
        >>> x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mindspore.float32)
        >>> equation = "ij->j"
        >>> output = ops.einsum(equation, x)
        >>> print(output)
        [5. 7. 9.]
        >>> x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mindspore.float32)
        >>> equation = "...->"
        >>> output = ops.einsum(equation, x)
        >>> print(output)
        [21.]
        >>> x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
        >>> y = Tensor(np.array([2.0, 4.0, 1.0]), mindspore.float32)
        >>> equation = "j,i->ji"
        >>> output = ops.einsum(equation, x, y)
        >>> print(output)
        [[ 2. 4. 1.]
         [ 4. 8. 2.]
         [ 6. 12. 3.]]
        >>> x = mindspore.Tensor([1, 2, 3, 4], mindspore.float32)
        >>> y = mindspore.Tensor([1, 2], mindspore.float32)
        >>> output = ops.einsum(x, [..., 1], y, [..., 2], [..., 1, 2])
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


def erfinv(input):
    r"""
    Computes the inverse error function of input. The inverse error function is defined in the range `(-1, 1)` as:

    .. math::

        erfinv(erf(x)) = x

    Args:
        input (Tensor): The input tensor to compute to, with data type float32, float16 or float64.

    Returns:
        Tensor, has the same shape and dtype as `input`.

    Raises:
        TypeError: If dtype of `input` is not float16, float32 or float64.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([0, 0.5, -0.9]), mindspore.float32)
        >>> output = ops.erfinv(x)
        >>> print(output)
        [ 0.          0.47695306 -1.1630805 ]
    """
    return _get_cache_prim(P.Erfinv)()(input)


def less_equal(input, other):
    r"""
    Computes the boolean value of :math:`input\_x <= other` element-wise.

    .. math::

        out_{i} =\begin{cases}
            & \text{True,    if } input\_x_{i}<=other_{i} \\
            & \text{False,   if } input\_x_{i}>other_{i}
            \end{cases}

    .. note::
        - Inputs of `input` and `other` comply with the implicit type conversion rules to make the data types
          consistent.
        - The inputs must be two tensors or one tensor and one scalar.
        - When the inputs are two tensors, dtypes of them cannot be both bool, and the shapes of them
          can be broadcast.
        - When the inputs are one tensor and one scalar, the scalar could only be a constant.

    Args:
        input (Union[Tensor, number.Number, bool]): The first input is a number.Number or
            a bool or a tensor whose data type is
            `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_ or
            `bool_ <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_.
        other (Union[Tensor, number.Number, bool]): The second input, when the first input is a Tensor,
            the second input should be a number.Number or bool value, or a Tensor whose data type is number or bool\_.
            When the first input is Scalar, the second input must be a Tensor whose data type is number or bool\_.

    Returns:
        Tensor, the shape is the same as the one after broadcasting, and the data type is bool.

    Raises:
        TypeError: If neither `input` nor `other` is a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1, 2, 3]), mindspore.int32)
        >>> other = Tensor(np.array([1, 1, 4]), mindspore.int32)
        >>> output = ops.less_equal(x, other)
        >>> print(output)
        [ True False  True]
    """
    return _get_cache_prim(P.LessEqual)()(input, other)


def cumprod(input, dim, dtype=None):
    r"""
    Computes the cumulative product of the `input` tensor along dimension `dim`.
    For example, if `input` is a vector of size `N`, the result will also be a vector of size `N`, with elements.

    .. math::

        y_i = x_1 * x_2 * x_3 * ... * x_i

    Args:
        input (Tensor[Number]): The input tensor.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.
        dim (int): The dimensions to compute the cumulative product. Only constant value is allowed.
        dtype: The desired data type of output. Default: None.

    Returns:
        Tensor, has the same shape and dtype as the `input` unless `dtype` is specified.

    Raises:
        TypeError: If `dim` is not an int.
        TypeError: If `dtype` conversion is not acceptable.
        ValueError: If `dim` is None.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1, 2, 3], np.float32))
        >>> output = ops.cumprod(x, 0)
        >>> print(output)
        [1. 2. 6.]
    """
    cumprod_op = _get_cache_prim(P.CumProd)()
    output = cumprod_op(input, dim)
    if dtype:
        output = _get_cache_prim(P.Cast)()(output, dtype)
    return output


def greater(input, other):
    r"""
    Computes the boolean value of :math:`input > other` element-wise.

    Args:
        input (Union[Tensor, number.Number, bool]): The first input is a number.Number or
            a bool or a tensor whose data type is
            `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_ or
            `bool_ <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_ .
        other (Union[Tensor, number.Number, bool]): The second input, when the first input is a Tensor,
            the second input should be a number.Number or bool value, or a Tensor whose data type is number or bool\_.
            When the first input is Scalar, the second input must be a Tensor whose data type is number or bool\_.

    Returns:
        Tensor, the shape is the same as the one after broadcasting, and the data type is bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1, 2, 3]), mindspore.int32)
        >>> y = Tensor(np.array([1, 1, 4]), mindspore.int32)
        >>> output = ops.greater(x, y)
        >>> print(output)
        [False True False]
    """
    greater_op = _get_cache_prim(P.Greater)()
    return greater_op(input, other)


def greater_equal(input, other):
    r"""
    Computes the boolean value of :math:`input >= other` element-wise.

    Args:
        input (Union[Tensor, number.Number, bool]): The first input is a number.Number or
            a bool or a tensor whose data type is
            `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_ or
            `bool_ <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_ .
        other (Union[Tensor, number.Number, bool]): The second input, when the first input is a Tensor,
            the second input should be a number.Number or bool value, or a Tensor whose data type is number or bool\_.
            When the first input is Scalar, the second input must be a Tensor whose data type is number or bool\_.

    Returns:
        Tensor, the shape is the same as the one after broadcasting, and the data type is bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1, 2, 3]), mindspore.int32)
        >>> y = Tensor(np.array([1, 1, 4]), mindspore.int32)
        >>> output = ops.greater_equal(x, y)
        >>> print(output)
        [True True False]
    """
    greater_equal_op = _get_cache_prim(P.GreaterEqual)()
    return greater_equal_op(input, other)


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
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        input (Tensor): The first input tensor. With type of float32 or float64.
        other (Tensor): The second input tensor. With float32 or float64 type. `other` should have
          the same dtype with `input`.

    Returns:
        Tensor, has the same dtype as `input` and `other`.

    Raises:
        TypeError: If `input` or `other` is not a Tensor.
        TypeError: If dtype of input `other` and a is not float32 nor float64.
        TypeError: If `other` has different dtype with `input`.
        ValueError: If `input` could not be broadcast to a tensor with shape of `other`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> a = Tensor(np.array([2.0, 4.0, 6.0, 8.0]).astype(np.float32))
        >>> x = Tensor(np.array([2.0, 3.0, 4.0, 5.0]).astype(np.float32))
        >>> output = ops.igamma(a, x)
        >>> print(output)
        [0.593994 0.35276785 0.21486944 0.13337152]
    """
    igamma_op = _get_cache_prim(Igamma)()
    return igamma_op(input, other)


def igammac(input, other):
    r"""
    Calculates upper regularized incomplete Gamma function.

    If we define `input` as `a` and `other` as `x`, the upper regularized incomplete Gamma function is defined as:

    .. math::
        Q(a, x) = Gamma(a, x) / Gamma(a) = 1 - P(a, x)

    where

    .. math::
        Gamma(a, x) = int_{x}^{\infty} t^{a-1} exp(-t) dt

    is the upper incomplete Gama function.

    Above :math:`P(a, x)` is the lower regularized complete Gamma function.

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        input (Tensor): The first input tensor. With type of float32 or float64.
        other (Tensor): The second input tensor. With float32 or float64 type. `other` should have
            the same dtype with `input`.

    Returns:
        Tensor, has the same dtype as `input` and `other`.

    Raises:
        TypeError: If `input` or `other` is not a Tensor.
        TypeError: If dtype of input `other` and a is not float32 nor float64.
        TypeError: If `other` has different dtype with `input`.
        ValueError: If `input` could not be broadcast to a tensor with shape of `other`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> a = Tensor(np.array([2.0, 4.0, 6.0, 8.0]).astype(np.float32))
        >>> x = Tensor(np.array([2.0, 3.0, 4.0, 5.0]).astype(np.float32))
        >>> output = ops.igammac(a, x)
        >>> print (output)
        [0.40600586 0.6472318 0.7851304 0.8666283]
    """
    igammac_op = _get_cache_prim(Igammac)()
    return igammac_op(input, other)


def lgamma(x):
    r"""
    Computes the natural logarithm of the absolute value of the gamma function on input.

    .. math::
        \text{out}_{i} = \ln \Gamma(|\text{input}_{i}|)

    Args:
        x (Tensor): The input tensor. With type of float16 or float32 or float64.

    Returns:
        Tensor, has the same dtype as `x`.

    Raises:
        TypeError: If x is not a Tensor.
        TypeError: If dtype of `x` is not float16 or float32 or float64.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.5, 3.2, 8.5]), mindspore.float32)
        >>> output = ops.lgamma(x)
        >>> print(output)
        [0.5723649 0.8854049 9.549267 ]
    """
    lgamma_op = _get_cache_prim(P.Lgamma)()
    return lgamma_op(x)


def digamma(x):
    r"""
    Computes the grad of the lgamma function on input.

    .. math::
        P(x) = grad(ln(gamma(x)))

    .. warning::
        This is an experimental prototype that is subject to change and/or deletion.

    Args:
        x (Tensor): The input tensor. With type of float16 or float32 or float64.

    Returns:
        Tensor, has the same dtype as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16 or float32 or float64.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1.5, 0.5, 9]).astype(np.float16))
        >>> output = ops.digamma(x)
        >>> print(output)
        [ 0.0365 -1.964   2.14  ]
    """
    digamma_op = _get_cache_prim(P.Digamma)()
    return digamma_op(x)


def polygamma(a, x):
    r"""
    Computes the :math:`a^{th}` derivative of the polygamma function on `x`.

    .. math::
        \psi^{(a)}(x) = \frac{d^{(a)}}{dx^{(a)}} \psi(x)

    Args:
        a (Tensor): the order of the polygamma function, types: int32, int64, the shape of a is 0.
        x (Tensor): the tensor to compute the polygamma function.

    Returns:
        Tensor, has the same dtype as `x`.

    Raises:
        TypeError: If x is not a Tensor.
        TypeError: If dtype of input x is not one of: float16, float32, float64.
        TypeError: If dtype of input a is not one of: int32, int64.
        TypeError: If shape of input a is not 0.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1.0, -0.5]), mindspore.float32)
        >>> a = Tensor(np.array(1), mindspore.int64)
        >>> polygamma = ops.Polygamma()
        >>> output = polygamma(a, x)
        >>> print(output)
        [1.644934 8.934802]
        >>> a = Tensor(np.array(2), mindspore.int64)
        >>> output = polygamma(a, x)
        >>> print(output)
        [-2.404114  -0.8287967]
        >>> a = Tensor(np.array(3), mindspore.int64)
        >>> output = polygamma(a, x)
        >>> print(output)
        [  6.4939404 193.40909  ]
        >>> a = Tensor(np.array(4), mindspore.int64)
        >>> output = polygamma(a, x)
        >>> print(output)
        [-24.886265   -3.4742498]
    """
    polygamma_op = _get_cache_prim(P.Polygamma)()
    return polygamma_op(a, x)


def isinf(input):
    r"""
    Determines which elements are inf or -inf for each position.

    .. math::

        out_i = \begin{cases}
        & \text{ if } x_{i} = \text{Inf},\ \ True \\
        & \text{ if } x_{i} \ne \text{Inf},\ \ False
        \end{cases}

    where :math:`Inf` means not a number.

    Args:
        input (Tensor): The input tensor.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Returns:
        Tensor, has the same shape of input, and the dtype is bool.

    Raises:
        TypeError: If `input` is not a Tensor.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([np.log(-1), 1, np.log(0)]), mindspore.float32)
        >>> output = ops.isinf(x)
        >>> print(output)
        [False False True]
    """
    isinf_op = _get_cache_prim(P.IsInf)()
    return isinf_op(input)


def _is_sign_inf(x, fn):
    """Tests element-wise for infinity with sign."""
    shape = x.shape
    zeros_tensor = _get_cache_prim(P.Zeros)()(shape, mstype.float32)
    ones_tensor = _get_cache_prim(P.Ones)()(shape, mstype.float32)
    is_inf = _get_cache_prim(P.IsInf)()(x)
    is_sign = fn(x, zeros_tensor)
    res = ops.select(is_inf, ones_tensor, zeros_tensor)
    res = ops.select(is_sign, res, zeros_tensor)
    return _get_cache_prim(P.Cast)()(res, mstype.bool_)


def isposinf(x):
    """
    Tests element-wise for positive infinity.

    Args:
        x (Tensor): Input values.

    Returns:
       Tensor, true where `x` is positive infinity, false otherwise.

    Raises:
        TypeError: If the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> output = ops.isposinf(Tensor([-float("inf"), float("inf"), 1.2], mstype.float32))
        >>> print(output)
        [False  True False]
    """
    if not isinstance(x, (Tensor, Tensor_)):
        raise TypeError(f"For isposinf, the input x must be a Tensor, but got {type(x)}")
    return _is_sign_inf(x, tensor_gt)


def isneginf(x):
    """
    Tests element-wise for negative infinity.

    Args:
        x (Tensor): Input Tensor.

    Returns:
       Tensor, true where `x` is negative infinity, false otherwise.

    Raises:
        TypeError: If the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> output = ops.isneginf(Tensor([-float("inf"), float("inf"), 1.2], mstype.float32))
        >>> print(output)
        [ True False False]
    """
    if not isinstance(x, (Tensor, Tensor_)):
        raise TypeError(f"For isneginf, the input x must be a Tensor, but got {type(x)}")
    return _is_sign_inf(x, tensor_lt)


def logical_xor(input, other):
    r"""
    Computes the "logical XOR" of two tensors element-wise.

    .. math::

        out_{i} = x_{i} \oplus y_{i}

    Args:
        input (Tensor): The first input is a tensor whose data type can be implicitly converted to bool.
        other (Tensor): The second input is a tensor whose data type can be implicitly converted to bool
            to compute XOR with the first input.

    Returns:
        Tensor, the shape is the same as the one after broadcasting, and the data type is bool.

    Raises:
        TypeError: If neither `input` nor `other` is a Tensor whose data type is bool.
        ValueError: If the shape of two inputs cannot be broadcast.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> x = Tensor(np.array([True, False, True]), mindspore.bool_)
        >>> y = Tensor(np.array([True, True, False]), mindspore.bool_)
        >>> output = ops.logical_xor(x, y)
        >>> print(output)
        [False True True]
    """
    if isinstance(input, Tensor) and input.dtype != mstype.bool_:
        input = input.astype(mstype.bool_)
    if isinstance(other, Tensor) and other.dtype != mstype.bool_:
        other = other.astype(mstype.bool_)
    logical_xor_op = _get_cache_prim(P.LogicalXor)()
    return logical_xor_op(input, other)


def imag(input):
    r"""
    Returns a new tensor containing imaginary value of the `input`.
    If `input` is real, it will return zeros.

    Args:
        input (Tensor): The input tensor to compute to.

    Returns:
        Tensor, the shape is the same as the `input`.

    Raises:
       TypeError: If `input` is not a Tensor.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.asarray(np.complex(1.3 + 0.4j)), mindspore.complex64)
        >>> output = ops.imag(x)
        >>> print(output)
        0.4
    """
    return _get_cache_prim(P.Imag)()(input)


def nansum(x, axis=None, keepdims=False, *, dtype=None):
    """
    Computes sum of `x` over a given dimension, treating NaNs as zero.

    Args:
        x (Tensor): The input Tensor.
        axis (Union[int, tuple(int)], optional): The dimensions to reduce. Supposed the rank of `x` is r,
            axis must be in the range [-rank(x), rank(x)). Default: None, all dimensions are reduced.
        keepdims (bool, optional): Whether the output Tensor keeps dimensions or not. Default: False.

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): The dtype of output Tensor. Default: None.

    Returns:
        Tensor, the sum of input `x` in the given dimension dim, treating NaNs as zero.

        - If axis is None, keepdims is False,
          the output is a 0-D Tensor representing the sum of all elements in the input Tensor.
        - If axis is int, set as 2, and keepdims is False,
          the shape of output is :math:`(x_1, x_3, ..., x_R)`.
        - If axis is tuple(int) or list(int), set as (2, 3), and keepdims is False,
          the shape of output is :math:`(x_1, x_4, ..., x_R)`.

    Raises:
        TypeError: If `x` is not Tensor.
        TypeError: If `keepdims` is not a bool.
        TypeError: If the dtype of `x` or `dtype` is complex type.
        ValueError: If 'axis' not in [-rank(`x`), rank(`x`)).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[float("nan"), 2, 3], [1, 2, float("nan")]]), mindspore.float32)
        >>> output1 = ops.nansum(x, axis=0, keepdims=False, dtype=mindspore.float32)
        >>> output2 = ops.nansum(x, axis=0, keepdims=True, dtype=mindspore.float32)
        >>> print(output1)
        [1. 4. 3.]
        >>> print(output2)
        [[1. 4. 3.]]
    """
    if not isinstance(x, Tensor):
        raise TypeError(f"For nansum, input must be Tensor, but got {type(x)}.")
    if x.is_complex():
        raise TypeError(f'For nansum, input are not supported complex type, but got {type(x)}.')
    if dtype is not None and dtype in mstype.complex_type:
        raise TypeError(f'For nansum, dtype not supported complex type, but got {dtype}.')
    if axis is None:
        axis = ()
    if x.dtype == mstype.bool_:
        x = x.astype(mstype.int64)
    is_nan = _get_cache_prim(P.IsNan)()(x)
    x = ops.masked_fill(x, is_nan, 0)
    x = _get_cache_prim(P.ReduceSum)(keepdims)(x, axis)
    if dtype is not None and x.dtype != dtype:
        x = x.astype(dtype)
    return x


def diag_embed(x, offset=0, dim1=-2, dim2=-1):
    r"""
    Creates a tensor with diagonals filled by `x`. The remaining elements are filled by 0.
    If the shape of `x` is :math:`[x_{0}, x_{1}, ..., x_{n-1}, x_{n}]`, the output shape is: the vector obtained
    by inserting :math:`x_{n}+|offset|` into the vector :math:`[x_{0}, x_{1}, ..., x_{n-1}]`
    at position `dim1` and `dim2`.

    Args:
        x (Tensor): Values to fill diagonal.
        offset (int): Offset of the diagonal. :math:`offset=0` refers to the main diagonal. If :math:`offset>0`,
            fill the diagonals that are `offset` units upward from the main diagonal. If :math:`offset<0`, fill the
            diagonals that are `offset` units downward from the main diagonal. Default: 0.
        dim1 (int): The first dimension in `x` with respect to which to fill diagonal. Default: -2.
        dim2 (int): The second dimension in `x` with respect to which to fill diagonal. Default: -1.

    Returns:
        Tensor, has the same dtype as `x`, but the shape of output is one dimension higher than the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not supported.
        TypeError: If `offset` is not an int.
        TypeError: If `dim1` is not an int.
        TypeError: If `dim2` is not an int.
        ValueError: If the dimension of input is not 1-6D.
        ValueError: If `dim1` is not in range of [-len(x.shape), len(x.shape)).
        ValueError: If `dim2` is not in range of [-len(x.shape), len(x.shape)).
        ValueError: If `dim1` and `dim2` are identical.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> x = Tensor(np.array([2,3,4]), mindspore.float32)
        >>> output = ops.diag_embed(x)
        >>> print(output)
        [[2. 0. 0.]
         [0. 3. 0.]
         [0. 0. 4.]]
    """

    transpose_op = Transpose()
    matrix_set_diag_op = MatrixSetDiagV3(align="LEFT_RIGHT")
    zeros = ops.Zeros()
    if not isinstance(x, (Tensor, Tensor_)):
        raise TypeError("For 'diag_embed', 'x' must be Tensor.")
    dtypeop = P.DType()
    input_dtype = dtypeop(x)
    if not (input_dtype in (mstype.int8, mstype.int16, mstype.int32, mstype.int64, mstype.uint8, mstype.uint16,
                            mstype.uint32, mstype.uint64, mstype.float16, mstype.float32, mstype.float64)):
        raise TypeError("For 'diag_embed', the dtype of 'x' must be int8, int16, int32, int64, "
                        f"uint8, uint16, uint32, uint64, float16, float32 or float64, but got '{input_dtype}'.")
    _check_attr_dtype("offset", offset, [int], "diag_embed")
    _check_attr_dtype("dim1", dim1, [int], "diag_embed")
    _check_attr_dtype("dim2", dim2, [int], "diag_embed")
    if len(x.shape) > 6:
        raise ValueError("For 'diag_embed', the dimension of 'x' must be 1-6D.")
    x_shape = x.shape
    output_dim = len(x_shape) + 1
    if dim1 < -output_dim or dim1 > (output_dim - 1):
        raise ValueError(f"For 'diag_embed', 'dim1' must be in range of [{-output_dim}, {output_dim - 1}), "
                         f"but got {dim1}.")
    if dim2 < -output_dim or dim2 > (output_dim - 1):
        raise ValueError(f"For 'diag_embed', 'dim2' must be in range of [{-output_dim}, {output_dim - 1}), "
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
    output = zeros(output_shape_trans, x.dtype)
    k = P.Cast()(offset, mstype.int32)
    output = matrix_set_diag_op(output, x, k)
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


def sum(x, dim=None, keepdim=False, *, dtype=None):
    """
    Calculate sum of Tensor elements over a given dim.

    Args:
        x (Tensor): The input tensor.
        dim (Union[None, int, tuple(int), list(int)]): Dimensions along which a sum is performed.
            If None, sum all the elements of the input tensor.
            If the `dim` is a tuple or list of ints, a sum is performed on all the dimensions specified in the tuple.
            Must be in the range :math:`[-x.ndim, x.ndim)` . Default: None.
        keepdim (bool): Whether the output tensor has dim retained or not.
            If True, keep these reduced dimensions and the length is 1.
            If False, don't keep these dimensions. Default: False.

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): The desired data type of returned Tensor. Default: None.

    Returns:
       A Tensor, sum of elements over a given dim in `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If `dim` is not an int, tulpe(int), list(int) or None.
        ValueError: If `dim` is not in the range :math:`[-x.ndim, x.ndim)` .
        TypeError: If `keepdim` is not a bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]],
        ...                      [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
        ...                      [[7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9]]]), mstype.float32)
        >>> out = ops.sum(x)
        >>> print(out)
        270.0
        >>> out = ops.sum(x, dim=2)
        >>> print(out)
        [[ 6. 12. 18.]
         [24. 30. 36.]
         [42. 48. 54.]]
        >>> out = ops.sum(x, dim=2, keepdim=True)
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
    """
    if not isinstance(x, Tensor):
        raise TypeError("For 'sum', 'x' must be Tensor.")
    if dim is not None:
        if not isinstance(dim, (int, tuple, list)):
            raise TypeError("For 'sum', 'dim' must be int, tuple(int), list(int) or None.")
    if not isinstance(keepdim, bool):
        raise TypeError("For 'sum', 'keepdim' must be bool.")

    reduce_sum = P.ReduceSum(keep_dims=keepdim)
    if dim is not None:
        out = reduce_sum(x, dim)
    else:
        out = reduce_sum(x)
    if dtype is not None:
        out = out.astype(dtype)
    return out


def tanhshrink(x):
    '''
    Tanhshrink Activation, :math:`Tanhshrink(x)=x-Tanh(x)` .
    See :class:`mindspore.nn.Tanhshrink` for more details.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> import mindspore.ops as ops
        >>> from mindspore import Tensor
        >>> import numpy as np
        >>> x = Tensor(np.array([1, 2, 3, 2, 1]), ms.float16)
        >>> output = ops.tanhshrink(x)
        >>> print(output)
        [0.2383 1.036  2.004  1.036  0.2383]
    '''
    if not isinstance(x, Tensor):
        raise TypeError(f"For tanhshrink, the input must be a Tensor, but got {type(x)}.")

    if x.dtype in mstype.int_type + mstype.uint_type:
        x = x.astype(mstype.float64)

    tanh_op = _get_cache_prim(P.Tanh)()
    return x - tanh_op(x)


def matrix_power(x, n):
    """
    Raises a square matrix to the (integer) power `n` .

    - When :math:`n=0` , returns the identity matrix, which has the same shape as `x` .
    - When :math:`n<0` and `x` is invertible, returns the inverse of `x` to the power of :math:`-n` .

    Args:
        x (Tensor): A 3-D Tensor. Supported data types are float16 and float32.
            The shape is :math:`(b, m, m)` , represents b m-D square matrices.
        n (int): The exponent, a required int.

    Returns:
        A 3-D Tensor. Data type and shape are the same as `x` 's.

    Raises:
        TypeError: If the data type of `n` is not int.
        TypeError: If the data type of `x` is neither float32 nor float16.
        TypeError: If `x` is not a Tensor.
        ValueError: If `x` is not a 3-D tensor.
        ValueError: If shape[1] and shape[2] of `x` are not the same.
        ValueError: If `n` is negative but got input `x` has singular matrices.

    Supported Platforms:
        ``CPU``

    Examples:
        >>> x = Tensor([[[0, 1], [-1, 0]], [[1, 0], [0, -1]]], dtype=ms.float32)
        >>> y = ops.matrix_power(x, 2)
        >>> print(y)
        [[[-1.  0.]
          [-0. -1.]]
         [[ 1.  0.]
          [ 0.  1.]]]
    """
    matrix_power_ops = _get_cache_prim(P.MatrixPower)(n=n)
    return matrix_power_ops(x)


__all__ = [
    'addn',
    'absolute',
    'abs',
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
    'neg_tensor',
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
    'mul',
    'multiply',
    'nan_to_num',
    'nansum',
    'digamma',
    'lgamma',
    'tensor_div',
    'div',
    'divide',
    'true_divide',
    'tensor_floordiv',
    'floor_div',
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
    'atleast_2d',
    'cartesian_prod',
    'atleast_3d',
    'view_as_real',
    'vstack',
    'combinations',
    'dist',
    'copysign',
    'hann_window',
    'log2',
    'slogdet',
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
    'baddbmm',
    'bmm',
    'trapz',
    'cholesky',
    'cholesky_inverse',
    'conj',
    'cosine_similarity',
    'cov',
    'cross',
    'einsum',
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
    'triu_indices'
]
__all__.sort()
