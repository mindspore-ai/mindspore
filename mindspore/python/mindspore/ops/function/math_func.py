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
from itertools import zip_longest
from collections import deque
import numpy as np
import mindspore.ops as ops
from mindspore.common import dtype as mstype
from mindspore.ops.primitive import constexpr
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops.operations._inner_ops import Cummin
from mindspore.ops.operations.math_ops import STFT
from mindspore.ops.operations.math_ops import ReduceStd
from mindspore.ops.operations.math_ops import Logit
from mindspore.ops.operations.math_ops import LuUnpack
from mindspore.ops.operations.math_ops import Roll
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
    Orgqr,
    Renorm,
    Hypot,
    Heaviside,
    Lcm,
    Gcd,
    Sinc,
    NanToNum,
    SparseSegmentMean,
    TriuIndices,
    InplaceIndexAdd,
    InplaceUpdateV2,
    Igamma,
    Igammac,
    Angle,
)
from mindspore.common.tensor import Tensor
from mindspore._checkparam import Validator as validator
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore._c_expression import Tensor as Tensor_


@constexpr
def _make_tensor(val, dtype):
    """Returns the tensor with value `val` and dtype `dtype`."""
    return Tensor(val, dtype)


@constexpr
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
not_equal = P.NotEqual()
size_ = P.Size()
transpose_ = P.Transpose()

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
            `number <https://www.mindspore.cn/docs/en/r2.0.0-alpha/api_python/mindspore.html#mindspore.dtype>`_ or
            `bool_ <https://www.mindspore.cn/docs/en/r2.0.0-alpha/api_python/mindspore.html#mindspore.dtype>`_.
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


def exp2(x):
    """
    Computes the base two exponential function of input.

    Calculates ``2^x``.

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor.

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
        TypeError: If the dtype of self Tensor is bool type.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> from mindspore import dtype as mstype
        >>> x = Tensor(np.array([-5.0, 1.5, 3.0, 100.0]), ms.float32)
        >>> print(ops.positive(x))
        [ -5.    1.5   3.  100. ]
    """
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

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32)
        >>> print(ops.numel(input_x))
        4
    """
    return size_(x)


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
            `number <https://www.mindspore.cn/docs/en/r2.0.0-alpha/api_python/mindspore.html#mindspore.dtype>`_ or
            `bool_ <https://www.mindspore.cn/docs/en/r2.0.0-alpha/api_python/mindspore.html#mindspore.dtype>`_.
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
        other (Union[Tensor, number.Number]): The tensor or number to be subtracted.

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
            `number <https://www.mindspore.cn/docs/en/r2.0.0-alpha/api_python/mindspore.html#mindspore.dtype>`_ or
            `bool_ <https://www.mindspore.cn/docs/en/r2.0.0-alpha/api_python/mindspore.html#mindspore.dtype>`_.
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
        ``Ascend`` ``CPU`` ``GPU``

    Examples:
        >>> x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
        >>> y = Tensor(np.array([4.0, 5.0, 6.0]), mindspore.float32)
        >>> output = ops.div(x, y)
        >>> print(output)
        [0.25 0.4 0.5]
    """
    if rounding_mode is not None and rounding_mode not in ['floor', 'trunc']:
        raise ValueError("For ops.div, rounding_mode value should be None, 'floor' or 'trunc'.")

    output = _get_cache_prim(P.Div)()(input, other)

    if rounding_mode == 'floor':
        output = _get_cache_prim(P.Floor)()(output)

    if rounding_mode == 'trunc':
        output = _get_cache_prim(P.Trunc)()(output)

    return output


def divide(x, other, *, rounding_mode=None):
    """
    Alias for :func:`mindspore.ops.div` .
    """
    return div(x, other, rounding_mode)


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
            `number <https://www.mindspore.cn/docs/en/r2.0.0-alpha/api_python/mindspore.html#mindspore.dtype>`_ or
            `bool_ <https://www.mindspore.cn/docs/en/r2.0.0-alpha/api_python/mindspore.html#mindspore.dtype>`_.
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
        TypeError: If dtype of `x` is not float16, float32 or float64 on GPU and CPU.
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
        ``Ascend`` ``CPU``

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

    Outputs:
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
        x (Tensor): The input tensor whose dtype is bool.
            :math:`(N,*)` where :math:`*` means,any number of additional dimensions.

    Returns:
        Tensor, the shape is the same as the `x`, and the dtype is bool.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not a bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([True, False, True]), mindspore.bool_)
        >>> output = ops.logical_not(x)
        >>> print(output)
        [False  True False]
    """
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
        x (Union[Tensor, bool]): The first input is a bool or a tensor whose data type is bool.
        y (Union[Tensor, bool]): The second input is a bool when the first input is a tensor or
            a tensor whose data type is bool.

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
        x (Union[Tensor, bool]): The first input is a bool or a tensor whose data type is bool.
        y (Union[Tensor, bool]): The second input is a bool when the first input is a tensor or
            a tensor whose data type is bool.

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
    return logical_and_(x, y)


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

        out_i = sinc(x_i)

    Args:
        x (Tensor): The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.

    Returns:
        Tensor, has the same shape as the `x`. The dtype of output is float32 when dtype of `x` is in
        [uint8, uint8, uint16, int16, uint32, int32, uint64, int64, bool]. Otherwise output has the
        same dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``CPU``

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
        ``Ascend`` ``CPU`` ``GPU``

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
    When the inputs are two tensors,
    dtypes of them cannot be bool at the same time, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    .. warning::
        - On Ascend, the data type of `x` and `y` must be float16 or float32.

    Args:
        x (Union[Tensor, number.Number, bool]): The first input is a number.Number or
            a bool or a tensor whose data type is
            `number <https://www.mindspore.cn/docs/en/r2.0.0-alpha/api_python/mindspore.html#mindspore.dtype>`_ or
            `bool_ <https://www.mindspore.cn/docs/en/r2.0.0-alpha/api_python/mindspore.html#mindspore.dtype>`_.
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


def arctan2(x, other):
    r"""
    For details, please refer to :func:`mindspore.ops.atan2`.
    """
    _atan2 = _get_cache_prim(P.Atan2)()
    return _atan2(x, other)


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


def atan2(x, y):
    r"""
    Returns arctangent of x/y element-wise.

    It returns :math:`\theta\ \in\ [-\pi, \pi]`
    such that :math:`x = r*\sin(\theta), y = r*\cos(\theta)`, where :math:`r = \sqrt{x^2 + y^2}`.

    Args of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    If they have different data types, the lower precision data type will be converted to
    the relatively highest precision data type.

    Args:
        x (Tensor): The input tensor.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
            The data type should be one of the following types: float16, float32, float64
        y (Tensor): The input tensor. It has the same shape with `x`.

    Returns:
        Tensor, the shape is the same as the one after broadcasting,and the data type is same as `x`.

    Raises:
        TypeError: If `x` or `y` is not a Tensor.
        RuntimeError: If the data type of `x` and `y` conversion of Parameter is required
                      when data type conversion of Parameter is not supported.

    Supported Platforms:
        ``Ascend`` ``CPU`` ``GPU``

    Examples:
        >>> x = Tensor(np.array([0, 1]), mindspore.float32)
        >>> y = Tensor(np.array([1, 1]), mindspore.float32)
        >>> output = ops.atan2(x, y)
        >>> print(output)
        [0.        0.7853982]
    """
    _atan2 = _get_cache_prim(P.Atan2)()
    return _atan2(x, y)


def bitwise_and(x, y):
    r"""
    Returns bitwise `and` of two tensors element-wise.

    .. math::

        out_i = x_{i} \wedge y_{i}

    Args of `x` and `y` comply with the implicit type conversion rules to
    make the data types consistent.
    If they have different data types, the lower priority data type will be converted to
    the relatively highest priority data type.

    Args:
        x (Tensor): The first input tensor with shape :math:`(N,*)` where :math:`*` means
            any number of additional dimensions. The supported data types are:
            int8, uint8, int16, uint16, int32, uint32, int64 and uint64.
        y (Tensor): The second input tensor with the same dtype as `x`.

    Returns:
        Tensor, has the same type as the `x`.

    Raises:
        TypeError: If `x` or `y` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([0, 0, 1, -1, 1, 1, 1]), mindspore.int16)
        >>> y = Tensor(np.array([0, 1, 1, -1, -1, 2, 3]), mindspore.int16)
        >>> output = ops.bitwise_and(x, y)
        >>> print(output)
        [ 0  0  1 -1  1  0  1]
    """
    return bitwise_and_(x, y)


def bitwise_or(x, y):
    r"""
    Returns bitwise `or` of two tensors element-wise.

    .. math::

        out_i = x_{i} \mid y_{i}

    Args of `x` and `y` comply with the implicit type conversion rules to
    make the data types consistent.
    If they have different data types, the lower priority data type will be converted to
    the relatively highest priority data type.

    Args:
        x (Tensor): The first input tensor with shape :math:`(N,*)` where :math:`*` means
            any number of additional dimensions. The supported data types are:
            int8, uint8, int16, uint16, int32, uint32, int64 and uint64.
        y (Tensor): The second input tensor with the same dtype as `x`.

    Returns:
        Tensor, has the same type as the `x`.

    Raises:
        TypeError: If `x` or `y` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([0, 0, 1, -1, 1, 1, 1]), mindspore.int16)
        >>> y = Tensor(np.array([0, 1, 1, -1, -1, 2, 3]), mindspore.int16)
        >>> output = ops.bitwise_or(x, y)
        >>> print(output)
        [ 0  1  1 -1 -1  3  3]
    """
    return bitwise_or_(x, y)


def bitwise_xor(x, y):
    r"""
    Returns bitwise `xor` of two tensors element-wise.

    .. math::

        out_i = x_{i} \oplus y_{i}

    Args of `x` and `y` comply with the implicit type conversion rules to
    make the data types consistent.
    If they have different data types, the lower priority data type will be converted to
    the relatively highest priority data type.

    Args:
        x (Tensor): The first input tensor with shape :math:`(N,*)` where :math:`*` means
            any number of additional dimensions. The supported data types are:
            int8, uint8, int16, uint16, int32, uint32, int64 and uint64.
        y (Tensor): The second input tensor with the same dtype as `x`.

    Returns:
        Tensor, has the same type as the `x`.

    Raises:
        TypeError: If `x` or `y` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([0, 0, 1, -1, 1, 1, 1]), mindspore.int16)
        >>> y = Tensor(np.array([0, 1, 1, -1, -1, 2, 3]), mindspore.int16)
        >>> output = ops.bitwise_xor(x, y)
        >>> print(output)
        [ 0  1  0  0 -2  3  2]
    """
    return bitwise_xor_(x, y)


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
        [1.26606588  1.06348337  1.06348337  1.26606588]
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
    return matrix_determinant_(x)


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
        ``CPU``

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


def truncate_div(x, y):
    """
    Divides the first input tensor by the second input tensor element-wise for integer types, negative numbers will
    round fractional quantities towards zero.

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
        ``Ascend`` ``CPU`` ``GPU``

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
        ``Ascend`` ``CPU`` ``GPU``

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
        by multiplying mantissas in input with powers of intger 2
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
            `number <https://www.mindspore.cn/docs/en/r2.0.0-alpha/api_python/mindspore.html#mindspore.dtype>`_ or
            `bool_ <https://www.mindspore.cn/docs/en/r2.0.0-alpha/api_python/mindspore.html#mindspore.dtype>`_.
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
            `number <https://www.mindspore.cn/docs/en/r2.0.0-alpha/api_python/mindspore.html#mindspore.dtype>`_ or
            `bool_ <https://www.mindspore.cn/docs/en/r2.0.0-alpha/api_python/mindspore.html#mindspore.dtype>`_ .
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
    return not_equal(x, y)


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
    Returns a new tensor with boolean elements representing whether each element of `x` is real-valued.
    All real value types are considered real numbers.
    A complex value is considered real when its imaginary part is 0.

    Inputs:
        - **x** (Tensor) - The input tensor.

    Outputs:
        Tensor, has the same shape of input, and the dtype is bool.

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
        [ True False  True]
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


def nan_to_num(x, nan=0.0, posinf=None, neginf=None):
    """
    Replaces `NaN`, positive infinity, and negative infinity values in the `x` with the values
    specified by `nan`, `posinf`, and `neginf`, respectively. By default, NaN is replaced by 0,
    positive infinity is replaced by the largest finite value representable by the x dtype,
    and negative infinity is replaced by the smallest finite value representable by the x dtype.

    Args:
        x (Tensor): The shape of tensor is :math:`(x_1, x_2, ..., x_R)`. With float32 or float16 data type.
        nan (float): The value to replace `NaN`. Default value is 0.0.
        posinf (float): If a Number, the value to replace positive infinity values with. If None, positive
          infinity values are replaced with the greatest finite value representable by `x`'s dtype.
          Default value is None.
        neginf (float): if a Number, the value to replace negative infinity values with. If None, negative
          infinity values are replaced with the lowest finite value representable by `x`'s dtype.
          Default value is None.

    Returns:
        Tensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16 or float32.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([float('nan'), float('inf'), -float('inf'), 3.14]), mindspore.float32)
        >>> output = ops.nan_to_num(x, 1.0, 2.0, 3.0)
        >>> print(output)
        [1.  2.  3.  3.14]
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
        >>> net = ops.Orgqr()
        >>> y = net(x, tau)
        >>> print(y)
        [[-0.54999995 -0.2128925   0.8137956 ]
         [ 0.47119996 -0.8752807   0.08240613]
         [ 0.69749993  0.42560163  0.57772595]]
    """

    orgqr_ = Orgqr()
    return orgqr_(x, tau)


def hypot(x, other):
    """
    Computes hypotenuse of input tensors element-wise as legs of a right triangle.
    The shape of two inputs should be broadcastable, and data type of them should be
    one of: float32, float64

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
        ``CPU``

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
        values (Tensor): The values to use where x is zero. Values can be broadcast with x.
            'x' should have the same dtype with 'values'.

    Returns:
        Tensor, has the same type as 'x' and 'values'.

    Raises:
        TypeError: If `x` or `values` is not Tensor.
        TypeError: If data type `x` and `values` is different.
        ValueError: If shape of two inputs are not broadcastable.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([-1.5, 0., 2.]))
        >>> values = Tensor(np.array([0.5]))
        >>> y = ops.heaviside(x, values)
        >>> print(y)
        [ 0.  0.5 1. ]
    """

    heaviside_ = Heaviside()
    return heaviside_(x, values)


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

    Outputs:
        out (Tensor, optional) : optional output matrix.

    Raises:
        TypeError: If `x1` is not a Tensor.
        TypeError: If `x2` is not a Tensor.
        ValueError: Expected 1-D input `x1`, but got n-D.
        ValueError: Expected 1-D input `x2`, but got n-D.

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

    If mat is a :math:`(N, M)` tensor, vec is a 1-D tensor of size :math:`M`, out will be 1-D of size :math:`N`.

    Args:
        mat (Tensor): Input matrix of the tensor. The shape of the tensor is :math:`(N, M)`.
        vec (Tensor): Input vector of the tensor. The shape of the tensor is :math:`(M,)`.

    Returns:
        Tensor, the shape of the output tensor is :math:`(N,)`.

    Raises:
        TypeError: If `mat`, `vec` is not a Tensor.
        ValueError: If `mat` is not a 2-D Tensor.
            If `vec` is not a 1-D Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> mat = Tensor(np.array([[3., 4.], [1., 6.], [1., 3.]]).astype(np.float32))
        >>> vec = Tensor(np.array([1., 2.]).astype(np.float32))
        >>> output = mv(mat, vec)
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
        ValueError: If `batch1`, `batch2` cannot apply batch matrix multiplication.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
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
    """
    matmul_op = _get_cache_prim(P.MatMul)()
    return beta * x + alpha * (matmul_op(mat1, mat2))


def addmv(x, mat, vec, beta=1, alpha=1):
    """
    Multiplies matrix `mat` and vector `vec`. The vector `x` is added to the final result.

    If mat is a :math:`(N, M)` tensor, vec is a 1-D tensor of size :math:`M`, then `x` must be broadcastable
    with a 1-D tensor of size :math:`N` and `out` will be 1-D tensor of size :math:`N`.

    The optional values `beta` and `alpha` are the matrix-vector product between `mat` and `vec` and the scale
    factor for the added tensor `x` respectively. If `beta` is 0, then `x` will be ignored.

    .. math::
        output = β x + α (mat @ vec)

    Args:
        x (Tensor): Vector to be added. The shape of the tensor is :math:`(N,)`.
        mat (Tensor): The first tensor to be multiplied. The shape of the tensor is :math:`(N, M)`.
        vec (Tensor): The second tensor to be multiplied. The shape of the tensor is :math:`(M,)`.
        beta (scalar[int, float, bool], optional): Multiplier for `x` (β). The `beta` must be int or
            float or bool, Default: 1.
        alpha (scalar[int, float, bool], optional): Multiplier for `mat` @ `vec` (α). The `alpha` must
            be int or float or bool, Default: 1.

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
        >>> output = addmv(x, mat, vec)
        >>> print(output)
        [30. 27.]
    """

    dtypeop = P.DType()
    input_dtype = dtypeop(x)
    if not (isinstance(x, Tensor) and isinstance(mat, Tensor) and isinstance(vec, Tensor)):
        raise TypeError("For Addmv, inputs must be all tensors.")
    if not (input_dtype == dtypeop(mat) and input_dtype == dtypeop(vec)):
        raise TypeError("For Addmv, the inputs should be the same dtype.")
    _check_input_1d(x.shape, "x", "Addmv")
    _check_input_1d(vec.shape, "vec", "Addmv")
    _check_input_2d(mat.shape, "mat", "Addmv")
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
    """
    _dtype = x.dtype
    _dim = x.ndim
    perm = [i for i in range(_dim)]
    perm[-2], perm[-1] = perm[-1], perm[-2]
    t = ops.transpose(x, tuple(perm))
    if _dtype in (mstype.complex64, mstype.complex128):
        return t.conj()
    return t


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
    _check_input_1d(vec1.shape, "vec1", "Addr")
    _check_input_1d(vec2.shape, "vec2", "Addr")
    _check_input_2d(x.shape, "x", "Addr")
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
        ``GPU`` ``CPU``

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

    Inputs:
        - **x1** (Tensor) - The first input tensor.
        - **x2** (Tensor) - The second input tensor.

    Outputs:
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
        ``GPU``

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

    Inputs:
        - x (Tensor) - x is a tensor.

    Outputs:
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
    if not isinstance(axis, int):
        raise TypeError(f"The date type of 'axis' must be Int, but got {axis}.")
    if axis < -len_axis or axis > len_axis:
        raise ValueError(f"The value of axis must be in [{-len_axis}, {len_axis}], but got {axis}.")
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

    Outputs:
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
    return [_expand(arr, 2) for arr in inputs]


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

    if not isinstance(x, Tensor):
        other = _type_convert(Tensor, other)
    other = _broadcast_to_shape(other, P.Shape()(x))

    if _check_same_type(P.DType()(x), mstype.bool_):
        raise TypeError("copysign does not accept dtype bool.")
    if _check_same_type(P.DType()(other), mstype.bool_):
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


def amin(x, axis=(), keep_dims=False):
    r"""
    Reduces all dimensions of a tensor by returning the minimum value in `x`, by default. And also can
    reduce a dimension of `x` along specified `axis`.  `keep_dims` determines whether the dimensions of
    output and input are the same.

    Args:
        x (Tensor[Number]): The input tensor. The dtype of the tensor to be reduced is number.
          :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.
        axis (Union[int, tuple(int), list(int)]): The dimensions to reduce. Default: (), reduce all dimensions.
          Only constant value is allowed. Assume the rank of `x` is r, and the value range is [-r,r)..
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
    return _get_cache_prim(P.ReduceMin)(keep_dims)(x, axis)


def amax(x, axis=(), keep_dims=False):
    r"""
    Reduces all dimensions of a tensor by returning the maximum value in `x`, by default. And also can
    reduce a dimension of `x` along specified `axis`.  `keep_dims` determines whether the dimensions of
    output and input are the same.

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
    return _get_cache_prim(P.ReduceMax)(keep_dims)(x, axis)


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


def norm(input_x, axis, p=2, keep_dims=False, epsilon=1e-12):
    r"""
    Returns the matrix norm or vector norm of a given tensor.

    .. math::
        output = sum(abs(input)**p)**(1/p)

    Args:
        input_x (Tensor): Input tensor. The dtype must be float32 or float16.
        axis (Union[int, list, tuple]): Specifies which dimension or dimensions of input to calculate the norm across.
        p (int): The order of norm. Default: 2. `p` is greater than or equal to 0.
        keep_dims (bool): Whether the output tensors have dim retained or not. Default: False.
        epsilon (float): A value added to the denominator for numerical stability. Default: 1e-12.

    Returns:
        Tensor, has the same dtype as `input`, which shape depends on the args axis. For example, if the size of input
        is (2, 3, 4), axis is [0, 1], Outputs' shape will be (4,).

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If dtype of `input` is not one of: float16, float32.
        TypeError: If `p` is not an int.
        TypeError: If `axis` is not an int, a tuple or a list.
        TypeError: If `axis` is a tuple or a list, but the element of `axis` is not an int.
        TypeError: If `keep_dims` is not a bool.
        TypeError: If `epsilon` is not a float.
        ValueError: If the element of `axis` is out of the range (-len(input.shape), len(input.shape)).
        ValueError: If the length of shape of `axis` is bigger than the length of shape of `input`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).astype(np.float32))
        >>> output = ops.norm(input_x, [0, 1], p=2)
        >>> print(output)
        [ 9.165152 10.954452]
    """
    lp_norm_inner = _get_cache_prim(P.LpNorm)(axis, p, keep_dims, epsilon)
    return lp_norm_inner(input_x)


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
        >>> pivots, L, U = F.lu_unpack(LU_data, LU_pivots, unpack_data, unpack_pivots)
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
    pivots, l, u = LuUnpack(LU_data, LU_pivots)
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
        ``Ascend`` ``CPU`` ``GPU``

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


@constexpr
def _check_attr_dtype(param_name, input_dtype, allow_dtypes, cls_name):
    validator.check_value_type(param_name, input_dtype, allow_dtypes, cls_name)


@constexpr
def _check_positive_float(arg_value, arg_name, cls_name):
    validator.check_positive_float(arg_value, arg_name, cls_name)


@constexpr
def _check_int_range(arg_value, lower_limit, upper_limit, arg_name=None, prim_name=None):
    validator.check_int_range(arg_value, lower_limit, upper_limit, Rel.INC_LEFT, arg_name, prim_name)


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
        _check_int_range(dim, -len(logits.shape), len(logits.shape), 'dim', "gumbel_softmax")

    log_op = _get_cache_prim(P.Log)()
    const_op = _get_cache_prim(P.ScalarToTensor)()

    sample_shape = _get_cache_prim(P.Shape)()(logits)
    uniform = C.uniform(sample_shape, const_op(0.0, mstype.float32), const_op(1.0, mstype.float32))
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


def stft(x, n_fft, hop_length=None, win_length=None, window=None, center=True,
         pad_mode="REFLECT", normalized=False, onesided=None, return_complex=None):
    r"""
    STFTs can be used as a way of quantifying the change
    of a nonstationary signal’s frequency and phase content over time.
    Ignoring the optional batch dimension, this method computes the following expression:
    math:`X[\omega, m]=\sum_{k=0}^{\text {win_length-1 }}
    \text { window }[k] \text { input }[m \times \text { hop_length }+
    k] \exp \left(-j \frac{2 \pi \cdot \omega k}{\text { win_length }}\right)`
    where m is the index of the sliding window, and
    math:`\omegaω` is the frequency math:`0 \leq \omega < \text{n\_fft}0≤ω<n_fft`
    Args:
        x (Tensor): Time sequence of stft, must be either a 1-D time tensor or a 2-D tensor.
        n_fft (int): The size of Fourier transform.
        hop_length (int, optional): The distance between neighboring sliding window
            frames. Default: ``None`` (treated as equal to ``floor(n_fft / 4)``).
        win_length (int, optional): the size of window frame and STFT filter.
            Default: ``None``  (treated as equal to :attr:`n_fft`).
        window (Tensor, optional): the optional window function.
            Default: ``None`` (treated as window of all :math:`1` s).
        center (bool, optional): whether to pad :attr:`input` on both sides.
            Default: ``True``.
        pad_mode (string, optional): controls the padding method used when
            :attr:`center` is ``True``. Default: ``"REFLECT"``.
        normalized (bool, optional): controls whether to return the normalized STFT results
             Default: ``False``.
        onesided (bool, optional): controls whether to return half of results to
            avoid redundancy for real inputs.
            Default: ``True`` for real :attr:`input` and :attr:`window`, ``False`` otherwise.
        return_complex (bool, optional): whether to return a complex tensor, or
            a real tensor with an extra last dimension for the real and
            imaginary components.
            Default: ``True`` for complex :attr:`input` or :attr:`window`, ``False`` otherwise.

    Returns:
        Tensor.

        - **output** (Tensor) - A tensor containing the STFT result with shape described above.

    Raises:
        TypeError: If the x is not a tensor.
        ValueError: If x arguments have values not specified above.

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


@constexpr
def _max(*args):
    """Returns the maximum value."""
    return max(*args)


@constexpr
def _min(*args):
    """Returns the minimum value."""
    return min(*args)


@constexpr
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


@constexpr
def _check_matmul_shapes(shape1, shape2, prim_name=None):
    """Checks shape1 and shape2 are valid to perform matmul, and returns output shape after broadcasting."""
    msg_prefix = f"For '{prim_name}', the" if prim_name else "The"
    ndim1, ndim2 = len(shape1), len(shape2)
    if ndim1 < 1 or ndim2 < 1:
        raise ValueError(f"{msg_prefix} dimension of input operands must be at least 1, but got "
                         f"the length of shape1: {ndim1}, the length of shape2: {ndim2}.")
    if ndim2 >= 2 and shape1[-1] != shape2[-2]:
        raise ValueError(f"{msg_prefix} shape1[-1] must be equal to shape2[-2] when the length of shape2 "
                         f"is greater than or equal to 2, but got shape1[-1]: {shape1[-1]}, "
                         f"shape2[-2]: {shape2[-2]}.")
    shape_out = deque()
    for items in zip_longest(reversed(shape1[:-2]), reversed(shape2[:-2]), fillvalue=1):
        max_size = max(items)
        for item in items:
            if item not in (1, max_size):
                raise ValueError(f"{msg_prefix} operands could not be broadcast together with shape1 {shape1} and "
                                 f"shape2 {shape2}.")
        shape_out.appendleft(max_size)
    return tuple(shape_out)


@constexpr
def _tile_size(shape, out_shape, ndim):
    """Returns tile_size such that shape*tile_size = out_shape"""
    size = [1] * ndim
    for idx, (i, j) in enumerate(zip(shape, out_shape)):
        if i != j:
            size[idx] = j
    return tuple(size)


@constexpr
def _check_need_broadcast(shape1, shape2):
    """Returns True if broadcast is necessary for batchmatmul."""
    return shape1[:-2] != shape2[:-2]


@constexpr
def _check_input_1d(input_shape, param_name, func_name):
    if len(input_shape) != 1:
        raise ValueError(f"{func_name} {param_name} should be 1d, but got shape {input_shape}")
    return True


@constexpr
def _check_input_2d(input_shape, param_name, func_name):
    if len(input_shape) != 2:
        raise ValueError(f"{func_name} {param_name} should be 2d, but got shape {input_shape}")
    return True


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


def baddbmm(x, batch1, batch2, beta=1, alpha=1):
    r"""
    Performs a batch matrix-matrix product of matrices in batch1 and batch2. input is added to the final result.
    The formula is defined as follows:

    .. math::
        \text{out}_{i} = \beta \text{input}_{i} + \alpha (\text{batch1}_{i} \mathbin{@} \text{batch2}_{i})

    Args:
        x (Tensor): The tensor to be added.
        batch1 (Tensor): The first batch of matrices to be multiplied.
        batch2 (Tensor): The second batch of matrices to be multiplied.
        beta (Union[float, int], optional): multiplier for input. The default is 1.
        alpha (Union[float, int], optional): multiplier for `batch1 @ batch2`. The default is 1.

    Returns:
        Tensor, the output tensor.

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
        If the input value of operator Log2 is within the range (0, 0.01] or [0.95, 1.05], the output accuracy may
        be affacted. If the input value of operator Log2 is less than or equal to 0, it will not raise Error.

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
        k (int): Number of times to rotate.
        dims (a list or tuple): Axis to rotate.

    Returns:
        Tensor.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If `k` is not integer.
        TypeError: If `dims` is not tuple of integers or list of ints.
        ValueError: If the length of `dims` is not `2`.
        ValueError: If any dims is out of range.
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
            Default: None. If `dims` is None, the Tensor will be flattened before rolling and then restored to the
            original shape.

    Returns:
        Tensor, has the same shape and type as `x`.

    Raises:
        TypeError: If `shifts` is not an int, a tuple or a list.
        TypeError: If `dims` is not an int, a tuple or a list.

    Supported Platforms:
        ``Ascend`` ``GPU``

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
        If the input value of operator Log10 is within the range (0, 0.01] or [0.95, 1.05], the output accuracy may
        be affacted. If the input value of operator Log10 is less than or equal to 0, it will not raise Error.

    .. note::
        The dimension of the input Tensor on Ascend should be less than or equal to 8, and the dimension of the
        input Tensor on the CPU or GPU should be less than 8.

    Args:
        x (Tensor): Input Tensor of any dimension. The value must be greater than 0.

    Returns:
        Tensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16 or float32 or float64 on CPU and GPU.
        TypeError: If dtype of `x` is not float16 or float32 on Ascend.

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

    If `x` is a :math:`(a_{0}` x :math:`a_{1}` x ... x :math:`a_{n})` tensor
    and `y` is a :math:`(b_{0}` x :math:`b_{1}` x ... x :math:`b_{n})` tensor,
    the result will be a :math:`(a_{0}*b_{0}` x :math:`a_{1}*b_{1}` x ... x :math:`a_{n}*b_{n})`
    tensor with the following entries:

    .. math::
            (x ⊗ y)_{k_{0},k_{1},...k_{n}} =
            x_{i_{0},i_{1},...i_{n}} * y_{j_{0},j_{1},...j_{n}},

    where :math:`k_{t} = i_{t} * b_{t} + j_{t}` for 0 ≤ `t` ≤ `n`. If one
    tensor has fewer dimensions than the other it is unsqueezed
    until it has the same number of dimensions.

    Note:
        Supports real-valued and complex-valued inputs.

    Args:
        x (Tensor): Input tensor.
        y (Tensor): Input tensor.

    Returns:
        Tensor.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If `y` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``CPU`` ``GPU``

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


def all(x, axis=(), keep_dims=False):
    r"""
    Reduces a dimension of a tensor by the "logicalAND" of all elements in the dimension, by default. And also can
    reduce a dimension of `x` along the axis. Determine whether the dimensions of the output and input are the same by
    controlling `keep_dims`.

    Args:
        x (Tensor[bool]): The input tensor. The dtype of the tensor to be reduced is bool.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.
        axis (Union[int, tuple(int), list(int)]): The dimensions to reduce. Default: (), reduce all dimensions.
            Only constant value is allowed. Must be in the range [-rank(x), rank(x)).
        keep_dims (bool): If true, keep these reduced dimensions and the length is 1.
                          If false, don't keep these dimensions. Default : False.

    Returns:
        Tensor, the dtype is bool.

        - If axis is (), and keep_dims is False,
          the output is a 0-D tensor representing the "logical and" of all elements in the input tensor.
        - If axis is int, set as 2, and keep_dims is False,
          the shape of output is :math:`(x_1, x_3, ..., x_R)`.
        - If axis is tuple(int), set as (2, 3), and keep_dims is False,
          the shape of output is :math:`(x_1, x_4, ..., x_R)`.

    Raises:
        TypeError: If `keep_dims` is not a bool.
        TypeError: If `x` is not a Tensor.
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

    return _get_cache_prim(P.ReduceAll)(keep_dims)(x, axis)


def any(x, axis=(), keep_dims=False):
    r"""
    Reduces a dimension of a tensor by the "logical OR" of all elements in the dimension, by default. And also can
    reduce a dimension of `x` along the axis. Determine whether the dimensions of the output and input are the same by
    controlling `keep_dims`.

    Args:
        x (Tensor[bool]): The input tensor. The dtype of the tensor to be reduced is bool.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.
        axis (Union[int, tuple(int), list(int)]): The dimensions to reduce. Default: (), reduce all dimensions.
            Only constant value is allowed. Must be in the range [-rank(x), rank(x)).
        keep_dims (bool): If true, keep these reduced dimensions and the length is 1.
                         If false, don't keep these dimensions. Default : False.

    Returns:
        Tensor, the dtype is bool.

        - If axis is (), and keep_dims is False,
          the output is a 0-D tensor representing the "logical or" of all elements in the input tensor.
        - If axis is int, set as 2, and keep_dims is False,
          the shape of output is :math:`(x_1, x_3, ..., x_R)`.
        - If axis is tuple(int), set as (2, 3), and keep_dims is False,
          the shape of output is :math:`(x_1, x_4, ..., x_R)`.

    Raises:
        TypeError: If `keep_dims` is not a bool.
        TypeError: If `x` is not a Tensor.
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

    return _get_cache_prim(P.ReduceAny)(keep_dims)(x, axis)


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
        - The input data does not support 0.
        - When the elements of input exceed 2048, the accuracy of operator cannot guarantee the requirement of
          double thousandths in the mini form.
        - Due to different architectures, the calculation results of this operator on NPU and CPU may be inconsistent.
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
        Computes the trapezoidal rule along dim.

        Integrates `y` (x) along given dim. By default x-dim distances between points will be 1.0,
        alternatively they can be provided with x array or with dx scalar.

        .. math::

            \mathop{ \int }\nolimits_{{}}^{{}}{y}{ \left( {x} \right) } \text{d} x

        Args:
            y (Tensor): Input tensor to integrate.
            x (Tensor, optional): The sample points corresponding to the `y` values. If `x` is None,
            the sample points are assumed to be evenly spaced `dx` apart. The default is None.
            If x is not None, after subtracting 1 from the axis specified by dim, the shape of x should be same
            or can be broadcast to y.
            dx (float, optional): The spacing between sample points when `x` is None. The default is 1.0.
            dim (int, optional): The dim along which to integrate. Defaults to -1.

        Returns:
            Tensor of float, definite integral as approximated by trapezoidal rule.
            If y is a one-dimensional array, the result is a floating-point number. If y is an n-dimensional array,
            the result is an N-1-dimensional array because the dimension associated with the axis has been deleted.

        Raises:
            ValueError: If dim is out of range of ``[-y.ndim, y.ndim)``.
            RuntimeError: If x's ndim is 1, and x's shape[0] is not equal to y's shape[dim].
            TypeError: If y is not a Tensor.
            TypeError: If x is not None and is not a Tensor.
            TypeError: If dx is not a float number.
            TypeError: If dim is not a Integer.

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
        ``GPU`` ``CPU``

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
        ``CPU``

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
        ``Ascend`` ``CPU`` ``GPU``

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
            `number <https://www.mindspore.cn/docs/en/r2.0.0-alpha/api_python/mindspore.html#mindspore.dtype>`_ or
            `bool_ <https://www.mindspore.cn/docs/en/r2.0.0-alpha/api_python/mindspore.html#mindspore.dtype>`_.
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
        ``Ascend`` ``CPU`` ``GPU``

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
            `number <https://www.mindspore.cn/docs/en/r2.0.0-alpha/api_python/mindspore.html#mindspore.dtype>`_ or
            `bool_ <https://www.mindspore.cn/docs/en/r2.0.0-alpha/api_python/mindspore.html#mindspore.dtype>`_ .
        other (Union[Tensor, number.Number, bool]): The second input, when the first input is a Tensor,
            the second input should be a number.Number or bool value, or a Tensor whose data type is number or bool\_.
            When the first input is Scalar, the second input must be a Tensor whose data type is number or bool\_.

    Returns:
        Tensor, the shape is the same as the one after broadcasting, and the data type is bool.

    Supported Platforms:
        ``Ascend`` ``CPU`` ``GPU``

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
            `number <https://www.mindspore.cn/docs/en/r2.0.0-alpha/api_python/mindspore.html#mindspore.dtype>`_ or
            `bool_ <https://www.mindspore.cn/docs/en/r2.0.0-alpha/api_python/mindspore.html#mindspore.dtype>`_ .
        other (Union[Tensor, number.Number, bool]): The second input, when the first input is a Tensor,
            the second input should be a number.Number or bool value, or a Tensor whose data type is number or bool\_.
            When the first input is Scalar, the second input must be a Tensor whose data type is number or bool\_.

    Returns:
        Tensor, the shape is the same as the one after broadcasting, and the data type is bool.

    Supported Platforms:
        ``Ascend`` ``CPU`` ``GPU``

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
        ``Ascend`` ``CPU`` ``GPU``

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
        ``Ascend`` ``CPU`` ``GPU``

    Examples:
        >>> a = Tensor(np.array([2.0, 4.0, 6.0, 8.0]).astype(np.float32))
        >>> x = Tensor(np.array([2.0, 3.0, 4.0, 5.0]).astype(np.float32))
        >>> output = ops.igammac(a, x)
        >>> print (output)
        [0.40600586 0.6472318 0.7851304 0.8666283]
    """
    igammac_op = _get_cache_prim(Igammac)()
    return igammac_op(input, other)


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
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([np.log(-1), 1, np.log(0)]), mindspore.float32)
        >>> output = ops.isinf(x)
        >>> print(output)
        [False False True]
    """
    isinf_op = _get_cache_prim(P.IsInf)()
    return isinf_op(input)


def logical_xor(input, other):
    r"""
    Computes the "logical XOR" of two tensors element-wise.

    .. math::

        out_{i} = x_{i} \oplus y_{i}

    Args:
        input (Tensor): The first input is a tensor whose data type is bool.
        other (Tensor): The second input is a tensor to compute XOR with the first input.
          Datatype must be bool.

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
        ``CPU`` ``GPU``

    Examples:
        >>> x = Tensor(np.asarray(np.complex(1.3 + 0.4j)), mindspore.complex64)
        >>> output = ops.imag(x)
        >>> print(output)
        0.4
    """
    return _get_cache_prim(P.Imag)()(input)


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
    'neg_tensor',
    'neg',
    'negative',
    'tensor_lt',
    'less',
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
    'tensor_div',
    'div',
    'divide',
    'true_divide',
    'tensor_floordiv',
    'floor_div',
    'floordiv',
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
    'log',
    'log_matrix_determinant',
    'matrix_determinant',
    'linspace',
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
    'logsumexp',
    'ldexp',
    'sqrt',
    'square',
    'sin',
    'cos',
    'tan',
    'asin',
    'acos',
    'arccos',
    'atan',
    'sinh',
    'cosh',
    'tanh',
    'asinh',
    'acosh',
    'atanh',
    'atan2',
    'round',
    'bitwise_and',
    'bitwise_or',
    'bitwise_xor',
    'inv',
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
    'matmul',
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
    'atleast_2d',
    'vstack',
    'copysign',
    'log2',
    'xlogy',
    'log10',
    'log1p',
    'approximate_equal',
    'frac',
    'kron',
    'rot90',
    'remainder',
    'accumulate_n',
    'iou',
    'baddbmm',
    'bmm',
    'trapz',
    'cholesky',
    'cholesky_inverse',
    'conj',
    'cross',
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
    'matrix_exp'
]
__all__.sort()
