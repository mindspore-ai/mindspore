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

from mindspore.ops import operations as P
from ..operations.math_ops import (BesselJ0, BesselJ1, BesselK0, BesselK0e, BesselY0, BesselY1, BesselK1,
                                   BesselK1e)

#####################################
# Public Operation Functions.
#####################################
absolute = P.Abs()
tensor_add = P.Add()
neg_tensor = P.Neg()
tensor_sub = P.Sub()
tensor_mul = P.Mul()
tensor_div = P.RealDiv()
tensor_floordiv = P.FloorDiv()
floordiv = tensor_floordiv
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

#####################################
# Private Operation Functions.
#####################################
addn_ = P.AddN()
log_ = P.Log()
floor_ = P.Floor()
logical_not_ = P.LogicalNot()
logical_or_ = P.LogicalOr()
logical_and_ = P.LogicalAnd()
sin_ = P.Sin()
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
atan2_ = P.Atan2()
bitwise_and_ = P.BitwiseAnd()
bitwise_or_ = P.BitwiseOr()
bitwise_xor_ = P.BitwiseXor()
invert_ = P.Invert()
erf_ = P.Erf()
erfc_ = P.Erfc()
bessel_j1_ = BesselJ1()
bessel_j0_ = BesselJ0()
bessel_i0_ = P.BesselI0()
bessel_i0e_ = P.BesselI0e()
bessel_k0_ = BesselK0()
bessel_k0e_ = BesselK0e()
bessel_y0_ = BesselY0()
bessel_y1_ = BesselY1()
bessel_i1_ = P.BesselI1()
bessel_i1e_ = P.BesselI1e()
bessel_k1_ = BesselK1()
bessel_k1e_ = BesselK1e()
equal_ = P.Equal()
isfinite_ = P.IsFinite()
isnan_ = P.IsNan()
same_type_shape_ = P.SameTypeShape()
maximum_ = P.Maximum()
minimum_ = P.Minimum()
lerp_ = P.Lerp()
tensor_round_ = P.Round()
matrix_determinant_ = P.MatrixDeterminant()
log_matrix_determinant_ = P.LogMatrixDeterminant()


#####################################
# Element-wise Operation Functions.
#####################################

def addn(x):
    """
    Computes addition of all input tensors element-wise.

    All input tensors must have the same shape.

    Args:
        x (Union(tuple[Tensor], list[Tensor])): A tuple or list composed of Tensor, the data type is
            `bool_ <https://www.mindspore.cn/docs/api/en/master/api_python/mindspore.html#mindspore.dtype>`_ or
            `number <https://www.mindspore.cn/docs/api/en/master/api_python/mindspore.html#mindspore.dtype>`_ .

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
    return absolute(x)


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


def neg(x):
    """
    Returns a tensor with negative values of the input tensor element-wise.

    .. math::

        out_{i} = - x_{i}

    Args:
        x (Tensor): The input tensor whose dtype is number.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.

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


def div(x, y):
    """
    Divides the first input tensor by the second input tensor in floating-point type element-wise.

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be bool at the same time, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

    .. math::

        out_{i} = x_{i} / y_{i}

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

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
        >>> y = Tensor(np.array([4.0, 5.0, 6.0]), mindspore.float32)
        >>> output = ops.div(x, y)
        >>> print(output)
        [0.25 0.4  0.5 ]
    """
    return tensor_div(x, y)


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

    where the :math:`floor` indicates the Floor operator, for more details, please refer to the Floor operator.

    Args:
        x (Union[Tensor, Number, bool]): The first input is a number or
            a bool or a tensor whose data type is number or bool.
        y (Union[Tensor, Number, bool]): The second input is a number or
            a bool when the first input is a tensor or a tensor whose data type is number or bool.

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

    where the :math:`floor` indicates the Floor operator, for more details, please refer to the Floor operator.

    .. warning::
        - The input data does not support 0.
        - When the elements of input exceeds 2048 , the accuracy of operator cannot guarantee the requirement of
          double thousandths in the mini form.
        - Due to different architectures, the calculation results of this operator on NPU and CPU may be inconsistent.
        - If shape is expressed as (D1,D2... ,Dn), then D1\*D2... \*DN<=1000000,n<=8.

    Args:
        x (Union[Tensor, Number, bool]): The first input is a number or
            a bool or a tensor whose data type is number or bool.
        y (Union[Tensor, Number, bool]): The second input is a number or
            a bool when the first input is a tensor or a tensor whose data type is number or bool.

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
        x (Tensor): The input tensor.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.

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
        x (Tensor): The input tensor. With float16 or float32 data type.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.

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
        x (Tensor): The input tensor. Its element data type must be float16 or float32.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.

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
    return floor_(x)


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


def logical_or(x):
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
    return logical_or_(x)


def logical_and(x):
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
    return logical_and_(x)


def sin(x):
    r"""
    Computes sine of the input element-wise.

    .. math::

        out_i = sin(x_i)

    Args:
        x (Tensor): The shape of tensor is
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Returns:
        Tensor, has the same shape as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.62, 0.28, 0.43, 0.62]), mindspore.float32)
        >>> output = ops.sin(x)
        >>> print(output)
        [0.5810352 0.27635565 0.41687083 0.5810352]
    """
    return sin_(x)


def cos(x):
    r"""
    Computes cosine of input element-wise.

    .. math::
        out_i = cos(x_i)

    .. warning::
        Currently support Float16, Float32 data type. If use Float64, there may
        be a problem of missing precision.

    Args:
        x (Tensor): The shape of tensor is
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions.

    Returns:
        Tensor, has the same shape as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.

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
        x (Tensor): The shape of tensor is
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
            Data type must be float16 or float32.

    Returns:
        Tensor, has the same shape as `x`.

    Raises:
        TypeError: If dtype of `x` is neither float16 nor float32.
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([-1.0, 0.0, 1.0]), mindspore.float32)
        >>> output = ops.tan(x)
        >>> print(output)
        [-1.5574081 0. 1.5574081]
    """
    return tan_(x)


def asin(x):
    r"""
    Computes arcsine of input tensors element-wise.

    .. math::

        out_i = sin^{-1}(x_i)

    Args:
        x (Tensor): The shape of tensor is
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
            The data type should be one of the following types: float16, float32, float64.

    Returns:
        Tensor, has the same shape and dtype as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16, float32, float64.

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
            The data type should be one of the following types: float16, float32, float64.

    Returns:
        Tensor, has the same shape and dtype as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16, float32 or float64.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.74, 0.04, 0.30, 0.56]), mindspore.float32)
        >>> output = ops.acos(x)
        >>> print(output)
        [0.737726  1.5307857 1.2661036 0.9764105]
    """
    return acos_(x)


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
        x (Tensor): The shape of tensor is
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.

    Returns:
        Tensor, has the same shape as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``CPU``

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
        x (Tensor): The shape of tensor is
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.

    Returns:
        Tensor, has the same shape as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.24, 0.83, 0.31, 0.09]), mindspore.float32)
        >>> output = ops.cosh(x)
        >>> print(output)
        [1.0289385 1.364684 1.048436 1.0040528]
    """
    return cosh_(x)


def tanh(input_x):
    r"""
    Tanh activation function.

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
        x (Tensor): The shape of tensor is
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.

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
        x (Tensor): The shape of tensor is
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.

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
        ``Ascend`` ``CPU``

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
    return atan2_(x, y)


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
        x (Tensor): The input tensor with int16, int32 or uint16 data type.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions.
        y (Tensor): The input tensor with same type as the `x`.

    Returns:
        Tensor, has the same type as the `x`.

    Raises:
        TypeError: If `x` or `y` is not a Tensor.
        RuntimeError: If the data type of `x` and `y` conversion of Parameter is required
                      when data type conversion of Parameter is not supported.

    Supported Platforms:
        ``Ascend`` ``CPU``

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
        x (Tensor): The input tensor with int16, int32 or uint16 data type.
        y (Tensor): The input tensor with same type as the `x`.

    Returns:
        Tensor, has the same type as the `x`.

    Raises:
        TypeError: If `x` or `y` is not a Tensor.
        RuntimeError: If the data type of `x`, `y` conversion of Parameter is required
                      when data type conversion of Parameter is not supported.

    Supported Platforms:
        ``Ascend`` ``CPU``

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
        x (Tensor): The input tensor with int16, int32 or uint16 data type.
        y (Tensor): The input tensor with same type as the `x`.

    Returns:
        Tensor, has the same type as the `x`.

    Raises:
        TypeError: If `x` or `y` is not a Tensor.
        RuntimeError: If the data type of `x`, `y` conversion of Parameter is required
                      when data type conversion of Parameter is not supported.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([0, 0, 1, -1, 1, 1, 1]), mindspore.int16)
        >>> y = Tensor(np.array([0, 1, 1, -1, -1, 2, 3]), mindspore.int16)
        >>> output = ops.bitwise_xor(x, y)
        >>> print(output)
        [ 0  1  0  0 -2  3  2]
    """
    return bitwise_xor_(x, y)


def invert(x):
    r"""
    Flips all bits of input tensor element-wise.

    .. math::

        out_i = ~x_{i}

    Args:
        x (Tensor): The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
            The data type should be one of the following types: int16, uint16.

    Returns:
        Tensor, has the same shape as `x`.

    Raises:
        TypeError: If dtype of `x` is neither int16 nor uint16.

    Supported Platforms:
        ``Ascend``

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
        x (Tensor): Input Tensor of Gaussian error function. Its dimension must be less than 8 and
            data type must be float16 or float32.

    Returns:
        Tensor, has the same shape and dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is neither float16 nor float32.

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
        x (Tensor): The input tensor. The data type must be float16 or float32.
            :math:`(N,*)` where :math:`*` means, any number of additional dimensions, its rank should be less than 8.

    Returns:
        Tensor, has the same shap dtype as the `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        TypeError: If dtype of `x` is not float16 or float32.

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
        ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.5, 1., 2., 4.]), mindspore.float32)
        >>> output = ops.bessel_j0(x)
        >>> print(output)
        [1.8427168e+00 1.0000000e+00 1.5728319e-01 4.6912432e-03 2.2351742e-05]
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
        ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.5, 1., 2., 4.]), mindspore.float32)
        >>> output = ops.bessel_j1(x)
        >>> print(output)
        [0.24226846,  0.44005059,  0.57672481, -0.06604333]
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
        ``CPU``

    Examples:
        >>> x = Tensor(np.array([-1, -0.5, 0.5, 1]), mindspore.float32)
        >>> output = ops.bessel_i0(x)
        >>> print(output)
        [1.26606588, 1.06348337, 1.06348337, 1.26606588]
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
        ``CPU``

    Examples:
        >>> x = Tensor(np.array([-1, -0.5, 0.5, 1]), mindspore.float32)
        >>> output = ops.bessel_i0e(x)
        >>> print(output)
        [0.46575961, 0.64503527, 0.64503527, 0.46575961]
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
        ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.5, 1., 2., 4.]), mindspore.float32)
        >>> output = ops.bessel_k0(x)
        >>> print(output)
        [0.92441907, 0.42102444, 0.11389387, 0.01115968]
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
        ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.5, 1., 2., 4.]), mindspore.float32)
        >>> output = ops.bessel_k0e(x)
        >>> print(output)
        [1.52410939, 1.14446308, 0.84156822, 0.60929767]
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
        ``CPU``

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
        ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.5, 1., 2., 4.]), mindspore.float32)
        >>> output = ops.bessel_y1(x)
        >>> print(output)
        [-1.47147239  -0.78121282  -0.10703243  0.39792571]
    """
    return bessel_y1_(x)


def matrix_determinant(x):
    """
    Computes the determinant of one or more square matrices.

    Args:
        x (Tensor): A matrix to be calculated. The matrix must be at least two dimensions, and the last two
          dimensions must be the same size.

    Returns:
       y (Tensor): The shape is `x_shape[:-2]`, the dtype is same as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        ValueError: If the last two dimensions of `x` is not same size.
        ValueError: If the dimension of `x` is less than 2.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[[-4.5, -1.5], [7.0, 6.0]], [[2.5, 0.5], [3.0, 9.0]]]), mindspore.float32)
        >>> output = ops.matrix_determinant(input_x)
        >>> print(output)
        [-16.5 21. ]
    """
    return matrix_determinant_(x)


def log_matrix_determinant(x):
    """
    Computes the sign and the log of the absolute value of the determinant of one or more square matrices.

    Args:
        x (Tensor): A matrix to be calculated. The matrix must be at least two dimensions, and the last two
          dimensions must be the same size.

    Returns:
       sign (Tensor): The signs of the log determinants. The shape is `x_shape[:-2]`, the dtype is same as `x`.
       y (Tensor): The absolute values of the log determinants. The shape is `x_shape[:-2]`, the dtype is same
         as `x`.

    Raises:
        TypeError: If `x` is not a Tensor.
        ValueError: If the last two dimensions of `x` is not same size.
        ValueError: If the dimension of `x` is less than 2.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[[-4.5, -1.5], [7.0, 6.0]], [[2.5, 0.5], [3.0, 9.0]]]), mindspore.float32)
        >>> output = ops.log_matrix_determinant(input_x)
        >>> print(output)
        (Tensor(shape=[2], dtype=Float32, value= [-1.00000000e+00,  1.00000000e+00]), Tensor(shape=[2], dtype=Float32,
        value= [ 2.80336046e+00,  3.04452229e+00]))
    """
    return log_matrix_determinant_(x)


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
            a bool when the first input is a tensor or a tensor whose data type is number or bool.

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
        [False  True False]
    """
    return tensor_gt(x, y)


def ge(x, y):
    r"""
    Computes the boolean value of :math:`x >= y` element-wise.

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be bool at the same time, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.

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
    return tensor_ge(x, y)


def equal(x, y):
    r"""
    Computes the equivalence between two tensors element-wise.

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors, the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar, the scalar could only be a constant.

    .. math::

        out_{i} =\begin{cases}
            & \text{True,    if } x_{i} = y_{i} \\
            & \text{False,   if } x_{i} \ne y_{i}
            \end{cases}

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

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors, the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar, the scalar could only be a constant.

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
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([np.log(-1), 1, np.log(0)]), mindspore.float32)
        >>> output = ops.isnan(x)
        >>> print(output)
        [ True False False]
    """
    return isnan_(x)


def same_type_shape(input_x, input_y):
    """
    Checks whether the data type and shape of two tensors are the same.

    Args:
        input_x (Tensor): The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        input_y (Tensor): The shape of tensor is :math:`(x_1, x_2, ..., x_S)`.

    Returns:
        Tensor, the shape of tensor is :math:`(x_1, x_2, ..., x_R)`,
        if data type and shape of `input_x` and `input_y` are the same.

    Raises:
        TypeError: If the data types of `input_x` and `input_y` are not the same.
        ValueError: If the shapes of `input_x` and `input_y` are not the same.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32)
        >>> input_y = Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32)
        >>> output = ops.same_type_shape(input_x, input_y)
        >>> print(output)
        [[2. 2.]
         [2. 2.]]
    """
    return same_type_shape_(input_x, input_y)


def maximum(x, y):
    """
    Computes the maximum of input tensors element-wise.

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be bool at the same time, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.
    If one of the elements being compared is a NaN, then that element is returned.

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

    Inputs of `x` and `y` comply with the implicit type conversion rules to make the data types consistent.
    The inputs must be two tensors or one tensor and one scalar.
    When the inputs are two tensors,
    dtypes of them cannot be bool at the same time, and the shapes of them could be broadcast.
    When the inputs are one tensor and one scalar,
    the scalar could only be a constant.
    If one of the elements being compared is a NaN, then that element is returned.

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
        ValueError: If `x` and `y` are not the same shape.

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


def cdist(x, y, p=2.0):
    """
    Computes batched the p-norm distance between each pair of the two collections of row vectors.

    Refer to :func:`mindspore.ops.cdist` for more detal

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[[1.0, 1.0], [2.0, 2.0]]]).astype(np.float32))
        >>> input_y = Tensor(np.array([[[3.0, 3.0], [3.0, 3.0]]]).astype(np.float32))
        >>> output = ops.cdist(input_x, input_y, 2.0)
        >>> print(output)
        [[[2.8284273 2.8284273]
          [1.4142137 1.4142137]]]
    """

    cdist_ = P.Cdist(p)
    return cdist_(x, y)


def lerp(start, end, weight):
    """
    Does a linear interpolation of two tensors start and end based on a float or tensor weight.

    If `weight` is a tensor, the shapes of three inputs need to be broadcast;
    If `weight` is a float, the shapes of `start` and `end` need to be broadcast.

    .. math::

        output_{i} = start_{i} + weight_{i} * (end_{i} - start_{i})

    Args:
        start (Tensor): The tensor with the starting points. Data type must be float16 or float32.
        end (Tensor): The tensor with the ending points. Data type must be float16 or float32.
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
        ``Ascend`` ``CPU``

    Examples:
        >>> start = Tensor(np.array([1., 2., 3., 4.]), mindspore.float32)
        >>> end = Tensor(np.array([10., 10., 10., 10.]), mindspore.float32)
        >>> output = ops.lerp(start, end, 0.5)
        >>> print(output)
        [5.5 6. 6.5 7. ]
    """
    return lerp_(start, end, weight)


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
        ``CPU``

    Examples:
        >>> x = Tensor(np.array([-1, -0.5, 0.5, 1]), mindspore.float32)
        >>> output = ops.bessel_i1(x)
        >>> print(output)
        [-0.5651591, -0.25789431, 0.25789431, 0.5651591]
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
        ``CPU``

    Examples:
        >>> x = Tensor(np.array([-1, -0.5, 0.5, 1]), mindspore.float32)
        >>> output = ops.bessel_i1e(x)
        >>> print(output)
        [-0.20791042, -0.15642083, 0.15642083, 0.20791042]
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
        ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.5, 1., 2., 4.]), mindspore.float32)
        >>> output = ops.bessel_k1(x)
        >>> print(output)
        [1.65644112, 0.60190723, 0.13986588, 0.0124835]
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
        ``CPU``

    Examples:
        >>> x = Tensor(np.array([0.5, 1., 2., 4.]), mindspore.float32)
        >>> output = ops.bessel_k1e(x)
        >>> print(output)
        [2.73100971, 1.63615349, 1.03347685, 0.68157595]
    """
    return bessel_k1e_(x)

#####################################
# Reduction Operation Functions.
#####################################


def lp_norm(input_x, axis, p=2, keep_dims=False, epsilon=1e-12):
    r"""
    Returns the matrix norm or vector norm of a given tensor.

    .. math::
        output = sum(abs(input)**p)**(1/p)

    Args:
        input_x (Tensor): Input tensor.
        axis (Union[int,list,tuple]): Specifies which dimension or dimensions of input to calculate the norm across.
        p (int): The order of norm. Default: 2.
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
        ValueError: If the element of `axis` is out of the range [-len(input.shape), len(input.shape)).
        ValueError: If the length of shape of `axis` is bigger than the length of shape of `input`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> input_x = Tensor(np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).astype(np.float32))
        >>> output = ops.lp_norm(input_x, 2, [0, 1])
        >>> print(output)
        [ 9.165152 10.954452]
    """
    lp_norm_ = P.LpNorm(axis, p, keep_dims, epsilon)
    return lp_norm_(input_x)


__all__ = [
    'addn',
    'absolute',
    'abs',
    'tensor_add',
    'add',
    'neg_tensor',
    'neg',
    'tensor_lt',
    'less',
    'tensor_le',
    'le',
    'lerp',
    'lp_norm',
    'tensor_gt',
    'gt',
    'tensor_ge',
    'ge',
    'tensor_sub',
    'sub',
    'tensor_mul',
    'mul',
    'tensor_div',
    'div',
    'tensor_floordiv',
    'floor_div',
    'floordiv',
    'tensor_pow',
    'pow',
    'pows',
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
    'isfinite',
    'isnan',
    'log',
    'log_matrix_determinant',
    'matrix_determinant',
    'same_type_shape',
    'maximum',
    'minimum',
    'floor',
    'logical_not',
    'logical_or',
    'logical_and',
    'sin',
    'cos',
    'tan',
    'asin',
    'acos',
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
    'invert',
    'erf',
    'erfc',
    'cdist',
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
    'bessel_k1e'
]
__all__.sort()
