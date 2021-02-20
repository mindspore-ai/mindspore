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
"""logical operations, the function docs are adapted from Numpy API."""


from .math_ops import _apply_tensor_op
from ..ops import functional as F
from ..common import dtype as mstype
from .._c_expression import typing

from .array_creations import zeros, ones
from .utils import _check_input_tensor


def not_equal(x1, x2, out=None, where=True, dtype=None):
    """
    Returns (x1 != x2) element-wise.

    Args:
        x1 (Tensor): First input tensor to be compared.
        x2 (Tensor): Second input tensor to be compared.
        out (Tensor or None, optional), default is None.
        where (Tensor or None, optional): For any non-default value of type other
            than :class:`Tensor` or :class:`None`, the output retains its original value.
            This condition is broadcasted over the input. At locations where the
            condition is `True`, the out array will be set to the ufunc result.
            Elsewhere, the out array will retain its original value. Note that
            if an uninitialized out array is created via the default ``out=None``,
            locations within it where the condition is `False` will remain
            uninitialized.
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
       Tensor or scalar, element-wise comparison of `x1` and `x2`. Typically of type
       bool, unless `dtype` is passed. This is a scalar if both `x1` and `x2` are
       scalars.

    Raises:
        TypeError: If the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.asarray([1, 2])
        >>> b = np.asarray([[1, 3],[1, 4]])
        >>> print(np.not_equal(a, b))
        >>> [[False  True]
             [False  True]]
    """
    _check_input_tensor(x1, x2)
    return _apply_tensor_op(F.not_equal, x1, x2, out=out, where=where, dtype=dtype)


def less_equal(x1, x2, out=None, where=True, dtype=None):
    """
    Returns the truth value of ``(x1 <= x2)`` element-wise.

    Note:
        Numpy arguments `casting`, `order`, `dtype`, `subok`, `signature`, and `extobj` are
        not supported.
        When `where` is provided, `out` must have a tensor value. `out` is not supported
        for storing the result, however it can be used in combination with `where` to set
        the value at indices for which `where` is set to False.

    Args:
        x1 (Tensor): Input array.
        x2 (Tensor): Input array. If ``x1.shape != x2.shape``, they must be
            broadcastable to a common shape (which becomes the shape of the output).
        out (Tensor or None, optional): defaults to None.
        where (Tensor or None, optional): For any non-default value of type other
            than :class:`Tensor` or :class:`None`, the output retains its original value.
            This condition is broadcasted over the input. At locations where the
            condition is `True`, the out array will be set to the ufunc result.
            Elsewhere, the out array will retain its original value. Note that
            if an uninitialized out array is created via the default ``out=None``,
            locations within it where the condition is `False` will remain
            uninitialized.
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
       Tensor or scalar, element-wise comparison of `x1` and `x2`. Typically of type
       bool, unless ``dtype=object`` is passed. This is a scalar if both `x1` and `x2` are
       scalars.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> output = np.less_equal(np.array([4, 2, 1]), np.array([2, 2, 2]))
        >>> print(output)
        [False  True  True]
    """
    _check_input_tensor(x1, x2)
    return _apply_tensor_op(F.tensor_le, x1, x2, out=out, where=where, dtype=dtype)


def less(x1, x2, out=None, where=True, dtype=None):
    """
    Returns the truth value of ``(x1 < x2)`` element-wise.

    Note:
        Numpy arguments `casting`, `order`, `dtype`, `subok`, `signature`, and `extobj` are
        not supported.
        When `where` is provided, `out` must have a tensor value. `out` is not supported
        for storing the result, however it can be used in combination with `where` to set
        the value at indices for which `where` is set to False.

    Args:
        x1 (Tensor): input array.
        x2 (Tensor): Input array. If ``x1.shape != x2.shape``, they must be
            broadcastable to a common shape (which becomes the shape of the output).
        out (Tensor or None, optional): defaults to None.
        where (Tensor or None, optional): For any non-default value of type other
            than :class:`Tensor` or :class:`None`, the output retains its original value.
            This condition is broadcasted over the input. At locations where the
            condition is `True`, the out array will be set to the ufunc result.
            Elsewhere, the out array will retain its original value. Note that
            if an uninitialized out array is created via the default ``out=None``,
            locations within it where the condition is `False` will remain
            uninitialized.
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
       Tensor or scalar, element-wise comparison of `x1` and `x2`. Typically of type
       bool, unless ``dtype=object`` is passed. This is a scalar if both `x1` and `x2` are
       scalars.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> output = np.less(np.array([1, 2]), np.array([2, 2]))
        >>> print(output)
        [ True False]
    """
    return _apply_tensor_op(F.tensor_lt, x1, x2, out=out, where=where, dtype=dtype)


def greater_equal(x1, x2, out=None, where=True, dtype=None):
    """
    Returns the truth value of ``(x1 >= x2)`` element-wise.

    Note:
        Numpy arguments `casting`, `order`, `dtype`, `subok`, `signature`, and `extobj` are
        not supported.
        When `where` is provided, `out` must have a tensor value. `out` is not supported
        for storing the result, however it can be used in combination with `where` to set
        the value at indices for which `where` is set to False.

    Args:
        x1 (Tensor): Input array.
        x2 (Tensor): Input array. If ``x1.shape != x2.shape``, they must be
            broadcastable to a common shape (which becomes the shape of the output).
        out (Tensor or None, optional): defaults to None.
        where (Tensor or None, optional): For any non-default value of type other
            than :class:`Tensor` or :class:`None`, the output retains its original value.
            This condition is broadcasted over the input. At locations where the
            condition is `True`, the out array will be set to the ufunc result.
            Elsewhere, the out array will retain its original value. Note that
            if an uninitialized out array is created via the default ``out=None``,
            locations within it where the condition is `False` will remain
            uninitialized.
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
       Tensor or scalar, element-wise comparison of `x1` and `x2`. Typically of type
       bool, unless ``dtype=object`` is passed. This is a scalar if both `x1` and `x2` are
       scalars.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> output = np.greater_equal(np.array([4, 2, 1]), np.array([2, 2, 2]))
        >>> print(output)
        [ True  True False]
    """
    return _apply_tensor_op(F.tensor_ge, x1, x2, out=out, where=where, dtype=dtype)


def greater(x1, x2, out=None, where=True, dtype=None):
    """
    Returns the truth value of ``(x1 > x2)`` element-wise.

    Note:
        Numpy arguments `casting`, `order`, `dtype`, `subok`, `signature`, and `extobj` are
        not supported.
        When `where` is provided, `out` must have a tensor value. `out` is not supported
        for storing the result, however it can be used in combination with `where` to set
        the value at indices for which `where` is set to False.

    Args:
        x1 (Tensor): Input array.
        x2 (Tensor): Input array. If ``x1.shape != x2.shape``, they must be
            broadcastable to a common shape (which becomes the shape of the output).
        out (Tensor or None, optional): defaults to None.
        where (Tensor or None, optional): For any non-default value of type other
            than :class:`Tensor` or :class:`None`, the output retains its original value.
            This condition is broadcasted over the input. At locations where the
            condition is `True`, the out array will be set to the ufunc result.
            Elsewhere, the out array will retain its original value. Note that
            if an uninitialized out array is created via the default ``out=None``,
            locations within it where the condition is `False` will remain
            uninitialized.
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
       Tensor or scalar, element-wise comparison of `x1` and `x2`. Typically of type
       bool, unless ``dtype=object`` is passed. This is a scalar if both `x1` and `x2` are
       scalars.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> output = np.greater(np.array([4, 2]), np.array([2, 2]))
        >>> print(output)
        [ True False]
    """
    return _apply_tensor_op(F.tensor_gt, x1, x2, out=out, where=where, dtype=dtype)


def equal(x1, x2, out=None, where=True, dtype=None):
    """
    Returns the truth value of ``(x1 == x2)`` element-wise.

    Note:
        Numpy arguments `casting`, `order`, `dtype`, `subok`, `signature`, and `extobj` are
        not supported.
        When `where` is provided, `out` must have a tensor value. `out` is not supported
        for storing the result, however it can be used in combination with `where` to set
        the value at indices for which `where` is set to False.

    Args:
        x1 (Tensor): Input array.
        x2 (Tensor): Input array. If ``x1.shape != x2.shape``, they must be
            broadcastable to a common shape (which becomes the shape of the output).
        out (Tensor or None, optional): defaults to None.
        where (Tensor or None, optional): For any non-default value of type other
            than :class:`Tensor` or :class:`None`, the output retains its original value.
            This condition is broadcasted over the input. At locations where the
            condition is `True`, the out array will be set to the ufunc result.
            Elsewhere, the out array will retain its original value. Note that
            if an uninitialized out array is created via the default ``out=None``,
            locations within it where the condition is `False` will remain
            uninitialized.
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
       Tensor or scalar, element-wise comparison of `x1` and `x2`. Typically of type
       bool, unless ``dtype=object`` is passed. This is a scalar if both `x1` and `x2` are
       scalars.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> output = np.equal(np.array([0, 1, 3]), np.arange(3))
        >>> print(output)
        [ True  True False]
    """
    return _apply_tensor_op(F.equal, x1, x2, out=out, where=where, dtype=dtype)


def isfinite(x, out=None, where=True, dtype=None):
    """
    Tests element-wise for finiteness (not infinity or not Not a Number).

    The result is returned as a boolean array.

    Note:
        Numpy arguments `casting`, `order`, `dtype`, `subok`, `signature`, and `extobj` are
        not supported.
        When `where` is provided, `out` must have a tensor value. `out` is not supported
        for storing the result, however it can be used in combination with `where` to set
        the value at indices for which `where` is set to False.
        On GPU, the supported dtypes are np.float16, and np.float32.

    Args:
        x (Tensor): Input values.
        out (Tensor or None, optional): defaults to None.
        where (Tensor or None, optional): For any non-default value of type other
            than :class:`Tensor` or :class:`None`, the output retains its original value.
            This condition is broadcasted over the input. At locations where the
            condition is `True`, the out array will be set to the ufunc result.
            Elsewhere, the out array will retain its original value. Note that
            if an uninitialized out array is created via the default ``out=None``,
            locations within it where the condition is `False` will remain
            uninitialized.
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
       Tensor or scalar, true where `x` is not positive infinity, negative infinity,
       or NaN; false otherwise. This is a scalar if `x` is a scalar.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> output = np.isfinite(np.array([np.inf, 1., np.nan]).astype('float32'))
        >>> print(output)
        [False  True False]
        >>> output = np.isfinite(np.log(np.array(-1.).astype('float32')))
        >>> print(output)
        False
    """
    return _apply_tensor_op(F.isfinite, x, out=out, where=where, dtype=dtype)


def _isnan(x):
    """Computes isnan without applying keyword arguments."""
    return F.not_equal(x, x)


def isnan(x, out=None, where=True, dtype=None):
    """
    Tests element-wise for NaN and return result as a boolean array.

    Note:
        Numpy arguments `casting`, `order`, `dtype`, `subok`, `signature`, and `extobj` are
        not supported.
        When `where` is provided, `out` must have a tensor value. `out` is not supported
        for storing the result, however it can be used in combination with `where` to set
        the value at indices for which `where` is set to False.
        On GPU, the supported dtypes are np.float16, and np.float32.

    Args:
        x (Tensor): Input values.
        out (Tensor or None, optional): defaults to None.
        where (Tensor or None, optional): For any non-default value of type other
            than :class:`Tensor` or :class:`None`, the output retains its original value.
            This condition is broadcasted over the input. At locations where the
            condition is `True`, the out array will be set to the ufunc result.
            Elsewhere, the out array will retain its original value. Note that
            if an uninitialized out array is created via the default ``out=None``,
            locations within it where the condition is `False` will remain
            uninitialized.
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
       Tensor or scalar, true where `x` is NaN, false otherwise. This is a scalar if
       `x` is a scalar.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> output = np.isnan(np.array(np.nan, np.float32))
        >>> print(output)
        True
        >>> output = np.isnan(np.array(np.inf, np.float32))
        >>> print(output)
        False
    """
    return _apply_tensor_op(_isnan, x, out=out, where=where, dtype=dtype)


def _isinf(x):
    """Computes isinf without applying keyword arguments."""
    shape = F.shape(x)
    zeros_tensor = zeros(shape, mstype.float32)
    ones_tensor = ones(shape, mstype.float32)
    not_inf = F.isfinite(x)
    is_nan = _isnan(x)
    res = F.select(not_inf, zeros_tensor, ones_tensor)
    res = F.select(is_nan, zeros_tensor, res)
    return F.cast(res, mstype.bool_)


def isinf(x, out=None, where=True, dtype=None):
    """
    Tests element-wise for positive or negative infinity.

    Returns a boolean array of the same shape as `x`, True where ``x == +/-inf``, otherwise False.

    Note:
        Numpy arguments `casting`, `order`, `dtype`, `subok`, `signature`, and `extobj` are
        not supported.
        When `where` is provided, `out` must have a tensor value. `out` is not supported
        for storing the result, however it can be used in combination with `where` to set
        the value at indices for which `where` is set to False.
        On GPU, the supported dtypes are np.float16, and np.float32.

    Args:
        x (Tensor): Input values.
        out (Tensor or None, optional): defaults to None.
        where (Tensor or None, optional): For any non-default value of type other
            than :class:`Tensor` or :class:`None`, the output retains its original value.
            This condition is broadcasted over the input. At locations where the
            condition is `True`, the out array will be set to the ufunc result.
            Elsewhere, the out array will retain its original value. Note that
            if an uninitialized out array is created via the default ``out=None``,
            locations within it where the condition is `False` will remain
            uninitialized.
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
       Tensor or scalar, true where `x` is positive or negative infinity, false
       otherwise. This is a scalar if `x` is a scalar.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> output = np.isinf(np.array(np.inf, np.float32))
        >>> print(output)
        True
        >>> output = np.isinf(np.array([np.inf, -np.inf, 1.0, np.nan], np.float32))
        >>> print(output)
        [ True  True False False]
    """
    return _apply_tensor_op(_isinf, x, out=out, where=where, dtype=dtype)


def _is_sign_inf(x, fn):
    """Tests element-wise for inifinity with sign."""
    shape = F.shape(x)
    zeros_tensor = zeros(shape, mstype.float32)
    ones_tensor = ones(shape, mstype.float32)
    not_inf = F.isfinite(x)
    is_sign = fn(x, zeros_tensor)
    res = F.select(not_inf, zeros_tensor, ones_tensor)
    res = F.select(is_sign, res, zeros_tensor)
    return F.cast(res, mstype.bool_)


def isposinf(x):
    """
    Tests element-wise for positive infinity, returns result as bool array.

    Note:
        Numpy argument `out` is not supported.
        On GPU, the supported dtypes are np.float16, and np.float32.

    Args:
        x (Tensor): Input values.

    Returns:
       Tensor or scalar, true where `x` is positive infinity, false otherwise.
       This is a scalar if `x` is a scalar.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> output = np.isposinf(np.array([-np.inf, 0., np.inf], np.float32))
        >>> print(output)
        [False False  True]
    """
    _check_input_tensor(x)
    return _is_sign_inf(x, F.tensor_gt)


def isneginf(x):
    """
    Tests element-wise for negative infinity, returns result as bool array.

    Note:
        Numpy argument `out` is not supported.
        On GPU, the supported dtypes are np.float16, and np.float32.

    Args:
        x (Tensor): Input values.

    Returns:
       Tensor or scalar, true where `x` is negative infinity, false otherwise.
       This is a scalar if `x` is a scalar.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> output = np.isneginf(np.array([-np.inf, 0., np.inf], np.float32))
        >>> print(output)
        [ True False False]
    """
    return _is_sign_inf(x, F.tensor_lt)


def isscalar(element):
    """
    Returns True if the type of element is a scalar type.

    Note:
        Only object types recognized by the mindspore parser are supported,
        which includes objects, types, methods and functions defined within
        the scope of mindspore. Other built-in types are not supported.

    Args:
        element (any): Input argument, can be of any type and shape.

    Returns:
       Boolean, True if `element` is a scalar type, False if it is not.

    Raises:
        TypeError: if the type of `element` is not supported by mindspore parser.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> output = np.isscalar(3.1)
        >>> print(output)
        True
        >>> output = np.isscalar(np.array(3.1))
        >>> print(output)
        False
        >>> output = np.isscalar(False)
        >>> print(output)
        True
        >>> output = np.isscalar('numpy')
        >>> print(output)
        True
    """
    return isinstance(F.typeof(element), (typing.Number, typing.Int, typing.UInt,
                                          typing.Float, typing.Bool, typing.String))
