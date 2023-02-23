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
from __future__ import absolute_import

from mindspore.ops import functional as F
from mindspore.common import dtype as mstype
from mindspore.common import Tensor
from mindspore.ops import operations as P

from mindspore.numpy.math_ops import _apply_tensor_op
from mindspore.numpy.array_creations import zeros, ones, asarray
from mindspore.numpy.utils import _check_input_tensor, _to_tensor, _isnan
from mindspore.numpy.utils_const import _raise_type_error, _check_same_type, \
    _check_axis_type, _canonicalize_axis, _can_broadcast, _isscalar


def not_equal(x1, x2, dtype=None):
    """
    Returns (x1 != x2) element-wise.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`,
        and `extobj` are not supported.

    Args:
        x1 (Tensor): First input tensor to be compared.
        x2 (Tensor): Second input tensor to be compared.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
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
        [[False  True]
        [False  True]]
    """
    _check_input_tensor(x1, x2)
    return _apply_tensor_op(F.not_equal, x1, x2, dtype=dtype)


def less_equal(x1, x2, dtype=None):
    """
    Returns the truth value of ``(x1 <= x2)`` element-wise.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`,
        and `extobj` are not supported.

    Args:
        x1 (Tensor): Input array.
        x2 (Tensor): Input array. If ``x1.shape != x2.shape``, they must be
            broadcastable to a common shape (which becomes the shape of the output).
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
       Tensor or scalar, element-wise comparison of `x1` and `x2`. Typically of type
       bool, unless `dtype` is passed. This is a scalar if both `x1` and `x2` are
       scalars.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> output = np.less_equal(np.array([4, 2, 1]), np.array([2, 2, 2]))
        >>> print(output)
        [False  True  True]
    """
    _check_input_tensor(x1, x2)
    return _apply_tensor_op(F.tensor_le, x1, x2, dtype=dtype)


def less(x1, x2, dtype=None):
    """
    Returns the truth value of ``(x1 < x2)`` element-wise.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`,
        and `extobj` are not supported.

    Args:
        x1 (Tensor): input array.
        x2 (Tensor): Input array. If ``x1.shape != x2.shape``, they must be
            broadcastable to a common shape (which becomes the shape of the output).
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
       Tensor or scalar, element-wise comparison of `x1` and `x2`. Typically of type
       bool, unless `dtype` is passed. This is a scalar if both `x1` and `x2` are
       scalars.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> output = np.less(np.array([1, 2]), np.array([2, 2]))
        >>> print(output)
        [ True False]
    """
    return _apply_tensor_op(F.tensor_lt, x1, x2, dtype=dtype)


def greater_equal(x1, x2, dtype=None):
    """
    Returns the truth value of ``(x1 >= x2)`` element-wise.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`,
        and `extobj` are not supported.

    Args:
        x1 (Tensor): Input array.
        x2 (Tensor): Input array. If ``x1.shape != x2.shape``, they must be
            broadcastable to a common shape (which becomes the shape of the output).
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
       Tensor or scalar, element-wise comparison of `x1` and `x2`. Typically of type
       bool, unless `dtype` is passed. This is a scalar if both `x1` and `x2` are
       scalars.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> output = np.greater_equal(np.array([4, 2, 1]), np.array([2, 2, 2]))
        >>> print(output)
        [ True  True False]
    """
    return _apply_tensor_op(F.tensor_ge, x1, x2, dtype=dtype)


def greater(x1, x2, dtype=None):
    """
    Returns the truth value of ``(x1 > x2)`` element-wise.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`,
        and `extobj` are not supported.

    Args:
        x1 (Tensor): Input array.
        x2 (Tensor): Input array. If ``x1.shape != x2.shape``, they must be
            broadcastable to a common shape (which becomes the shape of the output).
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
       Tensor or scalar, element-wise comparison of `x1` and `x2`. Typically of type
       bool, unless `dtype` is passed. This is a scalar if both `x1` and `x2` are
       scalars.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> output = np.greater(np.array([4, 2]), np.array([2, 2]))
        >>> print(output)
        [ True False]
    """
    return _apply_tensor_op(F.tensor_gt, x1, x2, dtype=dtype)


def equal(x1, x2, dtype=None):
    """
    Returns the truth value of ``(x1 == x2)`` element-wise.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`,
        and `extobj` are not supported.

    Args:
        x1 (Tensor): Input array.
        x2 (Tensor): Input array. If ``x1.shape != x2.shape``, they must be
            broadcastable to a common shape (which becomes the shape of the output).
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
       Tensor or scalar, element-wise comparison of `x1` and `x2`. Typically of type
       bool, unless `dtype` is passed. This is a scalar if both `x1` and `x2` are
       scalars.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> output = np.equal(np.array([0, 1, 3]), np.arange(3))
        >>> print(output)
        [ True  True False]
    """
    return _apply_tensor_op(F.equal, x1, x2, dtype=dtype)


def isfinite(x, dtype=None):
    """
    Tests element-wise for finiteness (not infinity or not Not a Number).

    The result is returned as a boolean array.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`,
        and `extobj` are not supported.
        On GPU, the supported dtypes are np.float16, and np.float32.

    Args:
        x (Tensor): Input values.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
       Tensor or scalar, true where `x` is not positive infinity, negative infinity,
       or NaN; false otherwise. This is a scalar if `x` is a scalar.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> output = np.isfinite(np.array([np.inf, 1., np.nan]).astype('float32'))
        >>> print(output)
        [False  True False]
    """
    return _apply_tensor_op(F.isfinite, x, dtype=dtype)


def isnan(x, dtype=None):
    """
    Tests element-wise for NaN and return result as a boolean array.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`,
        and `extobj` are not supported.
        Only np.float32 is currently supported.

    Args:
        x (Tensor): Input values.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
       Tensor or scalar, true where `x` is NaN, false otherwise. This is a scalar if
       `x` is a scalar.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> output = np.isnan(np.array(np.nan, np.float32))
        >>> print(output)
        True
        >>> output = np.isnan(np.array(np.inf, np.float32))
        >>> print(output)
        False
    """
    return _apply_tensor_op(_isnan, x, dtype=dtype)


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


def isinf(x, dtype=None):
    """
    Tests element-wise for positive or negative infinity.

    Returns a boolean array of the same shape as `x`, True where ``x == +/-inf``, otherwise False.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`,
        and `extobj` are not supported.
        Only np.float32 is currently supported.

    Args:
        x (Tensor): Input values.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
       Tensor or scalar, true where `x` is positive or negative infinity, false
       otherwise. This is a scalar if `x` is a scalar.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> output = np.isinf(np.array(np.inf, np.float32))
        >>> print(output)
        True
        >>> output = np.isinf(np.array([np.inf, -np.inf, 1.0, np.nan], np.float32))
        >>> print(output)
        [ True  True False False]
    """
    return _apply_tensor_op(_isinf, x, dtype=dtype)


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
        Only np.float32 is currently supported.

    Args:
        x (Tensor): Input values.

    Returns:
       Tensor or scalar, true where `x` is positive infinity, false otherwise.
       This is a scalar if `x` is a scalar.

    Raises:
        TypeError: If the input is not a tensor.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> output = np.isposinf(np.array([-np.inf, 0., np.inf, np.nan], np.float32))
        >>> print(output)
        [False False  True False]
    """
    _check_input_tensor(x)
    return _is_sign_inf(x, F.tensor_gt)


def isneginf(x):
    """
    Tests element-wise for negative infinity, returns result as bool array.

    Note:
        Numpy argument `out` is not supported.
        Only np.float32 is currently supported.

    Args:
        x (Tensor): Input values.

    Returns:
       Tensor or scalar, true where `x` is negative infinity, false otherwise.
       This is a scalar if `x` is a scalar.

    Raises:
        TypeError: If the input is not a tensor.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> output = np.isneginf(np.array([-np.inf, 0., np.inf, np.nan], np.float32))
        >>> print(output)
        [ True False False False]
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
        TypeError: If the type of `element` is not supported by mindspore parser.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
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
    obj_type = F.typeof(element)
    return not isinstance(obj_type, Tensor) and _isscalar(obj_type)


def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=True):
    """
    Returns a boolean tensor where two tensors are element-wise equal within a tolerance.

    The tolerance values are positive, typically very small numbers. The relative
    difference (:math:`rtol * abs(b)`) and the absolute difference `atol` are added together
    to compare against the absolute difference between `a` and `b`.

    Note:
        For finite values, isclose uses the following equation to test whether two
        floating point values are equivalent.
        :math:`absolute(a - b) <= (atol + rtol * absolute(b))`
        On Ascend, input arrays containing inf or NaN are not supported.

    Args:
        a (Union[Tensor, list, tuple]): Input first tensor to compare.
        b (Union[Tensor, list, tuple]): Input second tensor to compare.
        rtol (numbers.Number): The relative tolerance parameter (see Note).
        atol (numbers.Number): The absolute tolerance parameter (see Note).
        equal_nan (bool): Whether to compare ``NaN`` as equal. If True, ``NaN`` in
            `a` will be considered equal to ``NaN`` in `b` in the output tensor.
            Default: `False`.

    Returns:
        A ``bool`` tensor of where `a` and `b` are equal within the given tolerance.

    Raises:
        TypeError: If inputs have types not specified above.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> a = np.array([0,1,2,float('inf'),float('inf'),float('nan')])
        >>> b = np.array([0,1,-2,float('-inf'),float('inf'),float('nan')])
        >>> print(np.isclose(a, b))
        [ True  True False False  True False]
        >>> print(np.isclose(a, b, equal_nan=True))
        [ True  True False False  True  True]
    """
    a, b = _to_tensor(a, b)
    is_close = P.IsClose(rtol=rtol, atol=atol, equal_nan=equal_nan)
    return is_close(a, b)


def in1d(ar1, ar2, invert=False):
    """
    Tests whether each element of a 1-D array is also present in a second array.

    Returns a boolean array the same length as `ar1` that is True where an element
    of `ar1` is in `ar2` and False otherwise.

    Note:
        Numpy argument `assume_unique` is not supported since the implementation does
        not rely on the uniqueness of the input arrays.

    Args:
        ar1 (Union[int, float, bool, list, tuple, Tensor]): Input array with shape `(M,)`.
        ar2 (Union[int, float, bool, list, tuple, Tensor]): The values against which
            to test each value of `ar1`.
        invert (boolean, optional): If True, the values in the returned array are
            inverted (that is, False where an element of `ar1` is in `ar2` and True
            otherwise). Default is False.

    Returns:
       Tensor, with shape `(M,)`. The values ``ar1[in1d]`` are in `ar2`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> test = np.array([0, 1, 2, 5, 0])
        >>> states = [0, 2]
        >>> mask = np.in1d(test, states)
        >>> print(mask)
        [ True False  True False  True]
        >>> mask = np.in1d(test, states, invert=True)
        >>> print(mask)
        [False  True False  True False]
    """
    ar1, ar2 = _to_tensor(ar1, ar2)
    ar1 = F.expand_dims(ar1.ravel(), -1)
    ar2 = ar2.ravel()
    included = F.equal(ar1, ar2)
    # F.reduce_sum only supports float
    res = F.reduce_sum(included.astype(mstype.float32), -1).astype(mstype.bool_)
    if invert:
        res = F.logical_not(res)
    return res


def isin(element, test_elements, invert=False):
    """
    Calculates element in `test_elements`, broadcasting over `element` only. Returns a
    boolean array of the same shape as `element` that is True where an element of
    `element` is in `test_elements` and False otherwise.

    Note:
        Numpy argument `assume_unique` is not supported since the implementation does
        not rely on the uniqueness of the input arrays.

    Args:
        element (Union[int, float, bool, list, tuple, Tensor]): Input array.
        test_elements (Union[int, float, bool, list, tuple, Tensor]): The values against
            which to test each value of `element`.
        invert (boolean, optional): If True, the values in the returned array are
            inverted, as if calculating `element` not in `test_elements`. Default is False.

    Returns:
       Tensor, has the same shape as `element`. The values ``element[isin]`` are in
       `test_elements`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> element = 2*np.arange(4).reshape((2, 2))
        >>> test_elements = [1, 2, 4, 8]
        >>> mask = np.isin(element, test_elements)
        >>> print(mask)
        [[False  True]
        [ True False]]
        >>> mask = np.isin(element, test_elements, invert=True)
        >>> print(mask)
        [[ True False]
        [False  True]]
    """
    element = _to_tensor(element)
    res = in1d(element, test_elements, invert=invert)
    return F.reshape(res, F.shape(element))


def logical_not(a, dtype=None):
    """
    Computes the truth value of NOT `a` element-wise.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        a (Tensor): The input tensor whose dtype is bool.
        dtype (:class:`mindspore.dtype`, optional): Default: :class:`None`. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar.
        Boolean result with the same shape as `a` of the NOT operation on elements of `a`.
        This is a scalar if `a` is a scalar.

    Raises:
        TypeError: If the input is not a tensor or its dtype is not bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.array([True, False])
        >>> output = np.logical_not(a)
        >>> print(output)
        [False  True]
    """
    return _apply_tensor_op(F.logical_not, a, dtype=dtype)


def logical_or(x1, x2, dtype=None):
    """
    Computes the truth value of `x1` OR `x2` element-wise.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`,
        and `extobj` are not supported.

    Args:
        x1 (Tensor): Input tensor.
        x2 (Tensor): Input tensor. If ``x1.shape != x2.shape``, they must be
            broadcastable to a common shape (which becomes the shape of the output).
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
       Tensor or scalar, element-wise comparison of `x1` and `x2`. Typically of type
       bool, unless ``dtype=object`` is passed. This is a scalar if both `x1` and `x2` are
       scalars.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x1 = np.array([True, False])
        >>> x2 = np.array([False, True])
        >>> output = np.logical_or(x1, x2)
        >>> print(output)
        [ True  True]
    """
    return _apply_tensor_op(F.logical_or, x1, x2, dtype=dtype)


def logical_and(x1, x2, dtype=None):
    """
    Computes the truth value of `x1` AND `x2` element-wise.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`,
        and `extobj` are not supported.

    Args:
        x1 (Tensor): Input tensor.
        x2 (Tensor): Input tensor. If ``x1.shape != x2.shape``, they must be
            broadcastable to a common shape (which becomes the shape of the output).
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar.
        Boolean result of the logical AND operation applied to the elements of `x1` and `x2`;
        the shape is determined by broadcasting. This is a scalar if both `x1` and `x2` are scalars.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x1 = np.array([True, False])
        >>> x2 = np.array([False, False])
        >>> output = np.logical_and(x1, x2)
        >>> print(output)
        [False False]
    """
    return _apply_tensor_op(F.logical_and, x1, x2, dtype=dtype)


def logical_xor(x1, x2, dtype=None):
    """
    Computes the truth value of `x1` XOR `x2`, element-wise.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`,
        and `extobj` are not supported.

    Args:
        x1 (Tensor): Input tensor.
        x2 (Tensor): Input tensor. If ``x1.shape != x2.shape``, they must be
            broadcastable to a common shape (which becomes the shape of the output).
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar.
        Boolean result of the logical AND operation applied to the elements of `x1` and `x2`;
        the shape is determined by broadcasting. This is a scalar if both `x1` and `x2` are scalars.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x1 = np.array([True, False])
        >>> x2 = np.array([False, False])
        >>> output = np.logical_xor(x1, x2)
        >>> print(output)
        [True False]
    """
    _check_input_tensor(x1)
    _check_input_tensor(x2)
    y1 = F.logical_or(x1, x2)
    y2 = F.logical_or(F.logical_not(x1), F.logical_not(x2))
    return _apply_tensor_op(F.logical_and, y1, y2, dtype=dtype)


def array_equal(a1, a2, equal_nan=False):
    """
    Returns `True` if input arrays have same shapes and all elements equal.

    Note:
        In mindspore, a bool tensor is returned instead, since in Graph mode, the
        value cannot be traced and computed at compile time.

        Since on Ascend, :class:`nan` is treated differently, currently the argument
        `equal_nan` is not supported on Ascend.

    Args:
        a1/a2 (Union[int, float, bool, list, tuple, Tensor]): Input arrays.
        equal_nan (bool): Whether to compare NaN's as equal. Default: `False`.

    Returns:
        Scalar bool tensor, value is `True` if inputs are equal, `False` otherwise.

    Raises:
        TypeError: If inputs have types not specified above.

    Supported Platforms:
        ``GPU`` ``CPU`` ``Ascend``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = [0,1,2]
        >>> b = [[0,1,2], [0,1,2]]
        >>> print(np.array_equal(a,b))
        False
    """
    a1 = asarray(a1)
    a2 = asarray(a2)
    if not isinstance(equal_nan, bool):
        _raise_type_error("equal_nan must be bool.")
    if a1.shape == a2.shape:
        res = equal(a1, a2)
        if equal_nan:
            res = logical_or(res, logical_and(isnan(a1), isnan(a2)))
        return res.all()
    return _to_tensor(False)


def array_equiv(a1, a2):
    """
    Returns `True` if input arrays are shape consistent and all elements equal.

    Shape consistent means they are either the same shape, or one input array can
    be broadcasted to create the same shape as the other one.

    Note:
        In mindspore, a bool tensor is returned instead, since in Graph mode, the
        value cannot be traced and computed at compile time.

    Args:
        a1/a2 (Union[int, float, bool, list, tuple, Tensor]): Input arrays.

    Returns:
        Scalar bool tensor, value is `True` if inputs are equivalent, `False` otherwise.

    Raises:
        TypeError: If inputs have types not specified above.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = [0,1,2]
        >>> b = [[0,1,2], [0,1,2]]
        >>> print(np.array_equiv(a,b))
        True
    """
    a1 = asarray(a1)
    a2 = asarray(a2)
    if _can_broadcast(a1.shape, a2.shape):
        return equal(a1, a2).all()
    return _to_tensor(False)


def signbit(x, dtype=None):
    """
    Returns element-wise True where signbit is set (less than zero).

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and
        `extobj` are not supported.

    Args:
        x (Union[int, float, bool, list, tuple, Tensor]): The input value(s).
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor.

    Raises:
        TypeError: If input is not array_like or `dtype` is not `None` or `bool`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.array([1, -2.3, 2.1]).astype('float32')
        >>> output = np.signbit(x)
        >>> print(output)
        [False  True False]
    """
    if dtype is not None and not _check_same_type(dtype, mstype.bool_):
        _raise_type_error("Casting was not allowed for signbit.")
    x = _to_tensor(x)
    res = F.less(x, 0)
    if dtype is not None and not _check_same_type(F.dtype(res), dtype):
        res = F.cast(res, dtype)
    return res


def sometrue(a, axis=None, keepdims=False):
    """
    Tests whether any array element along a given axis evaluates to True.

    Returns single boolean unless axis is not None

    Args:
        a (Union[int, float, bool, list, tuple, Tensor]): Input tensor or object that can be converted to an array.
        axis (Union[None, int, tuple(int)]): Axis or axes along which a logical OR reduction is
            performed. Default: None.
            If None, perform a logical OR over all the dimensions of the input array.
            If negative, it counts from the last to the first axis.
            If tuple of integers, a reduction is performed on multiple axes, instead of a single axis or
            all the axes as before.
        keepdims (bool): Default: False.
            If True, the axes which are reduced are left in the result as dimensions with size one.
            With this option, the result will broadcast correctly against the input array.
            If the default value is passed, then keepdims will not be passed through to the any method of
            sub-classes of ndarray, however any non-default value will be. If the sub-class method does not
            implement keepdims any exceptions will be raised.

    Returns:
        Returns single boolean unless axis is not None

    Raises:
        TypeError: If input is not array_like or `axis` is not int or tuple of integers or
            `keepdims` is not integer or `initial` is not scalar.
        ValueError: If any axis is out of range or duplicate axes exist.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.array([1, -2.3, 2.1]).astype('float32')
        >>> output = np.sometrue(x)
        >>> print(output)
        True
    """
    if not isinstance(keepdims, int):
        _raise_type_error("integer argument expected, but got ", keepdims)
    if axis is not None:
        _check_axis_type(axis, True, True, False)
        axis = _canonicalize_axis(axis, a.ndim)
    a = _to_tensor(a)
    keepdims = keepdims not in (0, False)
    return F.not_equal(a, 0).any(axis, keepdims)
