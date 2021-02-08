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
"""math operations, the function docs are adapted from Numpy API."""
import operator
import functools

from ..ops import operations as P
from ..ops import functional as F
from ..ops import composite as C
from ..ops.primitive import constexpr
from ..common import dtype as mstype
from ..common import Tensor

from .dtypes import nan, pi

from .array_creations import asarray_const, ones, zeros, empty, full
from .array_ops import where as where_
from .array_ops import ravel, expand_dims

from .utils_const import _infer_out_shape, _check_axis_valid, _get_device, \
    _check_shape_aligned, _raise_type_error, _check_same_type, _check_is_float, \
    _raise_value_error, _check_matmul_shapes, _promote, _check_axis_type, _canonicalize_axis, \
    _max, _is_shape_empty, _check_is_int
from .utils import _is_scalar, _expand, _broadcast_to, _broadcast_to_shape, _get_size, \
    _check_input_tensor


ZERO_TENSOR = asarray_const(0)


_mean_default = P.ReduceMean()
_mean_keepdims = P.ReduceMean(True)
_matmul = P.MatMul(False, False)
_matmul_T = P.MatMul(False, True)
_reduce_sum_default = P.ReduceSum()
_reduce_sum_keepdims = P.ReduceSum(True)
_reduce_min_default = P.ReduceMin()
_reduce_min_keepdims = P.ReduceMin(True)
_reduce_max_default = P.ReduceMax()
_reduce_max_keepdims = P.ReduceMax(True)

def absolute(x, out=None, where=True, dtype=None):
    """
    Calculates the absolute value element-wise.

    Note:
        Numpy arguments `casting`, `order`, `dtype`, `subok`, `signature`, and `extobj` are
        not supported.
        When `where` is provided, `out` must have a tensor value. `out` is not supported
        for storing the result, however it can be used in combination with `where` to set
        the value at indices for which `where` is set to False.
        Currently the backend kernel only supports float calculation, if the input
        is not a `float`, then it will be casted to :class:`mstype.float32` and casted back.

    Args:
        x (Tensor): Tensor to be used for calculation.
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
        Tensor.

    Raises:
        TypeError: If input arguments have types not specified above.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.asarray([1, 2, 3, -4, -5], np.float32)
        >>> output = np.absolute(x)
        >>> print(output)
        [1. 2. 3. 4. 5.]
    """
    original_dtype = x.dtype
    if not _check_is_float(original_dtype) and dtype is None:
        x = x.astype(mstype.float32)
        return _apply_tensor_op(F.absolute, x, out=out, where=where, dtype=dtype).astype(original_dtype)
    return _apply_tensor_op(F.absolute, x, out=out, where=where, dtype=dtype)


def count_nonzero(x, axis=None, keepdims=False):
    """
    Counts the number of non-zero values in the tensor `x`.

    Args:
        x (Tensor): The tensor for which to count non-zeros.
        axis (Union[int,tuple], optional): Axis or tuple of axes along which to
            count non-zeros. Default is None, meaning that non-zeros will be counted
            along a flattened version of `x`.
        keepdims (bool, optional): If this is set to True, the axes that are counted
            are left in the result as dimensions with size one. With this option,
            the result will broadcast correctly against `x`.

    Returns:
        Tensor, indicating number of non-zero values in the `x` along a given axis.
        Otherwise, the total number of non-zero values in `x` is returned.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.asarray([1, 2, 3, -4, 0, 3, 2, 0])
        >>> output = np.count_nonzero(x)
        >>> print(output)
        6
    """
    if _is_shape_empty(x.shape):
        return ZERO_TENSOR
    if axis is None:
        axis = ()
    return C.count_nonzero(x=x, axis=axis, keep_dims=keepdims)


def clip(x, xmin, xmax, out=None, where=True, dtype=None):
    """
    Clips (limits) the values in an array.

    Given an interval, values outside the interval are clipped to the interval edges.
    For example, if an interval of :math:`[0, 1]` is specified, values smaller than 0 become 0,
    and values larger than 1 become 1.

    Args:
        x (Tensor): Tensor containing elements to clip.
        xmin (Tensor, scalar, None): Minimum value. If None, clipping is not performed
            on lower interval edge. Not more than one of `xmin` and `xmax` may be None.
        xmax (Tensor, scalar, None): Maximum value. If None, clipping is not performed
            on upper interval edge. Not more than one of `xmin` and `xmax` may be None.
            If `xmin` or `xmax` are tensors, then the three tensors will be broadcasted
            to match their shapes.
        out (Tensor or None): optional, default to None.
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
        Tensor, a tensor with the elements of `x`, but where values
        < `xmin` are replaced with `xmin`, and those > `xmax` with `xmax`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.asarray([1, 2, 3, -4, 0, 3, 2, 0])
        >>> output = np.clip(x, 0, 2)
        >>> print(output)
        [1 2 2 0 0 2 2 0]
    """
    if xmin is None and xmax is None:
        _raise_value_error("One of max or min must be given.")
    if xmin is not None:
        x = maximum(x, xmin, out=out, where=where, dtype=dtype)
    if xmax is not None:
        x = minimum(x, xmax, out=out, where=where, dtype=dtype)
    return x


def deg2rad(x, out=None, where=True, dtype=None):
    """
    Converts angles from degrees to radians.

    Args:
        x (Tensor): Angles in degrees.
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
        Tensor, the corresponding angle in radians. This is a tensor scalar if `x`
        is a tensor scalar.

    Raises:
        TypeError: if `x` is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.asarray([1, 2, 3, -4, -5])
        >>> output = np.deg2rad(x)
        >>> print(output)
        [ 0.01745329  0.03490658  0.05235988 -0.06981317 -0.08726647]
    """
    _check_input_tensor(x)

    def convert(a):
        return a * pi / 180.0
    return _apply_tensor_op(convert, x, out=out, where=where, dtype=dtype)


def rad2deg(x, out=None, where=True, dtype=None):
    """
    Converts angles from radians to degrees.

    Args:
        x (Tensor): Angles in radians.
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
        Tensor, the corresponding angle in degrees. This is a tensor scalar if `x`
        is a tensor scalar.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = np.asarray([1, 2, 3, -4, -5])
        >>> output = np.rad2deg(x)
        >>> print(output)
        [  57.295776  114.59155   171.88733  -229.1831   -286.47888 ]
    """
    _check_input_tensor(x)

    def convert(a):
        return a * 180.0 / pi
    return _apply_tensor_op(convert, x, out=out, where=where, dtype=dtype)


def add(x1, x2, out=None, where=True, dtype=None):
    """
    Adds arguments element-wise.

    Note:
        Numpy arguments `casting`, `order`, `dtype`, `subok`, `signature`, and `extobj` are
        not supported.
        When `where` is provided, `out` must have a tensor value. `out` is not supported
        for storing the result, however it can be used in combination with `where` to set
        the value at indices for which `where` is set to False.

    Args:
        x1 (Tensor): input to be added.
        x2 (Tensor): input to be added.
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
        Tensor or scalar, the sum of `x1` and `x2`, element-wise. This is a scalar
        if both `x1` and `x2` are scalars.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x1 = np.full((3, 2), [1, 2])
        >>> x2 = np.full((3, 2), [3, 4])
        >>> output = np.add(x1, x2)
        >>> print(output)
        [[4, 6],
        [4, 6],
        [4, 6]]
    """
    # broadcast is not fully supported in tensor_add on CPU,
    # so we use tensor_sub as a substitute solution
    if _get_device() == 'CPU':
        _check_input_tensor(x1, x2)
        return subtract(x1, F.neg_tensor(x2), out=out, where=where, dtype=dtype)
    return _apply_tensor_op(F.tensor_add, x1, x2, out=out, where=where, dtype=dtype)


def subtract(x1, x2, out=None, where=True, dtype=None):
    """
    Subtracts arguments, element-wise.

    Note:
        Numpy arguments `casting`, `order`, `dtype`, `subok`, `signature`, and `extobj` are
        not supported.
        When `where` is provided, `out` must have a tensor value. `out` is not supported
        for storing the result, however it can be used in combination with `where` to set
        the value at indices for which `where` is set to False.

    Args:
        x1 (Tensor): the input to be subtracted from.
        x2 (Tensor): the input to be subtracted by.
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
        Tensor or scalar, the difference of `x1` and `x2`, element-wise. This is a
        scalar if both `x1` and `x2` are scalars.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x1 = np.full((3, 2), [1, 2])
        >>> x2 = np.full((3, 2), [3, 4])
        >>> output = np.subtract(x1, x2)
        >>> print(output)
        [[-2, -2],
        [-2, -2],
        [-2, -2]]
    """
    return _apply_tensor_op(F.tensor_sub, x1, x2, out=out, where=where, dtype=dtype)


def multiply(x1, x2, out=None, where=True, dtype=None):
    """
    Multiplies arguments element-wise.

    Note:
        Numpy arguments `casting`, `order`, `dtype`, `subok`, `signature`, and `extobj` are
        not supported.
        When `where` is provided, `out` must have a tensor value. `out` is not supported
        for storing the result, however it can be used in combination with `where` to set
        the value at indices for which `where` is set to False.

    Args:
        x1 (Tensor): input tensor to be multiplied.
        x2 (Tensor): input tensor to be multiplied.
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
        Tensor or scalar, the product of `x1` and `x2`, element-wise. This is a scalar
        if both `x1` and `x2` are scalars.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x1 = np.full((3, 2), [1, 2])
        >>> x2 = np.full((3, 2), [3, 4])
        >>> output = np.multiply(x1, x2)
        >>> print(output)
        [[3, 8],
        [3, 8],
        [3, 8]]
    """
    if _get_device() == 'CPU':
        _check_input_tensor(x1, x2)
        # broadcast is not fully supported on CPU backend,
        # and explicit broadcasting is performed
        shape_out = _infer_out_shape(F.shape(x1), F.shape(x2))
        x1 = _broadcast_to_shape(x1, shape_out)
        x2 = _broadcast_to_shape(x2, shape_out)
    return _apply_tensor_op(F.tensor_mul, x1, x2, out=out, where=where, dtype=dtype)


def divide(x1, x2, out=None, where=True, dtype=None):
    """
    Returns a true division of the inputs, element-wise.

    Instead of the Python traditional ‘floor division’, this returns a true
    division.

    Note:
        Numpy arguments `casting`, `order`, `dtype`, `subok`, `signature`, and `extobj` are
        not supported.
        When `where` is provided, `out` must have a tensor value. `out` is not supported
        for storing the result, however it can be used in combination with `where` to set
        the value at indices for which `where` is set to False.

    Args:
        x1 (Tensor): the divident.
        x2 (Tensor): the divisor.
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
        Tensor or scalar, this is a scalar if both `x1` and `x2` are scalars.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x1 = np.full((3, 2), [1, 2])
        >>> x2 = np.full((3, 2), [3, 4])
        >>> output = np.divide(x1, x2)
        >>> print(output)
        [[0.33333333, 0.5],
        [0.33333333, 0.5],
        [0.33333333, 0.5]]
    """
    if not _check_is_float(F.dtype(x1)) and not _check_is_float(F.dtype(x2)):
        x1 = F.cast(x1, mstype.float32)
        x2 = F.cast(x2, mstype.float32)
    return _apply_tensor_op(F.tensor_div, x1, x2, out=out, where=where, dtype=dtype)


def true_divide(x1, x2, out=None, where=True, dtype=None):
    """
    Returns a true division of the inputs, element-wise.

    Instead of the Python traditional ‘floor division’, this returns a true
    division.

    Note:
        Numpy arguments `casting`, `order`, `dtype`, `subok`, `signature`, and `extobj` are
        not supported.
        When `where` is provided, `out` must have a tensor value. `out` is not supported
        for storing the result, however it can be used in combination with `where` to set
        the value at indices for which `where` is set to False.

    Args:
        x1 (Tensor): the divident.
        x2 (Tensor): the divisor.
        out (Tensor or None, optional)
        where (Tensor, optional):
            This condition is broadcast over the input. At locations where the
            condition is True, the out array will be set to the ufunc result.
            Elsewhere, the out array will retain its original value. Note that
            if an uninitialized out array is created via the default out=None,
            locations within it where the condition is False will remain
            uninitialized.
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, this is a scalar if both `x1` and `x2` are scalars.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x1 = np.full((3, 2), [1, 2])
        >>> x2 = np.full((3, 2), [3, 4])
        >>> output = np.true_divide(x1, x2)
        >>> print(output)
        [[0.33333333, 0.5],
        [0.33333333, 0.5],
        [0.33333333, 0.5]]
    """
    return divide(x1, x2, out=out, where=where, dtype=dtype)


def power(x1, x2, out=None, where=True, dtype=None):
    """
    First array elements raised to powers from second array, element-wise.

    Raises each base in `x1` to the positionally-corresponding power in `x2`.

    Note:
        Numpy arguments `casting`, `order`, `dtype`, `subok`, `signature`, and `extobj` are
        not supported.
        When `where` is provided, `out` must have a tensor value. `out` is not supported
        for storing the result, however it can be used in combination with `where` to set
        the value at indices for which `where` is set to False.
        On GPU, the supported dtypes are np.float16, and np.float32.

    Args:
        x1 (Tensor): the bases.
        x2 (Tensor): the exponents.
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
        Tensor or scalar, the bases in `x1` raised to the exponents in `x2`. This
        is a scalar if both `x1` and `x2` are scalars.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x1 = np.full((3, 2), [1, 2]).astype('float32')
        >>> x2 = np.full((3, 2), [3, 4]).astype('float32')
        >>> output = np.power(x1, x2)
        >>> print(output)
        [[ 1, 16],
        [ 1, 16],
        [ 1, 16]]
    """
    return _apply_tensor_op(F.tensor_pow, x1, x2, out=out, where=where, dtype=dtype)


def float_power(x1, x2, out=None, where=True, dtype=None):
    """
    First array elements raised to powers from second array, element-wise.

    Raise each base in `x1` to the positionally-corresponding power in `x2`. `x1` and
    `x2` must be broadcastable to the same shape. This differs from the power
    function in that integers, float16, and float64 are promoted to floats with
    a minimum precision of float32 so that the result is always inexact. The
    intent is that the function will return a usable result for negative powers
    and seldom overflow for positive powers.

    Note:
        Numpy arguments `casting`, `order`, `dtype`, `subok`, `signature`, and `extobj` are
        not supported.
        When `where` is provided, `out` must have a tensor value. `out` is not supported
        for storing the result, however it can be used in combination with `where` to set
        the value at indices for which `where` is set to False.
        Integers and floats are promoted to float32 instead of float64.

    Args:
        x1 (Tensor): the bases.
        x2 (Tensor): the exponenets.
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
        Tensor or scalar, the bases in `x1` raised to the exponents in `x2`. This
        is a scalar if both `x1` and `x2` are scalars.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x1 = np.arange(6)
        >>> x2 = np.array(3)
        >>> output = np.float_power(x1, x2)
        >>> print(output)
        [  0.   1.   8.  27.  64. 125.]
    """
    if not _check_same_type(F.dtype(x1), mstype.float32):
        x1 = F.cast(x1, mstype.float32)
    if not _check_same_type(F.dtype(x2), mstype.float32):
        x2 = F.cast(x2, mstype.float32)

    return _apply_tensor_op(F.tensor_pow, x1, x2, out=out, where=where, dtype=dtype)


def minimum(x1, x2, out=None, where=True, dtype=None):
    """
    Element-wise minimum of tensor elements.

    Compares two tensors and returns a new tensor containing the element-wise minima.

    Note:
        Numpy arguments `casting`, `order`, `dtype`, `subok`, `signature`, and `extobj` are
        not supported.
        When `where` is provided, `out` must have a tensor value. `out` is not supported
        for storing the result, however it can be used in combination with `where` to set
        the value at indices for which `where` is set to False.
        Unlike numpy, when one of the elements is a NaN, the second element is
        always returned regardless of whether the second element is a NaN, instead
        of returning NaN.

    Args:
        x1 (Tensor): first input tensor to be compared.
        x2 (Tensor): second input tensor to be compared.
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
       Tensor, element-wise minimum of `x1` and `x2`.

    Raises:
        TypeError: If inputs have types not specified above.
        ValueError: If the shapes of `x1` and `x2` cannot be broadcast.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.asarray([1, 2])
        >>> b = np.asarray([[1, 3],[1, 4]])
        >>> print(np.minimum(a, b))
        [[1 2]
        [1 2]]
    """
    if isinstance(x1, (int, float, bool, list, tuple, Tensor)) and \
       isinstance(x2, (int, float, bool, list, tuple, Tensor)):
        x1 = asarray_const(x1)
        x2 = asarray_const(x2)
    else:
        _raise_type_error("Input x1 and x2 are expected to be array_like")
    # if both are scalars, expand x1 to 1d tensor, since cpu kernel doesn't support
    # comparisons with 2 scalars
    if x1.ndim == 0 and x2.ndim == 0:
        x1 = expand_dims(x1, 0)
        return _apply_tensor_op(F.minimum, x1, x2, out=out, where=where, dtype=dtype).squeeze()
    if x1.ndim == 0:
        dtype = x2.dtype
    elif x2.ndim == 0:
        dtype = x1.dtype
    return _apply_tensor_op(F.minimum, x1, x2, out=out, where=where, dtype=dtype)


def mean(a, axis=None, keepdims=False, dtype=None):
    """
    Computes the arithmetic mean along the specified axis.

    Returns the average of the array elements. The average is taken
    over the flattened array by default, otherwise over the specified
    axis.

    Note:
        Numpy arguments `out` is not supported.
        On GPU, the supported dtypes are np.float16, and np.float32.

    Args:
        a (Tensor): input tensor containing numbers whose mean is desired.
                    If a is not an array, a conversion is attempted.
        axis (None or int or tuple of ints, optional): Axis or axes along
                    which the means are computed. The default is to compute
                    the mean  of the flattened array. If this is a tuple of
                    ints, a mean is performed over multiple axes.
        keepdims (bool, optional): If this is set to True, the axes which
                    are reduced are left in the result as dimensions with
                    size one. With this option, the result will broadcast
                    correctly against the input tensor.
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, an array containing the mean values.

    Raises:
        ValueError: if axes are out of the range of ``[-a.ndim, a.ndim)``, or
            if the axes contain duplicates.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.arange(6, dtype='float32')
        >>> output = np.mean(a, 0)
        >>> print(output)
        2.5
    """

    axis = _check_axis_valid(axis, F.rank(a))
    shape_a = F.shape(a)
    if dtype is None:
        dtype = F.dtype(a)

    if _is_shape_empty(shape_a):
        if keepdims:
            shape_out = _shape_reduced_keepdims(shape_a, axis)
        else:
            shape_out = _shape_reduced(shape_a, axis)
        if _is_shape_empty(shape_out):
            return empty(F.dtype(a), shape_out)
        return full(shape_out, nan, dtype)

    if _is_scalar(shape_a):
        if keepdims:
            return a
        shape_out = _shape_reduced(shape_a, axis)
        return F.reshape(a, shape_out)

    if keepdims:
        res = _mean_keepdims(a, axis)
    else:
        res = _mean_default(a, axis)
    if not _check_same_type(dtype, F.dtype(res)):
        res = F.cast(res, dtype)
    return res


def inner(a, b):
    """
    Returns the inner product of two tensors.

    Ordinary inner product of vectors for 1-D tensors (without complex
    conjugation), in higher dimensions a sum product over the last
    axes.

    Note:
        Numpy argument `out` is not supported.
        On GPU, the supported dtypes are np.float16, and np.float32.
        On CPU, the supported dtypes are np.float16, np.float32, and
        np.float64.

    Args:
        a (Tensor): input tensor. If `a` and `b` are nonscalar, their last
                    dimensions must match.
        b (Tensor): input tensor. If `a` and `b` are nonscalar, their last
                    dimensions must match.

    Returns:
        Tensor or scalar.

    Raises:
        ValueError: if ``x1.shape[-1] != x2.shape[-1]``.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.ones((5, 3))
        >>> b = np.ones((2, 7, 3))
        >>> output = np.inner(a, b)
        >>> print(output)
        [[[3. 3. 3. 3. 3. 3. 3.]
        [3. 3. 3. 3. 3. 3. 3.]]
        [[3. 3. 3. 3. 3. 3. 3.]
        [3. 3. 3. 3. 3. 3. 3.]]
        [[3. 3. 3. 3. 3. 3. 3.]
        [3. 3. 3. 3. 3. 3. 3.]]
        [[3. 3. 3. 3. 3. 3. 3.]
        [3. 3. 3. 3. 3. 3. 3.]]
        [[3. 3. 3. 3. 3. 3. 3.]
        [3. 3. 3. 3. 3. 3. 3.]]]
    """
    if F.rank(a) == 0 or F.rank(b) == 0:
        return F.tensor_mul(a, b)

    _check_shape_aligned(F.shape(a), F.shape(b))
    aligned_shape_a = (F.shape_mul(F.shape(a)[:-1]), F.shape(a)[-1])
    aligned_shape_b = (F.shape_mul(F.shape(b)[:-1]), F.shape(a)[-1])
    a_aligned = F.reshape(a, aligned_shape_a)
    b_aligned = F.reshape(b, aligned_shape_b)

    res = _matmul_T(a_aligned, b_aligned)
    res = F.reshape(res, F.shape(a)[:-1] + F.shape(b)[:-1])
    return res


def dot(a, b):
    """
    Returns the dot product of two arrays.

    Specifically,
    If both `a` and `b` are 1-D arrays, it is inner product of vectors
    (without complex conjugation).
    If both `a` and `b` are 2-D arrays, it is matrix multiplication.
    If either `a` or `b` is 0-D (scalar), it is equivalent to multiply.
    If `a` is an `N-D` array and `b` is a 1-D array, it is a sum product
    over the last axis of `a` and `b`.
    If `a` is an `N-D` array and `b` is an `M-D` array (where ``M>=2``), it is a
    sum product over the last axis of `a` and the second-to-last axis of `b`:
    ``dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])``

    Note:
        Numpy argument `out` is not supported.
        On GPU, the supported dtypes are np.float16, and np.float32.
        On CPU, the supported dtypes are np.float16, np.float32, and
        np.float64.

    Args:
        a (Tensor): input tensor
        b (Tensor): input tensor

    Returns:
        Tensor or scalar, the dot product of `a` and `b`. If `a` and `b` are
        both scalars or both 1-D arrays then a scalar is returned;
        otherwise an array is returned

    Raises:
        ValueError: If the last dimension of `a` is not the same size
            as the second-to-last dimension of `b`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.full((1, 3), 7).astype('float32')
        >>> b = np.full((2, 3, 4), 5).astype('float32')
        >>> output = np.dot(a, b)
        >>> print(output)
        [[[105, 105, 105, 105],
        [105, 105, 105, 105]]]
    """
    ndim_a, ndim_b = F.rank(a), F.rank(b)
    if ndim_a > 0 and ndim_b >= 2:
        perm = F.make_range(ndim_b)
        perm = perm[:-2] + (perm[-1],) + (perm[-2],)
        b = F.transpose(b, perm)
    return inner(a, b)


def outer(a, b):
    """
    Computes the outer product of two vectors.

    Given two vectors, ``a = [a0, a1, ..., aM]`` and ``b = [b0, b1, ..., bN]``,
    the outer product is:
    ``[[a0*b0  a0*b1 ... a0*bN ]``

    ``[a1*b0    .              ]``

    ``[ ...          .         ]``

    ``[aM*b0            aM*bN ]]``

    Note:
        Numpy argument ``out`` is not supported.
        On GPU, the supported dtypes are np.float16, and np.float32.
        On CPU, the supported dtypes are np.float16, np.float32, and
        np.float64.

    Args:
        a (Tensor): first input vector. Input is flattened if not
                    already 1-dimensional.
        b (Tensor): second input vector. Input is flattened if not
                    already 1-dimensional.

    Returns:
        Tensor or scalar, ``out[i, j] = a[i] * b[j]``.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.full(7, 2).astype('float32')
        >>> b = np.full(4, 3).astype('float32')
        >>> output = np.outer(a, b)
        >>> print(output)
        [[6, 6, 6, 6],
        [6, 6, 6, 6],
        [6, 6, 6, 6],
        [6, 6, 6, 6],
        [6, 6, 6, 6],
        [6, 6, 6, 6],
        [6, 6, 6, 6]]
    """
    _check_input_tensor(a, b)
    if F.rank(a) != 1:
        a = ravel(a)
    if F.rank(b) != 1:
        b = ravel(b)
    a = F.reshape(a, (F.shape(a)[0], 1))
    b = _expand(b, 2)
    return _matmul(a, b)


def tensordot(a, b, axes=2):
    """
    Computes tensor dot product along specified axes.

    Given two tensors, `a` and `b`, and an array_like object containing two array_like
    objects, `(a_axes, b_axes)`, sum the products of `a`’s and `b`’s elements (components)
    over the axes specified by `a_axes` and `b_axes`. The third argument can be a single
    non-negative integer_like scalar, `N`; if it is such, then the last `N` dimensions of
    `a` and the first `N` dimensions of `b` are summed over.
    Three common use cases are:

        ``axes = 0`` : tensor product

        ``axes = 1`` : tensor dot product

        ``axes = 2`` : (default) tensor double contraction

    When axes is integer_like, the sequence for evaluation will be: first the `-Nth`
    axis in `a` and 0th axis in `b`, and the -1th axis in `a` and `Nth` axis in `b` last.
    When there is more than one axis to sum over - and they are not the last (first)
    axes of `a` `(b)` - the argument axes should consist of two sequences of the same
    length, with the first axis to sum over given first in both sequences, the second
    axis second, and so forth.
    The shape of the result consists of the non-contracted axes of the first tensor,
    followed by the non-contracted axes of the second.

    Note:
        On CPU, the supported dypes are np.float16 and np.float32.
        On GPU, the supported dypes are np.float16 and np.float32.

    Args:
        a (Tensor): Tensor to "dot".
        b (Tensor): Tensor to “dot”.
        axes (int or sequence of ints):

            integer_like: If an int `N`, sum over the last `N` axes of `a` and the first `N`
            axes of `b` in order. The sizes of the corresponding axes must match.

            sequence of ints: Or, a list of axes to be summed over, first sequence
            applying to `a`, second to `b`. Both elements `array_like` must be of the same
            length.

    Returns:
        Tensor, or list of tensors, the tensor dot product of the input.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> a = np.ones((3, 4, 5))
        >>> b = np.ones((4, 3, 2))
        >>> output = np.tensordot(a, b, axes=([1,0],[0,1]))
        >>> print(output.shape)
        (5, 2)
    """
    if F.rank(a)*F.rank(b) == 0 and axes == 0:
        return F.tensor_mul(a, b)
    return C.tensor_dot(a, b, axes)


def std(x, axis=None, ddof=0, keepdims=False):
    """
    Computes the standard deviation along the specified axis.
    The standard deviation is the square root of the average of the squared deviations
    from the mean, i.e., :math:`std = sqrt(mean(abs(x - x.mean())**2))`.

    Returns the standard deviation, which is computed for the flattened array by default,
    otherwise over the specified axis.

    Note:
        Numpy arguments `dtype` and `out` are not supported.

    Args:
        x (Tensor): A Tensor to be calculated.
        axis (Union[None, int, tuple(int)]): Axis or axes along which the standard
            deviation is computed. Default: `None`.

            If `None`, compute the standard deviation of the flattened array.
        ddof (int): Means Delta Degrees of Freedom. The divisor used in calculations is :math:`N - ddof`,
            where :math:`N` represents the number of elements. Default: 0.
        keepdims: Default: `False`.

    Returns:
        Standard deviation tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> input_x = np.array([1., 2., 3., 4.])
        >>> output = np.std(input_x)
        >>> print(output)
        1.118034
    """
    if _is_shape_empty(x.shape):
        return full((), nan, F.dtype(x))

    if not isinstance(ddof, int):
        _raise_type_error("integer argument expected, but got ", ddof)
    if axis is None:
        axis = ()
    else:
        _check_axis_type(axis, True, True, False)
        axis = _canonicalize_axis(axis, x.ndim)

    x_mean = _mean_keepdims(x, axis)
    x_sub = F.tensor_sub(x, x_mean)
    x_pow = F.tensor_pow(x_sub, 2)
    if keepdims:
        x_sum = _reduce_sum_keepdims(x_pow, axis)
    else:
        x_sum = _reduce_sum_default(x_pow, axis)

    if isinstance(axis, int):
        nums = x.shape[axis]
    else:
        nums = _get_size(x, axis)

    x_std = F.tensor_pow(F.tensor_div(x_sum, nums - ddof), 0.5)
    return x_std


def var(x, axis=None, ddof=0, keepdims=False):
    """
    Computes the variance along the specified axis.
    The variance is the average of the squared deviations from the mean, i.e.,
    :math:`var = mean(abs(x - x.mean())**2)`.

    Returns the variance, which is computed for the flattened array by default,
    otherwise over the specified axis.

    Note:
        Numpy arguments `dtype` and `out` are not supported.

    Args:
        x (Tensor): A Tensor to be calculated.
        axis (Union[None, int, tuple(int)]): Axis or axes along which the variance is computed.
            The default is to compute the variance of the flattened array. Default: `None`.
        ddof (int): Means Delta Degrees of Freedom. Default: 0.
            The divisor used in calculations is :math:`N - ddof`, where :math:`N` represents the number of elements.
        keepdims (bool): Default: `False`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Returns:
        Standard deviation tensor.

    Examples:
        >>> import mindspore.numpy as np
        >>> input_x = np.array([1., 2., 3., 4.])
        >>> output = np.var(input_x)
        >>> print(output)
        1.25
    """
    if _is_shape_empty(x.shape):
        return full((), nan, F.dtype(x))

    x_std = std(x, axis, ddof, keepdims)
    return F.tensor_pow(x_std, 2)


def ptp(x, axis=None, out=None, keepdims=False):
    """
    Range of values (maximum - minimum) along an axis.
    The name of the function comes from the acronym for ‘peak to peak’.

    Note:
        Numpy arguments `dtype` and `out` are not supported.

    Args:
        x (Tensor): Input tensor.
        axis (Union[None, int, tuple(int)]): Axis or axes along which the range is computed.
            The default is to compute the variance of the flattened array. Default: None.
        keepdims (bool): Default is False.

    Returns:
        Tensor.

    Raises:
        TypeError: if inputs have types not specified above.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.array([[4.0, 9.0, 2.0, 10.0], [6.0, 9.0, 7.0, 12.0]])
        >>> print(np.ptp(x, axis=1))
        [8. 6.]
        >>> print(np.ptp(x, axis=0))
        [2. 0. 5. 2.]
    """
    _check_input_tensor(x)
    if axis is None:
        axis = ()
    else:
        _check_axis_type(axis, True, True, False)
        axis = _canonicalize_axis(axis, x.ndim)

    if keepdims:
        x_min = _reduce_min_keepdims(x, axis)
        x_max = _reduce_max_keepdims(x, axis)
    else:
        x_min = _reduce_min_default(x, axis)
        x_max = _reduce_max_default(x, axis)
    return F.tensor_sub(x_max, x_min)


def average(x, axis=None, weights=None, returned=False):
    """
    Computes the weighted average along the specified axis.

    Args:
        x (Tensor): A Tensor to be averaged.
        axis (Union[None, int, tuple(int)]): Axis along which to average `x`. Default: `None`.
            If the axis is `None`, it will average over all of the elements of the tensor `x`.
            If the axis is negative, it counts from the last to the first axis.
        weights (Tensor): Weights associated with the values in `x`. Default: `None`.
            If `weights` is `None`, all the data in `x` are assumed to have a weight equal to one.
            If `weights` is 1-D tensor, the length must be the same as the given axis.
            Otherwise, `weights` should have the same shape as `x`.
        returned (bool): Default: `False`.
            If `True`, the tuple (average, sum_of_weights) is returned.
            If `False`, only the average is returned.

    Returns:
        Averaged Tensor. If returned is `True`, return tuple.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> input_x = np.array([[1., 2.], [3., 4.]])
        >>> output = np.average(input_x, axis=0, weights=input_x, returned=True)
        >>> print(output)
        (Tensor(shape=[2], dtype=Float32, value= [ 2.50000000e+00,  3.33333325e+00]),
         Tensor(shape=[2], dtype=Float32, value= [ 4.00000000e+00,  6.00000000e+00]))
    """
    if axis is None:
        axis = ()
    else:
        _check_axis_type(axis, True, True, False)
        axis = _canonicalize_axis(axis, x.ndim)

    if weights is None:
        return mean(x, axis)

    x_avg = full((), nan, F.dtype(x))
    sum_of_weights = None
    if x.shape == weights.shape:
        x_avg, sum_of_weights = comput_avg(x, axis, weights)
    elif F.rank(weights) == 1:
        if not isinstance(axis, int):
            _raise_type_error("Axis must be specified when shapes of x and weights differ.")
        weights = _broadcast_to_shape(weights, x.shape)
        x_avg, sum_of_weights = comput_avg(x, axis, weights)
    else:
        _raise_type_error("Weights should be None, 1-D or the same as input x, but got shape of", weights)

    if returned:
        return (x_avg, sum_of_weights)
    return x_avg


def comput_avg(x, axis, weights):
    """Computes average value of input x with given parameters."""
    x_mul = F.tensor_mul(x, weights)
    x_sum = _reduce_sum_default(x_mul, axis)
    sum_of_weights = _reduce_sum_default(weights, axis)
    x_avg = F.tensor_div(x_sum, sum_of_weights)
    return x_avg, sum_of_weights


def matmul(x1, x2, dtype=None):
    """
    Returns the matrix product of two arrays.

    Note:
        Numpy arguments `out`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.
        On GPU, the supported dtypes are np.float16 and np.float32.
        On CPU, the supported dtypes are np.float16 and np.float32.

    Args:
        x1 (Tensor): Input tensor, scalar not allowed.
        x2 (Tensor): Input tensor, scalar not allowed.
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, the matrix product of the inputs. This is a scalar only
        when both `x1`, `x2` are 1-d vectors.

    Raises:
        ValueError: If the last dimension of `x1` is not the same size as the
            second-to-last dimension of `x2`, or if a scalar value is passed in.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x1 = np.arange(2*3*4).reshape(2, 3, 4).astype('float32')
        >>> x2 = np.arange(4*5).reshape(4, 5).astype('float32')
        >>> output = np.matmul(x1, x2)
        >>> print(output)
        [[[  70.   76.   82.   88.   94.]
        [ 190.  212.  234.  256.  278.]
        [ 310.  348.  386.  424.  462.]]
        [[ 430.  484.  538.  592.  646.]
        [ 550.  620.  690.  760.  830.]
        [ 670.  756.  842.  928. 1014.]]]
    """
    # performs type promotion
    dtype1 = F.dtype(x1)
    dtype2 = F.dtype(x2)
    dtype_out = _promote(dtype1, dtype2)
    if not _check_same_type(dtype1, dtype_out):
        x1 = F.cast(x1, dtype_out)
    if not _check_same_type(dtype2, dtype_out):
        x2 = F.cast(x2, dtype_out)

    ndim1_orig, ndim2_orig = F.rank(x1), F.rank(x2)
    shape1_orig, shape2_orig = F.shape(x1), F.shape(x2)
    _check_matmul_shapes(shape1_orig, shape2_orig)
    ndim_aligned = _max(ndim1_orig, ndim2_orig)
    transpose_b = ndim2_orig == 1
    shape_backbone = _infer_out_shape(
        shape1_orig[:-2], shape2_orig[:-2])
    # infers the shape of the output
    shape_out = shape_backbone + _infer_shape_rem(shape1_orig, shape2_orig,
                                                  ndim1_orig, ndim2_orig, transpose_b)

    x1 = _expand(x1, _max(ndim_aligned, 2))
    x2 = _expand(x2, _max(ndim_aligned, 2))
    shape1_aligned, shape2_aligned = F.shape(x1), F.shape(x2)

    if ndim_aligned <= 2:
        res = P.MatMul(False, transpose_b)(x1, x2)
    else:
        # broadcasts x1.shape[:-2] with x2.shape[:-2]
        shape_aligned = shape_backbone + _infer_shape_rem(shape1_aligned, shape2_aligned,
                                                          ndim_aligned, ndim_aligned,
                                                          transpose_b)
        x1 = _broadcast_to(x1, shape1_aligned[:-2], shape_aligned[:-2], ndim_aligned)
        x2 = _broadcast_to(x2, shape2_aligned[:-2], shape_aligned[:-2], ndim_aligned)
        res = P.BatchMatMul(False, transpose_b)(x1, x2)

    if dtype is not None and not _check_same_type(dtype_out, dtype):
        res = F.cast(res, dtype)
    return F.reshape(res, shape_out)


def square(x, out=None, where=True, dtype=None):
    """
    Returns the element-wise square of the input.

    Note:
        Numpy arguments `casting`, `order`, `dtype`, `subok`, `signature`, and `extobj` are
        not supported.
        When `where` is provided, `out` must have a tensor value. `out` is not supported
        for storing the result, however it can be used in combination with `where` to set
        the value at indices for which `where` is set to False.
        On GPU, the supported dtypes are np.float16 and np.float32.

    Args:
        x (Tensor): Input data.
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
        Tensor or scalar, element-wise ``x*x``, of the same shape and dtype as `x`.
        This is a scalar if `x` is a scalar..

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = np.square(np.arange(6).reshape(2, 3).astype('float32'))
        >>> print(x)
        [[ 0.  1.  4.]
        [ 9. 16. 25.]]
    """
    return _apply_tensor_op(F.square, x, out=out, where=where, dtype=dtype)


def sqrt(x, out=None, where=True, dtype=None):
    """
    Returns the non-negative square-root of an array, element-wise.

    Note:
        Numpy arguments `casting`, `order`, `dtype`, `subok`, `signature`, and `extobj` are
        not supported.
        When `where` is provided, `out` must have a tensor value. `out` is not supported
        for storing the result, however it can be used in combination with `where` to set
        the value at indices for which `where` is set to False.
        On GPU, the supported dtypes are np.float16 and np.float32.

    Args:
        x (Tensor): The values whose square-roots are required.
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
        Tensor or scalar, an array of the same shape as `x`, containing the positive
        square-root of each element in `x`. For negative elements, nan is returned.
        This is a scalar if `x` is a scalar.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = np.arange(6).reshape(2, 3).astype('float32')
        >>> x_squared = np.square(x)
        >>> output = np.sqrt(x_squared)
        >>> print(output)
        [[ 0. 1. 2.]
        [ 3. 4. 5.]]
    """
    return _apply_tensor_op(F.sqrt, x, out=out, where=where, dtype=dtype)


def reciprocal(x, out=None, where=True, dtype=None):
    """
    Returns the reciprocal of the argument, element-wise.

    Calculates ``1/x``.

    Note:
        Numpy arguments `casting`, `order`, `dtype`, `subok`, `signature`, and `extobj` are
        not supported.
        When `where` is provided, `out` must have a tensor value. `out` is not supported
        for storing the result, however it can be used in combination with `where` to set
        the value at indices for which `where` is set to False.

    Args:
        x (Tensor): Input array. For integer arguments with absolute value larger
            than 1 the result is always zero because of the way Python handles
            integer division. For integer zero the result is an overflow.
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
        Tensor or scalar, this is a scalar if `x` is a scalar.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = np.arange(1, 7).reshape(2, 3).astype('float32')
        >>> output = np.reciprocal(x)
        >>> print(output)
        [[1.         0.5        0.33333334]
        [0.25       0.2        0.16666667]]
    """
    return _apply_tensor_op(lambda x: F.tensor_div(1, x), x, out=out, where=where, dtype=dtype)


def log(x, out=None, where=True, dtype=None):
    """
    Returns the natural logarithm, element-wise.

    The natural logarithm log is the inverse of the exponential function, so that
    ``log(exp(x)) = x``. The natural logarithm is logarithm in base e.

    Note:
        Numpy arguments `casting`, `order`, `dtype`, `subok`, `signature`, and `extobj` are
        not supported.
        When `where` is provided, `out` must have a tensor value. `out` is not supported
        for storing the result, however it can be used in combination with `where` to set
        the value at indices for which `where` is set to False.
        On GPU, the supported dtypes are np.float16, and np.float32.
        On CPU, the supported dtypes are np.float16, np.float32, and np.float64.

    Args:
        x (Tensor): Input array. For integer arguments with absolute value larger
            than 1 the result is always zero because of the way Python handles
            integer division. For integer zero the result is an overflow.
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
        Tensor or scalar, the natural logarithm of `x`, element-wise. This is a
        scalar if `x` is a scalar.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = np.array([1, 2, 3]).astype('float32')
        >>> output = np.log(x)
        >>> print(output)
        [1.09861   1.3862929 1.6094407]
    """
    return _apply_tensor_op(F.log, x, out=out, where=where, dtype=dtype)


def maximum(x1, x2, out=None, where=True, dtype=None):
    """
    Returns the element-wise maximum of array elements.

    Compares two arrays and returns a new array containing the element-wise maxima.

    Note:
        Numpy arguments `casting`, `order`, `dtype`, `subok`, `signature`, and `extobj` are
        not supported.
        When `where` is provided, `out` must have a tensor value. `out` is not supported
        for storing the result, however it can be used in combination with `where` to set
        the value at indices for which `where` is set to False.
        Unlike numpy, when one of the elements is a NaN, the second element is
        always returned regardless of whether the second element is a NaN, instead
        of returning NaN.

    Args:
        x1 (Tensor): Input array
        x2 (Tensor): The array holding the elements to be compared. If
            ``x1.shape != x2.shape``, they must be broadcastable to a common shape
            (which becomes the shape of the output).
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
        Tensor or scalar, the maximum of `x1` and `x2`, element-wise. This is a scalar
        if both `x1` and `x2` are scalars.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> output = np.maximum(np.array([2, 3, 4]), np.array([1, 5, 2]))
        >>> print(output)
        [2 5 4]
    """
    if isinstance(x1, (int, float, bool, list, tuple, Tensor)) and \
       isinstance(x2, (int, float, bool, list, tuple, Tensor)):
        x1 = asarray_const(x1)
        x2 = asarray_const(x2)
    else:
        _raise_type_error("Input x1 and x2 are expected to be array_like")
    # F.maximum does not support when both operands are scalar
    if x1.ndim == 0 and x2.ndim == 0:
        x1 = expand_dims(x1, 0)
        return _apply_tensor_op(F.maximum, x1, x2, out=out, where=where, dtype=dtype).squeeze()
    if x1.ndim == 0:
        dtype = x2.dtype
    elif x2.ndim == 0:
        dtype = x1.dtype
    return _apply_tensor_op(F.maximum, x1, x2, out=out, where=where, dtype=dtype)


def heaviside(x1, x2, out=None, where=True, dtype=None):
    """
    Computes the Heaviside step function.

    Note:
        Numpy arguments `casting`, `order`, `dtype`, `subok`, `signature`, and `extobj` are
        not supported.
        When `where` is provided, `out` must have a tensor value. `out` is not supported
        for storing the result, however it can be used in combination with `where` to set
        the value at indices for which `where` is set to False.

    Args:
        x1 (Tensor): Input values.
        x2 (Tensor): The value of the function when `x1` is 0. If
            ``x1.shape != x2.shape``, they must be broadcastable to a common shape
            (which becomes the shape of the output).
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
        Tensor or scalar, the output array, element-wise Heaviside step function
        of `x1`. This is a scalar if both `x1` and `x2` are scalars.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> output = np.heaviside(np.array([-1.5, 0, 2.0]), np.array(0.5))
        >>> print(output)
        [0.  0.5 1. ]
        >>> output = np.heaviside(np.array([-1.5, 0, 2.0]), np.array(1))
        >>> print(output)
        [0. 1. 1.]
    """

    def _heaviside(x1, x2):
        """Computes heaviside without passing keyword arguments"""
        # performs type promotion
        dtype1 = F.dtype(x1)
        dtype2 = F.dtype(x2)
        dtype_out = _promote(dtype1, dtype2)
        if not _check_same_type(dtype1, dtype_out):
            x1 = F.cast(x1, dtype_out)
        if not _check_same_type(dtype2, dtype_out):
            x2 = F.cast(x2, dtype_out)

        # performs broadcast
        shape_out = _infer_out_shape(F.shape(x1), F.shape(x2))
        x1 = _broadcast_to_shape(x1, shape_out)
        x2 = _broadcast_to_shape(x2, shape_out)

        x2 = F.select(x1 < 0, zeros(shape_out, dtype_out), x2)
        x2 = F.select(x1 > 0, ones(shape_out, dtype_out), x2)
        return x2

    return _apply_tensor_op(_heaviside, x1, x2, out=out, where=where, dtype=dtype)


def amax(a, axis=None, keepdims=False, initial=None, where=True):
    """
    Returns the maximum of an array or maximum along an axis.

    Note:
        Numpy argument `out` is not supported.
        On GPU, the supported dtypes are np.float16, and np.float32.

    Args:
        a (Tensor): Input data.
        axis (None or int or tuple of ints, optional): defaults to None. Axis or
            axes along which to operate. By default, flattened input is used. If
            this is a tuple of ints, the maximum is selected over multiple axes,
            instead of a single axis or all the axes as before.
        keepdims (boolean, optional): defaults to False.
            If this is set to True, the axes which are reduced are left in the
            result as dimensions with size one. With this option, the result will
            broadcast correctly against the input array.
        initial (scalar, optional):
            The minimum value of an output element. Must be present to allow
            computation on empty slice.
        where (boolean Tensor, optional): defaults to True.
            A boolean array which is broadcasted to match the dimensions of array,
            and selects elements to include in the reduction. If non-default value
            is passed, initial must also be provided.

    Returns:
        Tensor or scalar, maximum of `a`. If `axis` is None, the result is a scalar
        value. If `axis` is given, the result is an array of dimension ``a.ndim - 1``.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> a = np.arange(4).reshape((2,2)).astype('float32')
        >>> output = np.amax(a)
        >>> print(output)
        3.0
        >>> output = np.amax(a, axis=0)
        >>> print(output)
        [2. 3.]
        >>> output = np.amax(a, axis=1)
        >>> print(output)
        [1. 3.]
        >>> output = np.amax(a, where=np.array([False, True]), initial=-1, axis=0)
        >>> print(output)
        [-1.  3.]
    """
    return _reduce(a, P.ReduceMax(keepdims), F.maximum, axis=axis, keepdims=keepdims,
                   initial=initial, where=where)


def amin(a, axis=None, keepdims=False, initial=None, where=True):
    """
    Returns the minimum of an array or minimum along an axis.

    Note:
        Numpy argument `out` is not supported.
        On GPU, the supported dtypes are np.float16, and np.float32.

    Args:
        a (Tensor): Input data.
        axis (None or int or tuple of ints, optional): defaults to None. Axis or
            axes along which to operate. By default, flattened input is used. If
            this is a tuple of ints, the maximum is selected over multiple axes,
            instead of a single axis or all the axes as before.
        keepdims (boolean, optional): defaults to False.
            If this is set to True, the axes which are reduced are left in the
            result as dimensions with size one. With this option, the result will
            broadcast correctly against the input array.
        initial (scalar, optional):
            The maximum value of an output element. Must be present to allow
            computation on empty slice.
        where (boolean Tensor, optional): defaults to True.
            A boolean array which is broadcasted to match the dimensions of array,
            and selects elements to include in the reduction. If non-default value
            is passed, initial must also be provided.

    Returns:
        Tensor or scalar, minimum of `a`. If axis is None, the result is a scalar
        value. If `axis` is given, the result is an array of dimension ``a.ndim - 1``.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> a = np.arange(4).reshape((2,2)).astype('float32')
        >>> output = np.amin(a)
        >>> print(output)
        0.0
        >>> output = np.amin(a, axis=0)
        >>> print(output)
        [0. 1.]
        >>> output = np.amin(a, axis=1)
        >>> print(output)
        [1. 3.]
        >>> output = np.amax(a, where=np.array([False, True]), initial=10, axis=0)
        >>> print(output)
        [10.  1.]
    """
    return _reduce(a, P.ReduceMin(keepdims), F.minimum, axis=axis, keepdims=keepdims,
                   initial=initial, where=where)


def hypot(x1, x2, out=None, where=True, dtype=None):
    """
    Given the “legs” of a right triangle, returns its hypotenuse.

    Equivalent to ``sqrt(x1**2 + x2**2)``, element-wise. If `x1` or `x2` is scalar_like
    (i.e., unambiguously cast-able to a scalar type), it is broadcast for use
    with each element of the other argument. (See Examples)

    Note:
        Numpy arguments `casting`, `order`, `dtype`, `subok`, `signature`, and `extobj` are
        not supported.
        When `where` is provided, `out` must have a tensor value. `out` is not supported
        for storing the result, however it can be used in combination with `where` to set
        the value at indices for which `where` is set to False.
        On GPU, the supported dtypes are np.float16 and np.float32.
        On CPU, the supported dtypes are np.float16, np.float32, and np.float64.

    Args:
        x1 (Tensor): Leg of the traingle(s).
        x2 (Tensor): Leg of the triangle(s). If ``x1.shape != x2.shape``, they
            must be broadcastable to a common shape (which becomes the shape of
            the output).
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
        Tensor or scalar, the hypotenuse of the triangle(s). This is a scalar if
        both `x1` and `x2` are scalars.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> output = np.hypot(3*np.ones((3, 3)), 4*np.ones((3, 3)))
        >>> print(output)
        [[5. 5. 5.]
        [5. 5. 5.]
        [5. 5. 5.]]
        >>> output = np.hypot(3*np.ones((3, 3)), np.array([4]))
        >>> print(output)
        [[5. 5. 5.]
        [5. 5. 5.]
        [5. 5. 5.]]
    """

    def _hypot(x1, x2):
        """Computes hypotenuse without passing keyword arguments"""
        if _get_device() == 'CPU':
            # broadcast is not fully supported in tensor_add on CPU,
            # so we use tensor_sub as a substitute solution
            return F.sqrt(F.tensor_sub(F.square(x1), F.neg_tensor(F.square(x2))))
        return F.sqrt(F.tensor_add(F.square(x1), F.square(x2)))

    return _apply_tensor_op(_hypot, x1, x2, out=out, where=where, dtype=dtype)


def floor(x, out=None, where=True, dtype=None):
    """
    Returns the floor of the input, element-wise.

    The floor of the scalar `x` is the largest integer `i`, such that ``i <= x``.

    Note:
        Numpy arguments `casting`, `order`, `dtype`, `subok`, `signature`, and `extobj` are
        not supported.
        When `where` is provided, `out` must have a tensor value. `out` is not supported
        for storing the result, however it can be used in combination with `where` to set
        the value at indices for which `where` is set to False.
        On GPU, the supported dtypes are np.float16 and np.float32.
        On CPU, the supported dtypes are np.float16, np.float32, and np.float64.

    Args:
        x (Tensor): input data.
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
        Tensor or scalar, the floor of each element in `x`. This is a scalar if `x`
        is a scalar.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> output = np.floor(np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]))
        >>> print(output)
        [-2. -2. -1.  0.  1.  1.  2.]
    """
    return _apply_tensor_op(F.floor, x, out=out, where=where, dtype=dtype)


def floor_divide(x1, x2, out=None, where=True, dtype=None):
    """
    Returns the largest integer smaller or equal to the division of the inputs.
    It is equivalent to the Python // operator and pairs with the
    Python % (remainder), function so that ``a = a % b + b * (a // b)`` up to roundoff.

    Note:
        Numpy arguments `casting`, `order`, `dtype`, `subok`, `signature`, and `extobj` are
        not supported.
        When `where` is provided, `out` must have a tensor value. `out` is not supported
        for storing the result, however it can be used in combination with `where` to set
        the value at indices for which `where` is set to False.

    Args:
        x1 (Tensor): Input array.
        x2 (Tensor): Input array.
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
        Tensor or scalar.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> output = np.floor_divide(np.array([1., 2., 3., 4.]), np.array(2.5))
        >>> print(output)
        [0. 0. 1. 1.]
    """
    return _apply_tensor_op(F.tensor_floordiv, x1, x2, out=out, where=where, dtype=dtype)


def _remainder(x1, x2, C_style=False):
    """Computes remainder without applying keyword arguments."""
    dtype = _promote(F.dtype(x1), F.dtype(x2))
    if not _check_is_float(dtype):
        x1 = F.cast(x1, mstype.float32)
        x2 = F.cast(x2, mstype.float32)

    quotient = F.tensor_div(x1, x2)
    if C_style:
        quotient = fix(quotient)
    else:
        quotient = F.floor(quotient)
    prod = F.tensor_mul(x2, quotient)
    res = F.tensor_sub(x1, prod)
    if _check_is_int(dtype):
        zeros_tensor = zeros(F.shape(quotient), F.dtype(quotient))
        x2_zeros = F.equal(x2, zeros_tensor)
        res = F.select(x2_zeros, zeros_tensor, res)

    if not _check_same_type(F.dtype(res), dtype):
        res = F.cast(res, dtype)
    return res


def remainder(x1, x2, out=None, where=True, dtype=None):
    """
    Returns element-wise remainder of division.

    Computes the remainder complementary to the floor_divide function. It is
    equivalent to the Python modulus operator ``x1 % x2`` and has the same sign
    as the divisor `x2`. The MATLAB function equivalent to np.remainder is mod.

    Note:
        Numpy arguments `casting`, `order`, `dtype`, `subok`, `signature`, and `extobj` are
        not supported.
        When `where` is provided, `out` must have a tensor value. `out` is not supported
        for storing the result, however it can be used in combination with `where` to set
        the value at indices for which `where` is set to False.

    Args:
        x1 (Tensor): input array.
        x2 (Tensor): input array.
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
        Tensor or scalar, the element-wise remainder of the quotient
        ``floor_divide(x1, x2)``. This is a scalar if both `x1` and `x2` are scalars.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> output = np.remainder(np.array([4, 7]), np.array([2, 3]))
        >>> print(output)
        [0 1]
        >>> output = np.remainder(np.arange(7), np.array(5))
        >>> print(output)
        [0 1 2 3 4 0 1]
    """
    return _apply_tensor_op(_remainder, x1, x2, out=out, where=where, dtype=dtype)


def fix(x):
    """
    Rounds to nearest integer towards zero.

    Rounds an array of floats element-wise to nearest integer towards zero. The
    rounded values are returned as floats.

    Note:
        Numpy argument `out` is not supported.

    Args:
        x (Tensor): An array of floats to be rounded.

    Returns:
        Tensor.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> output = np.fix(np.array([2.1, 2.9, -2.1, -2.9]))
        >>> print(output)
        [ 2.  2. -2. -2.]
    """
    _check_input_tensor(x)
    if not _check_is_float(F.dtype(x)):
        x = F.cast(x, mstype.float32)
    floored = F.floor(x)
    # TODO change to F.ceil once supported on CPU.
    ceiled = F.neg_tensor(F.floor(F.neg_tensor(x)))
    is_neg = F.tensor_lt(x, zeros(F.shape(x), F.dtype(x)))
    return F.select(is_neg, ceiled, floored)


def fmod(x1, x2, out=None, where=True, dtype=None):
    """
    Returns the element-wise remainder of division.

    This is the NumPy implementation of the C library function fmod, the remainder
    has the same sign as the dividend `x1`. It is equivalent to the Matlab(TM) rem
    function and should not be confused with the Python modulus operator ``x1 % x2``.

    Note:
        Numpy arguments `casting`, `order`, `dtype`, `subok`, `signature`, and `extobj` are
        not supported.
        When `where` is provided, `out` must have a tensor value. `out` is not supported
        for storing the result, however it can be used in combination with `where` to set
        the value at indices for which `where` is set to False.

    Args:
        x1 (Tensor)
        x2 (Tensor): input arrays.
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
        Tensor or scalar, the remainder of the division of `x1` by `x2`. This is a
        scalar if both `x1` and `x2` are scalars.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> output = np.fmod(np.array([-3, -2, -1, 1, 2, 3]), np.array(2))
        >>> print(output)
        [-1  0 -1  1  0  1]
    """
    return _apply_tensor_op(lambda x1, x2: _remainder(x1, x2, C_style=True), x1, x2,
                            out=out, where=where, dtype=dtype)


def trunc(x, out=None, where=True, dtype=None):
    """
    Returns the element-wise remainder of division.

    This is the NumPy implementation of the C library function fmod, the remainder
    has the same sign as the dividend `x1`. It is equivalent to the Matlab(TM) rem
    function and should not be confused with the Python modulus operator ``x1 % x2``.

    Note:
        Numpy arguments `casting`, `order`, `dtype`, `subok`, `signature`, and `extobj` are
        not supported.
        When `where` is provided, `out` must have a tensor value. `out` is not supported
        for storing the result, however it can be used in combination with `where` to set
        the value at indices for which `where` is set to False.

    Args:
        x (Tensor): input data.
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
        Tensor or scalar, the remainder of the division of `x1` by `x2`. This is a
        scalar if both `x1` and `x2` are scalars.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> output = np.trunc(np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]))
        >>> print(output)
        [-1. -1. -0.  0.  1.  1.  2.]
    """
    return _apply_tensor_op(fix, x, out=out, where=where, dtype=dtype)


def exp(x, out=None, where=True, dtype=None):
    """
    Calculates the exponential of all elements in the input array.

    Note:
        Numpy arguments `casting`, `order`, `dtype`, `subok`, `signature`, and `extobj` are
        not supported.
        When `where` is provided, `out` must have a tensor value. `out` is not supported
        for storing the result, however it can be used in combination with `where` to set
        the value at indices for which `where` is set to False.
        On GPU, the supported dtypes are np.float16, and np.float32.
        On CPU, the supported dtypes are np.float16, np.float32, np.float64.

    Args:
        x (Tensor): input data.
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
        Tensor or scalar, element-wise exponential of `x`. This is a scalar if both
        `x1` and `x2` are scalars.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> output = np.exp(np.arange(5).astype(np.float32))
        >>> print(output)
        [ 1.         2.718282   7.3890557 20.085537  54.598145 ]
    """
    return _apply_tensor_op(F.tensor_exp, x, out=out, where=where, dtype=dtype)


def expm1(x, out=None, where=True, dtype=None):
    """
    Calculates ``exp(x) - 1`` for all elements in the array.

    Note:
        Numpy arguments `casting`, `order`, `dtype`, `subok`, `signature`, and `extobj` are
        not supported.
        When `where` is provided, `out` must have a tensor value. `out` is not supported
        for storing the result, however it can be used in combination with `where` to set
        the value at indices for which `where` is set to False.
        On GPU, the supported dtypes are np.float16, and np.float32.
        On CPU, the supported dtypes are np.float16, and np.float32.

    Args:
        x (Tensor): input data.
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
        Tensor or scalar, element-wise exponential minus one, ``out = exp(x) - 1``.
        This is a scalar if both `x1` and `x2` are scalars.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> output = np.expm1(np.arange(5).astype(np.float32))
        >>> print(output)
       [ 0.         1.7182819  6.389056  19.085537  53.59815  ]
    """
    return _apply_tensor_op(F.tensor_expm1, x, out=out, where=where, dtype=dtype)


@constexpr
def _real_axes(ndim_orig, ndim_out, axes_orig):
    """Returns the real axes to be reduced after performing broadcast"""
    diff = ndim_out - ndim_orig
    axes = F.make_range(diff)
    axes_orig = map(functools.partial(operator.add, diff), axes_orig)
    return axes + tuple(axes_orig)


@constexpr
def _shape_reduced_keepdims(shape, axes):
    """
    Reduces dimensions corresponding to argument axes while
    keeping the number of dimensions unchanged.
    """
    ndim_out = F.tuple_len(shape)
    shape_out = [1]*ndim_out
    for i in range(ndim_out):
        if not i in axes:
            shape_out[i] = shape[i]
    return tuple(shape_out)


@constexpr
def _shape_reduced(shape, axes):
    """Removes dimensions corresponding to argument axes"""
    ndim_orig = F.tuple_len(shape)
    ndim_out = ndim_orig - F.tuple_len(axes)
    shape_out = [0]*ndim_out
    idx_out = 0
    for i in range(ndim_orig):
        if not i in axes:
            shape_out[idx_out] = shape[i]
            idx_out += 1
    return tuple(shape_out)


def _infer_shape_rem(shape1, shape2, ndim1, ndim2, transpose_b):
    """Infers the shape of the last two dimensions after performing matmul."""
    shape_rem = ()
    if ndim1 >= 2:
        shape_rem += (shape1[-2],)
    if transpose_b:
        if ndim2 >= 2:
            shape_rem += (shape2[-2],)
    else:
        if ndim1 >= 1:
            shape_rem += (shape2[-1],)
    return shape_rem


def _reduce(a, reduce_fn, cmp_fn, axis=None, keepdims=False, initial=None, where=True):
    """Applies comparison based on cmp_fn and reduction based on reduce_fn"""
    _check_input_tensor(a)

    shape = F.shape(a)
    ndim = F.rank(a)
    dtype = F.dtype(a)
    axes = _check_axis_valid(axis, ndim)

    if _is_shape_empty(shape):
        if not axes:
            return a
        if keepdims:
            shape_out = _shape_reduced_keepdims(shape, axes)
        else:
            shape_out = _shape_reduced(shape, axes)
        if _is_shape_empty(shape_out):
            return empty(F.dtype(a), shape_out)
        if initial is None:
            return _raise_value_error('initial value must be provided for zero-size arrays')
        return full(shape_out, initial, dtype)

    if initial is not None:
        initial = full(shape, initial, dtype)
        a = cmp_fn(a, initial)
    if not axes:
        return a
    if isinstance(where, Tensor):
        if initial is None:
            return _raise_value_error('initial value must be provided for where masks')
        ndim_orig = F.rank(a)
        a = where_(where, a, initial)
        axes = _real_axes(ndim_orig, F.rank(a), axes)

    return reduce_fn(a, axes)


def positive(a, out=None, where=True, dtype=None):
    """
    Numerical positive, element-wise.

    Note:
        Numpy arguments casting, order, subok, signature, and extobj are
        not supported.

    Args:
        a (Tensor): Input tensor.
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
        Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.asarray([1, -1])
        >>> output = np.positive(a)
        >>> print(output)
        [1, -1]
    """
    _check_input_tensor(a)
    neg_tensor = F.neg_tensor(a)
    return _apply_tensor_op(F.neg_tensor, neg_tensor, out=out, where=where, dtype=dtype)


def negative(a, out=None, where=True, dtype=None):
    """
    Numerical negative, element-wise.

    Note:
        Numpy arguments `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        a (Tensor): Input tensor.
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
        Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.asarray([1, -1])
        >>> output = np.negative(a)
        >>> print(output)
        [-1, 1]
    """
    _check_input_tensor(a)
    return _apply_tensor_op(F.neg_tensor, a, out=out, where=where, dtype=dtype)


def _apply_tensor_op(fn, *args, out=None, where=True, dtype=None):
    """Applies tensor operations based on fn"""
    _check_input_tensor(*args)
    res = fn(*args)

    # if out is set to a non-default value, return tensor will have the same
    # dtype as out, which overrides the dtype passed into the keyword argument
    if isinstance(out, Tensor):
        dtype_out = F.dtype(out)
    elif dtype is not None:
        dtype_out = dtype
    else:
        dtype_out = F.dtype(res)

    if isinstance(out, Tensor) and isinstance(where, Tensor):
        out = where_(where, res, out)
    elif out is None or where is not None:
        out = res

    if not _check_same_type(F.dtype(out), dtype_out):
        out = F.cast(out, dtype_out)

    return out
