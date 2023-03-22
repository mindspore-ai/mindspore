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
from __future__ import absolute_import
from __future__ import division

import operator
import functools
import itertools
import sys
from numpy import dtype as nptype

from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.ops.primitive import constexpr, _primexpr
from mindspore.common import dtype as mstype
from mindspore.common import Tensor
from mindspore._c_expression import typing

from mindspore.numpy.dtypes import nan, pi, dtype_map, inf

from mindspore.numpy.array_creations import asarray_const, ones, zeros, empty, full, full_like, diag, \
    arange, histogram_bin_edges, eye
from mindspore.numpy.array_ops import where as where_
from mindspore.numpy.array_ops import ravel, expand_dims, moveaxis, concatenate, flip, stack, atleast_1d, \
    split

from mindspore.numpy.utils_const import _infer_out_shape, _check_axis_valid, _get_device, \
    _raise_type_error, _check_same_type, _check_is_float, \
    _raise_value_error, _promote, _check_axis_type, _canonicalize_axis, \
    _is_shape_empty, _check_is_int, _expanded_shape, _check_axis_in_range, \
    _check_dtype, _list_comprehensions, _tuple_setitem, _add_unit_axes, _seq_prod, \
    _make_tensor, _promote_for_trigonometric, _raise_runtime_error, _max, _type_convert, \
    _raise_unimplemented_error, _abs, _in, _tuple_slice, _check_is_inf
from mindspore.numpy.utils import _expand, _broadcast_to, _broadcast_to_shape, _check_input_tensor, \
    _to_tensor, _to_tensor_origin_dtype, _isnan


ZERO_TENSOR = asarray_const(0)


_mean_keepdims = P.ReduceMean(True)
_matmul = P.MatMul(False, False)
_matmul_t = P.MatMul(False, True)
_reduce_sum_default = P.ReduceSum()
_reduce_sum_keepdims = P.ReduceSum(True)
_reduce_min_default = P.ReduceMin()
_reduce_min_keepdims = P.ReduceMin(True)
_reduce_max_default = P.ReduceMax()
_reduce_max_keepdims = P.ReduceMax(True)
_cumsum_default = P.CumSum()
_concat = P.Concat(-1)
_cumprod_default = P.CumProd()
_round = P.Round()
_rint = P.Rint()



def absolute(x, dtype=None):
    """
    Calculates the absolute value element-wise.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.
        Currently the backend kernel only supports float calculation, if the input
        is not a `float`, then it will be casted to :class:`mstype.float32` and casted back.

    Args:
        x (Tensor): Tensor to be used for calculation.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
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
    allowed_types = None
    if _get_device() == "Ascend":
        allowed_types = (mstype.float16, mstype.float32)
    else:
        allowed_types = (mstype.int32, mstype.float16, mstype.float32, mstype.float64)
    if original_dtype not in allowed_types and dtype is None:
        x = x.astype(mstype.float32)
        return _apply_tensor_op(F.absolute, x, dtype=dtype).astype(original_dtype)
    return _apply_tensor_op(F.absolute, x, dtype=dtype)


def count_nonzero(x, axis=None, keepdims=False):
    """
    Counts the number of non-zero values in the tensor `x`.

    Args:
        x (Tensor): The tensor for which to count non-zeros.
        axis (Union[int,tuple], optional): Axis or tuple of axes along which to
            count non-zeros. Default is None, meaning that non-zeros will be counted
            along a flattened version of `x`. Default: `None`.
        keepdims (bool, optional): If this is set to True, the axes that are counted
            are left in the result as dimensions with size one. With this option,
            the result will broadcast correctly against `x`. Default: `False`.

    Returns:
        Tensor, indicating number of non-zero values in the `x` along a given axis.
        Otherwise, the total number of non-zero values in `x` is returned.

    Raises:
        TypeError: If axis is not int or tuple.
        ValueError: If axis is not in range [-x.ndim, x.ndim).

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


def clip(x, xmin, xmax, dtype=None):
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
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor, a tensor with the elements of `x`, but where values
        < `xmin` are replaced with `xmin`, and those > `xmax` with `xmax`.

    Raises:
        TypeError: If inputs have types not specified above.
        ValueError: If the shapes of `x1` and `x2` cannot broadcast, or both `xmin` and `xmax` are `None`.

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
        x = maximum(x, xmin, dtype=dtype)
    if xmax is not None:
        x = minimum(x, xmax, dtype=dtype)
    return x


def deg2rad(x, dtype=None):
    """
    Converts angles from degrees to radians.

    Args:
        x (Tensor): Angles in degrees.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor, the corresponding angle in radians. This is a tensor scalar if `x`
        is a tensor scalar.

    Raises:
        TypeError: If `x` is not a tensor.

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
    return _apply_tensor_op(convert, x, dtype=dtype)


def rad2deg(x, dtype=None):
    """
    Converts angles from radians to degrees.

    Args:
        x (Tensor): Angles in radians.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor, the corresponding angle in degrees. This is a tensor scalar if `x`
        is a tensor scalar.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.asarray([1, 2, 3, -4, -5])
        >>> output = np.rad2deg(x)
        >>> print(output)
        [  57.295776  114.59155   171.88733  -229.1831   -286.47888 ]
    """
    _check_input_tensor(x)

    def convert(a):
        return a * 180.0 / pi
    return _apply_tensor_op(convert, x, dtype=dtype)


def add(x1, x2, dtype=None):
    """
    Adds arguments element-wise.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        x1 (Tensor): input to be added.
        x2 (Tensor): input to be added.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, the sum of `x1` and `x2`, element-wise. This is a scalar
        if both `x1` and `x2` are scalars.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x1 = np.full((3, 2), [1, 2])
        >>> x2 = np.full((3, 2), [3, 4])
        >>> output = np.add(x1, x2)
        >>> print(output)
        [[4 6]
        [4 6]
        [4 6]]
    """
    # broadcast is not fully supported in tensor_add on CPU,
    # so we use tensor_sub as a substitute solution
    if _get_device() == 'CPU':
        return subtract(x1, F.neg_tensor(_to_tensor(x2)), dtype=dtype)
    return _apply_tensor_op(F.tensor_add, x1, x2, dtype=dtype)


def subtract(x1, x2, dtype=None):
    """
    Subtracts arguments, element-wise.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        x1 (Tensor): The input to be subtracted from.
        x2 (Tensor): The input to be subtracted by.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, the difference of `x1` and `x2`, element-wise. This is a
        scalar if both `x1` and `x2` are scalars.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x1 = np.full((3, 2), [1, 2])
        >>> x2 = np.full((3, 2), [3, 4])
        >>> output = np.subtract(x1, x2)
        >>> print(output)
        [[-2 -2]
        [-2 -2]
        [-2 -2]]
    """
    return _apply_tensor_op(F.tensor_sub, x1, x2, dtype=dtype)


def multiply(x1, x2, dtype=None):
    """
    Multiplies arguments element-wise.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        x1 (Tensor): input tensor to be multiplied.
        x2 (Tensor): input tensor to be multiplied.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, the product of `x1` and `x2`, element-wise. This is a scalar
        if both `x1` and `x2` are scalars.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x1 = np.full((3, 2), [1, 2])
        >>> x2 = np.full((3, 2), [3, 4])
        >>> output = np.multiply(x1, x2)
        >>> print(output)
        [[3 8]
        [3 8]
        [3 8]]
    """
    if _get_device() == 'CPU':
        _check_input_tensor(x1, x2)
        # broadcast is not fully supported on CPU backend,
        # and explicit broadcasting is performed
        shape_out = _infer_out_shape(F.shape(x1), F.shape(x2))
        x1 = _broadcast_to_shape(x1, shape_out)
        x2 = _broadcast_to_shape(x2, shape_out)
    return _apply_tensor_op(F.tensor_mul, x1, x2, dtype=dtype)


def divide(x1, x2, dtype=None):
    """
    Returns a true division of the inputs, element-wise.

    Instead of the Python traditional "floor division", this returns a true
    division.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        x1 (Tensor): the divident.
        x2 (Tensor): the divisor.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, this is a scalar if both `x1` and `x2` are scalars.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x1 = np.full((3, 2), [1, 2])
        >>> x2 = np.full((3, 2), [3, 4])
        >>> output = np.divide(x1, x2)
        >>> print(output)
        [[0.33333334 0.5       ]
        [0.33333334 0.5       ]
        [0.33333334 0.5       ]]
    """
    x1, x2 = _to_tensor(x1, x2)
    if not _check_is_float(F.dtype(x1)) and not _check_is_float(F.dtype(x2)):
        x1 = F.cast(x1, mstype.float32)
        x2 = F.cast(x2, mstype.float32)
    return _apply_tensor_op(F.tensor_div, x1, x2, dtype=dtype)


def true_divide(x1, x2, dtype=None):
    """
    Returns a true division of the inputs, element-wise.

    Instead of the Python traditional "floor division", this returns a true
    division.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        x1 (Tensor): the dividend.
        x2 (Tensor): the divisor.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, this is a scalar if both `x1` and `x2` are scalars.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x1 = np.full((3, 2), [1, 2])
        >>> x2 = np.full((3, 2), [3, 4])
        >>> output = np.true_divide(x1, x2)
        >>> print(output)
        [[0.33333334 0.5       ]
        [0.33333334 0.5       ]
        [0.33333334 0.5       ]]
    """
    return divide(x1, x2, dtype=dtype)


def power(x1, x2, dtype=None):
    """
    First array elements raised to powers from second array, element-wise.

    Raises each base in `x1` to the positionally-corresponding power in `x2`.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.
        On GPU, the supported dtypes are np.float16, and np.float32.

    Args:
        x1 (Tensor): The bases.
        x2 (Tensor): The exponents.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, the bases in `x1` raised to the exponents in `x2`. This
        is a scalar if both `x1` and `x2` are scalars.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x1 = np.full((3, 2), [1, 2]).astype('float32')
        >>> x2 = np.full((3, 2), [3, 4]).astype('float32')
        >>> output = np.power(x1, x2)
        >>> print(output)
        [[ 1. 16.]
        [ 1. 16.]
        [ 1. 16.]]
    """
    return _apply_tensor_op(F.tensor_pow, x1, x2, dtype=dtype)


def float_power(x1, x2, dtype=None):
    """
    First array elements raised to powers from second array, element-wise.

    Raise each base in `x1` to the positionally-corresponding power in `x2`. `x1` and
    `x2` must be broadcastable to the same shape. This differs from the power
    function in that integers, float16, and float64 are promoted to floats with
    a minimum precision of float32 so that the result is always inexact. The
    intent is that the function will return a usable result for negative powers
    and seldom overflow for positive powers.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.
        Integers and floats are promoted to float32 instead of float64.

    Args:
        x1 (Tensor): the bases.
        x2 (Tensor): the exponents.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, the bases in `x1` raised to the exponents in `x2`. This
        is a scalar if both `x1` and `x2` are scalars.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
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

    return _apply_tensor_op(F.tensor_pow, x1, x2, dtype=dtype)


def minimum(x1, x2, dtype=None):
    """
    Element-wise minimum of tensor elements.

    Compares two tensors and returns a new tensor containing the element-wise minima.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.
        On Ascend, input arrays containing inf or NaN are not supported.

    Args:
        x1 (Tensor): first input tensor to be compared.
        x2 (Tensor): second input tensor to be compared.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
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
    if isinstance(x1, (int, float, bool, list, tuple)):
        x1 = asarray_const(x1)
    elif not isinstance(x1, Tensor):
        _raise_type_error("Input x1 is expected to be array_like")

    if isinstance(x2, (int, float, bool, list, tuple)):
        x2 = asarray_const(x2)
    elif not isinstance(x2, Tensor):
        _raise_type_error("Input x2 is expected to be array_like")

    # if both are scalars, expand x1 to 1d tensor, since cpu kernel doesn't support
    # comparisons with 2 scalars
    if x1.ndim == 0 and x2.ndim == 0:
        x1 = expand_dims(x1, 0)
        return _apply_tensor_op(functools.partial(_prop_nan, F.minimum), x1, x2, dtype=dtype).squeeze()
    if x1.ndim == 0:
        dtype = x2.dtype
    elif x2.ndim == 0:
        dtype = x1.dtype
    return _apply_tensor_op(functools.partial(_prop_nan, F.minimum), x1, x2, dtype=dtype)


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
        axis (None or int or tuple of integers, optional): Axis or axes along
                    which the means are computed. The default is to compute
                    the mean  of the flattened array. If this is a tuple of
                    ints, a mean is performed over multiple axes.
        keepdims (bool, optional): If this is set to True, the axes which
                    are reduced are left in the result as dimensions with
                    size one. With this option, the result will broadcast
                    correctly against the input tensor.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, an array containing the mean values.

    Raises:
        ValueError: If axes are out of the range of ``[-a.ndim, a.ndim)``, or
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
    return _reduce(a, P.ReduceMean(keepdims), axis=axis, keepdims=keepdims, dtype=dtype)


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
        ValueError: If ``x1.shape[-1] != x2.shape[-1]``.

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

    aligned_shape_a = (F.shape_mul(F.shape(a)[:-1]), F.shape(a)[-1])
    aligned_shape_b = (F.shape_mul(F.shape(b)[:-1]), F.shape(a)[-1])
    a_aligned = F.reshape(a, aligned_shape_a)
    b_aligned = F.reshape(b, aligned_shape_b)

    res = _matmul_t(a_aligned, b_aligned)
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
        [[[105. 105. 105. 105.]
        [105. 105. 105. 105.]]]
    """
    ndim_a, ndim_b = F.rank(a), F.rank(b)
    if ndim_a == 0 or ndim_b == 0:
        return F.tensor_mul(a, b)
    if ndim_a > 0 and ndim_b >= 2:
        perm = F.make_range(ndim_b)
        perm = perm[:-2] + (perm[-1],) + (perm[-2],)
        b = F.transpose(b, perm)

    if F.shape(a)[-1] != F.shape(b)[-1]:
        _raise_value_error('shapes are not aligned')
    a_aligned = F.reshape(a, (-1, F.shape(a)[-1]))
    b_aligned = F.reshape(b, (-1, F.shape(b)[-1]))

    res = _matmul_t(a_aligned, b_aligned)
    res = F.reshape(res, F.shape(a)[:-1] + F.shape(b)[:-1])
    return res


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
        TypeError: If the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.full(7, 2).astype('float32')
        >>> b = np.full(4, 3).astype('float32')
        >>> output = np.outer(a, b)
        >>> print(output)
        [[6. 6. 6. 6.]
        [6. 6. 6. 6.]
        [6. 6. 6. 6.]
        [6. 6. 6. 6.]
        [6. 6. 6. 6.]
        [6. 6. 6. 6.]
        [6. 6. 6. 6.]]
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
    objects, `(a_axes, b_axes)`, sum the products of `a`'s and `b`'s elements (components)
    over the axes specified by `a_axes` and `b_axes`. The third argument can be a single
    non-negative integer_like scalar, `N`; if it is such, then the last `N` dimensions of
    `a` and the first `N` dimensions of `b` are summed over.
    Three common use cases are:

    - ``axes = 0`` : tensor product

    - ``axes = 1`` : tensor dot product

    - ``axes = 2`` : (default) tensor double contraction

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
        b (Tensor): Tensor to "dot".
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
        >>> import mindspore.numpy as np
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
        Numpy arguments `dtype`, `out` and `where` are not supported.

    Args:
        x (Tensor): A Tensor to be calculated.
        axis (Union[None, int, tuple(int)]): Axis or axes along which the standard
            deviation is computed. Default: `None`.

            If `None`, compute the standard deviation of the flattened array.
        ddof (int): Means Delta Degrees of Freedom. The divisor used in calculations is :math:`N - ddof`,
            where :math:`N` represents the number of elements. Default: 0.
        keepdims: If this is set to True, the axes which are reduced are left in the result as
            dimensions with size one. With this option, the result will broadcast correctly against the input tensor.
            If the default value is passed, then keepdims will not be passed through to the std method of
            sub-classes of tensor, however any non-default value will be. If the sub-class’ method does not
            implement keepdims any exceptions will be raised. Default: `False`.

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
    x = _to_tensor(x)
    return x.std(axis, ddof, keepdims)


def var(x, axis=None, ddof=0, keepdims=False):
    """
    Computes the variance along the specified axis.
    The variance is the average of the squared deviations from the mean, i.e.,
    :math:`var = mean(abs(x - x.mean())**2)`.

    Returns the variance, which is computed for the flattened array by default,
    otherwise over the specified axis.

    Note:
        Numpy arguments `dtype`, `out` and `where` are not supported.

    Args:
        x (Tensor): A Tensor to be calculated.
        axis (Union[None, int, tuple(int)]): Axis or axes along which the variance is computed.
            The default is to compute the variance of the flattened array. Default: `None`.
        ddof (int): Means Delta Degrees of Freedom. Default: 0.
            The divisor used in calculations is :math:`N - ddof`, where :math:`N` represents the number of elements.
        keepdims (bool): If this is set to True, the axes which are reduced are left in the result as
            dimensions with size one. With this option, the result will broadcast correctly against the input tensor.
            If the default value is passed, then keepdims will not be passed through to the var method of
            sub-classes of tensor, however any non-default value will be. If the sub-class method does not
            implement keepdims any exceptions will be raised. Default: `False`.

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
    x = _to_tensor(x)
    return x.var(axis, ddof, keepdims)


def ptp(x, axis=None, keepdims=False):
    """
    Range of values (maximum - minimum) along an axis.
    The name of the function comes from the acronym for "peak to peak".

    Note:
        Numpy arguments `dtype` and `out` are not supported.

    Args:
        x (Tensor): Input tensor.
        axis (Union[None, int, tuple(int)]): Axis or axes along which the range is computed.
            The default is to compute the variance of the flattened array. Default: None.
        keepdims (bool): If this is set to True, the axes which are reduced are left in the result as
            dimensions with size one. With this option, the result will broadcast correctly against the input tensor.
            If the default value is passed, then keepdims will not be passed through to the ptp method of
            sub-classes of tensor, however any non-default value will be. Default is False.

    Returns:
        Tensor.

    Raises:
        TypeError: If inputs have types not specified above.

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
    return x.ptp(axis, keepdims)


def average(x, axis=None, weights=None, returned=False):
    """
    Computes the weighted average along the specified axis.

    Args:
        x (Tensor): A Tensor to be averaged.
        axis (Union[None, int, tuple(int)]): Axis along which to average `x`. Default: `None`.
            If the axis is `None`, it will average over all of the elements of the tensor `x`.
            If the axis is negative, it counts from the last to the first axis.
        weights (Union[None, Tensor]): Weights associated with the values in `x`. Default: `None`.
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
    _check_input_tensor(x)
    if axis is not None:
        _check_axis_type(axis, True, True, False)
        axis = _canonicalize_axis(axis, x.ndim)

    x_avg = full((), nan, F.dtype(x))
    sum_of_weights = None

    if weights is None:
        x_avg = mean(x, axis)
        sum_of_weights = compute_weights_for_mean(x, x_avg, axis)
    else:
        _check_input_tensor(weights)
        if x.shape == weights.shape:
            x_avg, sum_of_weights = comput_avg(x, axis, weights)
        elif F.rank(weights) == 1:
            if not isinstance(axis, int):
                _raise_type_error("Axis must be specified when shapes of x and weights differ.")
            perm = _expanded_shape(x.ndim, weights.shape[0], axis)
            weights = weights.reshape(perm)
            x_avg, sum_of_weights = comput_avg(x, axis, weights)
        else:
            _raise_type_error("Weights should be None, 1-D or the same shape as input x.")

    if returned:
        if x_avg.shape != sum_of_weights.shape:
            sum_of_weights = _broadcast_to(sum_of_weights, sum_of_weights.shape, x_avg.shape, x_avg.ndim)
        return (x_avg, sum_of_weights)
    return x_avg


def compute_weights_for_mean(x, x_avg, axis):
    """Computes weights for np.average."""
    if axis is None:
        sum_of_weights = full((), x.size, F.dtype(x))
    else:
        fill_value = 1
        if isinstance(axis, int) or (isinstance(axis, tuple) and F.tuple_len(axis) == 1):
            fill_value = x.shape[axis] if isinstance(axis, int) else x.shape[axis[0]]
        elif axis is None:
            for sh in x.shape:
                fill_value *= sh
        else:
            for ax in axis:
                fill_value *= x.shape[ax]
        sum_of_weights = full_like(x_avg, fill_value, F.dtype(x))
    return sum_of_weights


def comput_avg(x, axis, weights):
    """Computes average value of input x with given parameters."""
    axis = () if axis is None else axis
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
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
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
        >>> import mindspore.numpy as np
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
    return C.matmul(x1, x2, dtype=dtype)


def square(x, dtype=None):
    """
    Returns the element-wise square of the input.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.
        On GPU, the supported dtypes are np.float16 and np.float32.

    Args:
        x (Tensor): Input data.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, element-wise ``x*x``, of the same shape and dtype as `x`.
        This is a scalar if `x` is a scalar..

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.square(np.arange(6).reshape(2, 3).astype('float32'))
        >>> print(x)
        [[ 0.  1.  4.]
        [ 9. 16. 25.]]
    """
    return _apply_tensor_op(F.square, x, dtype=dtype)


def sqrt(x, dtype=None):
    """
    Returns the non-negative square-root of an array, element-wise.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.
        On GPU, the supported dtypes are np.float16 and np.float32.

    Args:
        x (Tensor): The values whose square-roots are required.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, an array of the same shape as `x`, containing the positive
        square-root of each element in `x`. For negative elements, nan is returned.
        This is a scalar if `x` is a scalar.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.arange(6).reshape(2, 3).astype('float32')
        >>> x_squared = np.square(x)
        >>> output = np.sqrt(x_squared)
        >>> print(output)
        [[ 0. 1. 2.]
        [ 3. 4. 5.]]
    """
    return _apply_tensor_op(F.sqrt, x, dtype=dtype)


def reciprocal(x, dtype=None):
    """
    Returns the reciprocal of the argument, element-wise.

    Calculates ``1/x``.

    Note:
        Numpy arguments `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.
        When `where` is provided, `out` must have a tensor value. `out` is not supported
        for storing the result, however it can be used in combination with `where` to set
        the value at indices for which `where` is set to False.

    Args:
        x (Tensor): Input array. For integer arguments with absolute value larger
            than 1 the result is always zero because of the way Python handles
            integer division. For integer zero the result is an overflow.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, this is a scalar if `x` is a scalar.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.arange(1, 7).reshape(2, 3).astype('float32')
        >>> output = np.reciprocal(x)
        >>> print(output)
        [[1.         0.5        0.33333334]
        [0.25       0.2        0.16666667]]
    """
    return _apply_tensor_op(lambda x: F.tensor_div(1, x), x, dtype=dtype)


def log(x, dtype=None):
    """
    Returns the natural logarithm, element-wise.

    The natural logarithm log is the inverse of the exponential function, so that
    ``log(exp(x)) = x``. The natural logarithm is logarithm in base e.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.
        On GPU, the supported dtypes are np.float16, and np.float32.
        On CPU, the supported dtypes are np.float16, np.float32, and np.float64.

    Args:
        x (Tensor): Input array.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, the natural logarithm of `x`, element-wise. This is a
        scalar if `x` is a scalar.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.array([2, 3, 4]).astype('float32')
        >>> output = np.log(x)
        >>> print(output)
        [0.69314575 1.09861    1.3862929 ]
    """
    return _apply_tensor_op(F.log, x, dtype=dtype)


def _prop_nan(fn, x1, x2):
    """Selects NaN if either element is NaN"""
    has_nan = F.logical_or(_isnan(x1), _isnan(x2))
    nan_tensor = F.fill(_promote(F.dtype(x1), F.dtype(x2)), F.shape(has_nan), nan)
    res = fn(x1, x2)
    return F.select(has_nan, nan_tensor, res)


def maximum(x1, x2, dtype=None):
    """
    Returns the element-wise maximum of array elements.

    Compares two arrays and returns a new array containing the element-wise maxima.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.
        On Ascend, input arrays containing inf or NaN are not supported.

    Args:
        x1 (Tensor): Input array
        x2 (Tensor): The array holding the elements to be compared. If
            ``x1.shape != x2.shape``, they must be broadcastable to a common shape
            (which becomes the shape of the output).
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, the maximum of `x1` and `x2`, element-wise. This is a scalar
        if both `x1` and `x2` are scalars.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> output = np.maximum(np.array([2, 3, 4]), np.array([1, 5, 2]))
        >>> print(output)
        [2 5 4]
    """
    if isinstance(x1, (int, float, bool, list, tuple)):
        x1 = asarray_const(x1)
    elif not isinstance(x1, Tensor):
        _raise_type_error("Input x1 is expected to be array_like")

    if isinstance(x2, (int, float, bool, list, tuple)):
        x2 = asarray_const(x2)
    elif not isinstance(x2, Tensor):
        _raise_type_error("Input x2 is expected to be array_like")

    # F.maximum does not support when both operands are scalar
    if x1.ndim == 0 and x2.ndim == 0:
        x1 = expand_dims(x1, 0)
        return _apply_tensor_op(functools.partial(_prop_nan, F.maximum), x1, x2, dtype=dtype).squeeze()
    if x1.ndim == 0:
        dtype = x2.dtype
    elif x2.ndim == 0:
        dtype = x1.dtype
    return _apply_tensor_op(functools.partial(_prop_nan, F.maximum), x1, x2, dtype=dtype)


def heaviside(x1, x2, dtype=None):
    """
    Computes the Heaviside step function.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        x1 (Tensor): Input values.
        x2 (Tensor): The value of the function when `x1` is 0. If
            ``x1.shape != x2.shape``, they must be broadcastable to a common shape
            (which becomes the shape of the output).
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, the output array, element-wise Heaviside step function
        of `x1`. This is a scalar if both `x1` and `x2` are scalars.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
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

    return _apply_tensor_op(_heaviside, x1, x2, dtype=dtype)


def amax(a, axis=None, keepdims=False, initial=None, where=True):
    """
    Returns the maximum of an array or maximum along an axis.

    Note:
        Numpy argument `out` is not supported.
        On GPU, the supported dtypes are np.float16, and np.float32.

    Args:
        a (Tensor): Input data.
        axis (None or int or tuple of integers, optional): Defaults to None. Axis or
            axes along which to operate. By default, flattened input is used. If
            this is a tuple of integers, the maximum is selected over multiple axes,
            instead of a single axis or all the axes as before.
        keepdims (boolean, optional): Defaults to False.
            If this is set to True, the axes which are reduced are left in the
            result as dimensions with size one. With this option, the result will
            broadcast correctly against the input array.
        initial (scalar, optional): Defaults to None.
            The minimum value of an output element. Must be present to allow
            computation on empty slice.
        where (boolean Tensor, optional): Defaults to True.
            A boolean array which is broadcasted to match the dimensions of array,
            and selects elements to include in the reduction. If non-default value
            is passed, initial must also be provided.

    Returns:
        Tensor or scalar, maximum of `a`. If `axis` is None, the result is a scalar
        value. If `axis` is given, the result is an array of dimension ``a.ndim - 1``.

    Raises:
        TypeError: If the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
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
    return a.max(axis, keepdims, initial=initial, where=where)


def amin(a, axis=None, keepdims=False, initial=None, where=True):
    """
    Returns the minimum of an array or minimum along an axis.

    Note:
        Numpy argument `out` is not supported.
        On GPU, the supported dtypes are np.float16, and np.float32.

    Args:
        a (Tensor): Input data.
        axis (None or int or tuple of integers, optional): Defaults to None. Axis or
            axes along which to operate. By default, flattened input is used. If
            this is a tuple of integers, the minimum is selected over multiple axes,
            instead of a single axis or all the axes as before.
        keepdims (bool, optional): Defaults to False.
            If this is set to True, the axes which are reduced are left in the
            result as dimensions with size one. With this option, the result will
            broadcast correctly against the input array.
        initial (Number, optional): Defaults to None.
            The maximum value of an output element. Must be present to allow
            computation on empty slice.
        where (bool Tensor, optional): Defaults to True.
            A boolean array which is broadcasted to match the dimensions of array,
            and selects elements to include in the reduction. If non-default value
            is passed, initial must also be provided.

    Returns:
        Tensor or scalar, minimum of `a`. If axis is None, the result is a scalar
        value. If `axis` is given, the result is an array of dimension ``a.ndim - 1``.

    Raises:
        TypeError: If the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.arange(4).reshape((2,2)).astype('float32')
        >>> output = np.amin(a)
        >>> print(output)
        0.0
        >>> output = np.amin(a, axis=0)
        >>> print(output)
        [0. 1.]
        >>> output = np.amin(a, axis=1)
        >>> print(output)
        [0. 2.]
        >>> output = np.amin(a, where=np.array([False, True]), initial=10, axis=0)
        >>> print(output)
        [10.  1.]
    """
    return a.min(axis, keepdims, initial=initial, where=where)


def hypot(x1, x2, dtype=None):
    """
    Given the "legs" of a right triangle, returns its hypotenuse.

    Equivalent to ``sqrt(x1**2 + x2**2)``, element-wise. If `x1` or `x2` is scalar_like
    (i.e., unambiguously cast-able to a scalar type), it is broadcast for use
    with each element of the other argument. (See Examples)

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.
        On GPU, the supported dtypes are np.float16 and np.float32.
        On CPU, the supported dtypes are np.float16, np.float32, and np.float64.

    Args:
        x1 (Tensor): Leg of the triangle(s).
        x2 (Tensor): Leg of the triangle(s). If ``x1.shape != x2.shape``, they
            must be broadcastable to a common shape (which becomes the shape of
            the output).
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, the hypotenuse of the triangle(s). This is a scalar if
        both `x1` and `x2` are scalars.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> output = np.hypot(3*np.ones((3, 3)), 4*np.ones((3, 3)))
        >>> print(output)
        [[5. 5. 5.]
        [5. 5. 5.]
        [5. 5. 5.]]
        >>> output = np.hypot(3*np.ones((3, 3)), np.array([4.0]))
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

    return _apply_tensor_op(_hypot, x1, x2, dtype=dtype)


def floor(x, dtype=None):
    """
    Returns the floor of the input, element-wise.

    The floor of the scalar `x` is the largest integer `i`, such that ``i <= x``.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.
        On GPU, the supported dtypes are np.float16 and np.float32.
        On CPU, the supported dtypes are np.float16, np.float32, and np.float64.

    Args:
        x (Tensor): input data.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, the floor of each element in `x`. This is a scalar if `x`
        is a scalar.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> output = np.floor(np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]))
        >>> print(output)
        [-2. -2. -1.  0.  1.  1.  2.]
    """
    return _apply_tensor_op(F.floor, x, dtype=dtype)


def floor_divide(x1, x2, dtype=None):
    """
    Returns the largest integer smaller or equal to the division of the inputs.
    It is equivalent to the Python // operator and pairs with the
    Python % (remainder), function so that ``a = a % b + b * (a // b)`` up to roundoff.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        x1 (Tensor): Input array.
        x2 (Tensor): Input array.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> output = np.floor_divide(np.array([1., 2., 3., 4.]), np.array(2.5))
        >>> print(output)
        [0. 0. 1. 1.]
    """
    return _apply_tensor_op(F.tensor_floordiv, x1, x2, dtype=dtype)


def _remainder(x1, x2, c_style=False):
    """Computes remainder without applying keyword arguments."""
    dtype = _promote(F.dtype(x1), F.dtype(x2))
    if not _check_is_float(dtype):
        x1 = F.cast(x1, mstype.float32)
        x2 = F.cast(x2, mstype.float32)

    quotient = F.tensor_div(x1, x2)
    if c_style:
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


def remainder(x1, x2, dtype=None):
    """
    Returns element-wise remainder of division.

    Computes the remainder complementary to the floor_divide function. It is
    equivalent to the Python modulus operator ``x1 % x2`` and has the same sign
    as the divisor `x2`. The MATLAB function equivalent to np.remainder is mod.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        x1 (Tensor): input array.
        x2 (Tensor): input array.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, the element-wise remainder of the quotient
        ``floor_divide(x1, x2)``. This is a scalar if both `x1` and `x2` are scalars.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> output = np.remainder(np.array([4, 7]), np.array([2, 3]))
        >>> print(output)
        [0 1]
        >>> output = np.remainder(np.arange(7), np.array(5))
        >>> print(output)
        [0 1 2 3 4 0 1]
    """
    return _apply_tensor_op(_remainder, x1, x2, dtype=dtype)


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
        TypeError: If the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> output = np.fix(np.array([2.1, 2.9, -2.1, -2.9]))
        >>> print(output)
        [ 2.  2. -2. -2.]
    """
    _check_input_tensor(x)
    if not _check_is_float(F.dtype(x)):
        x = F.cast(x, mstype.float32)
    floored = F.floor(x)
    # change to F.ceil once supported on CPU.
    ceiled = F.neg_tensor(F.floor(F.neg_tensor(x)))
    is_neg = F.tensor_lt(x, zeros(F.shape(x), F.dtype(x)))
    return F.select(is_neg, ceiled, floored)


def fmod(x1, x2, dtype=None):
    """
    Returns the element-wise remainder of division.

    This is the NumPy implementation of the C library function fmod, the remainder
    has the same sign as the dividend `x1`. It is equivalent to the Matlab(TM) rem
    function and should not be confused with the Python modulus operator ``x1 % x2``.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        x1 (Tensor): the first input arrays.
        x2 (Tensor): the second input arrays.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, the remainder of the division of `x1` by `x2`. This is a
        scalar if both `x1` and `x2` are scalars.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> output = np.fmod(np.array([-3, -2, -1, 1, 2, 3]), np.array(2))
        >>> print(output)
        [-1  0 -1  1  0  1]
    """
    return _apply_tensor_op(lambda x1, x2: _remainder(x1, x2, c_style=True), x1, x2, dtype=dtype)


def trunc(x, dtype=None):
    """
    Returns the truncated value of the input, element-wise.

    The truncated value of the scalar `x` is the nearest integer `i` which is closer to zero
    than `x` is. In short, the fractional part of the signed number `x` is discarded.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        x (Tensor): input data.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, the truncated value of each element in `x`. This is a scalar if `x` is
        a scalar.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> output = np.trunc(np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]))
        >>> print(output)
        [-1. -1. -0.  0.  1.  1.  2.]
    """
    return _apply_tensor_op(fix, x, dtype=dtype)


def exp(x, dtype=None):
    """
    Calculates the exponential of all elements in the input array.

    Note:
        Numpy arguments `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.
        When `where` is provided, `out` must have a tensor value. `out` is not supported
        for storing the result, however it can be used in combination with `where` to set
        the value at indices for which `where` is set to False.
        On GPU, the supported dtypes are np.float16, and np.float32.
        On CPU, the supported dtypes are np.float16, np.float32, np.float64.

    Args:
        x (Tensor): input data.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, element-wise exponential of `x`. This is a scalar if both
        `x1` and `x2` are scalars.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> output = np.exp(np.arange(5).astype(np.float32))
        >>> print(output)
        [ 1.         2.718282   7.3890557 20.085537  54.598145 ]
    """
    return _apply_tensor_op(F.tensor_exp, x, dtype=dtype)


def expm1(x, dtype=None):
    """
    Calculates ``exp(x) - 1`` for all elements in the array.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.
        On GPU, the supported dtypes are np.float16, and np.float32.
        On CPU, the supported dtypes are np.float16, and np.float32.

    Args:
        x (Tensor): input data.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, element-wise exponential minus one, ``out = exp(x) - 1``.
        This is a scalar if both `x1` and `x2` are scalars.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> output = np.expm1(np.arange(5).astype(np.float32))
        >>> print(output)
        [ 0.         1.7182819  6.389056  19.085537  53.59815  ]
    """
    return _apply_tensor_op(F.tensor_expm1, x, dtype=dtype)


def divmod_(x1, x2, dtype=None):
    """
    Returns element-wise quotient and remainder simultaneously.

    Args:
        x1(Union[Tensor]): Dividend tensor.
        x2(Union[Tensor, int, float, bool]): Divisor. If ``x1.shape != x2.shape``,
            they must be broadcastable to a common shape.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Element-wise quotient and remainder from floor division, in format of (quotient, remainder)

    Raises:
        TypeError: If `x1` and `x2` are not Tensor or scalar.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.array([1, 2, 3, 4, 5])
        >>> print(np.divmod(a, 1.5))
        (Tensor(shape=[5], dtype=Float32,
         value= [ 0.00000000e+00,  1.00000000e+00,  2.00000000e+00,  2.00000000e+00,  3.00000000e+00]),
         Tensor(shape=[5], dtype=Float32,
         value= [ 1.00000000e+00,  5.00000000e-01,  0.00000000e+00,  1.00000000e+00,  5.00000000e-01]))
    """
    q = F.tensor_floordiv(x1, x2)
    r = remainder(x1, x2)
    if dtype is not None:
        q = q.astype(dtype)
        r = r.astype(dtype)
    return (q, r)


def _handle_prepend_append(combined, tensor, additional_tensor, axis):
    """Concatenates prepend or append to tensor."""
    if isinstance(additional_tensor, (int, float, bool)):
        additional_tensor = asarray_const(additional_tensor)
    elif not isinstance(additional_tensor, Tensor):
        _raise_type_error("prepend must be scalar or Tensor, but got ", additional_tensor)
    additional_shape = tensor.shape
    additional_shape = _tuple_setitem(additional_shape, axis, 1)
    additional_tensor = _broadcast_to_shape(additional_tensor, additional_shape)
    combined += (additional_tensor,)
    return combined


def diff(a, n=1, axis=-1, prepend=None, append=None):
    """
    Calculates the n-th discrete difference along the given axis.

    The first difference is given by :math:`out[i] = a[i+1] - a[i]` along the given axis,
    higher differences are calculated by using `diff` iteratively.

    Note:
        Since zero-shaped Tensor is not supported in MindSpore, a value error is raised if
        an empty Tensor is encountered.

    Args:
        a (Tensor): Input tensor.
        n (int, optional): The number of times values are differenced. If zero,
            the input is returned as-is. Default: 1.
        axis (int, optional): The axis along which the difference is taken, default
            is the last axis. Default: -1.
        prepend/append (Tensor, optional): Values to prepend or append to a along
            `axis` prior to performing the difference. Scalar values are expanded to
            arrays with length 1 in the direction of `axis` and the shape of the input
            array in along all other axes. Otherwise the dimension and shape must
            match `a` except along axis. Default: `None`.

    Returns:
        The n-th differences. The shape of the output is the same as a except along
        `axis` where the dimension is smaller by `n`. The type of the output is the same
        as the type of the difference between any two elements of `a`. This is the same
        as the type of `a` in most cases.

    Raises:
        TypeError: If inputs have types not specified above.
        ValueError: If ``n < 0``.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> arr = np.array([1, 3, -1, 0, 4])
        >>> print(np.diff(arr, n=2))
        [-6  5  3]
    """
    # This implementation is inspired by jax.numpy
    _check_input_tensor(a)
    axis = _canonicalize_axis(axis, a.ndim)
    if not isinstance(n, int):
        _raise_type_error("Input n should be int, but got ", n)
    if n < 0:
        _raise_value_error("Input n must > 0.")
    if n == 0:
        return a

    combined = ()
    if prepend is not None:
        combined = _handle_prepend_append(combined, a, prepend, axis)

    combined += (a,)

    if append is not None:
        combined = _handle_prepend_append(combined, a, append, axis)

    if combined:
        a = concatenate(combined, axis)

    # if n > maximum length allowed, the tensor is empty, and is not supported
    if n >= a.shape[axis]:
        _raise_value_error("n is bigger then the specified dimension, this will result in an empty tensor.")

    original_dtype = a.dtype
    # will change once F.tensor_slice supports types other than float32
    if not _check_is_float(original_dtype):
        a = a.astype(mstype.float32)
    a = moveaxis(a, axis, -1)
    for _ in F.make_range(n):
        slice_start = _list_comprehensions(F.rank(a) - 1, 0, True)
        slice_size = F.shape(a)[:-1] + (F.shape(a)[-1] - 1,)
        minuend = F.tensor_slice(a, slice_start + (1,), slice_size)
        subtrahend = F.tensor_slice(a, slice_start + (0,), slice_size)
        a = F.tensor_sub(minuend, subtrahend)
    if not _check_is_float(original_dtype):
        a = a.astype(original_dtype)
    return moveaxis(a, -1, axis)


def ediff1d(ary, to_end=None, to_begin=None):
    """
    The differences between consecutive elements of a tensor.

    Args:
        ary (Tensor): If necessary, will be flattened before the differences are taken.
        to_end (Tensor or scalar, optional): Number(s) to append at the end of the
            returned differences.
        to_begin (Tensor or scalar, optional): Number(s) to prepend at the beginning
            of the returned differences.

    Returns:
        The differences.

    Raises:
        TypeError: If inputs have types not specified above.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> arr = np.array([1, 3, -1, 0, 4])
        >>> print(np.ediff1d(arr))
        [ 2 -4  1  4]
    """
    _check_input_tensor(ary)
    combined = ()

    if to_begin is not None:
        if isinstance(to_begin, Tensor):
            to_begin = to_begin.ravel()
        else:
            to_begin = _to_tensor(to_begin).ravel()
        to_begin = to_begin.astype(ary.dtype)
        combined += (to_begin,)

    combined += (diff(ary.ravel()),)

    if to_end is not None:
        if isinstance(to_end, Tensor):
            to_end = to_end.ravel()
        else:
            to_end = _to_tensor(to_end).ravel()
        to_end = to_end.astype(ary.dtype)
        combined += (to_end,)

    return P.Concat(0)(combined)


def trapz(y, x=None, dx=1.0, axis=-1):
    """
    Integrates along the given axis using the composite trapezoidal rule.

    Integrates `y` (x) along given axis.

    Args:
        y (Tensor): Input array to integrate.
        x (Union[int, float, bool, list, tuple, Tensor], optional): The sample points
            corresponding to the `y` values. If `x` is None, the sample points are
            assumed to be evenly spaced `dx` apart. The default is None.
        dx (scalar, optional): The spacing between sample points when `x` is None. The
            default is 1.0.
        axis (int, optional): The axis along which to integrate. Defaults to -1.

    Returns:
        Tensor of float, definite integral as approximated by trapezoidal rule.

    Raises:
        ValueError: If axis is out of range of ``[-y.ndim, y.ndim)``.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.arange(6).reshape(2, 3)
        >>> output = np.trapz(a,  x=[-2, 1, 2], axis=1)
        >>> print(output)
        [ 3. 15.]
        >>> output = np.trapz(a,  dx=3, axis=0)
        >>> print(output)
        [ 4.5  7.5 10.5]
    """
    y = _to_tensor(y)
    ndim = F.rank(y)
    _check_axis_in_range(axis, ndim)
    axis = axis + ndim if axis < 0 else axis
    y_start_axis_left = _list_comprehensions(axis, 0, True)
    y_start_axis_right = _list_comprehensions(ndim - axis - 1, 0, True)
    shape = F.shape(y)
    y_slice_size = _tuple_setitem(shape, axis, shape[axis] - 1)
    if x is not None:
        x = _to_tensor(x)
        dx = diff(x)
    else:
        dx = _to_tensor(dx)
    dx = _expand(dx, ndim - axis, axis=-1)
    dx = _broadcast_to_shape(dx, y_slice_size)
    if not _check_is_float(F.dtype(y)):
        # trapz returns float
        y = F.cast(y, mstype.float32)
    dx = F.cast(dx, F.dtype(y))

    # product of dx and y with the last column removed
    y_slice_left = F.tensor_slice(y, y_start_axis_left + (0,) + y_start_axis_right, y_slice_size)
    prod_left = F.tensor_mul(y_slice_left, dx)
    # product of dx and y with the first column removed
    y_slice_right = F.tensor_slice(y, y_start_axis_left + (1,) + y_start_axis_right, y_slice_size)
    prod_right = F.tensor_mul(y_slice_right, dx)
    prod_sum = F.tensor_div(F.tensor_add(prod_left, prod_right), _to_tensor(2.0).astype(F.dtype(y)))
    return F.reduce_sum(prod_sum, axis)


def _gcd(x1, x2):
    """Calculates gcd without applying keyword arguments."""
    dtype = _promote(F.dtype(x1), F.dtype(x2))
    if not _check_is_float(dtype):
        # F.reduce_sum only supports float
        x1 = F.cast(x1, mstype.float32)
        x2 = F.cast(x2, mstype.float32)
    x1 = F.absolute(x1)
    x2 = F.absolute(x2)
    cond_ge = F.tensor_ge(x1, x2)
    a = where_(cond_ge, x1, x2)
    b = where_(cond_ge, x2, x1)
    b = where_(F.equal(b, ZERO_TENSOR), a, b)
    r = _remainder(a, b)
    while F.tensor_gt(F.reduce_sum(r), ZERO_TENSOR):
        r = _remainder(a, b)
        has_terminated = F.equal(r, ZERO_TENSOR)
        a = where_(has_terminated, a, b)
        b = where_(has_terminated, b, r)
    if not _check_same_type(F.dtype(b), dtype):
        b = F.cast(b, dtype)
    return b


def gcd(x1, x2, dtype=None):
    """
    Returns the greatest common divisor of ``|x1|`` and ``|x2|``.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        x1 (Tensor): input data.
        x2 (Tensor): input data.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, the greatest common divisor of the absolute value of the inputs.
        This is a scalar if both `x1` and `x2` are scalars.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> output = np.gcd(np.arange(6), np.array(20))
        >>> print(output)
        [20  1  2  1  4  5]
    """
    return _apply_tensor_op(_gcd, x1, x2, dtype=dtype)


def lcm(x1, x2, dtype=None):
    """
    Returns the lowest common multiple of ``|x1|`` and ``|x2|``.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        x1 (Tensor): input data.
        x2 (Tensor): input data.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, the lowest common multiple of the absolute value of the inputs.
        This is a scalar if both `x1` and `x2` are scalars.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> output = np.lcm(np.arange(6), np.array(20))
        >>> print(output)
        [ 0 20 20 60 20 20]
    """
    def _lcm(x1, x2):
        """Calculates lcm without applying keyword arguments"""
        common_divisor = _gcd(x1, x2)
        dtype = _promote(F.dtype(x1), F.dtype(x2))
        x1 = x1.astype(mstype.float32)
        x2 = x2.astype(mstype.float32)
        q1 = F.tensor_div(x1, common_divisor)
        q2 = F.tensor_div(x2, common_divisor)
        res = F.tensor_mul(F.tensor_mul(q1, q2), common_divisor)
        has_zero = F.equal(multiply(x1, x2), ZERO_TENSOR)
        res = where_(has_zero, ZERO_TENSOR, res)
        return F.absolute(res).astype(dtype)

    return _apply_tensor_op(_lcm, x1, x2, dtype=dtype)


def convolve(a, v, mode='full'):
    """
    Returns the discrete, linear convolution of two one-dimensional sequences.

    Note:
        If `v` is longer than `a`, the tensors are swapped before computation.

    Args:
        a (Union[list, tuple, Tensor]): First one-dimensional input tensor.
        v (Union[list, tuple, Tensor]): Second one-dimensional input tensor.

        mode (str, optional): By default, mode is `\'full\'`. This returns the
            convolution at each point of overlap, with an output shape of :math:`(N+M-1,)`.
            At the end-points of the convolution, the signals do not overlap completely,
            and boundary effects may be seen.
            If `mode` is `\'same\'`, it returns output of length :math:`max(M, N)`. Boundary
            effects are still visible.
            If `mode` is `\'valid\'`, it returns output of length :math:`max(M, N) - min(M, N) + 1`.
            The convolution product is only given for points where the signals overlap
            completely. Values outside the signal boundary have no effect.

    Returns:
        Tensor, discrete, linear convolution of a and v.

    Raises:
        TypeError: If the inputs have types not specified above.
        ValueError: If a and v are empty or have wrong dimensions

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> output = np.convolve([1., 2., 3., 4., 5.], [2., 3.], mode="valid")
        >>> print(output)
        [ 7. 12. 17. 22.]
    """
    if not isinstance(a, Tensor):
        a = asarray_const(a)
    if not isinstance(v, Tensor):
        v = asarray_const(v)
    a_size = F.shape_mul(a.shape)
    v_size = F.shape_mul(v.shape)
    if a_size == 0 or v_size == 0:
        _raise_value_error("Inputs cannot be empty.")
    a = _expand(a, 1)
    v = _expand(v, 1)
    final_dtype = _promote(a.dtype, v.dtype)
    a = a.astype("float32")
    v = v.astype("float32")
    if a.ndim != 1 or v.ndim != 1:
        _raise_value_error("a and v must be 1-D tensor.")
    if a_size < v_size:
        a, v = v, a
        a_size, v_size = v_size, a_size
    v = v[::-1]
    return _compute_1d_conv(a, v, mode).astype(final_dtype)


def _handle_weights(weights, num_samples):
    """Checks fweight and aweight in np.cov."""
    weights = asarray_const(weights)
    if not _check_is_int(weights.dtype):
        _raise_type_error("weights must be integer")
    weights = weights.astype("float32")
    if weights.ndim > 1:
        _raise_runtime_error("cannot handle multidimensional weights")
    if weights.shape[0] != num_samples:
        _raise_runtime_error("incompatible numbers of samples and weights")
    return absolute(weights)


def _handle_inputs(cov_input, rowvar):
    """Checks input arrays for np.cov."""
    if not isinstance(cov_input, Tensor):
        cov_input = asarray_const(cov_input)
    if cov_input.ndim > 2:
        _raise_value_error("input array has dimension more than 2.")
    cov_input = cov_input.astype("float32")
    if cov_input.size == 0:
        _raise_value_error("The value of cov_input should not be None, but got {}.".format(cov_input))
    cov_input = _expand(cov_input, 2)
    if not isinstance(rowvar, bool):
        _raise_type_error("input rowvar should be boolean.")
    if not rowvar and cov_input.shape[0] != 1:
        cov_input = cov_input.T
    return cov_input


def _handle_facts(w, m, ddof, aweights):
    """Computes facts for np.cov"""
    fact = None
    if w is None:
        fact = m.shape[1] - ddof
    else:
        w_sum = _reduce_sum_default(w, -1)
        if ddof == 0:
            fact = w_sum
        elif aweights is None:
            fact = w_sum - ddof
        else:
            fact = w_sum - ddof * F.reduce_sum(w * aweights) / w_sum
    return fact


def cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None, dtype=None):
    """
    Estimates a covariance matrix, given data and weights.

    Covariance indicates the level to which two variables vary together. If we examine
    N-dimensional samples, :math:`X = [x_1, x_2, ... x_N]^T`, then the covariance matrix
    element :math:`C_{ij}` is the covariance of :math:`x_i` and :math:`x_j`. The element
    :math:`C_{ii}` is the variance of :math:`x_i`.

    Note:
        `fweights` and `aweights` must be all positive, in Numpy if negative values
        are detected, a value error will be raised, in MindSpore we converts all values
        to positive instead.

    Args:
        m (Union[Tensor, list, tuple]): A 1-D or 2-D tensor containing multiple variables
            and observations. Each row of `m` represents a variable, and each column
            represents a single observation of all those variables. Also see `rowvar` below.
        y (Union[Tensor, list, tuple], optional): An additional set of variables
            and observations. `y` has the same form as that of `m`, default is ``None``.
        rowvar(bool, optional): If `rowvar` is ``True`` (default), then each row represents
            a variable, with observations in the columns. Otherwise, the relationship
            is transposed: each column represents a variable, while the rows contain
            observations.
        bias (bool, optional): Default Normalization (``False``) is by :math:`(N - 1)`, where
            :math:`N` is the number of observations given (unbiased estimate). If bias is
            ``True``, then Normalization is by `N`. These values can be overridden by
            using the keyword `ddof`.
        ddof (int, optional): If not ``None``, the default value implied by `bias` is
            overridden. Note that :math:`ddof=1` will return the unbiased estimate, even
            if both fweights and aweights are specified, and :math:`ddof=0` will return
            the simple average. See the notes for the details. The default value
            is ``None``.
        fweights (Union[Tensor, list, tuple], optional): 1-D tensor of integer
            frequency weights; the number of times each observation vector should
            be repeated. The default value is ``None``.
        aweights (Union[Tensor, list, tuple], optional): 1-D tensor of observation
            vector weights. These relative weights are typically larger for observations
            considered more important and smaller for observations considered less
            important. If :math:`ddof=0` the tensor of weights can be used to assign probabilities
            to observation vectors. The default value is ``None``.
        dtype (Union[:class:`mindspore.dtype`, str], optional): Data-type of the
            result. By default, the return data-type will have mstype.float32 precision.
            Default is ``None``.

    Returns:
        Tensor, the covariance matrix of the variables.

    Raises:
        TypeError: If the inputs have types not specified above.
        ValueError: If `m` and `y` have wrong dimensions.
        RuntimeError: If `aweights` and `fweights` have dimensions > 2.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> output = np.cov([[2., 3., 4., 5.], [0., 2., 3., 4.], [7., 8., 9., 10.]])
        >>> print(output)
        [[1.6666666 2.1666667 1.6666666]
        [2.1666667 2.9166667 2.1666667]
        [1.6666666 2.1666667 1.6666666]]
    """
    # This implementation was inspired by original numpy implementation.
    m = _handle_inputs(m, rowvar)

    if m.shape[0] == 0:
        return empty((0, 0), dtype="float32")

    if y is not None:
        y = _handle_inputs(y, rowvar)
        m = concatenate((m, y), axis=0)

    if ddof is None:
        if not bias:
            ddof = 1
        else:
            ddof = 0

    # Handle fweights and aweights
    w = _handle_weights(fweights, m.shape[1]) if fweights is not None else None

    if aweights is not None:
        aweights = _handle_weights(aweights, m.shape[1])
        w = aweights if w is None else w * aweights

    avg = average(m, axis=1, weights=w)

    # Determine the Normalization
    fact = _handle_facts(w, m, ddof, aweights)

    m = m - F.expand_dims(avg, -1)
    if w is None:
        m_t = m.T
    else:
        m_t = (m * w).T
    res = true_divide(dot(m, m_t), fact).squeeze()
    if dtype is not None:
        return res.astype(dtype)
    return res


@constexpr
def _real_axes(ndim_orig, ndim_out, axes_orig):
    """Returns the real axes to be reduced after performing broadcast"""
    _diff = ndim_out - ndim_orig
    axes = F.make_range(_diff)
    axes_orig = map(functools.partial(operator.add, _diff), axes_orig)
    return axes + tuple(axes_orig)


@_primexpr
def _shape_reduced_keepdims(shape, axes):
    """
    Reduces dimensions corresponding to argument axes while
    keeping the number of dimensions unchanged.
    """
    ndim_out = F.tuple_len(shape)
    shape_out = [1]*ndim_out
    for i in range(ndim_out):
        if i not in axes:
            shape_out[i] = shape[i]
    return tuple(shape_out)


@_primexpr
def _shape_reduced(shape, axes):
    """Removes dimensions corresponding to argument axes"""
    ndim_orig = F.tuple_len(shape)
    ndim_out = ndim_orig - F.tuple_len(axes)
    shape_out = [0]*ndim_out
    idx_out = 0
    for i in range(ndim_orig):
        if i not in axes:
            shape_out[idx_out] = shape[i]
            idx_out += 1
    return tuple(shape_out)


def _reduce(a, reduce_fn, cmp_fn=None, axis=None, keepdims=False, initial=None, where=True, dtype=None):
    """
    Applies comparison based on cmp_fn and reduction based on reduce_fn.
    If cmp_fn is None, only reduction is performed.
    """
    a = _to_tensor(a)

    shape = F.shape(a)
    ndim = F.rank(a)
    if dtype is None:
        dtype = F.dtype(a)
    axes = _check_axis_valid(axis, ndim)
    if initial is not None:
        if ((isinstance(initial, Tensor) and F.rank(initial) > 0) or
                not isinstance(initial, (int, float, bool, Tensor))):
            _raise_type_error('initial should be scalar')

    if _is_shape_empty(shape):
        if not axes:
            return a
        if keepdims:
            shape_out = _shape_reduced_keepdims(shape, axes)
        else:
            shape_out = _shape_reduced(shape, axes)
        if _is_shape_empty(shape_out):
            return empty(shape_out, dtype)
        if initial is None:
            if cmp_fn is None:
                initial = nan
            else:
                _raise_value_error('initial value must be provided for zero-size arrays')
        return full(shape_out, initial, dtype)

    if initial is not None:
        initial = full(shape, initial, dtype)
        a = cmp_fn(a, initial)

    if isinstance(where, Tensor):
        if initial is None:
            _raise_value_error('initial value must be provided for where masks')
        ndim_orig = F.rank(a)
        a = where_(where, a, initial)
        axes = _real_axes(ndim_orig, F.rank(a), axes)

    return reduce_fn(a, axes).astype(dtype)


def nanmax(a, axis=None, dtype=None, keepdims=False):
    """
    Return the maximum of an array or maximum along an axis, ignoring any NaNs.

    Note:
        Numpy arguments `out` is not supported.
        For all NaN slices, a very small negative number is returned instead of NaN.

    Args:
        a (Union[int, float, list, tuple, Tensor]): Array containing numbers whose maximum
            is desired. If `a` is not an array, a conversion is attempted.
        axis (Union[int, tuple of int, None], optional): Axis or axes along which the maximum is
            computed. The default is to compute the maximum of the flattened array.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.
        keepdims (boolean, optional): Defaults to False. If this is set to True, the axes which
            are reduced are left in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the original `a`.

    Returns:
        Tensor.

    Raises:
        ValueError: If axes are out of the range of ``[-a.ndim, a.ndim)``, or
            if the axes contain duplicates.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.array([[1, 2], [3, np.nan]])
        >>> output = np.nanmax(a)
        >>> print(output)
        3.0
        >>> output = np.nanmax(a, axis=0)
        >>> print(output)
        [3. 2.]
    """
    a = _to_tensor(a)
    if not isinstance(keepdims, int):
        _raise_type_error("integer argument expected, got", keepdims)
    nan_mask = _isnan(a)
    a = F.select(nan_mask, full(F.shape(a), -sys.maxsize - 1, F.dtype(a)), a)
    reduce_fn = _reduce_max_keepdims if keepdims else _reduce_max_default
    return _reduce(a, reduce_fn, axis=axis, keepdims=keepdims, dtype=dtype)


def nanmin(a, axis=None, dtype=None, keepdims=False):
    """
    Returns the minimum of array elements over a given axis, ignoring any NaNs.

    Note:
        Numpy arguments `out` is not supported.
        For all-NaN slices, a very large number is returned instead of NaN.
        On Ascend, since checking for NaN is currently not supported, it is not recommended to
        use np.nanmin. If the array does not contain NaN, np.min should be used instead.

    Args:
        a (Union[int, float, list, tuple, Tensor]): Array containing numbers whose minimum
            is desired. If `a` is not an array, a conversion is attempted.
        axis (Union[int, tuple of int, None], optional): Axis or axes along which the minimum is
            computed. The default is to compute the minimum of the flattened array.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.
        keepdims (boolean, optional): Defaults to False. If this is set to True, the axes which
            are reduced are left in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the original `a`.

    Returns:
        Tensor.

    Raises:
        ValueError: If axes are out of the range of ``[-a.ndim, a.ndim)``, or
            if the axes contain duplicates.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.array([[1, 2], [3, np.nan]])
        >>> output = np.nanmin(a)
        >>> print(output)
        1.0
        >>> output = np.nanmin(a, axis=0)
        >>> print(output)
        [1. 2.]
    """
    a = _to_tensor(a)
    if not isinstance(keepdims, int):
        _raise_type_error("integer argument expected, got", keepdims)
    nan_mask = _isnan(a)
    a = F.select(nan_mask, full(F.shape(a), sys.maxsize, F.dtype(a)), a)
    reduce_fn = _reduce_min_keepdims if keepdims else _reduce_min_default
    return _reduce(a, reduce_fn, axis=axis, keepdims=keepdims, dtype=dtype)


def _reduce_nansum(x, axis, keepdims=False):
    """Computes reduce sum treating NaNs as zeros."""
    x = F.select(_isnan(x), zeros(F.shape(x), F.dtype(x)), x)
    if keepdims:
        return _reduce_sum_keepdims(x, axis)
    return _reduce_sum_default(x, axis)


def nansum(a, axis=None, dtype=None, keepdims=False):
    """
    Returns the sum of array elements over a given axis treating Not a Numbers (NaNs) as zero.

    Note:
        Numpy arguments `out` is not supported.

    Args:
        a (Union[int, float, list, tuple, Tensor]): Array containing numbers
            whose sum is desired. If `a` is not an array, a conversion is attempted.
        axis (Union[int, tuple of int, None], optional): Axis or axes along which the sum is
            computed. The default is to compute the sum of the flattened array.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.
        keepdims (boolean, optional): Defaults to False. If this is set to True, the axes which
            are reduced are left in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the original `a`.

    Returns:
        Tensor.

    Raises:
        ValueError: If axes are out of the range of ``[-a.ndim, a.ndim)``, or
            if the axes contain duplicates.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.array([[1, 1], [1, np.nan]])
        >>> output = np.nansum(a)
        >>> print(output)
        3.0
        >>> output = np.nansum(a, axis=0)
        >>> print(output)
        [2. 1.]
    """
    a = _to_tensor(a)
    nan_mask = _isnan(a)
    a = F.select(nan_mask, zeros(F.shape(a), F.dtype(a)), a)
    return _reduce(a, functools.partial(_reduce_nansum, keepdims=keepdims), axis=axis,
                   keepdims=keepdims, dtype=dtype)


def _count_nonnan(a, axis, keepdims=False):
    """Counts the number of elements excluding NaNs."""
    nonnan_mask = F.select(_isnan(a), zeros(F.shape(a), F.dtype(a)), ones(F.shape(a), F.dtype(a)))
    if keepdims:
        return _reduce_sum_keepdims(nonnan_mask, axis)
    return _reduce_sum_default(nonnan_mask, axis)


def nanmean(a, axis=None, dtype=None, keepdims=False):
    """
    Computes the arithmetic mean along the specified axis, ignoring NaNs.

    Returns the average of the array elements. The average is taken over the flattened
    array by default, otherwise over the specified axis. float32 intermediate and
    return values are used for integer inputs.

    Note:
        Numpy arguments `out` is not supported.

    Args:
        a (Union[int, float, list, tuple, Tensor]): Array containing numbers
            whose mean is desired. If `a` is not an array, a conversion is attempted.
        axis (Union[int, tuple of int, None], optional): Axis or axes along which the mean is
            computed. The default is to compute the mean of the flattened array.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.
        keepdims (boolean, optional): Defaults to False. If this is set to True, the axes which
            are reduced are left in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the original `a`.

    Returns:
        Tensor.

    Raises:
        ValueError: If axes are out of the range of ``[-a.ndim, a.ndim)``, or
            if the axes contain duplicates.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.array([[1, np.nan], [3, 4]])
        >>> output = np.nanmean(a)
        >>> print(output)
        2.6666667
        >>> output = np.nanmean(a, axis=0)
        >>> print(output)
        [2. 4.]
        >>> output = np.nanmean(a, axis=1)
        >>> print(output)
        [1.  3.5]
    """
    if dtype is None:
        dtype = mstype.float32
    a = _to_tensor(a)
    axis = _check_axis_valid(axis, F.rank(a))
    sum_a = nansum(a, axis=axis, dtype=dtype, keepdims=keepdims)
    return F.tensor_div(sum_a, _count_nonnan(a, axis, keepdims))


def _nanvar(a, axis, ddof=0, keepdims=False):
    """Computes nanvar without applying keyword arguments."""
    mean_a = nanmean(a, axis=axis, keepdims=True)
    pow_a = F.tensor_pow(F.tensor_sub(a, mean_a), 2)
    sum_a = _reduce_nansum(pow_a, axis, keepdims)
    count = _count_nonnan(a, axis, keepdims)
    return divide(sum_a, F.tensor_sub(count, ddof))


def nanvar(a, axis=None, dtype=None, ddof=0, keepdims=False):
    """
    Computes the variance along the specified axis, while ignoring NaNs.

    Returns the variance of the array elements, a measure of the spread of a distribution. The
    variance is computed for the flattened array by default, otherwise over the specified axis.

    Note:
        Numpy arguments `out` is not supported.
        On GPU, the supported dtypes are np.float16, and np.float32.

    Args:
        a (Union[int, float, list, tuple, Tensor]): Array containing numbers
            whose variance is desired. If `a` is not an array, a conversion is attempted.
        axis (Union[int, tuple of int, None], optional): Axis or axes along which the variance is
            computed. The default is to compute the variance of the flattened array.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.
        ddof (int, optional): "Delta Degrees of Freedom": the divisor used in the calculation is
            ``N - ddof``, where `N` represents the number of non-NaN elements. By default `ddof`
            is zero.
        keepdims (boolean, optional): Defaults to False. If this is set to True, the axes which
            are reduced are left in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the original `a`.

    Returns:
        Tensor.

    Raises:
        ValueError: If axes are out of the range of ``[-a.ndim, a.ndim)``, or
            if the axes contain duplicates.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.array([[1, np.nan], [3, 4]])
        >>> output = np.nanvar(a)
        >>> print(output)
        1.5555557
        >>> output = np.nanvar(a, axis=0)
        >>> print(output)
        [1. 0.]
        >>> output = np.nanvar(a, axis=1)
        >>> print(output)
        [0.   0.25]
    """
    if dtype is None:
        dtype = mstype.float32
    return _reduce(a, functools.partial(_nanvar, ddof=ddof, keepdims=keepdims), axis=axis,
                   keepdims=keepdims, dtype=dtype)


def nanstd(a, axis=None, dtype=None, ddof=0, keepdims=False):
    """
    Computes the standard deviation along the specified axis, while ignoring NaNs.

    Returns the standard deviation, a measure of the spread of a distribution, of the non-NaN
    array elements. The standard deviation is computed for the flattened array by default,
    otherwise over the specified axis.

    Note:
        Numpy arguments `out` is not supported.
        On GPU, the supported dtypes are np.float16, and np.float32.

    Args:
        a (Union[int, float, list, tuple, Tensor]): Calculates the standard deviation of the non-NaN values.
        axis (Union[int, tuple of int, None], optional): Axis or axes along which the standard
            deviation is computed. The default is to compute the standard deviation of the
            flattened array.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.
        ddof (int, optional): "Delta Degrees of Freedom": the divisor used in the calculation is
            ``N - ddof``, where `N` represents the number of non-NaN elements. By default `ddof`
            is zero.
        keepdims (boolean, optional): Defaults to False. If this is set to True, the axes which
            are reduced are left in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the original `a`.

    Returns:
        Tensor.

    Raises:
        ValueError: If axes are out of the range of ``[-a.ndim, a.ndim)``, or
            if the axes contain duplicates.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.array([[1, np.nan], [3, 4]])
        >>> output = np.nanstd(a)
        >>> print(output)
        1.2472192
        >>> output = np.nanstd(a, axis=0)
        >>> print(output)
        [1. 0.]
        >>> output = np.nanstd(a, axis=1)
        >>> print(output)
        [0.  0.5]
    """
    if dtype is None:
        dtype = mstype.float32
    return _reduce(a, lambda a, axis: F.sqrt(_nanvar(a, axis, ddof=ddof, keepdims=keepdims)),
                   axis=axis, keepdims=keepdims, dtype=dtype)


def exp2(x, dtype=None):
    """
    Calculates ``2**p`` for all p in the input array.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.
        On GPU, the supported dtypes are np.float16, and np.float32.

    Args:
        x (Tensor): input values.
        dtype (:class:`mindspore.dtype`, optional): Defaults to :class:`None`. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, element-wise 2 to the power `x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.array([2, 3]).astype(np.float32)
        >>> output = np.exp2(x)
        >>> print(output)
        [4. 8.]
    """
    return _apply_tensor_op(lambda x: F.tensor_pow(2, x), x, dtype=dtype)


def kron(a, b):
    """
    Kronecker product of two arrays.

    Computes the Kronecker product, a composite array made of blocks of the second
    array scaled by the first.

    Note:
        Booleans are not supported.

    Args:
        a (Union[int, float, list, tuple, Tensor]): input values.
        b (Union[int, float, list, tuple, Tensor]): input values.

    Returns:
        Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> output = np.kron([1,10,100], [5,6,7])
        >>> print(output)
        [  5   6   7  50  60  70 500 600 700]
        >>> output = np.kron([5,6,7], [1,10,100])
        >>> print(output)
        [  5  50 500   6  60 600   7  70 700]
        >>> output = np.kron(np.eye(2), np.ones((2,2)))
        >>> print(output)
        [[1. 1. 0. 0.]
        [1. 1. 0. 0.]
        [0. 0. 1. 1.]
        [0. 0. 1. 1.]]
    """
    a, b = _to_tensor(a, b)
    ndim = _max(F.rank(a), F.rank(b))
    if ndim == 0:
        return F.tensor_mul(a, b)
    a = _expand(a, ndim)
    b = _expand(b, ndim)
    shape_a = F.shape(a)
    shape_b = F.shape(b)

    # scales a by the shape of b
    kron_shape = _seq_prod(shape_a, shape_b)
    a = F.reshape(a, _add_unit_axes(shape_a, 2*ndim, True))
    a = F.tile(a, _add_unit_axes(shape_b, 2*ndim, False))
    a = moveaxis(a, F.make_range(ndim, 2*ndim), F.make_range(1, 2*ndim, 2))
    a = F.reshape(a, kron_shape)
    # scales b by the shape of a
    b = F.tile(b, shape_a)
    return F.tensor_mul(a, b)


def cross(a, b, axisa=- 1, axisb=- 1, axisc=- 1, axis=None):
    """
    Returns the cross product of two (arrays of) vectors.

    The cross product of `a` and `b` in :math:`R^3` is a vector perpendicular to both
    `a` and `b`. If `a` and `b` are arrays of vectors, the vectors are defined by the
    last axis of `a` and `b` by default, and these axes can have dimensions 2 or 3.
    Where the dimension of either `a` or `b` is 2, the third component of the input
    vector is assumed to be zero and the cross product calculated accordingly. In cases
    where both input vectors have dimension 2, the z-component of the cross product is
    returned.

    Args:
        a (Union[list, tuple, Tensor]): Components of the first vector(s).
        b (Union[list, tuple, Tensor]): Components of the second vector(s).
        axisa (int, optional): Axis of `a` that defines the vector(s). By default, the last
            axis.
        axisb (int, optional): Axis of `b` that defines the vector(s). By default, the last
            axis.
        axisc (int, optional): Axis of `c` containing the cross product vector(s). Ignored
            if both input vectors have dimension 2, as the return is scalar. By default,
            the last axis.
        axis (int, optional): If defined, the axis of `a`, `b` and `c` that defines the
            vector(s) and cross product(s). Overrides `axisa`, `axisb` and `axisc`.
            Defaults to None.

    Returns:
        Tensor, vector cross product(s).

    Raises:
        ValueError: when the dimensions of the vector(s) in `a` and/or `b` does not equal 2
            or 3.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.array([[1,2,3], [4,5,6]])
        >>> y = np.array([[4,5,6], [1,2,3]])
        >>> output = np.cross(x, y)
        >>> print(output)
        [[-3  6 -3]
        [ 3 -6  3]]
        >>> output = np.cross(x, y, axisc=0)
        >>> print(output)
        [[-3  3]
        [ 6 -6]
        [-3  3]]
    """
    a, b = _to_tensor(a, b)
    if axis is not None:
        axisa, axisb, axisc = axis, axis, axis

    _check_axis_in_range(axisa, F.rank(a))
    _check_axis_in_range(axisb, F.rank(b))
    a = moveaxis(a, axisa, -1)
    b = moveaxis(b, axisb, -1)
    shape_a = F.shape(a)
    shape_b = F.shape(b)
    if F.shape(a)[-1] not in (2, 3) or F.shape(b)[-1] not in (2, 3):
        _raise_value_error('incompatible dimensions for cross product (dimension must be 2 or 3)')
    a_has_z = shape_a[-1] == 3
    b_has_z = shape_b[-1] == 3
    shape_out = _infer_out_shape(shape_a[:-1], shape_b[:-1])
    if a_has_z or b_has_z:
        shape_out += (3,)
    _check_axis_in_range(axisc, len(shape_out))

    dtype = _promote(F.dtype(a), F.dtype(b))
    if _get_device() == 'CPU':
        # F.tensor_slice only supports float on CPU
        if not _check_is_float(F.dtype(a)):
            a = F.cast(a, mstype.float32)
        if not _check_is_float(F.dtype(b)):
            b = F.cast(b, mstype.float32)

    a_slice_start = _list_comprehensions(F.rank(a) - 1, 0, True)
    a_slice_size = shape_a[:-1] + (1,)
    b_slice_start = _list_comprehensions(F.rank(b) - 1, 0, True)
    b_slice_size = shape_b[:-1] + (1,)

    def _get_slice_product(idx_a, idx_b):
        return multiply(F.tensor_slice(a, a_slice_start + (idx_a,), a_slice_size),
                        F.tensor_slice(b, b_slice_start + (idx_b,), b_slice_size))

    cz = F.tensor_sub(_get_slice_product(0, 1), _get_slice_product(1, 0)) # ax*by - ay*bx
    if not a_has_z and not b_has_z:
        return F.reshape(cz, shape_out).astype(dtype)

    if a_has_z and b_has_z:
        cx = F.tensor_sub(_get_slice_product(1, 2), _get_slice_product(2, 1)) # ay*bz - az*by
        cy = F.tensor_sub(_get_slice_product(2, 0), _get_slice_product(0, 2)) # az*bx - ax*bz
    elif a_has_z:
        cx = F.neg_tensor(_get_slice_product(2, 1)) # -az*by
        cy = _get_slice_product(2, 0)               # az*bx
    else: # b_has_z
        cx = _get_slice_product(1, 2)               # ay*bz
        cy = F.neg_tensor(_get_slice_product(0, 2)) # -ax*bz
    res = _concat((cx, cy, cz)).reshape(shape_out)
    return moveaxis(res, -1, axisc).astype(dtype)


def ceil(x, dtype=None):
    """
    Returns the ceiling of the input, element-wise.

    The ceil of the scalar `x` is the smallest integer `i`, such that ``i >= x``.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.
        On GPU, the supported dtypes are np.float16, and np.float32.

    Args:
        x (Tensor): input values.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, the floor of each element in `x`. This is a scalar if `x` is a scalar.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
        >>> output = np.ceil(a)
        >>> print(output)
        [-1. -1. -0.  1.  2.  2.  2.]
    """
    return _apply_tensor_op(lambda x: F.neg_tensor(F.floor(F.neg_tensor(x.astype(mstype.float32)))),
                            x, dtype=dtype)


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


def positive(a, dtype=None):
    """
    Numerical positive, element-wise.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        a (Tensor): Input tensor.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.asarray([1, -1]).astype('float32')
        >>> output = np.positive(a)
        >>> print(output)
        [1. -1.]
    """
    _check_input_tensor(a)
    neg_tensor = F.neg_tensor(a)
    return _apply_tensor_op(F.neg_tensor, neg_tensor, dtype=dtype)


def negative(a, dtype=None):
    """
    Numerical negative, element-wise.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        a (Tensor): Input tensor.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.asarray([1, -1]).astype('float32')
        >>> output = np.negative(a)
        >>> print(output)
        [-1. 1.]
    """
    return _apply_tensor_op(F.neg_tensor, a, dtype=dtype)


def cumsum(a, axis=None, dtype=None):
    """
    Returns the cumulative sum of the elements along a given axis.

    Note:
        If ``a.dtype`` is :class:`int8`, :class:`int16` or :class:`bool`, the result
        `dtype` will be elevated to :class:`int32`.

    Args:
        a (Tensor): Input tensor.
        axis (int, optional): Axis along which the cumulative sum is computed. The
            default (None) is to compute the cumsum over the flattened array.
        dtype (:class:`mindspore.dtype`, optional): If not specified, stay the same as `a`,
            unless `a` has an integer dtype with a precision less than that of the
            default platform integer. In that case, the default platform integer
            is used. Default: `None`.

    Returns:
        Tensor.

    Raises:
        TypeError: If input arguments have types not specified above.
        ValueError: If axis is out of range.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> output = np.cumsum(np.ones((3,3)), axis=0)
        >>> print(output)
        [[1. 1. 1.]
         [2. 2. 2.]
         [3. 3. 3.]]
    """
    _check_input_tensor(a)
    return a.cumsum(axis, dtype)


def nancumsum(a, axis=None, dtype=None):
    """
    Return the cumulative sum of array elements over a given axis treating Not a Numbers (NaNs)
    as zero. The cumulative sum does not change when NaNs are encountered and leading NaNs are
    replaced by zeros.

    Zeros are returned for slices that are all-NaN or empty.

    Note:
        If ``a.dtype`` is :class:`int8`, :class:`int16` or :class:`bool`, the result
        `dtype` will be elevated to :class:`int32`.

    Args:
        a (Tensor): Input tensor.
        axis (int, optional): Axis along which the cumulative sum is computed. The
            default (None) is to compute the cumsum over the flattened array.
        dtype (:class:`mindspore.dtype`, optional): If not specified, stay the same as `a`,
            unless `a` has an integer dtype with a precision less than that of the
            default platform integer. In that case, the default platform integer
            is used.

    Returns:
        Tensor.

    Raises:
        TypeError: If input arguments have types not specified above.
        ValueError: If axis is out of range.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.array([[1, 2], [3, np.nan]])
        >>> output = np.nancumsum(a)
        >>> print(output)
        [1. 3. 6. 6.]
        >>> output = np.nancumsum(a, axis=0)
        >>> print(output)
        [[1. 2.]
        [4. 2.]]
        >>> output = np.nancumsum(a, axis=1)
        >>> print(output)
        [[1. 3.]
        [3. 3.]]
    """
    a = F.select(_isnan(a), zeros(F.shape(a), F.dtype(a)), a)
    return a.cumsum(axis, dtype)


def cbrt(x, dtype=None):
    """
    Returns the cube-root of a tensor, element-wise.

    Note:
        Numpy arguments `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        x (Tensor): Input tensor.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.asarray([1, -1, 3, -8, 64])
        >>> output = np.cbrt(a)
        >>> print(output)
        [ 1.        -1.         1.4422495 -2.         4.       ]
    """
    def _cbrt(x):
        compute_type = promote_types(x.dtype, "float32")
        x = x.astype(compute_type)
        # TODO: use P.Sign() once gpu support is added
        abs_x = F.absolute(x)
        sign_x = abs_x / x
        return sign_x * F.tensor_pow(abs_x, 1. / 3.)
    return _apply_tensor_op(_cbrt, x, dtype=dtype)


def log1p(x, dtype=None):
    """
    Returns the natural logarithm of one plus the input array, element-wise.

    Calculates ``log(1 + x)``.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        x (Tensor): Input array.
        dtype (:class:`mindspore.dtype`): Default: :class:`None`. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar. This is a scalar if `x` is a scalar.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.array([1, 2, 3]).astype('float16')
        >>> output = np.log1p(x)
        >>> print(output)
        [0.6934 1.099 1.387 ]
    """
    return _apply_tensor_op(lambda x: F.log(x + 1), x, dtype=dtype)


def logaddexp(x1, x2, dtype=None):
    """
    Logarithm of the sum of exponentiations of the inputs.

    Calculates ``log(exp(x1) + exp(x2))``. This function is useful in statistics where the
    calculated probabilities of events may be so small as to exceed the range of normal
    floating point numbers. In such cases the logarithm of the calculated probability is
    stored. This function allows adding probabilities stored in such a fashion.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        x1 (Tensor): Input array.
        x2 (Tensor): Input array. If ``x1.shape != x2.shape``, they must be broadcastable to
            a common shape (which becomes the shape of the output).
        dtype (:class:`mindspore.dtype`): Default: :class:`None`. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar. This is a scalar if both `x1` and `x2` are scalars.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x1 = np.array([1, 2, 3]).astype('float16')
        >>> x2 = np.array(2).astype('float16')
        >>> output = np.logaddexp(x1, x2)
        >>> print(output)
        [2.312 2.693 3.312]
    """
    def _logaddexp(x1, x2):
        return F.log(F.tensor_add(F.tensor_exp(x1), F.tensor_exp(x2)))
    return _apply_tensor_op(_logaddexp, x1, x2, dtype=dtype)


def log2(x, dtype=None):
    """
    Base-2 logarithm of `x`.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        x (Tensor): Input tensor.
        dtype (:class:`mindspore.dtype`, optional): Default: :class:`None`. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar. This is a scalar if `x` is a scalar.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.array([2, 4, 8]).astype('float16')
        >>> output = np.log2(x)
        >>> print(output)
        [1. 2. 3.]
    """
    tensor_2 = _make_tensor(2, x.dtype)

    def _log2(x):
        return F.log(x) / F.log(tensor_2)

    return _apply_tensor_op(_log2, x, dtype=dtype)


def logaddexp2(x1, x2, dtype=None):
    """
    Logarithm of the sum of exponentiations of the inputs in base of 2.

    Calculates ``log2(2**x1 + 2**x2)``.
    This function is useful in machine learning when the calculated probabilities of events
    may be so small as to exceed the range of normal floating point numbers.
    In such cases the base-2 logarithm of the calculated probability can be used instead.
    This function allows adding probabilities stored in such a fashion.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        x1 (Tensor): Input tensor.
        x2 (Tensor): Input tensor. If ``x1.shape != x2.shape``, they must be broadcastable to
            a common shape (which becomes the shape of the output).
        dtype (:class:`mindspore.dtype`): Default: :class:`None`. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar. This is a scalar if both `x1` and `x2` are scalars.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x1 = np.array([2, 4, 8]).astype('float16')
        >>> x2 = np.array(2).astype('float16')
        >>> output = np.logaddexp2(x1, x2)
        >>> print(output)
        [3. 4.32 8.02]
    """
    _check_input_tensor(x1, x2)
    add_exp = F.tensor_add(F.tensor_pow(2, x1), F.tensor_pow(2, x2))
    return log2(add_exp, dtype=dtype)


def log10(x, dtype=None):
    """
    Base-10 logarithm of `x`.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        x (Tensor): Input tensor.
        dtype (:class:`mindspore.dtype`, optional): Default: :class:`None`. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar. This is a scalar if `x` is a scalar.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.array([10, 100, 1000]).astype('float16')
        >>> output = np.log10(x)
        >>> print(output)
        [1. 2. 3.]
    """
    tensor_10 = _make_tensor(10, x.dtype)

    def _log10(x):
        return F.log(x) / F.log(tensor_10)

    return _apply_tensor_op(_log10, x, dtype=dtype)


def _cast_type_for_trigonometric(x):
    _check_input_tensor(x)
    if x.dtype != mstype.float16 or x.dtype != mstype.float32 or x.dtype != mstype.float64:
        dtype = _promote_for_trigonometric(x.dtype)
        x = F.cast(x, dtype)
    return x


def sin(x, dtype=None):
    """
    Trigonometric sine, element-wise.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        x (Tensor): Input tensor.
        dtype (:class:`mindspore.dtype`, optional): Default: :class:`None`. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar. This is a scalar if `x` is a scalar.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.array([-5, -1, 0, 2, 4, 100]).astype('float32')
        >>> output = np.sin(x)
        >>> print(output)
        [ 0.9589243  -0.84147096  0.   0.9092974  -0.7568025  -0.50636566]
    """
    x = _cast_type_for_trigonometric(x)
    return _apply_tensor_op(F.sin, x, dtype=dtype)


def cos(x, dtype=None):
    """
    Cosine element-wise.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        x (Tensor): Input tensor.
        dtype (:class:`mindspore.dtype`, optional): Default: :class:`None`. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar. This is a scalar if `x` is a scalar.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.arange(5).astype('float32')
        >>> print(np.cos(x))
        [ 1.          0.5403023  -0.41614684 -0.9899925  -0.6536436 ]
    """
    x = _cast_type_for_trigonometric(x)
    return _apply_tensor_op(F.cos, x, dtype=dtype)


def tan(x, dtype=None):
    """
    Computes tangent element-wise.

    Equivalent to :math:`np.sin(x)/np.cos(x)` element-wise.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        x (Tensor): Input tensor.
        dtype (:class:`mindspore.dtype`, optional): Default: :class:`None`. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar. This is a scalar if `x` is a scalar.

    Raises:
        TypeError: If the input is not a tensor or is :class:`tensor.dtype` is :class:`mindspore.float64`.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.array([-5, -1, 0, 2, 4, 100]).astype('float32')
        >>> print(np.tan(x))
        [ 3.380515   -1.5574077   0.         -2.1850398   1.1578213  -0.58721393]
    """
    x = _cast_type_for_trigonometric(x)
    return _apply_tensor_op(F.tan, x, dtype=dtype)


def arcsin(x, dtype=None):
    """
    Inverse sine, element-wise.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        x (Tensor): Input tensor. y-coordinate on the unit circle.
        dtype (:class:`mindspore.dtype`, optional): Default: :class:`None`. Overrides the dtype of the
            output Tensor.

    Returns:
        Output Tensor.

    Raises:
        TypeError: If the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.asarray([1, -1], np.float32)
        >>> output = np.arcsin(x)
        >>> print(output)
        [ 1.5707964 -1.5707964]
    """
    x = _cast_type_for_trigonometric(x)
    return _apply_tensor_op(F.asin, x, dtype=dtype)


def arccos(input, dtype=None):
    """
    Trigonometric inverse cosine, element-wise.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        input (Tensor): Input tensor. x-coordinate on the unit circle.
            For real arguments, the domain is :math:`[-1, 1]`.
        dtype (:class:`mindspore.dtype`, optional): Default: :class:`None`. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor.

    Raises:
        TypeError: If the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> input = np.asarray([1, -1], np.float32)
        >>> output = np.arccos(input)
        >>> print(output)
        [0.        3.1415927]
    """
    input = _cast_type_for_trigonometric(input)
    return _apply_tensor_op(F.acos, input, dtype=dtype)


def arctan(x, dtype=None):
    """
    Trigonometric inverse tangent, element-wise.

    The inverse of tan, so that if :math:`y = tan(x)` then :math:`x = arctan(y)`.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        x (Tensor): Input tensor.
        dtype (:class:`mindspore.dtype`, optional): Default: :class:`None`. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar. This is a scalar if `x` is a scalar.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.arange(5).astype('float32')
        >>> print(np.arctan(x))
        [0.        0.7853982 1.1071488 1.2490457 1.3258177]
    """
    x = _cast_type_for_trigonometric(x)
    return _apply_tensor_op(F.atan, x, dtype=dtype)


def sinh(x, dtype=None):
    """
    Hyperbolic sine, element-wise.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        x (Tensor): Input tensor.
        dtype (:class:`mindspore.dtype`, optional): Default: :class:`None`. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar. This is a scalar if `x` is a scalar.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.arange(5).astype('float32')
        >>> print(np.sinh(x))
        [ 0.         1.1752012  3.6268604 10.017875  27.289917 ]
    """
    x = _cast_type_for_trigonometric(x)
    return _apply_tensor_op(F.sinh, x, dtype=dtype)


def cosh(x, dtype=None):
    """
    Hyperbolic cosine, element-wise.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        x (Tensor): Input tensor.
        dtype (:class:`mindspore.dtype`, optional): Default: :class:`None`. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar. This is a scalar if `x` is a scalar.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.arange(5).astype('float32')
        >>> print(np.cosh(x))
        [ 1.         1.5430807  3.7621956 10.067662  27.308233 ]
    """
    x = _cast_type_for_trigonometric(x)
    return _apply_tensor_op(F.cosh, x, dtype=dtype)


def tanh(x, dtype=None):
    """
    Computes hyperbolic tangent element-wise.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        x (Tensor): Input tensor.
        dtype (:class:`mindspore.dtype`, optional): Default: :class:`None`. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar. This is a scalar if `x` is a scalar.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.arange(5).astype('float32')
        >>> print(np.tanh(x))
        [0.        0.7615942 0.9640276 0.9950548 0.9993293]
    """
    x = _cast_type_for_trigonometric(x)
    return _apply_tensor_op(F.tanh, x, dtype=dtype)


def arcsinh(x, dtype=None):
    """
    Inverse hyperbolic sine element-wise.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        x (Tensor): Input tensor.
        dtype (:class:`mindspore.dtype`, optional): Default: :class:`None`. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar. This is a scalar if `x` is a scalar.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.array([1., 2., 3., 4.], dtype=np.float32)
        >>> print(np.arcsinh(x))
        [0.8813736 1.4436355 1.8184465 2.0947125]
    """
    x = _cast_type_for_trigonometric(x)
    return _apply_tensor_op(F.asinh, x, dtype=dtype)


def arccosh(x, dtype=None):
    """
    Inverse hyperbolic cosine, element-wise.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        x (Tensor): Input tensor.
        dtype (:class:`mindspore.dtype`, optional): Default: :class:`None`. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar. This is a scalar if `x` is a scalar.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.arange(1, 5).astype('float32')
        >>> print(np.arccosh(x))
        [0.        1.316958  1.7627472 2.063437 ]
    """
    x = _cast_type_for_trigonometric(x)
    return _apply_tensor_op(F.acosh, x, dtype=dtype)


def arctanh(x, dtype=None):
    """
    Inverse hyperbolic tangent element-wise.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        x (Tensor): Input tensor.
        dtype (:class:`mindspore.dtype`, optional): Default: :class:`None`. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar. This is a scalar if `x` is a scalar.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.array([-0.99, -0.75, -0.5, 0, 0.5]).astype('float32')
        >>> print(np.arctanh(x))
        [-2.646653   -0.97295505 -0.54930615  0.          0.54930615]
    """
    x = _cast_type_for_trigonometric(x)
    return _apply_tensor_op(F.atanh, x, dtype=dtype)


def arctan2(x1, x2, dtype=None):
    """
    Element-wise arc tangent of :math:`x1/x2` choosing the quadrant correctly.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        x1 (Tensor): input tensor.
        x2 (Tensor): input tensor.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, the sum of `x1` and `x2`, element-wise. This is a scalar
        if both `x1` and `x2` are scalars.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x1 = np.array([-1, +1, +1, -1])
        >>> x2 = np.array([-1, -1, +1, +1])
        >>> output = np.arctan2(x1, x2)
        >>> print(output)
        [-2.3561945   2.3561945   0.78539819 -0.78539819]
    """
    x1 = _cast_type_for_trigonometric(x1)
    x2 = _cast_type_for_trigonometric(x2)
    return _apply_tensor_op(F.atan2, x1, x2, dtype=dtype)


def promote_types(type1, type2):
    """
    Returns the data type with the smallest size and smallest scalar kind.

    Note:
        The promotion rule is slightly different from original Numpy, but more like
        jax, due to the preference on ``32-bit`` over ``64-bit`` data types.

    Args:
        type1 (Union[:class:`mindspore.dtype`, str]): First data type.
        type2 (Union[:class:`mindspore.dtype`, str]): Second data type.

    Returns:
        The promoted data type.

    Raises:
        TypeError: If the input are not valid :class:`mindspore.dtype` input.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> output = np.promote_types(np.float32, np.float64)
        >>> print(output)
        Float64
    """
    type1 = _check_dtype(type1)
    type2 = _check_dtype(type2)
    return _promote(type1, type2)


def corrcoef(x, y=None, rowvar=True, dtype=None):
    r"""
    Returns Pearson product-moment correlation coefficients.

    Please refer to the documentation for cov for more detail. The relationship
    between the correlation coefficient matrix, R, and the covariance matrix, C, is
    :math:`R_{ij} = \frac{ C_{ij} } { \sqrt{ C_{ii} * C_{jj} } }`
    The values of R are between -1 and 1, inclusive.

    Note:
        Currently, complex numbers are not supported.

    Args:
        x (Union[int, float, bool, tuple, list, Tensor]): A 1-D or 2-D array containing
            multiple variables and observations. Each row of `x` represents a variable,
            and each column a single observation of all those variables. Also see rowvar below.
        y (Union[int, float, bool, tuple, list, Tensor], optional): An additional set
            of variables and observations. Default: `None`.
        rowvar (bool, optional): If rowvar is `True` (default), then each row represents
            a variable, with observations in the columns. Otherwise, the relationship
            is transposed: each column represents a variable, while the rows contain observations.
            Default: `True`.
        dtype (:class:`mindspore.dtype`, optional): Data-type of the result. By default,
            the return data-type will have at least float32 precision. Default: `None`.

    Returns:
        Tensor, The correlation coefficient matrix of the variables.

    Raises:
        TypeError: If the inputs have types not specified above.
        ValueError: If `x` and `y` have wrong dimensions.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> output = np.corrcoef([[2., 3., 4., 5.], [0., 2., 3., 4.], [7., 8., 9., 10.]])
        >>> print(output)
        [[1.         0.9827076  1.        ]
        [0.9827077  0.99999994 0.9827077 ]
        [1.         0.9827076  1.        ]]
    """
    # This implementation was adapted from original Numpy.
    c = cov(x, y, rowvar)
    if not c.shape:
        return F.tensor_div(c, c)
    d = diag(c)
    stddev = sqrt(d)
    c /= F.expand_dims(stddev, -1)
    c /= F.expand_dims(stddev, 0)
    c = clip(c, -1, 1)
    if dtype is not None:
        return c.astype(dtype)
    return c


def _slice_along_axis(f, axis, slice_start, slice_end):
    """
    Slice a tensor along a given axis, a helper function for gradient

    Args:
        f (Tensor): Input Tensor.
        axis (int): Specified axis.
        slice_start (int): The start of the slice.
        slice_end (int): The end of the int.

    Returns:
        Sliced tensor.
    """
    slice_size = slice_end - slice_start
    index_start = (0,) * f.ndim
    index_end = f.shape
    index_start = _tuple_setitem(index_start, axis, slice_start)
    index_end = _tuple_setitem(index_end, axis, slice_size)
    return F.tensor_slice(f, index_start, index_end)


def _gradient_along_axis(f, h, axis):
    """compute the gradients of `f` along a given axis, a helper function of gradient."""
    end = f.shape[axis]
    upper_edge = _slice_along_axis(f, axis, 1, 2) - _slice_along_axis(f, axis, 0, 1)
    lower_edge = _slice_along_axis(f, axis, end-1, end) - _slice_along_axis(f, axis, end-2, end-1)
    if end <= 2:
        a_grad = concatenate((upper_edge, lower_edge), axis)
    else:
        middle = (_slice_along_axis(f, axis, 2, end) - _slice_along_axis(f, axis, 0, end-2)) * 0.5
        a_grad = concatenate((upper_edge, middle, lower_edge), axis)
    return a_grad / h


def check_gradient_arguments(f, axis, edge_order):
    """check arguments for gradient"""
    if edge_order != 1:
        _raise_unimplemented_error("edge_order != 1 not implemented")
    if not isinstance(f, Tensor):
        f = asarray_const(f)
    if f.dtype != mstype.float64:
        f = f.astype(mstype.float32)
    if axis is None:
        axis = F.make_range(f.ndim)
    else:
        _check_axis_type(axis, True, True, True)
        axis = _canonicalize_axis(axis, f.ndim)
        axis = (axis,) if isinstance(axis, int) else axis
    return f, axis, edge_order


def gradient(f, *varargs, axis=None, edge_order=1):
    """
    Returns the gradient of a N-dimensional array.
    The gradient is computed using second order accurate central differences
    in the interior points and either first or second order accurate one-sides
    (forward or backwards) differences at the boundaries.
    The returned gradient hence has the same shape as the input array.

    Note:
        Currently we only support `edge_order` =1 and uniform spacing of `varargs`.

    Args:
        f (Union[tuple, list, Tensor]): An N-dimensional array containing samples of
            a scalar function.
        varargs (Union[tuple[number], tuple[tensor scalar]], optional)
            Spacing between f values. Default unitary spacing for all dimensions.
            Spacing can be specified using:
            1. single scalar to specify a sample distance for all dimensions.
            2. N scalars to specify a constant sample distance for each dimension.
        axis (Union[None, int, tuple(int), list(int)], optional): Gradient is calculated
            only along the given axis or axes. The default :class:`(axis = None)` is to calculate
            the gradient for all the axes of the input tensor. `axis` may be negative,
            in which case it counts from the last to the first `axis`.
        edge_order (int): Gradient is calculated using N-th order accurate differences
            at the boundaries. Default: 1.

    Returns:
        gradient, a list of tensors (or a single tensor if there is only one dimension
        to be calculated). Each derivative has the same shape as f.

    Raises:
        TypeError: If the inputs have types not specified above.
        ValueError: If `axis` values out of bounds, or shape of `f` has entries < 1.
        NotImplementedError: If `edge_order` != 1, or `varargs` contains non-scalar entries.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> output = np.gradient([[1, 2, 6], [3, 4, 5]], axis=-1)
        >>> print(output)
        [[1.  2.5 4. ]
        [1.  1.  1. ]]
    """
    # This implementation was adapted from Numpy and jax.numpy
    f, axis, edge_order = check_gradient_arguments(f, axis, edge_order)

    len_axes = len(axis)
    n = len(varargs)
    dx = None
    # check varargs and make varags the same length as axis
    if n == 0 or varargs is None:
        # no spacing
        dx = (1,) * len_axes
    elif n == 1:
        # single value for all axes
        dx = varargs * len_axes
    elif n == len_axes:
        dx = varargs
    else:
        _raise_type_error("Invalid number of arguments")

    a_grad = []

    for idx in F.make_range(len_axes):
        h = dx[idx]
        ax = axis[idx]
        if f.shape[ax] < 2:
            _raise_value_error("Shape of array too small to calculate a numerical gradient, "
                               "at least 2 elements are required.")
        # if h is not scalar
        if not (isinstance(h, (int, float, bool)) or (isinstance(h, Tensor) and h.ndim == 0)):
            _raise_unimplemented_error("Non-constant spacing not implemented")

        a_grad.append(_gradient_along_axis(f, h, ax))

    if len(axis) == 1:
        return a_grad[0]

    return a_grad


def sum_(a, axis=None, dtype=None, keepdims=False, initial=None):
    """
    Returns sum of array elements over a given axis.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and
        `extobj` are not supported.

    Args:
        x (Union[int, float, bool, list, tuple, Tensor]): Elements to sum.
        axis (Union[None, int, tuple(int)]): Axis or axes along which a sum is performed. Default: `None`.
            If `None`, sum all of the elements of the input array.
            If axis is negative it counts from the last to the first axis.
            If axis is a tuple of integers, a sum is performed on all of the axes specified in the tuple
            instead of a single axis or all the axes as before.
        dtype (:class:`mindspore.dtype`, optional): Defaults to `None`. Overrides the dtype of the
            output Tensor.
        keepdims (bool): If this is set to True, the axes which are reduced are left in the result as
            dimensions with size one. With this option, the result will broadcast correctly against the input array.
            If the default value is passed, then keepdims will not be passed through to the sum method of
            sub-classes of ndarray, however any non-default value will be. If the sub-class method does not
            implement keepdims any exceptions will be raised. Default: `False`.
        initial (scalar): Starting value for the sum, if `None`, which refers to the first element of the reduction.
            Default: `None`.

    Returns:
        Tensor. An array with the same shape as a, with the specified axis removed.
        If a is a 0-d array, or if axis is `None`, a scalar is returned.
        If an output array is specified, a reference to out is returned.

    Raises:
        TypeError: If input is not array_like or `axis` is not int or tuple of integers or
            `keepdims` is not integer or `initial` is not scalar.
        ValueError: If any axis is out of range or duplicate axes exist.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.sum([0.5, 1.5]))
        2.0
        >>> x = np.arange(10).reshape(2, 5).astype('float32')
        >>> print(np.sum(x, axis=1))
        [10. 35.]
    """
    a = _to_tensor(a)
    return a.sum(axis, dtype, keepdims, initial)


@constexpr
def _min_cost_chain_matmul(dims):
    """
    Returns indices of splits that has the minimal cost for matmul.
    s[i, j] holds the index of the split with minimal cost for arrays[i, i + 1, ... j]
    """
    dims = tuple(dims)
    n = len(dims) - 1
    m = [[0]*n for _ in range(n)]
    s = [[0]*n for _ in range(n)]
    for pos in range(1, n):
        for i in range(n - pos):
            j = i + pos
            m[i][j] = sys.maxsize
            for k in range(i, j):
                cost = m[i][k] + m[k + 1][j] + dims[i]*dims[k + 1]*dims[j + 1]
                if cost < m[i][j]:
                    m[i][j] = cost
                    s[i][j] = k
    return s


@_primexpr
def _get_dims(shapes):
    """
    Returns the chain of the dimensions in arrays.
    dims[i] == arrays[i - 1].shape[1] == arrays[i].shape[0]
    """
    shapes = tuple(shapes)
    if any(len(shape) != 2 for shape in shapes):
        raise ValueError('Array must be 2 dimensional')
    dims = tuple(map(operator.itemgetter(0), shapes))
    if any(shape[1] != dim for shape, dim in zip(shapes[:-1], dims[1:])):
        raise ValueError(f'shapes not aligned')
    return dims + (shapes[-1][1],)


def _multi_dot(arrays, i, j, order):
    """Computes multi dot recursively using minimal cost."""
    if i == j:
        return arrays[i]
    return dot(_multi_dot(arrays, i, order[i][j], order),
               _multi_dot(arrays, order[i][j] + 1, j, order))


def multi_dot(arrays):
    """
    Computes the dot product of two or more arrays in a single function call, while automatically
    selecting the fastest evaluation order.
    multi_dot chains numpy.dot and uses optimal parenthesization of the matrices. For more
    information, refer to the `wiki page <https://en.wikipedia.org/wiki/Matrix_chain_multiplication>`_.
    Depending on the shapes of the matrices, this can speed up the multiplication a lot.
    If the first argument is 1-D, it is treated as a row vector. If the last argument is 1-D, it
    is treated as a column vector. The other arguments must be 2-D.

    Note:
        Numpy argument `out` is not supported.

    Args:
        arrays (sequence of array_like): If the first argument is 1-D, it is treated as row
            vector. If the last argument is 1-D, it is treated as column vector. The other
            arguments must be 2-D.

    Returns:
        Tensor, the dot product of the supplied arrays.

    Raises:
        ValueError: arrays are not 2-D.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> A = np.ones((10000, 100))
        >>> B = np.ones((100, 1000))
        >>> C = np.ones((1000, 5))
        >>> D = np.ones((5, 333))
        >>> output = np.multi_dot([A, B, C, D])
        >>> print(output)
        [[500000. 500000. 500000. ... 500000. 500000. 500000.]
        [500000. 500000. 500000. ... 500000. 500000. 500000.]
        [500000. 500000. 500000. ... 500000. 500000. 500000.]
        ...
        [500000. 500000. 500000. ... 500000. 500000. 500000.]
        [500000. 500000. 500000. ... 500000. 500000. 500000.]
        [500000. 500000. 500000. ... 500000. 500000. 500000.]]
    """
    if len(arrays) < 2:
        _raise_value_error('Expecting at least 2 arrays')
    if isinstance(arrays, (tuple, list)):
        arrays = _to_tensor(*arrays)
    else:
        arrays = _to_tensor(arrays)
        num = len(arrays)
        arrays = F.reshape(arrays, (-1,) + _tuple_slice(F.shape(arrays), 2, None))
        arrays = split(arrays, num)
    if len(arrays) == 2:
        return dot(*arrays)

    shape_out = ()
    arrs = []
    for arr in arrays:
        arrs.append(arr)

    if F.rank(arrs[0]) == 1:
        arrs[0] = F.reshape(arrs[0], (1, arrs[0].size))
    else:
        shape_out += (F.shape(arrs[0])[0],)
    if F.rank(arrs[-1]) == 1:
        arrs[-1] = F.reshape(arrs[-1], (arrs[-1].size, 1))
    else:
        shape_out += (F.shape(arrs[-1])[1],)

    shapes = []
    for arr in arrs:
        shapes.append(F.shape(arr))
    dims = _get_dims(shapes)
    order = _min_cost_chain_matmul(dims)
    res = _multi_dot(arrs, 0, len(arrs) - 1, order)
    return F.reshape(res, shape_out)


def argmax(a, axis=None):
    """
    Returns the indices of the maximum values along an axis.

    Note:
        Numpy argument `out` is not supported.
        On Ascend, in case of multiple occurrences of the maximum values, the return
        indices may not necessarily correspond to the first occurrence.

    Args:
        a (Union[int, float, bool, list, tuple, Tensor]): Input array.
        axis (int, optional): By default, the index is into
            the flattened array, otherwise along the specified axis.
            Default: `None`.

    Returns:
        Tensor, array of indices into the array. It has the same
        shape as a.shape with the dimension along axis removed.

    Raises:
        ValueError: If axis is out of range.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.arange(10, 16).reshape(2, 3)
        >>> print(np.argmax(a))
        5
        >>> print(np.argmax(a, axis=0))
        [1 1 1]
        >>> print(np.argmax(a, axis=1))
        [2 2]
    """
    a = _to_tensor(a)
    if a.dtype == mstype.bool_:
        a = a.astype(mstype.int32)
    return a.argmax(axis)


def argmin(a, axis=None):
    """
    Returns the indices of the minimum values along an axis.

    Note:
        Numpy argument `out` is not supported.

    Args:
        a (Union[int, float, bool, list, tuple, Tensor]): Input array.
        axis (int, optional): By default, the index is into
            the flattened array, otherwise along the specified axis.
            Default: `None`.

    Returns:
        Tensor, array of indices into the array. It has the same
        shape as a.shape with the dimension along axis removed.

    Raises:
        ValueError: If axis is out of range.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.arange(10, 16).reshape(2, 3)
        >>> print(np.argmin(a))
        0
        >>> print(np.argmin(a, axis=0))
        [0 0 0]
        >>> print(np.argmin(a, axis=1))
        [0 0]
    """
    a = _to_tensor(a)
    return a.argmin(axis)


@constexpr
def _get_sort_range(size):
    """Returns the range for number of searches (log2(size)) on a sorted array with the given size."""
    return tuple(range(ceil(log2(_to_tensor(size + 1).astype(mstype.float32))).astype(mstype.int32)))


def searchsorted(a, v, side='left', sorter=None):
    """
    Finds indices where elements should be inserted to maintain order.
    Finds the indices into a sorted array `a` such that, if the corresponding elements
    in `v` were inserted before the indices, the order of `a` would be preserved.

    Args:
        a (Union[list, tuple, Tensor]): 1-D input array. If `sorter` is
            None, then it must be sorted in ascending order, otherwise `sorter` must be
            an array of indices that sort it.
        v (Union[int, float, bool, list, tuple, Tensor]): Values to insert into `a`.
        side ('left', 'right', optional): If 'left', the index of the first suitable
            location found is given. If 'right', return the last such index. If there is
            no suitable index, return either 0 or N (where N is the length of `a`).
        sorter (Union[int, float, bool, list, tuple, Tensor]): 1-D optional array of
            integer indices that sort array `a` into ascending order. They are typically
            the result of argsort.

    Returns:
        Tensor, array of insertion points with the same shape as `v`.

    Raises:
        ValueError: If argument for `side` or `sorter` is invalid.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import numpy as np
        >>> print(np.searchsorted([1,2,3,4,5], 3))
        2
        >>> print(np.searchsorted([1,2,3,4,5], 3, side='right'))
        3
        >>> print(np.searchsorted([1,2,3,4,5], [-10, 10, 2, 3]))
        [0 5 1 2]
    """
    if side not in ('left', 'right'):
        _raise_value_error('invalid value for keyword "side"')
    a = _to_tensor(a).astype(mstype.float32)
    if F.rank(a) != 1:
        _raise_value_error('`a` should be 1-D array')
    v = _to_tensor(v)
    shape = F.shape(v)
    if sorter is not None:
        if F.rank(sorter) != 1 or sorter.size != a.size:
            _raise_value_error('sorter must be 1-D array with the same size as `a`')
        sorter = _to_tensor(sorter)
        sorter = F.expand_dims(sorter, -1)
        a = F.gather_nd(a, sorter)
    less_op = F.tensor_le if side == 'left' else F.tensor_lt
    i = F.fill(mstype.int32, shape, 0)
    j = F.fill(mstype.int32, shape, a.size)
    two = F.fill(mstype.int32, shape, 2)

    for _ in _get_sort_range(a.size):
        mid = floor_divide(add(i, j), two)
        mask = less_op(v, F.gather_nd(a, F.expand_dims(mid, -1)))
        i = F.select(mask, i, mid)
        j = F.select(mask, mid, j)
    return j


def interp(x, xp, fp, left=None, right=None):
    """
    One-dimensional linear interpolation for monotonically increasing sample points.
    Returns the one-dimensional piecewise linear interpolant to a function with given
    discrete data points `(xp, fp)`, evaluated at `x`.

    Note:
        Numpy argument `period` is not supported.
        Complex values are not supported.

    Args:
        x (Union[int, float, bool, list, tuple, Tensor]): The x-coordinates at which
            to evaluate the interpolated values.
        xp (Union[int, float, bool, list, tuple, Tensor]): 1-D sequence of floats, the
            x-coordinates of the data points, must be increasing.
        fp (Union[int, float, bool, list, tuple, Tensor]): 1-D sequence of floats, the
            y-coordinates of the data points, same length as `xp`.
        left (float, optional): Value to return for ``x < xp[0]``, default is ``fp[0]``
            once obtained.
        right (float, optional): Value to return for ``x > xp[-1]``, default is ``fp[-1]``
            once obtained.

    Returns:
        Tensor, the interpolated values, same shape as `x`.

    Raises:
        ValueError: If `xp` or `fp` is not one-dimensional, or if `xp` and `fp` do not have
            the same length.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> xp = [1, 2, 3]
        >>> fp = [3, 2, 0]
        >>> print(np.interp([0, 1, 1.5, 2.72, 3.14], xp, fp))
        [3.         3.         2.5        0.55999994 0.        ]
        >>> UNDEF = -99.0
        >>> print(np.interp(3.14, xp, fp, right=UNDEF))
        -99.0
    """
    # implement period once sort is supported
    x, xp, fp = _to_tensor(x, xp, fp)
    if F.rank(xp) != 1 or F.rank(fp) != 1:
        _raise_value_error('xp and fp must be 1-d sequences')
    size = xp.size
    if fp.size != size:
        _raise_value_error('the y-coordinates must have the same length as `xp`')

    xp = xp.astype(mstype.float32)
    fp = fp.astype(mstype.float32)

    indices_1 = clip(searchsorted(xp, x), 0, size - 1)
    indices_0 = clip(indices_1 - _to_tensor(1), 0, size - 1)
    indices_0 = F.expand_dims(indices_0, -1)
    indices_1 = F.expand_dims(indices_1, -1)
    x_0 = F.gather_nd(xp, indices_0)
    x_1 = F.gather_nd(xp, indices_1)
    y_0 = F.gather_nd(fp, indices_0)
    y_1 = F.gather_nd(fp, indices_1)
    res = (y_0*(x_1 - x) + y_1*(x - x_0))/(x_1 - x_0)
    res = F.select(F.equal(x_0, x_1), y_0, res)

    idx_0 = _to_tensor([0])
    idx_last = _to_tensor([size - 1])
    if left is None:
        left = F.gather_nd(fp, idx_0)
    left = full(F.shape(x), left, mstype.float32)
    if right is None:
        right = F.gather_nd(fp, idx_last)
    right = full(F.shape(x), right, mstype.float32)
    res = F.select(F.tensor_lt(x, F.gather_nd(xp, idx_0)), left, res)
    res = F.select(F.tensor_gt(x, F.gather_nd(xp, idx_last)), right, res)
    return res


def _apply_tensor_op(fn, *args, dtype=None):
    """Applies tensor operations based on fn"""
    args = _to_tensor(*args)
    if isinstance(args, Tensor):
        res = fn(args)
    else:
        res = fn(*args)
    if dtype is not None and not _check_same_type(F.dtype(res), dtype):
        res = F.cast(res, dtype)
    return res


def sign(x, dtype=None):
    """
    Returns an element-wise indication of the sign of a number.

    The sign function returns `-1 if x < 0, 0 if x == 0, 1 if x > 0`. nan is returned for nan inputs.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.
        Complex inputs are not supported now.
        On Ascend, integer inputs are not supported.

    Args:
        x (Union[int, float, list, tuple, Tensor]): Input values.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        The sign of x. This is a tensor or a scalar when x is a scalar.

    Raises:
        TypeError: If dtype of the input is not in the given types or
            the input can not be converted to tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> output = np.sign(np.array([-1., 0., 1., 1.2]))
        >>> print(output)
        [-1.  0.  1.  1.]
    """
    if not isinstance(x, (int, float, list, tuple, Tensor)):
        _raise_type_error('integer, float, list, tuple or Tensor are expected, but got', x)
    x = _to_tensor(x)
    if _check_same_type(F.dtype(x), mstype.bool_):
        _raise_type_error("sign does not accept dtype bool.")

    _non_zero_sign = x / absolute(x)
    _zero = _broadcast_to_shape(_make_tensor(0, x.dtype), x.shape)
    is_zero = F.equal(x, 0)
    res = F.select(is_zero, _zero, _non_zero_sign)

    if dtype is not None and not _check_same_type(F.dtype(res), dtype):
        res = F.cast(res, dtype)
    return res


def copysign(x1, x2, dtype=None):
    """
    Changes the sign of `x1` to that of `x2`, element-wise.

    If `x2` is a scalar, its sign will be copied to all elements of `x1`.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.
        Complex inputs are not supported now.

    Args:
        x1 (Union[int, float, list, tuple, Tensor]): Values to change the sign of.
        x2 (Union[int, float, list, tuple, Tensor]): The sign of x2 is copied to x1. If `x1.shape != x2.shape`,
            they must be broadcastable to a common shape (which becomes the shape of the output).
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar. The values of `x1` with the sign of `x2`. This is a scalar if both `x1` and `x2` are scalars.

    Raises:
        TypeError: If dtype of the input is not in the given types or
            the input can not be converted to tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> output = np.copysign(np.array([1, -1, -1]), np.array([-1, 1, -1]))
        >>> print(output)
        [-1  1 -1]
    """
    if not isinstance(x1, (int, float, list, tuple, Tensor)):
        _raise_type_error('integer, float, list, tuple or Tensor are expected, but got', x1)
    if not isinstance(x2, (int, float, list, tuple, Tensor)):
        _raise_type_error('integer, float, list, tuple or Tensor are expected, but got', x2)
    x1, x2 = _to_tensor(x1, x2)
    shape_out = _infer_out_shape(F.shape(x1), F.shape(x2))
    x1 = _broadcast_to_shape(x1, shape_out)
    x2 = _broadcast_to_shape(x2, shape_out)
    if _check_same_type(F.dtype(x1), mstype.bool_) or _check_same_type(F.dtype(x2), mstype.bool_):
        _raise_type_error("sign does not accept dtype bool.")

    original_dtype = x1.dtype
    if not _check_is_float(original_dtype):
        pos_tensor = F.absolute(x1.astype('float32')).astype(original_dtype)
    else:
        pos_tensor = F.absolute(x1)

    neg_tensor = F.neg_tensor(pos_tensor)
    less_zero = F.less(x2, 0)
    res = F.select(less_zero, neg_tensor, pos_tensor)

    if dtype is not None and not _check_same_type(F.dtype(res), dtype):
        res = F.cast(res, dtype)
    return res


def digitize(x, bins, right=False):
    """
    Returns the indices of the bins to which each value in input array belongs.
    If values in `x` are beyond the bounds of `bins`, 0 or ``len(bins)`` is returned
    as appropriate.

    Args:
        x (Union[int, float, bool, list, tuple, Tensor]): Input array to be binned.
        bins (Union[list, tuple, Tensor]): Array of bins. It has to
            be 1-dimensional and monotonic.
        right (boolean, optional): Indicating whether the intervals include the right
            or the left bin edge. Default behavior is ``(right==False)`` indicating
            that the interval does not include the right edge. The left bin end is
            open in this case, i.e., ``bins[i-1] <= x < bins[i]`` is the default
            behavior for monotonically increasing bins.

    Returns:
        Tensor of ints, output array of indices, of same shape as `x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.array([1.2, 10.0, 12.4, 15.5, 20.])
        >>> bins = np.array([0, 5, 10, 15, 20])
        >>> inds = np.digitize(x, bins)
        >>> print(inds)
        [1 3 3 4 5]
    """
    x, bins = _to_tensor(x, bins)
    if F.rank(bins) != 1:
        _raise_value_error('bins should be 1-dimensional')
    if x.size == 0:
        return x
    if bins.size == 0:
        return zeros(F.shape(x), mstype.int32)
    side = 'left' if right else 'right'
    first_bin = bins[0]
    last_bin = bins[_type_convert(int, bins.size) - 1]
    cond = first_bin <= last_bin
    incr = searchsorted(bins, x, side)
    decr = _to_tensor(bins.size) - searchsorted(flip(bins), x, side)
    return where_(cond, incr, decr)


def bincount(x, weights=None, minlength=0, length=None):
    """
    Count number of occurrences of each value in array of non-negative ints.
    The number of bins (of size 1) is one larger than the largest value in `x`.
    If `minlength` is specified, there will be at least this number of bins in the
    output array (though it will be longer if necessary, depending on the contents
    of `x`). Each bin gives the number of occurrences of its index value in `x`. If
    `weights` is specified the input array is weighted by it, i.e. if a value `n`
    is found at position `i`, ``out[n] += weight[i]`` instead of ``out[n] += 1``.

    Note:
        The additional argument `length` specifies the number of bins (overriding
        ``x.max() + 1``), which must be provided in graph mode.
        If `x` contains negative values, no error will be raised, and negative values
        are treated as zeros instead.

    Args:
        x (Union[list, tuple, Tensor]): 1-d input array.
        weights (Union[int, float, bool, list, tuple, Tensor], optional): Weights,
            array of the same shape as `x`. Defaults to None.
        minlength (int, optional): A minimum number of bins for the output array.
            Defaults to 0.
        length (int, optional): Number of bins. Defaults to None.

    Returns:
        Tensor, the result of binning the input array. The length of out is equal to
        ``np.amax(x)+1``.

    Raises:
        ValueError: If `x` is not one-dimensional, or if `x` and `weights` do not have
            the same shape.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.bincount(np.arange(5)))
        [1. 1. 1. 1. 1.]
        >>> print(np.bincount(np.array([0, 1, 1, 3, 2, 1, 7])))
        [1. 3. 1. 1. 0. 0. 0. 1.]
        >>> w = np.array([0.3, 0.5, 0.2, 0.7, 1., -0.6]) # weights
        >>> x = np.array([0, 1, 1, 2, 2, 2])
        >>> print(np.bincount(x,  weights=w))
        [0.3 0.7 1.1]
    """
    x = _to_tensor(x)
    if F.rank(x) != 1:
        _raise_value_error('`x` should be one-dimensional')
    if not _check_is_int(F.dtype(x)):
        _raise_type_error('`x` should be an array of ints')
    x = clip(x, 0, None)
    if length is None:
        if F.isconstant(x):
            length = int(maximum(F.reduce_max(x.astype(mstype.float32)), minlength - 1).asnumpy()) + 1
        else:
            _raise_value_error('argument `length` must be provided in graph mode')
    idx = arange(length).reshape(length, 1)
    idx_mapping = F.equal(x, idx)
    if weights is not None:
        weights = _to_tensor(weights)
        if F.shape(x) != F.shape(weights):
            _raise_value_error('`x` and `weights` must have the same length')
        idx_mapping *= weights
    return F.reduce_sum(idx_mapping.astype(mstype.float32), 1).ravel()


def histogram(a, bins=10, range=None, weights=None, density=False): # pylint: disable=redefined-builtin
    """
    Computes the histogram of a dataset.

    Note:
        String values for `bins` is not supported.
        Deprecated numpy argument `normed` is not supported.

    Args:
        a (Union[int, float, bool, list, tuple, Tensor]): Input data. The histogram
            is computed over the flattened array.
        bins (Union[int, tuple, list, Tensor], optional): If `bins` is an int, it
            defines the number of equal-width bins in the given range (10, by
            default). If `bins` is a sequence, it defines the bin edges, including
            the rightmost edge, allowing for non-uniform bin widths.
        range((float, float), optional): The lower and upper range of the bins. If
            not provided, `range` is simply ``(a.min(), a.max())``. Values outside
            the range are ignored. The first element of the range must be less than
            or equal to the second.
        weights (Union[int, float, bool, list, tuple, Tensor], optional): An array
            of weights, of the same shape as `a`. If density is True, the weights
            are normalized, so that the integral of the density over the range
            remains 1.
        density (boolean, optional): If False, the result will contain the number of
            samples in each bin. If True, the result is the value of the probability
            density function at the bin, normalized such that the integral over the
            range is 1. Note that the sum of the histogram values will not be equal
            to 1 unless bins of unity width are chosen; it is not a probability mass
            function.

    Returns:
        (Tensor, Tensor), the values of the histogram and the bin edges.

    Raises:
        ValueError: If `x` and `weights` do not have the same size.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import numpy as np
        >>> print(np.histogram([1, 2, 1], bins=[0, 1, 2, 3]))
        (Tensor(shape=[3], dtype=Float32, value= [ 0.00000000e+00,  2.00000000e+00,  1.00000000e+00]),
        Tensor(shape=[4], dtype=Int32, value= [0, 1, 2, 3]))
        >>> print(np.histogram(np.arange(4), bins=np.arange(5), density=True))
        (Tensor(shape=[4], dtype=Float32, value=
        [ 2.50000000e-01,  2.50000000e-01,  2.50000000e-01,  2.50000000e-01]),
        Tensor(shape=[5], dtype=Int32, value= [0, 1, 2, 3, 4]))
        >>> print(np.histogram([[1, 2, 1], [1, 0, 1]], bins=[0,1,2,3]))
        (Tensor(shape=[3], dtype=Float32, value= [ 1.00000000e+00,  4.00000000e+00,  1.00000000e+00]),
        Tensor(shape=[4], dtype=Int32, value= [0, 1, 2, 3]))
    """
    a = _to_tensor(a)
    if weights is not None:
        weights = _to_tensor(weights)
        if F.shape(a) != F.shape(weights):
            _raise_value_error('weights should have the same shape as a')
        weights = weights.ravel()
    a = a.ravel()
    bin_edges = histogram_bin_edges(a, bins, range, weights)
    data_to_bins = searchsorted(bin_edges, a, 'right')
    bin_size = _type_convert(int, bin_edges.size)
    data_to_bins = where_(a == bin_edges[-1], _to_tensor(bin_size - 1), data_to_bins)
    count = bincount(data_to_bins, weights, length=bin_size)[1:]
    if count.size == 0:
        return count, bin_edges
    if density:
        count = F.cast(count, mstype.float32)
        count = count/diff(bin_edges)/F.reduce_sum(count)
    return count, bin_edges


@constexpr
def _factor_flattened_hist(nbin):
    """Returns the factor that will be applied to the histogram to be flattened."""
    factor = list((itertools.accumulate(nbin[1:][::-1], operator.mul)))[::-1]
    factor.append(1)
    return factor


def _get_histogramdd_count(ndim, bin_edges, sample, weights):
    """Returns count for histogramdd."""
    data_indices = []
    nbin = ()
    flattened_bin_size = 1
    for i in F.make_range(ndim):
        data_to_bins = searchsorted(bin_edges[i], sample[:, i], 'right')
        bin_size = _type_convert(int, bin_edges[i].size)
        data_to_bins = where_(sample[:, i] == bin_edges[i][-1], _to_tensor(bin_size - 1), data_to_bins)
        data_indices.append(data_to_bins)
        nbin += (bin_size + 1,)
        flattened_bin_size *= (bin_size + 1)

    factor = F.reshape(_to_tensor(_factor_flattened_hist(nbin)), (ndim, 1))
    stacked_indices = stack(data_indices) * factor
    if _get_device() == 'Ascend':
        stacked_indices = F.cast(stacked_indices, mstype.float32)
    flattened_hist = F.reduce_sum(stacked_indices.astype(mstype.float32), 0)
    count = bincount(flattened_hist.astype(mstype.int32), weights, length=flattened_bin_size)
    count = F.reshape(count, nbin)
    slices = _list_comprehensions(ndim, F.make_slice(1, -1, 1), True)
    count = count[slices]
    return count


def histogramdd(sample, bins=10, range=None, weights=None, density=False): # pylint: disable=redefined-builtin
    """
    Computes the multidimensional histogram of some data.

    Note:
        Deprecated numpy argument `normed` is not supported.

    Args:
        sample (Union[list, tuple, Tensor]): The data to be histogrammed, either `(N, D)`
            array, or `(D, N)` array_like. Note the unusual interpretation of sample
            when an array_like:

            When an array, each row is a coordinate in a `D-dimensional` space, such as
            ``histogramdd(np.array([p1, p2, p3]))``.

            When an array_like, each element is the list of values for single coordinate,
            such as ``histogramdd((X, Y, Z))``.

            The first form should be preferred.
        bins (Union[int, tuple, list], optional): The bin specification:

            A sequence of arrays describing the monotonically increasing bin edges along
            each dimension.

            The number of bins for each dimension ``(nx, ny, … =bins)``

            The number of bins for all dimensions ``(nx=ny=…=bins)``.
        range(Union[list, tuple], optional): A sequence of length `D`, each an optional
            ``(lower, upper)`` tuple giving the outer bin edges to be used if the edges
            are not given explicitly in bins. An entry of None in the sequence results in
            the minimum and maximum values being used for the corresponding dimension.
            The default, None, is equivalent to passing a tuple of `D` None values.
        weights (Union[list, tuple, Tensor], optional): An array with shape `(N,)` of values
            `w_i` weighing each sample ``(x_i, y_i, z_i, …)``.
        density (boolean, optional): If False, the default, returns the number of samples
            in each bin. If True, returns the probability density function at the bin,
            ``bin_count / sample_count / bin_volume``.

    Returns:
        (Tensor, list of Tensor), the values of the histogram and the bin edges.

    Raises:
        ValueError: If `range` does not have the same size as the number of samples.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import numpy as np
        >>> sample = np.arange(15).reshape(5, 3)
        >>> print(sample)
        [[ 0  1  2]
        [ 3  4  5]
        [ 6  7  8]
        [ 9 10 11]
        [12 13 14]]
        >>> print(np.histogramdd(sample, bins=(2, 3, 4)))
        (Tensor(shape=[2, 3, 4], dtype=Float32, value=
        [[[ 1.00000000e+00,  1.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]],
        [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  2.00000000e+00]]]),
        [Tensor(shape=[3], dtype=Float32, value= [ 0.00000000e+00,  6.00000000e+00,  1.20000000e+01]),
        Tensor(shape=[4], dtype=Float32, value=
        [ 1.00000000e+00,  5.00000000e+00,  9.00000000e+00,  1.30000000e+01]),
        Tensor(shape=[5], dtype=Float32, value=
        [ 2.00000000e+00,  5.00000000e+00,  8.00000000e+00,  1.10000000e+01,  1.40000000e+01])])
    """
    if isinstance(sample, (tuple, list)):
        sample = _to_tensor(*sample)
        sample = stack(sample, -1)
    elif not isinstance(sample, Tensor):
        _raise_type_error('sample should be (N, D) array, or (D, N) array_like')
    if F.rank(sample) != 2:
        _raise_value_error('when an array, sample should be 2-dimensional')
    ndim = F.shape(sample)[1]

    if isinstance(bins, int):
        bins = _list_comprehensions(ndim, bins)
    if isinstance(bins, (tuple, list, Tensor)):
        if len(bins) != ndim:
            _raise_value_error('The dimension of bins must be equal to the dimension of the sample')
    else:
        _raise_type_error('bins should be int or sequence')

    if range is None:
        range = _list_comprehensions(ndim, None, False, True)
    else:
        if len(range) != ndim:
            _raise_value_error('range argument must have one entry per dimension')

    bin_edges = []
    dedges = []
    for i in F.make_range(ndim):
        edges = histogram_bin_edges(sample[:, i], bins[i], range[i], weights)
        bin_edges.append(edges)
        dedges.append(diff(edges))

    count = _get_histogramdd_count(ndim, bin_edges, sample, weights)

    if density:
        s = F.reduce_sum(count.astype(mstype.float32))
        for i in F.make_range(ndim):
            shape = _expanded_shape(ndim, dedges[i].size, i)
            count /= _to_tensor(dedges[i]).reshape(shape)
        count /= s
    return count, bin_edges


def histogram2d(x, y, bins=10, range=None, weights=None, density=False): # pylint: disable=redefined-builtin
    """
    Computes the multidimensional histogram of some data.

    Note:
        Deprecated numpy argument `normed` is not supported.

    Args:
        x (Union[list, tuple, Tensor]): An array with shape `(N,)` containing the x
            coordinates of the points to be histogrammed.
        y (Union[list, tuple, Tensor]): An array with shape `(N,)` containing the y
            coordinates of the points to be histogrammed.
        bins (Union[int, tuple, list], optional): The bin specification:

            If int, the number of bins for the two dimensions ``(nx=ny=bins)``.

            If array_like, the bin edges for the two dimensions ``(x_edges=y_edges=bins)``.

            If [int, int], the number of bins in each dimension ``(nx, ny = bins)``.

            If [array, array], the bin edges in each dimension ``(x_edges, y_edges = bins)``.

            A combination [int, array] or [array, int], where int is the number of bins and
            array is the bin edges.
        range(Union[list, tuple], optional): has shape (2, 2), the leftmost and rightmost
            edges of the bins along each dimension (if not specified explicitly in the bins
            parameters): ``[[xmin, xmax], [ymin, ymax]]``. All values outside of this range
            will be considered outliers and not tallied in the histogram.
        weights (Union[list, tuple, Tensor], optional): An array with shape `(N,)` of values
            `w_i` weighing each sample `(x_i, y_i)`.
        density (boolean, optional): If False, the default, returns the number of samples
            in each bin. If True, returns the probability density function at the bin,
            ``bin_count / sample_count / bin_volume``.

    Returns:
        (Tensor, Tensor, Tensor), the values of the bi-directional histogram and the bin edges
        along the first and second dimensions.

    Raises:
        ValueError: If `range` does not have the same size as the number of samples.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import numpy as np
        >>> x = np.arange(5)
        >>> y = np.arange(2, 7)
        >>> print(np.histogram2d(x, y, bins=(2, 3)))
        (Tensor(shape=[2, 3], dtype=Float32, value=
        [[ 2.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  1.00000000e+00,  2.00000000e+00]]),
        Tensor(shape=[3], dtype=Float32, value= [ 0.00000000e+00,  2.00000000e+00,  4.00000000e+00]),
        Tensor(shape=[4], dtype=Float32, value=
        [ 2.00000000e+00,  3.33333349e+00,  4.66666698e+00,  6.00000000e+00]))
    """
    count, bin_edges = histogramdd((x, y), bins=bins, range=range, weights=weights, density=density)
    return count, bin_edges[0], bin_edges[1]


def matrix_power(a, n):
    """
    Raises a square matrix to the (integer) power `n`.

    For positive integers `n`, the power is computed by repeated matrix squarings and
    matrix multiplications.
    If :math:`n == 0`, the identity matrix of the same shape as `M` is returned.

    Note:
        Stacks of object matrices are not currently supported and
        :math:`n < 0` is not supported.

    Args:
        a (Union[int, float, bool, list, tuple, Tensor]): Input matrix.
        n (int): The exponent can be any integer or long integer, positive or zero.

    Returns:
        Tensor.

    Raises:
        TypeError: If the input can not be converted to a tensor or
            the exponent is not integer.
        ValueError: If the input includes less than 2 dimensions or
            the last 2 dimensions are not square.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import numpy as np
        >>> a = np.arange(16).reshape(4, 4).astype('float32')
        >>> print(np.matrix_power(a, 2))
        [[ 56.  62.  68.  74.]
         [152. 174. 196. 218.]
         [248. 286. 324. 362.]
         [344. 398. 452. 506.]]
    """
    a = _to_tensor(a)
    if not isinstance(n, int):
        _raise_type_error("exponent must be an integer")
    if a.ndim < 2:
        _raise_value_error("Array must be at least two-dimensional")
    if a.shape[-2] != a.shape[-1]:
        _raise_value_error("Last 2 dimensions of the array must be square")

    if n < 0:
        _raise_value_error("n < 0 is not supported now.")
    if n == 0:
        return _broadcast_to_shape(eye(a.shape[-1], a.shape[-1], dtype=a.dtype), a.shape)
    if n == 1:
        return a
    res = a
    while n > 1:
        res = C.matmul(res, a)
        n = n - 1
    return res


def around(a, decimals=0):
    """
    Evenly round to the given number of decimals.

    Note:
        Numpy argument `out` is not supported.
        Complex numbers are not supported.

    Args:
        a (Union[int, float, list, tuple, Tensor]): Input data.
        decimals (int): Number of decimal places to round to. Default: 0.

    Returns:
        Tensor. A tensor of the same type as a, containing the rounded values.
        The result of rounding a float is a float.

    Raises:
        TypeError: If the input can not be converted to a tensor or
            the `decimals` argument is not integer.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.array([-1.3, 0.0, 0.5, 1.5, 2.5])
        >>> print(np.around(a))
        [-1. 0. 0. 2. 2.]
    """
    a = _to_tensor_origin_dtype(a)
    if not isinstance(decimals, int):
        _raise_type_error("decimals must be an integer")
    if decimals < 0:
        _raise_value_error("decimals < 0 is not supported now.")
    if decimals == 0:
        return _round(a)
    return F.tensor_div(_round(a * 10**decimals), 10**decimals)


def _to_poly1d(x):
    x = atleast_1d(_to_tensor(x))
    if F.rank(x) > 1:
        _raise_value_error('input array must be scalar or 1-d sequence')
    return x


def polyadd(a1, a2):
    """
    Finds the sum of two polynomials.
    Returns the polynomial resulting from the sum of two input polynomials.

    Note:
        Numpy object poly1d is currently not supported.

    Args:
        a1 (Union[int, float, list, tuple, Tensor): Input polynomial.
        a2 (Union[int, float, list, tuple, Tensor): Input polynomial.

    Returns:
        Tensor, the sum of the inputs.

    Raises:
        ValueError: If the input array has more than 1 dimensions.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.polyadd([1, 2], [9, 5, 4]))
        [9 6 6]
    """
    a1 = _to_poly1d(a1)
    a2 = _to_poly1d(a2)
    diff_size = a1.size - a2.size
    if diff_size == 0:
        return add(a1, a2)
    if diff_size > 0:
        return concatenate((a1[:diff_size], add(a1[diff_size:], a2)))
    return concatenate((a2[:-diff_size], add(a1, a2[-diff_size:])))


def polysub(a1, a2):
    """
    Difference (subtraction) of two polynomials.
    Given two polynomials `a1` and `a2`, returns ``a1 - a2``.

    Note:
        Numpy object poly1d is currently not supported.

    Args:
        a1 (Union[int, float, list, tuple, Tensor): Minuend polynomial.
        a2 (Union[int, float, list, tuple, Tensor): Subtrahend polynomial.

    Returns:
        Tensor, the difference of the inputs.

    Raises:
        ValueError: If the input array has more than 1 dimensions.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.polysub([2, 10, -2], [3, 10, -4]))
        [-1  0  2]
    """
    return polyadd(a1, F.neg_tensor(_to_tensor(a2)))


def polyval(p, x):
    """
    Evaluates a polynomial at specific values.
    If `p` is of length `N`, this function returns the value:
    ``p[0]*x**(N-1) + p[1]*x**(N-2) + ... + p[N-2]*x + p[N-1]``
    If `x` is a sequence, then ``p(x)`` is returned for each element of `x`. If `x`
    is another polynomial then the composite polynomial ``p(x(t))`` is returned.

    Note:
        Numpy object poly1d is currently not supported.

    Args:
        p (Union[int, float, bool, list, tuple, Tensor): 1D array of polynomial
            coefficients (including coefficients equal to zero) from highest
            degree to the constant term.
        x (Union[int, float, bool, list, tuple, Tensor): A number, an array of
            numbers, at which to evaluate `p`.

    Returns:
        Tensor.
    Raises:
        ValueError: If `p` has more than 1 dimensions.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.polyval([3.,0.,1.], 5.))
        76.0
    """
    p = _to_poly1d(p)
    x = _to_tensor(x)
    shape = F.shape(x)
    exp_p = arange(_type_convert(int, p.size) - 1, -1, -1).astype(mstype.float32)
    var_p = (x.reshape(shape + (1,)))**exp_p
    return F.reduce_sum(p*var_p, -1)


def polyder(p, m=1):
    """
    Returns the derivative of the specified order of a polynomial.

    Note:
        Numpy object poly1d is currently not supported.

    Args:
        p (Union[int, float, bool, list, tuple, Tensor): Polynomial to differentiate.
            A sequence is interpreted as polynomial coefficients.
        m (int, optional): Defaults to 1, order of differentiation.

    Returns:
        Tensor, a new polynomial representing the derivative.

    Raises:
        ValueError: If `p` has more than 1 dimensions.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.polyder([1, 1, 1, 1]))
        [3 2 1]
    """
    p = _to_poly1d(p)
    if m < 0:
        _raise_value_error('Order of derivative must be positive')
    if m >= p.size:
        return _to_tensor([])
    for _ in range(m):
        coeff = _to_tensor(F.make_range(_type_convert(int, p.size) - 1, 0, -1))
        p = p[:-1]*coeff
    return p


def polymul(a1, a2):
    """
    Finds the product of two polynomials.

    Note:
        Numpy object poly1d is currently not supported.

    Args:
        a1 (Union[int, float, bool, list, tuple, Tensor): Input polynomial.
        a2 (Union[int, float, bool, list, tuple, Tensor): Input polynomial.

    Returns:
        Tensor, a new polynomial representing the derivative.

    Raises:
        ValueError: If the input array has more than 1 dimensions.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.polymul([3, 1, 2], [2, 5]))
        [ 6 17  9 10]
    """
    a1 = _to_poly1d(a1)
    a2 = _to_poly1d(a2)
    return convolve(a1, a2)


def polyint(p, m=1, k=None):
    """
    Returns an antiderivative (indefinite integral) of a polynomial.

    Note:
        Numpy object poly1d is currently not supported.

    Args:
        p (Union[int, float, bool, list, tuple, Tensor): Polynomial to integrate. A
            sequence is interpreted as polynomial coefficients.
        m (int, optional): Defaults to 1, Order of the antiderivative.
        k (Union[int, list of int]y, optinoal): Integration constants. They are given
            in the order of integration: those corresponding to highest-order terms
            come first. If None (default), all constants are assumed to be zero. If
            ``m = 1``, a single scalar can be given instead of a list.

    Returns:
        Tensor, a new polynomial representing the antiderivative.

    Raises:
        ValueError: If `p` has more than 1 dimensions.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.polyint([1, 1, 1]))
        [0.33333334 0.5        1.         0.        ]
    """
    p = _to_poly1d(p)
    if m < 0:
        _raise_value_error('Order of derivative must be positive')
    if m == 0:
        return p
    if k is None:
        k = zeros(m, F.dtype(p))
    k = atleast_1d(_to_tensor(k))
    if k.size == 1:
        k = F.tile(k, (m,))
    k = F.expand_dims(k, -1)
    for i in range(m):
        coeff = _to_tensor(F.make_range(_type_convert(int, p.size), 0, -1))
        p = concatenate((true_divide(p, coeff), k[i]))
    return p


@constexpr
def _get_dtype(x):
    """Returns the dtype of x."""
    if isinstance(x, bool):
        return mstype.bool_
    if isinstance(x, int):
        return mstype.int32
    if isinstance(x, float):
        return mstype.float32
    if isinstance(x, typing.Number):
        return x
    if isinstance(x, str):
        t = dtype_map.get(x, None)
        if t is None:
            t = dtype_map.get(str(nptype(x)))
        return t
    raise TypeError('data type not understood')


def result_type(*arrays_and_dtypes):
    """
    Returns the type that results from applying the type promotion rules to the arguments.

    Note:
        The promotion rule is slightly different from original Numpy, but more like
        jax, due to the preference on ``32-bit`` over ``64-bit`` data types.
        Complex dtypes are not supported.

    Args:
        *arrays_and_dtypes (Union[int, float, bool, list, tuple, Tensor, :class:`mindspore.dtype`, str]):
            The operands of some operation whose result type is needed.

    Returns:
        :class:`mindspore.dtype`, the result type.

    Raises:
        TypeError: If the input is not a valid data type.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.result_type('i2', np.float32, True))
        Float32
    """
    def get_dtype(x):
        if isinstance(x, Tensor):
            return F.dtype(_to_tensor(x))
        return _get_dtype(x)

    dtype_out = get_dtype(arrays_and_dtypes[0])
    for i in arrays_and_dtypes[1:]:
        dtype_out = _promote(dtype_out, get_dtype(i))
    return dtype_out


def unwrap(p, discont=3.141592653589793, axis=-1):
    """
    Unwraps by changing deltas between values to ``2*pi`` complement.
    Unwraps radian phase `p` by changing absolute jumps greater than `discont` to their
    ``2*pi`` complement along the given axis.

    Note:
        For absolute jumps that are within a very close range to pi, unwrapping may be done
        differently than numpy due to differences in round-off.

    Args:
        p (Union[int, float, bool, list, tuple, Tensor): Input array.
        discont (float, optional): Maximum discontinuity between values, default is pi.
        axis (int, optional): Axis along which unwrap will operate, default is -1.

    Returns:
        Tensor.

    Raises:
        ValueError: If the axis is out of range.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> phase = np.add(np.linspace(0, np.pi, num=5), [0, 0, 0, np.pi, np.pi])
        >>> print(phase)
        [0.        0.7853982 1.5707964 5.4977875 6.2831855]
        >>> print(np.unwrap(phase))
        [ 0.0000000e+00  7.8539819e-01  1.5707964e+00 -7.8539848e-01 -4.7683716e-07]
    """
    if not isinstance(discont, (int, float)):
        _raise_type_error('discont should be a float')
    p = _to_tensor(p)
    ndim = F.rank(p)
    axis = _check_axis_in_range(axis, ndim)
    dd = diff(p, axis=axis)
    ddmod = remainder(add(dd, pi), 2*pi) - pi
    ddmod = where_(F.logical_and(ddmod == -pi, dd > 0), pi, ddmod)
    ph_correct = ddmod - dd
    ph_correct = where_(absolute(dd) < discont, 0, ph_correct)
    slice_all = _list_comprehensions(F.rank(p), F.make_slice(None, None, None), True)
    slice0 = _tuple_setitem(slice_all, axis, F.make_slice(0, 1, None))
    slice1 = _tuple_setitem(slice_all, axis, F.make_slice(1, None, None))
    head = p[slice0]
    tail = add(p[slice1], cumsum(ph_correct, axis))
    return concatenate((head, tail), axis=axis)


def cumprod(a, axis=None, dtype=None):
    """
    Returns the cumulative product of elements along a given axis.

    Note:
        Numpy argument `out` is not supported.

    Args:
        a (Union[int, float, bool, list, tuple, Tensor]): Input tensor.
        axis (int, optional): Axis along which the cumulative product is computed.
            By default the input is flattened. Default: `None`.
        dtype (:class:`mindspore.dtype`, optional): Default: `None`. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor.

    Raises:
        TypeError: If the input can not be converted to tensor or `axis` is not integer.
        ValueError: If axis is out of range.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.array([1, 2, 3])
        >>> print(np.cumprod(x))
        [1 2 6]
    """
    a = _to_tensor_origin_dtype(a)
    original_dtype = F.dtype(a)

    if axis is not None and not isinstance(axis, int):
        _raise_type_error("integer axis is expected, but got", axis)
    if axis is None:
        a = a.ravel()
        axis = 0
    _check_axis_in_range(axis, a.ndim)

    a = a.astype('float32') if original_dtype != mstype.float64 else a
    if dtype is None:
        if original_dtype in [mstype.int8, mstype.int16, mstype.bool_]:
            dtype = mstype.int32
        elif original_dtype in [mstype.uint8, mstype.uint16]:
            dtype = mstype.uint32
        else:
            dtype = original_dtype
    return _cumprod_default(a, axis).astype(dtype, copy=False)


def _process_index(index, dims, mode='raise'):
    """Generates index (Tensor) according to different modes."""
    if mode == "raise":
        _raise_unimplemented_error("'raise' mode is not implemented")
    if mode not in ['clip', 'wrap']:
        _raise_value_error("invalid mode. Expected 'wrap' or 'clip'")
    ori_shape = index.shape
    tup = ()
    for i, idx in enumerate(index):
        d = dims[i]
        if mode == "clip":
            idx = clip(idx, 0, d - 1)
        elif mode == "wrap":
            idx = remainder(idx, d)
        idx = F.expand_dims(idx, 0) if idx.ndim < 1 else idx
        tup += (idx,)
    return P.Concat(0)(tup).reshape(ori_shape)


def _get_strides(dims, order='C'):
    """Generates strides (1-D tensor) according to `dims` (1-D tensor)."""
    if order not in ['C', 'F']:
        _raise_value_error("invalid order. Expected 'C' or 'F'")
    tup = (_to_tensor([1]),)
    dims = dims[1:][::-1] if order == 'C' else dims[:-1]
    for d in dims:
        tensor = tup[-1] * d
        if tensor.ndim < 1:
            tensor = F.expand_dims(tensor, 0)
        tup += (tensor,)
    tup = tup[::-1] if order == 'C' else tup
    return P.Concat(0)(tup)


def ravel_multi_index(multi_index, dims, mode='clip', order='C'):
    """
    Converts a tuple of index arrays into an array of flat indices,
    applying boundary modes to the multi-index.

    Note:
        `raise` mode is not supported. Default mode is `clip`.

    Args:
        multi_index (tuple of array_like):
            A tuple of integer arrays, one array for each dimension.
        dims (Union[int, tuple of integers]): The shape of array into which the indices from multi_index apply.
        mode ({`wrap`, `clip`}): Specifies how out-of-bounds indices are handled. Default: `clip`.

            - `wrap`: wrap around
            - `clip`: clip to the range

            In `clip` mode, a negative index which would normally wrap will clip to 0 instead.
        order ({`C`, `F`}): Determines whether the multi-index should be viewed as indexing in
            row-major (C-style) or column-major (Fortran-style) order.

    Returns:
        Raveled_indices array. An array of indices into the flattened version of an array of dimensions dims.

    Raises:
        TypeError: If `multi_index` or `dims` can not be converted to tensor or
            `dims` is not a sequence of integer values.
        ValueError: If the length of `multi_index` and that of `dims` are not equal.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> arr = np.array([[3, 6, 6], [4, 5, 1]])
        >>> output = np.ravel_multi_index(arr, (7, 6))
        >>> print(output)
        [22. 41. 37.]
        >>> output = np.ravel_multi_index((3, 1, 4, 1), (6, 7, 8, 9))
        >>> print(output)
        1621.0
    """
    if isinstance(dims, int):
        dims = (dims,)
    dims = _to_tensor(dims)
    if dims.ndim > 1 or dims.dtype in (mstype.float16, mstype.float32, mstype.float64, mstype.bool_):
        _raise_type_error("only 1-D integer arrays are accepted.")
    multi_index = _to_tensor(multi_index)
    if len(multi_index) != len(dims):
        _raise_value_error("parameter multi_index must be a sequence of length ", len(dims))
    if multi_index.dtype in (mstype.float16, mstype.float32, mstype.float64):
        _raise_type_error("only int indices permitted")

    multi_index = _process_index(multi_index, dims, mode)
    strides = _get_strides(dims, order)
    s_shape = strides.shape + _list_comprehensions(multi_index.ndim - 1, 1, True)
    strides = _broadcast_to_shape(strides.reshape(s_shape), multi_index.shape)
    return sum_((multi_index * strides).astype('float32'), axis=0)


def _vector_norm(x, _ord, axis, keepdims):
    """Returns norm of a vector."""
    if _in(_ord, ('fro', 'nuc')):
        _raise_value_error('Frobenius norm and nuclear norm are only defined for vectors')
    if _ord is None:
        _ord = 2
    if _ord == inf:
        res = P.ReduceMax(keepdims)(absolute(x), axis)
    elif _ord == -inf:
        res = P.ReduceMin(keepdims)(absolute(x), axis)
    elif _ord == 0:
        res = P.ReduceSum(keepdims)(F.not_equal(x, 0).astype(mstype.float32), axis)
    else:
        res = power(P.ReduceSum(keepdims)(power(absolute(x), _ord), axis), 1./_ord)
    return res


def _matrix_norm(x, _ord, axis, keepdims):
    """Returns norm of a matrix."""
    if _ord == 0:
        _raise_value_error('for 0 axis, norm is defined only for 2-D matrices')
    if _ord == 'nuc':
        _raise_unimplemented_error('nuclear norm is not implemented')
    if _in(_ord, (2, -2)):
        _raise_unimplemented_error('2-norm is not implemented for matrices')
    if _in(_ord, (None, 'fro')):
        return F.sqrt(P.ReduceSum(keepdims)(F.square(x), axis))
    axis0, axis1 = axis
    if not keepdims:
        if _check_is_inf(_abs(_ord)) and axis0 > axis1:
            axis0 -= 1
        elif _abs(_ord) == 1 and axis1 > axis0:
            axis1 -= 1
    if _check_is_inf(_ord):
        return P.ReduceMax(keepdims)(P.ReduceSum(keepdims)(absolute(x), axis1), axis0)
    if _check_is_inf(_ord, True):
        return P.ReduceMin(keepdims)(P.ReduceSum(keepdims)(absolute(x), axis1), axis0)
    if _ord == 1:
        return P.ReduceMax(keepdims)(P.ReduceSum(keepdims)(absolute(x), axis0), axis1)
    if _ord == -1:
        return P.ReduceMin(keepdims)(P.ReduceSum(keepdims)(absolute(x), axis0), axis1)
    return _raise_value_error('invalid norm order for matrices')


def norm(x, ord=None, axis=None, keepdims=False): # pylint: disable=redefined-builtin
    """
    Matrix or vector norm.
    This function is able to return one of eight different matrix norms, or one of an
    infinite number of vector norms (described below), depending on the value of the
    ord parameter.

    Note:
        Nuclear norm and 2-norm are not supported for matrices.

    Args:
        x (Union[int, float, bool, list, tuple, Tensor]): Input array. If `axis` is None,
            `x` must be 1-D or 2-D, unless `ord` is None. If both `axis` and `ord` are None,
            the 2-norm of ``x.ravel`` will be returned.
        ord (Union[None, 'fro', 'nuc', inf, -inf, int, float], optional): Order of the norm.
            inf means numpy’s inf object. The default is None.
        axis (Union[None, int, 2-tuple of integers], optional): If `axis` is an integer, it
            specifies the axis of `x` along which to compute the vector norms. If `axis` is
            a 2-tuple, it specifies the axes that hold 2-D matrices, and the matrix norms of
            these matrices are computed. If `axis` is None then either a vector norm (when x
            is 1-D) or a matrix norm (when `x` is 2-D) is returned. The default is None.
        keepdims (boolean, optional): If this is set to True, the axes which are normed over
            are left in the result as dimensions with size one. With this option the result
            will broadcast correctly against the original `x`.

    Returns:
        Tensor, norm of the matrix or vector(s).

    Raises:
        ValueError: If the norm order is not defined.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.norm(np.arange(9).astype(np.float32)))
        14.282857
    """
    if not isinstance(ord, (int, float)) and not _in(ord, (None, 'fro', 'nuc', inf, -inf)):
        _raise_value_error('invalid value for `ord`')
    x = _to_tensor(x)
    ndim = F.rank(x)
    if axis is None:
        if ord is None:
            x = x.ravel()
        if F.rank(x) not in (1, 2):
            _raise_value_error('for None axis, array must a vector or a 2-D matrix')
        axis = F.make_range(F.rank(x))
    axis = _check_axis_valid(axis, F.rank(x))

    if len(axis) == 1:
        res = _vector_norm(x, ord, axis, keepdims)
    elif len(axis) == 2:
        res = _matrix_norm(x, ord, axis, keepdims)
    else:
        return _raise_value_error('invalid number of dimensions to norm')

    if keepdims and ndim > F.rank(res):
        res = _expand(res, ndim)
    return res


def bitwise_and(x1, x2, dtype=None):
    """
    Computes the bit-wise AND of two arrays element-wise.
    Computes the bit-wise AND of the underlying binary representation of the integers in
    the input arrays. This ufunc implements the C/Python operator &.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        x1 (Tensor): Input array.
        x2 (Tensor): Input array. Only integer and boolean types are handled. If
            ``x1.shape != x2.shape``, they must be broadcastable to a common shape (which becomes
            the shape of the output).
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, this is a scalar if both x1 and x2 are scalars.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.bitwise_and(13, 17))
        1
    """
    return _apply_tensor_op(F.bitwise_and, x1, x2, dtype=dtype)


def bitwise_or(x1, x2, dtype=None):
    r"""
    Computes the bit-wise OR of two arrays element-wise.
    Computes the bit-wise OR of the underlying binary representation of the integers in
    the input arrays. This ufunc implements the C/Python operator \|.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        x1 (Tensor): Input array.
        x2 (Tensor): Input array. Only integer and boolean types are handled. If
            ``x1.shape != x2.shape``, they must be broadcastable to a common shape (which becomes
            the shape of the output).
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, this is a scalar if both x1 and x2 are scalars.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.bitwise_or(13, 16))
        29
    """
    return _apply_tensor_op(F.bitwise_or, x1, x2, dtype=dtype)


def bitwise_xor(x1, x2, dtype=None):
    """
    Computes the bit-wise XOR of two arrays element-wise.
    Computes the bit-wise XOR of the underlying binary representation of the integers in
    the input arrays. This ufunc implements the C/Python operator ^.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        x1 (Tensor): Input array.
        x2 (Tensor): Input array. Only integer and boolean types are handled. If
            ``x1.shape != x2.shape``, they must be broadcastable to a common shape (which becomes
            the shape of the output).
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, this is a scalar if both x1 and x2 are scalars.

    Supported Platforms:
        ``Ascend`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.bitwise_xor(13, 17))
        28
    """
    return _apply_tensor_op(F.bitwise_xor, x1, x2, dtype=dtype)


def invert(x, dtype=None):
    """
    Computes bit-wise inversion, or bit-wise NOT, element-wise.
    Computes the bit-wise NOT of the underlying binary representation of the integers in
    the input arrays. This ufunc implements the C/Python operator ~.
    For signed integer inputs, the two's complement is returned. In a two's-complement system
    negative numbers are represented by the two's complement of the absolute value. This is
    the most common method of representing signed integers on computers
    `[1] <https://en.wikipedia.org/wiki/Two's_complement>`_. A N-bit two's-complement system
    can represent every integer in the range ``-2^{N-1}`` to ``+2^{N-1}-1``.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.
        Supported dtypes on Ascend: np.int16, np.uint16.

    Args:
        x (Tensor): Only integer and boolean types are handled.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.invert(np.array(13, dtype=np.uint16)))
        65522
    """
    return _apply_tensor_op(F.invert, x, dtype=dtype)


def rint(x, dtype=None):
    """
    Rounds elements of the array to the nearest integer.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.
        Ascend does not support dtype `float64` currently.

    Args:
        x (Union[float, list, tuple, Tensor]): Input tensor.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Output tensor is same shape and type as x. This is a scalar if x is a scalar.

    Raises:
        TypeError: If `x` can not be converted to tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.array([-1.7, -1.5, 0.2, 1.5, 1.7, 2.0])
        >>> print(np.rint(x))
        [-2. -2. 0. 2. 2. 2.]
    """
    x = _to_tensor_origin_dtype(x)
    res = _rint(x)
    if dtype is not None and not _check_same_type(F.dtype(res), dtype):
        res = F.cast(res, dtype)
    return res


def correlate(a, v, mode='valid'):
    """
    Cross-correlation of two 1-dimensional sequences.

    This function computes the correlation as generally defined in signal processing texts:

    :math:`c_{av}[k] = sum_n a[n+k] * conj(v[n])`

    with `a` and `v` sequences being zero-padded where necessary and conj being the conjugate.

    Note:
        Currently, complex numbers are not supported.

    Args:
        a (Union[list, tuple, Tensor]): First input sequence.
        v (Union[list, tuple, Tensor]): Second input sequence.
        mode (str, optional): By default, mode is `\'valid\'`.
            If `mode` is `\'valid\'`, it returns output of length :math:`max(M, N) - min(M, N) + 1`.
            The convolution product is only given for points where the signals overlap
            completely. Values outside the signal boundary have no effect.
            If `mode` is `\'full\'`, it returns the convolution at each point of overlap, with
            an output shape of :math:`(N + M - 1,)`.
            At the end-points of the convolution, the signals do not overlap completely,
            and boundary effects may be seen.
            If `mode` is `\'same\'`, it returns output of length :math:`max(M, N)`. Boundary
            effects are still visible.

    Returns:
        Tensor. Discrete cross-correlation of `a` and `v`.

    Raises:
        TypeError: If the inputs can not be converted to tensor.
        ValueError: If `a` and `v` are empty or have wrong dimensions

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> output = np.correlate([1, 2, 3], [0, 1, 0.5])
        >>> print(output)
        [3.5]
        >>> output = np.correlate([1, 2, 3], [0, 1, 0.5], mode="same")
        >>> print(output)
        [2.  3.5 3. ]
        >>> output = np.correlate([1, 2, 3, 4, 5], [1, 2], mode="same")
        >>> print(output)
        [ 2.  5.  8. 11. 14.]
    """
    a, v = _to_tensor(a, v)
    if a.ndim != 1 or v.ndim != 1:
        _raise_value_error("only support 1-dimensional inputs.")
    if a.size == 0 or v.size == 0:
        _raise_value_error("Inputs cannot be empty.")

    promote_dtype = _promote(a.dtype, v.dtype)
    # P.Conv2D requires that the two tensors have the same data type.
    # If the promote data type is not supported, it will be converted to float32.
    # The supported dtype list may vary in the future.
    if promote_dtype not in [mstype.float32, mstype.float16]:
        promote_dtype = mstype.float32
    a = a.astype(promote_dtype)
    v = v.astype(promote_dtype)
    if a.size < v.size:
        a, v = v, a
        return _compute_1d_conv(a, v, mode)[::-1]
    return _compute_1d_conv(a, v, mode)


def _compute_1d_conv(a, v, mode):
    """Returns a 1-D sequence which is the cross-correlate of two 1-D sequences (`a` and `v`)."""
    v_size = F.shape_mul(v.shape)
    if mode not in ('same', 'full', 'valid'):
        _raise_value_error("mode must be one of ['full', 'same', 'valid']")
    if v_size > 1:
        if mode == 'same':
            pad_left = _to_tensor(_list_comprehensions(v_size // 2, 0.0, True))
            pad_right = _to_tensor(_list_comprehensions(v_size - v_size // 2 - 1, 0.0, True))
            a = P.Concat(0)((pad_left, a, pad_right))
        elif mode == 'full':
            pad = _to_tensor(_list_comprehensions(v_size - 1, 0.0, True))
            a = P.Concat(0)((pad, a, pad))
    a = a.reshape(1, 1, 1, a.size)
    v = v.reshape(1, 1, 1, v.size)
    _conv = P.Conv2D(1, (1, v.size))
    return _conv(a, v).reshape(-1)


def radians(x, dtype=None):
    """
    Converts angles from degrees to radians.

    Args:
        x (Tensor): Angles in degrees.
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor, the corresponding radian values. This is a tensor scalar if `x`
        is a tensor scalar.

    Raises:
        TypeError: If `x` is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.asarray([1, 2, 3, -4, -5])
        >>> output = np.radians(x)
        >>> print(output)
        [ 0.01745329  0.03490658  0.05235988 -0.06981317 -0.08726647]
    """
    return deg2rad(x, dtype=dtype)
