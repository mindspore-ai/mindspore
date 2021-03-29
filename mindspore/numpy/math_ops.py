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

from .array_creations import asarray_const, ones, zeros, empty, full, full_like
from .array_ops import where as where_
from .array_ops import ravel, expand_dims, moveaxis, concatenate

from .utils_const import _infer_out_shape, _check_axis_valid, _get_device, \
    _check_shape_aligned, _raise_type_error, _check_same_type, _check_is_float, \
    _raise_value_error, _promote, _check_axis_type, _canonicalize_axis, \
    _is_shape_empty, _check_is_int, _expanded_shape, _check_axis_in_range, \
    _check_dtype, _list_comprehensions, _tuple_setitem, _add_unit_axes, _seq_prod, \
    _make_tensor, _promote_for_trigonometric, _raise_runtime_error, _max
from .utils import _expand, _broadcast_to, _broadcast_to_shape, _get_size, \
    _check_input_tensor, _to_tensor, _isnan


ZERO_TENSOR = asarray_const(0)


_mean_keepdims = P.ReduceMean(True)
_matmul = P.MatMul(False, False)
_matmul_T = P.MatMul(False, True)
_reduce_sum_default = P.ReduceSum()
_reduce_sum_keepdims = P.ReduceSum(True)
_reduce_min_default = P.ReduceMin()
_reduce_min_keepdims = P.ReduceMin(True)
_reduce_max_default = P.ReduceMax()
_reduce_max_keepdims = P.ReduceMax(True)
_cumsum_default = P.CumSum()
_concat = P.Concat(-1)

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
        return _apply_tensor_op(F.absolute, x, dtype=dtype).astype(original_dtype)
    return _apply_tensor_op(F.absolute, x, dtype=dtype)


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
        x = maximum(x, xmin, dtype=dtype)
    if xmax is not None:
        x = minimum(x, xmax, dtype=dtype)
    return x


def deg2rad(x, dtype=None):
    """
    Converts angles from degrees to radians.

    Args:
        x (Tensor): Angles in degrees.
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
    return _apply_tensor_op(convert, x, dtype=dtype)


def rad2deg(x, dtype=None):
    """
    Converts angles from radians to degrees.

    Args:
        x (Tensor): Angles in radians.
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
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
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
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
        _check_input_tensor(x1, x2)
        return subtract(x1, F.neg_tensor(x2), dtype=dtype)
    return _apply_tensor_op(F.tensor_add, x1, x2, dtype=dtype)


def subtract(x1, x2, dtype=None):
    """
    Subtracts arguments, element-wise.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        x1 (Tensor): the input to be subtracted from.
        x2 (Tensor): the input to be subtracted by.
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
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
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
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

    Instead of the Python traditional ‘floor division’, this returns a true
    division.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        x1 (Tensor): the divident.
        x2 (Tensor): the divisor.
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
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
    if not _check_is_float(F.dtype(x1)) and not _check_is_float(F.dtype(x2)):
        x1 = F.cast(x1, mstype.float32)
        x2 = F.cast(x2, mstype.float32)
    return _apply_tensor_op(F.tensor_div, x1, x2, dtype=dtype)


def true_divide(x1, x2, dtype=None):
    """
    Returns a true division of the inputs, element-wise.

    Instead of the Python traditional ‘floor division’, this returns a true
    division.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        x1 (Tensor): the divident.
        x2 (Tensor): the divisor.
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
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
        x1 (Tensor): the bases.
        x2 (Tensor): the exponents.
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
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
        x2 (Tensor): the exponenets.
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
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
        [[[105. 105. 105. 105.]
        [105. 105. 105. 105.]]]
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
    objects, `(a_axes, b_axes)`, sum the products of `a`’s and `b`’s elements (components)
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
    if not isinstance(keepdims, int):
        _raise_type_error("integer argument expected, but got ", keepdims)
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


def ptp(x, axis=None, keepdims=False):
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
    if not isinstance(keepdims, bool):
        _raise_type_error('keepdims should be boolean')
    if axis is None:
        axis = ()
    else:
        _check_axis_type(axis, True, True, False)
        axis = _check_axis_valid(axis, x.ndim)

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
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
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
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
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
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
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
        x (Tensor): Input array. For integer arguments with absolute value larger
            than 1 the result is always zero because of the way Python handles
            integer division. For integer zero the result is an overflow.
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
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
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
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
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
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
    return _reduce(a, P.ReduceMax(keepdims), cmp_fn=F.maximum, axis=axis, keepdims=keepdims,
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
            this is a tuple of ints, the minimum is selected over multiple axes,
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
    return _reduce(a, P.ReduceMin(keepdims), cmp_fn=F.minimum, axis=axis, keepdims=keepdims,
                   initial=initial, where=where)


def hypot(x1, x2, dtype=None):
    """
    Given the “legs” of a right triangle, returns its hypotenuse.

    Equivalent to ``sqrt(x1**2 + x2**2)``, element-wise. If `x1` or `x2` is scalar_like
    (i.e., unambiguously cast-able to a scalar type), it is broadcast for use
    with each element of the other argument. (See Examples)

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.
        On GPU, the supported dtypes are np.float16 and np.float32.
        On CPU, the supported dtypes are np.float16, np.float32, and np.float64.

    Args:
        x1 (Tensor): Leg of the traingle(s).
        x2 (Tensor): Leg of the triangle(s). If ``x1.shape != x2.shape``, they
            must be broadcastable to a common shape (which becomes the shape of
            the output).
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
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
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
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
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
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
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
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
        TypeError: if the input is not a tensor.

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
    # TODO change to F.ceil once supported on CPU.
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
        x1 (Tensor)
        x2 (Tensor): input arrays.
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
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
    return _apply_tensor_op(lambda x1, x2: _remainder(x1, x2, C_style=True), x1, x2, dtype=dtype)


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
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
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
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
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
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
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
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Element-wise quotient and remainder from floor division, in format of (quotient, remainder)

    Raises:
        TypeError: if `x1` and `x2` are not Tensor or scalar.

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


def diff(a, n=1, axis=-1, prepend=None, append=None):
    """
    Calculates the n-th discrete difference along the given axis.

    The first difference is given by :math:`out[i] = a[i+1] - a[i]` along the given axis,
    higher differences are calculated by using `diff` iteratively.

    Args:
        a (Tensor): Input tensor.
        n (int, optional): The number of times values are differenced. If zero,
            the input is returned as-is.
        axis (int, optional): The axis along which the difference is taken, default
            is the last axis.
        prepend/append (Tensor, optional): Values to prepend or append to a along
            `axis` prior to performing the difference. Scalar values are expanded to
            arrays with length 1 in the direction of `axis` and the shape of the input
            array in along all other axes. Otherwise the dimension and shape must
            match `a` except along axis.

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
        if isinstance(prepend, (int, float, bool)):
            prepend = asarray_const(prepend)
            prepend_shape = a.shape
            prepend_shape = _tuple_setitem(prepend_shape, axis, 1)
            prepend = _broadcast_to_shape(prepend, prepend_shape)
        elif not isinstance(prepend, Tensor):
            _raise_type_error("prepend must be scalar or Tensor, but got ", prepend)
        combined += (prepend,)

    combined += (a,)

    if append is not None:
        if isinstance(append, (int, float, bool)):
            append = asarray_const(append)
            append_shape = a.shape
            append_shape = _tuple_setitem(append_shape, axis, 1)
            append = _broadcast_to_shape(append, append_shape)
        elif not isinstance(append, Tensor):
            _raise_type_error("append must be scalar or Tensor, but got ", append)
        combined += (append,)

    if combined:
        a = concatenate(combined, axis)

    # if n > maximum length allowed, returns empty tensor, with shape matched with
    # the original tensor
    if n > a.shape[axis]:
        empty_shape = a.shape
        empty_shape = _tuple_setitem(empty_shape, axis, 0)
        return empty(empty_shape, a.dtype)

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
            default is 1.
        axis (int, optional): The axis along which to integrate.

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
    if _get_device() == 'CPU' and not _check_is_float(dtype):
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
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
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
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
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
        q1 = F.tensor_div(x1, common_divisor)
        q2 = F.tensor_div(x2, common_divisor)
        res = F.tensor_mul(F.tensor_mul(q1, q2), common_divisor)
        dtype = F.dtype(res)
        if _get_device() == 'CPU' and not _check_is_float(dtype):
            # F.absolute only supports float
            res = F.cast(res, mstype.float32)
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
        TypeError: if the inputs have types not specified above.
        ValueError: if a and v are empty or have wrong dimensions

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> output = np.convolve([1., 2., 3., 4., 5.], [2., 3.], mode="valid")
        >>> print(output)
        [ 3.  6.  9. 12.]
    """
    if not isinstance(a, Tensor):
        a = asarray_const(a)
    if not isinstance(v, Tensor):
        v = asarray_const(v)
    if a.size == 0 or v.size == 0:
        _raise_value_error("Inputs cannot be empty.")
    a = _expand(a, 1)
    v = _expand(v, 1)
    final_dtype = _promote(a.dtype, v.dtype)
    a = a.astype("float32")
    v = v.astype("float32")
    if a.ndim != 1 or v.ndim != 1:
        _raise_value_error("a and v must be 1-D tensor.")
    if a.size < v.size:
        a, v = v, a
    v = v[::-1]
    if mode not in ('same', 'full', 'valid'):
        _raise_value_error("mode must be one of ['full', 'same', 'valid']")
    if v.size > 1:
        if mode == 'same':
            pad_left = _to_tensor(_list_comprehensions(v.size // 2, 0.0, True))
            pad_right = _to_tensor(_list_comprehensions(v.size - v.size // 2 - 1, 0.0, True))
            a = P.Concat(axis=0)((pad_left, a, pad_right))
        elif mode == 'full':
            pad = _to_tensor(_list_comprehensions(v.size - 1, 0.0, True))
            a = P.Concat(axis=0)((pad, a, pad))
    a = a.reshape(1, 1, 1, a.size)
    v = v.reshape(1, 1, 1, v.size)
    _conv = P.Conv2D(out_channel=1, kernel_size=(1, v.size), pad_mode="valid")
    return _conv(a, v).reshape(-1).astype(final_dtype)


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
    cov_input = _expand(cov_input, 2)
    if not rowvar and cov_input.shape[0] != 1:
        cov_input = cov_input.T
    return cov_input


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
            and observations. `y` has the same form as that of `m`.
        rowvar(bool, optional): If `rowvar` is ``True`` (default), then each row represents
            a variable, with observations in the columns. Otherwise, the relationship
            is transposed: each column represents a variable, while the rows contain
            observations.
        bias (bool, optional): Default normalization (``False``) is by :math:`(N - 1)`, where
            :math:`N` is the number of observations given (unbiased estimate). If bias is
            ``True``, then normalization is by `N`. These values can be overridden by
            using the keyword `ddof`.
        ddof (int, optional): If not ``None``, the default value implied by `bias` is
            overridden. Note that :math:`ddof=1` will return the unbiased estimate, even
            if both fweights and aweights are specified, and :math:`ddof=0` will return
            the simple average. See the notes for the details. The default value
            is ``None``.
        fweights (Union[Tensor, list, tuple], optional): 1-D tensor of integer
            frequency weights; the number of times each observation vector should
            be repeated.
        aweights (Union[Tensor, list, tuple], optional): 1-D tensor of observation
            vector weights. These relative weights are typically larger for observations
            considered more important and smaller for observations considered less
            important. If :math:`ddof=0` the tensor of weights can be used to assign probabilities
            to observation vectors.
        dtype (Union[:class:`mindspore.dtype`, str], optional): Data-type of the
            result. By default, the return data-type will have mstype.float32 precision.

    Returns:
        Tensor, the covariance matrix of the variables.

    Raises:
        TypeError: if the inputs have types not specified above.
        ValueError: if `m` and `y` have wrong dimensions.
        RuntimeError: if `aweights` and `fweights` have dimensions > 2.

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

    # Determine the normalization
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

    m = m - F.expand_dims(avg, -1)
    if w is None:
        m_T = m.T
    else:
        m_T = (m * w).T
    res = true_divide(dot(m, m_T), fact).squeeze()
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


def _reduce(a, reduce_fn, cmp_fn=None, axis=None, keepdims=False, initial=None, where=True, dtype=None):
    """
    Applies comparison based on cmp_fn and reduction based on reduce_fn.
    If cmp_fn is None, only reduction is performed.
    """
    _check_input_tensor(a)

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
                return _raise_value_error('initial value must be provided for zero-size arrays')
        return full(shape_out, initial, dtype)

    if initial is not None:
        initial = full(shape, initial, dtype)
        a = cmp_fn(a, initial)
    if not axes:
        return a.astype(dtype)
    if isinstance(where, Tensor):
        if initial is None:
            return _raise_value_error('initial value must be provided for where masks')
        ndim_orig = F.rank(a)
        a = where_(where, a, initial)
        axes = _real_axes(ndim_orig, F.rank(a), axes)

    return reduce_fn(a, axes).astype(dtype)


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
        a (Union[int, float, bool, list, tuple, Tensor]): Array containing numbers
            whose sum is desired. If `a` is not an array, a conversion is attempted.
        axis (Union[int, tuple of int, None], optional): Axis or axes along which the sum is
            computed. The default is to compute the sum of the flattened array.
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
            output Tensor.
        keepdims (boolean, optional): defaults to False. If this is set to True, the axes which
            are reduced are left in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the original `a`.

    Returns:
        Tensor.

    Raises:
        ValueError: if axes are out of the range of ``[-a.ndim, a.ndim)``, or
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
    if dtype is None and _get_device() == 'CPU' and not _check_is_float(F.dtype(a)):
        # F.reduce_sum only supports float on CPU
        dtype = F.dtype(a)
        a = F.cast(a, mstype.float32)
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
        a (Union[int, float, bool, list, tuple, Tensor]): Array containing numbers
            whose mean is desired. If `a` is not an array, a conversion is attempted.
        axis (Union[int, tuple of int, None], optional): Axis or axes along which the mean is
            computed. The default is to compute the mean of the flattened array.
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
            output Tensor.
        keepdims (boolean, optional): defaults to False. If this is set to True, the axes which
            are reduced are left in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the original `a`.

    Returns:
        Tensor.

    Raises:
        ValueError: if axes are out of the range of ``[-a.ndim, a.ndim)``, or
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
    return F.tensor_div(sum_a, F.tensor_sub(count, ddof))


def nanvar(a, axis=None, dtype=None, ddof=0, keepdims=False):
    """
    Computes the variance along the specified axis, while ignoring NaNs.

    Returns the variance of the array elements, a measure of the spread of a distribution. The
    variance is computed for the flattened array by default, otherwise over the specified axis.

    Note:
        Numpy arguments `out` is not supported.
        On GPU, the supported dtypes are np.float16, and np.float32.

    Args:
        a (Union[int, float, bool, list, tuple, Tensor]): Array containing numbers
            whose variance is desired. If `a` is not an array, a conversion is attempted.
        axis (Union[int, tuple of int, None], optional): Axis or axes along which the variance is
            computed. The default is to compute the variance of the flattened array.
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
            output Tensor.
        ddof (int, optional): “Delta Degrees of Freedom”: the divisor used in the calculation is
            ``N - ddof``, where `N` represents the number of non-NaN elements. By default `ddof`
            is zero.
        keepdims (boolean, optional): defaults to False. If this is set to True, the axes which
            are reduced are left in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the original `a`.

    Returns:
        Tensor.

    Raises:
        ValueError: if axes are out of the range of ``[-a.ndim, a.ndim)``, or
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
        a (Union[int, float, bool, list, tuple, Tensor]): Calculates the standard deviation of the non-NaN values.
        axis (Union[int, tuple of int, None], optional): Axis or axes along which the standard
            deviation is computed. The default is to compute the standard deviation of the
            flattened array.
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
            output Tensor.
        ddof (int, optional): “Delta Degrees of Freedom”: the divisor used in the calculation is
            ``N - ddof``, where `N` represents the number of non-NaN elements. By default `ddof`
            is zero.
        keepdims (boolean, optional): defaults to False. If this is set to True, the axes which
            are reduced are left in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the original `a`.

    Returns:
        Tensor.

    Raises:
        ValueError: if axes are out of the range of ``[-a.ndim, a.ndim)``, or
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
        dtype (:class:`mindspore.dtype`, optional): defaults to :class:`None`. Overrides the dtype of the
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

    Args:
        a (Union[int, float, bool, list, tuple, Tensor]): input values.
        b (Union[int, float, bool, list, tuple, Tensor]): input values.

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
        a (Union[int, float, bool, list, tuple, Tensor]): Components of the first vector(s).
        b (Union[int, float, bool, list, tuple, Tensor]): Components of the second vector(s).
        axisa (int, optional): Axis of `a` that defines the vector(s). By default, the last
            axis.
        axisb (int, optional): Axis of `b` that defines the vector(s). By default, the last
            axis.
        axisc (int, optional): Axis of `c` containing the cross product vector(s). Ignored
            if both input vectors have dimension 2, as the return is scalar. By default,
            the last axis.
        axis (int, optional): If defined, the axis of `a`, `b` and `c` that defines the
            vector(s) and cross product(s). Overrides `axisa`, `axisb` and `axisc`.

    Returns:
        Tensor, vector cross product(s).

    Raises:
        ValueError: when the dimensions of the vector(s) in `a` and/or `b` equal 2 or 3.

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
        cy = _get_slice_product(0, 2)               # az*bx
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
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
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
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
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
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
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
            is used.

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
    original_dtype = F.dtype(a)
    # If original tensor is int, and has precision less then int32, convert to int32
    if _check_same_type(original_dtype, mstype.bool_) or \
       _check_same_type(original_dtype, mstype.int8) or \
       _check_same_type(original_dtype, mstype.int16):
        original_dtype = mstype.int32
    a = a.astype(mstype.float32)
    if axis is None:
        a = a.ravel()
        axis = 0
    _check_axis_in_range(axis, a.ndim)
    if dtype is not None and not _check_same_type(original_dtype, dtype):
        return _cumsum_default(a, axis).astype(dtype, copy=False)
    return _cumsum_default(a, axis).astype(original_dtype, copy=False)


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
    return cumsum(a, axis=axis, dtype=dtype)


def cbrt(x, dtype=None):
    """
    Returns the cube-root of a tensor, element-wise.

    Note:
        Numpy arguments `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        x (Tensor): Input tensor.
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
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
        TypeError: If the input is not a tensor or is :class:`tensor.dtype` is :class:`mindsproe.float64`.

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
        Tensor.

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


def arccos(x, dtype=None):
    """
    Trigonometric inverse cosine, element-wise.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and `extobj` are
        not supported.

    Args:
        x (Tensor): Input tensor. x-coordinate on the unit circle.
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
        >>> x = np.asarray([1, -1], np.float32)
        >>> output = np.arccos(x)
        >>> print(output)
        [0.        3.1415927]
    """
    x = _cast_type_for_trigonometric(x)
    return _apply_tensor_op(F.acos, x, dtype=dtype)


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
        >>> print(np.tan(x))
        [ 0.          1.5574077  -2.1850398  -0.14254655  1.1578213 ]
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
        >>> x = np.arange(5).astype('float32')
        >>> print(np.arcsinh(x))
        [0.        0.8813736 1.4436355 1.8184465 2.0947125]
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
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor or scalar, the sum of `x1` and `x2`, element-wise. This is a scalar
        if both `x1` and `x2` are scalars.

    Supported Platforms:
        ``Ascend`` ``CPU``

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
        TypeError: if the input are not valid :class:`mindspore.dtype` input.

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
