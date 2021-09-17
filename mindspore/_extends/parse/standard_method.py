# This is the Python adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
#
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
"""standard_method"""

from dataclasses import dataclass

from mindspore import Tensor, Parameter
from mindspore import dtype as mstype

from ..._checkparam import Validator as validator
from ...ops import functional as F
from ...ops import operations as P
from ...ops.composite import tail, core, MultitypeFuncGraph, env_get, hyper_add, \
    zeros_like, ones_like, repeat_elements
from ...ops.composite.base import _append
from ...ops.composite.multitype_ops import _constexpr_utils as const_utils
from ...ops.composite.multitype_ops import _compile_utils as compile_utils
from ...ops.primitive import constexpr


__all__ = ['MultitypeFuncGraph', 'env_get', 'hyper_add', 'zeros_like', 'ones_like']

shape_ = P.Shape()
dtype_ = P.DType()
abs_ = P.Abs()
ndim_ = P.Rank()
cumsum_ = P.CumSum()
size_op_ = P.Size()
_reduce_sum_default = P.ReduceSum()
_reduce_sum_keepdims = P.ReduceSum(True)
_mean_keepdims = P.ReduceMean(True)

itemsize_map = {mstype.bool_: 1, mstype.int8: 1, mstype.uint8: 1,
                mstype.float16: 2, mstype.int16: 2, mstype.uint16: 2,
                mstype.float32: 4, mstype.int32: 4, mstype.uint32: 4,
                mstype.float64: 8, mstype.int64: 8, mstype.uint64: 8}

nan_tensor = Tensor(float('nan'), dtype=mstype.float32)


def mean(x, axis=(), keep_dims=False):
    """
    Reduces a dimension of a tensor by averaging all elements in the dimension.

    Args:
        axis (Union[None, int, tuple(int), list(int)]): Dimensions of reduction,
            when axis is None or empty tuple, reduce all dimensions. Default: ().
        keep_dims (bool): Whether to keep the reduced dimensions. Default: False.

    Returns:
        Tensor, has the same data type as input tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> input_x = Tensor(np.array([1, 2, 3], dtype=np.float32))
        >>> output = input_x.mean()
        >>> print(output)
        2.0
    """
    if axis is None:
        axis = ()
    reduce_mean = P.ReduceMean(keep_dims)
    return reduce_mean(x, axis)


def all_(x, axis=(), keep_dims=False):
    """
    Check all array elements along a given axis evaluate to True.

    Args:
        x (Tensor): A Tensor to be reduced.
        axis (Union[None, int, tuple(int)): Dimensions of reduction.
        keep_dims (bool): Whether to keep the reduced dimensions.

    Returns:
        Tensor, has the same data type as x.
    """

    if axis is None:
        axis = ()
    reduce_all = P.ReduceAll(keep_dims)
    return reduce_all(x, axis)


def any_(x, axis=(), keep_dims=False):
    """
    Check any array element along a given axis evaluate to True.

    Args:
        x (Tensor): A Tensor to be reduced.
        axis (Union[None, int, tuple(int)): Dimensions of reduction.
        keep_dims (bool): Whether to keep the reduced dimensions.

    Returns:
        Tensor, has the same data type as x.
    """
    if axis is None:
        axis = ()
    reduce_any = P.ReduceAny(keep_dims)
    return reduce_any(x, axis)


def size_(x):
    """
    Return the number of elements in tensor `x`.

    Note:
        To strictly follow Numpy's behaviour, return 1 for tensor scalar.

    Args:
        x (Tensor): Input tensor.

    Returns:
        size(int).
    """
    if not shape_(x):
        return size_op_(x) + 1
    return size_op_(x)


def itemsize_(x):
    """
    Return length of one tensor element in bytes.

    Args:
        x (Tensor): Input tensor.

    Returns:
        itemsize(int).
    """
    return get_itemsize(x.dtype)


def nbytes_(x):
    """
    Return total number of bytes taken by the tensor.

    Args:
        x (Tensor): Input tensor.

    Returns:
        nbytes(int).
    """
    return itemsize_(x) * F.shape_mul(shape_(x))


def strides_(x):
    """
    Return the tuple of bytes to step in each dimension when traversing a tensor.

    Args:
        x (Tensor): Input tensor.

    Returns:
        strides (tuple[int]).
    """
    strides = ()
    ndim = P.Rank()(x)
    tensor_shape = shape_(x)
    for i in F.make_range(0, ndim):
        stride = itemsize_(x)
        for j in F.make_range(i + 1, ndim):
            stride *= tensor_shape[j]
        strides += (stride,)
    return strides


def astype(x, dtype, copy=True): # pylint: disable=redefined-outer-name
    """
    Return a copy of the tensor, casted to a specified type.

    Args:
        dtype (Union[:class:`mindspore.dtype`, str]): Designated tensor dtype, can be in format
            of :class:`mindspore.dtype.float32` or `float32`.
            Default: :class:`mindspore.dtype.float32`.
        copy (bool, optional): By default, astype always returns a newly allocated
            tensor. If this is set to false, the input tensor is returned instead
            of a copy if possible. Default: True.

    Returns:
        Tensor, with the designated dtype.

    Raises:
        TypeError: If `dtype` has types not specified above, or values cannot be understood.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> x = Tensor(np.ones((1,2,2,1), dtype=np.float32))
        >>> x = x.astype("int32")
        >>> print(x.dtype)
        Int32
    """
    dtype = check_astype_dtype_const(dtype)
    if not copy and dtype == x.dtype:
        return x
    return F.cast(x, dtype)


def transpose(x, *axis):
    r"""
    Return a view of the tensor with axes transposed.

    For a 1-D tensor this has no effect, as a transposed vector is simply the
    same vector. For a 2-D tensor, this is a standard matrix transpose. For a
    n-D tensor, if axes are given, their order indicates how the axes are permuted.
    If axes are not provided and tensor.shape = (i[0], i[1],...i[n-2], i[n-1]),
    then tensor.transpose().shape = (i[n-1], i[n-2], ... i[1], i[0]).

    Args:
        axes(Union[None, tuple(int), list(int), int], optional): If axes is None or
            blank, tensor.transpose() will reverse the order of the axes. If axes is tuple(int)
            or list(int), tensor.transpose() will transpose the tensor to the new axes order.
            If axes is int, this form is simply intended as a convenience alternative to the
            tuple/list form.

    Returns:
        Tensor, has the same dimension as input tensor, with axes suitably permuted.

    Raises:
        TypeError: If input arguments have types not specified above.
        ValueError: If the number of `axes` is not euqal to a.ndim.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> x = Tensor(np.ones((1,2,3), dtype=np.float32))
        >>> x = x.transpose()
        >>> print(x.shape)
        (3, 2, 1)
    """
    ndim = F.rank(x)
    perm = check_transpose_axis_const(axis, ndim)
    return F.transpose(x, perm)


# `tensor.T` is used as a property in graph mode
T_ = transpose


def reshape(x, *shape):
    """
    Give a new shape to a tensor without changing its data.

    Args:
        shape(Union[int, tuple(int), list(int)]): The new shape should be compatible
            with the original shape. If an integer, then the result will be a 1-D
            array of that length. One shape dimension can be -1. In this case, the
            value is inferred from the length of the array and remaining dimensions.

    Returns:
        Tensor, with new specified shape.

    Raises:
        TypeError: If new_shape is not integer, list or tuple, or `x` is not tensor.
        ValueError: If new_shape is not compatible with the original shape.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor
        >>> from mindspore import dtype as mstype
        >>> x = Tensor([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]], dtype=mstype.float32)
        >>> output = x.reshape((3, 2))
        >>> print(output)
        [[-0.1  0.3]
        [ 3.6  0.4]
        [ 0.5 -3.2]]
    """
    new_shape = check_reshape_shp_const(shape)
    return F.reshape(x, new_shape)


def ravel(x):
    """
    Return a contiguous flattened tensor.

    Returns:
        Tensor, a 1-D tensor, containing the same elements of the input.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> x = Tensor(np.ones((2,3,4), dtype=np.float32))
        >>> output = x.ravel()
        >>> print(output.shape)
        (24,)
    """
    return reshape(x, (-1,))


def flatten(x, order='C'):
    r"""
    Return a copy of the tensor collapsed into one dimension.

    Args:
        order (str, optional): Can choose between 'C' and 'F'. 'C' means to
            flatten in row-major (C-style) order. 'F' means to flatten in column-major
            (Fortran-style) order. Only 'C' and 'F' are supported. Default: 'C'.

    Returns:
        Tensor, has the same data type as input.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Raises:
        TypeError: If `order` is not string type.
        ValueError: If `order` is string type, but not 'C' or 'F'.

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> x = Tensor(np.ones((2,3,4), dtype=np.float32))
        >>> output = x.flatten()
        >>> print(output.shape)
        (24,)
    """
    order = check_flatten_order_const(order)
    if order == 'C':
        return F.reshape(x, (-1,))

    perm = F.make_range(0, F.rank(x))
    new_order = F.tuple_reversed(perm)
    return F.reshape(F.transpose(x, new_order), (-1,))


def swapaxes(x, axis1, axis2):
    """
    Interchange two axes of a tensor.

    Args:
        axis1 (int): First axis.
        axis2 (int): Second axis.

    Returns:
        Transposed tensor, has the same data type as the input.

    Raises:
        TypeError: If `axis1` or `axis2` is not integer.
        ValueError: If `axis1` or `axis2` is not in the range of :math:`[-ndim, ndim-1]`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> x = Tensor(np.ones((2,3,4), dtype=np.float32))
        >>> output = x.swapaxes(0, 2)
        >>> print(output.shape)
        (4,3,2)
    """
    axis1, axis2 = check_swapaxes_axis_const((axis1, axis2), x.ndim)

    if axis1 == axis2:
        return x
    if axis1 > axis2:
        axis1, axis2 = axis2, axis1

    perm = F.make_range(0, x.ndim)
    new_perm = None
    if axis2 + 1 < x.ndim:
        new_perm = perm[0:axis1] + perm[axis2:axis2 + 1] + \
            perm[axis1 + 1:axis2] + perm[axis1:axis1 + 1] + perm[axis2 + 1:]
    else:
        new_perm = perm[0:axis1] + perm[axis2:axis2 + 1] + \
            perm[axis1 + 1:axis2] + perm[axis1:axis1 + 1]

    return F.transpose(x, new_perm)


def squeeze(x, axis=None):
    """
    Remove single-dimensional entries from the shape of a tensor.

    Args:
        axis (Union[None, int, list(int), tuple(int)], optional): Default is None.

    Returns:
        Tensor, with all or a subset of the dimensions of length 1 removed.

    Raises:
        TypeError: If input arguments have types not specified above.
        ValueError: If specified axis has shape entry :math:`> 1`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> x = Tensor(np.ones((1,2,2,1), dtype=np.float32))
        >>> x = x.squeeze()
        >>> print(x.shape)
        (2, 2)
    """
    shape = F.shape(x)
    if axis is None:
        return F.squeeze(x)
    # yield squeezed shape based on the axes
    new_shape = prepare_shape_for_squeeze_const(shape, axis)
    return F.reshape(x, new_shape)


def argmax(x, axis=None):
    """
    Returns the indices of the maximum values along an axis.

    Args:
        axis (int, optional): By default, the index is into
            the flattened array, otherwise along the specified axis.

    Returns:
        Tensor, array of indices into the array. It has the same
        shape as a.shape with the dimension along axis removed.

    Raises:
        ValueError: if axis is out of range.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> a = Tensor(np.arange(10, 16).reshape(2, 3).astype("float32"))
        >>> print(a.argmax())
        5
    """
    # P.Argmax only supports float
    x = x.astype(mstype.float32)
    if axis is None:
        x = ravel(x)
        axis = 0
    else:
        axis = check_axis_in_range_const(axis, F.rank(x))
    return P.Argmax(axis)(x)


def argmin(x, axis=None):
    """
    Returns the indices of the minimum values along an axis.

    Args:
        a (Union[int, float, bool, list, tuple, Tensor]): Input array.
        axis (int, optional): By default, the index is into
            the flattened array, otherwise along the specified axis.

    Returns:
        Tensor, array of indices into the array. It has the same
        shape as a.shape with the dimension along axis removed.

    Raises:
        ValueError: if axis is out of range.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> a = Tensor(np.arange(10, 16).reshape(2, 3).astype("float32"))
        >>> print(a.argmin())
        0
    """
    # P.Argmax only supports float
    x = x.astype(mstype.float32)
    if axis is None:
        x = ravel(x)
        axis = 0
    else:
        axis = check_axis_in_range_const(axis, F.rank(x))
    # P.Argmin is currently not supported
    return P.Argmax(axis)(F.neg_tensor(x))


def cumsum(x, axis=None, dtype=None):
    """
    Returns the cumulative sum of the elements along a given axis.

    Note:
        If ``x.dtype`` is :class:`int8`, :class:`int16` or :class:`bool`, the result
        `dtype` will be elevated to :class:`int32`, :class:`int64` is not supported.

    Args:
        x (Tensor): Input tensor.
        axis (int, optional): Axis along which the cumulative sum is computed. The
            default (None) is to compute the cumsum over the flattened array.
        dtype (:class:`mindspore.dtype`, optional): If not specified, stay the same as original,
            tensor, unless it has an integer dtype with a precision less than :class:`float32`.
            In that case, :class:`float32` is used.

    Returns:
        Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> a = Tensor(np.ones((3,3)).astype("float32"))
        >>> output = a.cumsum(axis=0)
        >>> print(output)
        [[1. 1. 1.]
        [2. 2. 2.]
        [3. 3. 3.]]
    """
    original_dtype = x.dtype
    # If original tensor is int, and has precision less then int32, convert
    # to int32
    if x.dtype in (mstype.bool_, mstype.int8, mstype.int16, mstype.uint8, mstype.int16):
        x = x.astype(mstype.int32)
    if axis is None:
        x = x.ravel()
        axis = 0
    check_axis_in_range_const(axis, x.ndim)
    if dtype is not None and original_dtype != dtype:
        return cumsum_(x, axis).astype(dtype, copy=False)
    return cumsum_(x, axis)


def copy(x):
    """
    Returns a copy of the tensor.

    Note:
        The current implementation does not support `order` argument.

    Args:
        x (Tensor): Input tensor.

    Returns:
        Copied tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> a = Tensor(np.ones((3,3)).astype("float32"))
        >>> output = a.copy()
        >>> print(output)
        [[1. 1. 1.]
        [1. 1. 1.]
        [1. 1. 1.]]
    """
    if x.size == 0:
        return x
    origin_dtype = x.dtype
    if origin_dtype == mstype.bool_:
        return F.logical_not(F.logical_not(x))
    if origin_dtype != mstype.float64:
        x = x.astype(mstype.float32)
    x = x / 1.0
    x = x.astype(origin_dtype)
    return x


def max(x, axis=None, keepdims=False, initial=None, where=True): # pylint: disable=redefined-builtin
    """
    Returns the maximum of a tensor or maximum along an axis.

    Args:
        x (Tensor): Input Tensor.
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
        Tensor or scalar, maximum of input tensor. If `axis` is None, the result is a scalar
        value. If `axis` is given, the result is an array of dimension ``a.ndim - 1``.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
            ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> import mindspore.numpy as np
        >>> a = Tensor(np.arange(4).reshape((2,2)).astype('float32'))
        >>> output = a.max()
        >>> print(output)
        3.0
    """
    return compile_utils.reduce_(x, P.ReduceMax(keepdims), cmp_fn=F.maximum,
                                 axis=axis, keepdims=keepdims, initial=initial, where=where)


def min(x, axis=None, keepdims=False, initial=None, where=True): # pylint: disable=redefined-builtin
    """
    Returns the minimum of a tensor or minimum along an axis.

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
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> import mindspore.numpy as np
        >>> a = Tensor(np.arange(4).reshape((2,2)).astype('float32'))
        >>> output = a.min()
        >>> print(output)
        0.0
    """
    return compile_utils.reduce_(x, P.ReduceMin(keepdims), cmp_fn=F.minimum,
                                 axis=axis, keepdims=keepdims, initial=initial, where=where)


def resize(x, *new_shape):
    """
    Changes shape and size of array in-place.

    Note:
        Instead of changing the size of the input array and returns nothing as in numpy,
        this method returns a new Tensor with the input size.
        Numpy argument `refcheck` is not supported.

    Args:
        new_shape (Union[ints, tuple of ints]): Shape of resized array.

    Returns:
        Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import numpy as np
        >>> x = np.array([[0, 1], [2, 3]])
        >>> x = x.resize(2, 3)
        >>> print(x)
        [[0 1 2]
        [3 0 0]]
    """
    if not new_shape:
        return x
    if len(new_shape) == 1:
        if isinstance(new_shape[0], tuple):
            new_shape = new_shape[0]
    flattened = x.ravel()
    cur_size = F.shape_mul(x.shape)
    new_size = F.shape_mul(new_shape)
    diff_size = new_size - cur_size
    if diff_size > 0:
        pad_val = F.fill(x.dtype, (diff_size,), 0)
        res = P.Concat()((flattened, pad_val))
    else:
        res = flattened[:new_size]
    return res.reshape(new_shape)


def diagonal(x, offset=0, axis1=0, axis2=1):
    """
    Returns specified diagonals.

    Args:
        offset (int, optional): Offset of the diagonal from the main diagonal.
            Can be positive or negative. Defaults to main diagonal.
        axis1 (int, optional): Axis to be used as the first axis of the 2-D
            sub-arrays from which the diagonals should be taken. Defaults to
            first axis (0).
        axis2 (int, optional): Axis to be used as the second axis of the 2-D
            sub-arrays from which the diagonals should be taken. Defaults to
            second axis.

    Returns:
        Tensor, if `a` is 2-D, then `a` 1-D array containing the diagonal.

    Raises:
        ValueError: if the input tensor has less than two dimensions.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.arange(4).reshape(2,2)
        >>> print(a)
        [[0 1]
        [2 3]]
        >>> output = a.diagonal()
        >>> print(output)
        [0 3]
    """
    ndim = x.ndim
    if ndim < 2:
        const_utils.raise_value_error('diagonal requires an array of at least two dimensions')
    dtype = x.dtype

    axes = check_axis_valid((axis1, axis2), ndim)
    perm = ()
    for i in range(ndim):
        if i not in axes:
            perm += (i,)
    perm += axes
    x = x.transpose(perm)

    shape = x.shape
    n, m = shape[-2:]

    e = F.eye(n, m, dtype)
    if offset >= m or offset <= -n:
        e = F.fill(dtype, (n, m), 0)
    elif offset != 0:
        e = e.astype(mstype.float32)
        if offset > 0:
            e_left = F.fill(dtype, (n, offset), 0)
            e_right = e[..., 0:m-offset:1]
            e = P.Concat(1)((e_left, e_right)).astype(dtype)
        elif offset < 0:
            e_upper = F.fill(dtype, (-offset, m), 0)
            e_lower = e[0:n+offset:1, ...]
            e = P.Concat(0)((e_upper, e_lower)).astype(dtype)
    e = P.BroadcastTo(shape)(e)

    prod = F.tensor_mul(x, e)
    res = F.reduce_sum(prod.astype(mstype.float32), -1)

    begin = ()
    for i in range(ndim-2):
        begin += (0,)
    last_dim_begin = max_(0, -offset)
    begin += (last_dim_begin,)
    size = res.shape[:-1]
    last_dim_end = min_(
        shape[-2], max_(0, shape[-1] - offset)) - last_dim_begin
    if last_dim_end <= 0:
        return empty_compile(dtype, (0,))
    size += (last_dim_end,)
    res = F.tensor_slice(res, begin, size)
    return res.astype(dtype)


def trace(x, offset=0, axis1=0, axis2=1, dtype=None):
    """
    Returns the sum along diagonals of the array.

    Args:
        offset (int, optional): Offset of the diagonal from the main diagonal.
            Can be positive or negative. Defaults to main diagonal.
        axis1 (int, optional): Axis to be used as the first axis of the 2-D
            sub-arrays from which the diagonals should be taken. Defaults to
            first axis (0).
        axis2 (int, optional): Axis to be used as the second axis of the 2-D
            sub-arrays from which the diagonals should be taken. Defaults to
            second axis.
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor, sum_along_diagonals.

    Raises:
        ValueError: if the input tensor has less than two dimensions.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.eye(3)
        >>> print(x.trace())
        3.0
    """
    d = x.diagonal(offset, axis1=axis1, axis2=axis2)
    shape = d.shape
    if dtype is None:
        dtype = d.dtype
    if shape[-1] == 0:
        return F.fill(dtype, shape[:-1], 0)
    res = F.reduce_sum(d.astype(mstype.float32), -1)
    return res.astype(dtype)


def take(x, indices, axis=None, mode='clip'):
    """
    Takes elements from an array along an axis.

    Args:
        a (Tensor): Source array with shape `(Ni…, M, Nk…)`.
        indices (Tensor): The indices with shape `(Nj...)` of the values to extract.
        axis (int, optional): The axis over which to select values. By default,
            the flattened input array is used.
        mode (‘raise’, ‘wrap’, ‘clip’, optional):
            - edge: Pads with the edge values of `arr`.
            - raise: Raises an error;
            - wrap: Wraps around;
            - clip: Clips to the range. `clip` mode means that all indices that are
              too large are replaced by the index that addresses the last element
              along that axis. Note that this disables indexing with negative numbers.

    Returns:
        Tensor, the indexed result.

    Raises:
        ValueError: if axis is out of range.
        TypeError: if the input is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.array([4, 3, 5, 7, 6, 8])
        >>> indices = np.array([0, 1, 4])
        >>> output = a.take(indices)
        >>> print(output)
        [4 3 6]
    """
    if mode not in ('raise', 'wrap', 'clip'):
        const_utils.raise_value_error('raise should be one of "raise", "wrap", or "clip"')
    if axis is None:
        a = x.ravel()
        axis = 0
    else:
        a = x
    ndim = a.ndim
    axis = check_axis_in_range_const(axis, ndim)

    shape_a = a.shape
    shape_indices = indices.shape
    size_indices = indices.size
    indices = compile_utils.check_indices(shape_a[axis], indices, mode)

    # reshapes indices to shape (Ni..., Nj..., Nk)
    shape_ni = tuple_slice(shape_a, None, axis)
    shape_nk = tuple_slice(shape_a, axis + 1, None)
    shape_out = shape_ni + shape_indices + shape_nk
    shape_indices = expanded_shape(ndim, size_indices, axis)
    indices = indices.reshape(shape_indices)
    shape_indices = shape_ni + (indices.size,) + shape_nk
    indices = P.BroadcastTo(shape_indices)(indices)

    res = F.gather_d(a, axis, indices)
    return res.reshape(shape_out)


def choose(x, choices, mode='clip'):
    """
    Construct an array from an index array and a list of arrays to choose from.

    Args:
        choices (sequence of arrays): Choice arrays. `a` and all of the `choices` must
            be broadcastable to the same shape. If `choices` is itself an array, then
            its outermost dimension (i.e., the one corresponding to ``choices.shape[0]``)
            is taken as defining the “sequence”.
        mode (‘raise’, ‘wrap’, ‘clip’, optional): Specifies how indices outside
            ``[0, n-1]`` will be treated:

            ‘raise’ – raise an error (default);

            ‘wrap’ – wrap around;

            ‘clip’ – clip to the range. ‘clip’ mode means that all indices that are
            too large are replaced by the index that addresses the last element
            along that axis. Note that this disables indexing with negative numbers.

    Returns:
        Tensor, the merged result.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Raises:
        ValueError: if ``len(condlist) != len(choicelist)``.

    Examples:
        >>> import mindspore.numpy as np
        >>> choices = [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]]
        >>> x = np.array([2, 3, 1, 0])
        >>> print(x.choose(choices))
        [20 31 12  3]
    """
    if check_is_tensor(F.typeof(choices)):
        shape_choice = infer_out_shape(x.shape, choices.shape[1:])
        choices = P.BroadcastTo((choices.shape[0],) + shape_choice)(choices)
    else:
        # broadcasts choices to the same shape if choices is a sequence
        choicelist = []
        shapes = ()
        for choice in choices:
            if not check_is_tensor(F.typeof(choice)):
                choice = const_utils.make_tensor(choice)
            shapes += (choice.shape,)
            choicelist.append(choice)
        shape_choice = infer_out_shape(x.shape, *shapes)
        tmp = []
        for choice in choicelist:
            tmp.append(P.BroadcastTo(shape_choice)(choice))
        choices = F.stack(tmp)

    if x.ndim == 0 or choices.ndim == 0:
        const_utils.raise_value_error('input cannot be scalars')
    a = P.BroadcastTo(shape_choice)(x)
    dtype = choices.dtype
    # adjusts dtype for F.tensor_mul and F.gather_nd
    a = a.astype(mstype.int32)
    choices = choices.astype(mstype.int32)
    a = compile_utils.check_indices(choices.shape[0], a, mode, allow_negative_index=False)

    grids = []
    ndim = len(a.shape)
    for i in range(ndim):
        dim_grid = const_utils.make_tensor(F.make_range(a.shape[i]), mstype.int32)
        dim_shape = expanded_shape(ndim, a.shape[i], i)
        dim_grid = P.BroadcastTo(a.shape)(dim_grid.reshape(dim_shape))
        grids.append(dim_grid)
    grid = P.Stack(-1)(grids)
    indices = P.Concat(-1)((a.reshape(a.shape + (1,)), grid))
    return F.gather_nd(choices, indices).astype(dtype)


def searchsorted(x, v, side='left', sorter=None):
    """
    Finds indices where elements should be inserted to maintain order.

    Args:
        v (Union[int, float, bool, list, tuple, Tensor]): Values to insert into `a`.
        side ('left', 'right', optional): If ‘left’, the index of the first suitable
            location found is given. If ‘right’, return the last such index. If there is
            no suitable index, return either 0 or N (where N is the length of `a`).
        sorter (Union[int, float, bool, list, tuple, Tensor]): 1-D optional array of
            integer indices that sort array `a` into ascending order. They are typically
            the result of argsort.

    Returns:
        Tensor, array of insertion points with the same shape as `v`.

    Raises:
        ValueError: if argument for `side` or `sorter` is invalid.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import numpy as np
        >>> x = np.array([1,2,3,4,5])
        >>> print(x.searchsorted(3))
        2
    """
    if side not in ('left', 'right'):
        const_utils.raise_value_error('invalid value for keyword "side"')
    a = x.astype(mstype.float32)
    if not check_is_tensor(F.typeof(v)):
        v = const_utils.make_tensor(v)
    shape = v.shape
    if sorter is not None:
        if sorter.ndim != 1 or sorter.size != a.size:
            const_utils.raise_value_error('sorter must be 1-D array with the same size as `a`')
        sorter = const_utils.make_tensor(sorter)
        sorter = sorter.reshape(sorter.shape + (1,))
        a = F.gather_nd(a, sorter)
    less_op = F.tensor_le if side == 'left' else F.tensor_lt
    i = F.fill(mstype.int32, shape, 0)
    j = F.fill(mstype.int32, shape, a.size)

    sort_range = F.make_range(get_log2_size(F.shape_mul(shape) + 1))
    for _ in sort_range:
        mid = (i - F.neg_tensor(j))//2
        mask = less_op(v, F.gather_nd(a, mid.reshape(mid.shape + (1,))))
        i = F.select(mask, i, mid)
        j = F.select(mask, mid, j)
    return j


def fill(x, value):
    """
    Fills the array with a scalar value.

    Note:
        Unlike Numpy, tensor.fill() will always returns a new tensor, instead of
        filling the original tensor.

    Args:
        value (Union[None, int, float, bool]): All elements of a will be assigned this value.

    Returns:
        Tensor, with the original dtype and shape as input tensor.

    Raises:
        TypeError: If input arguments have types not specified above.
        ValueError: If `shape` has entries < 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> a = Tensor(np.arange(4).reshape((2,2)).astype('float32'))
        >>> print(a.fill(1.0))
        [[1. 1.]
        [1. 1.]]
    """
    if value is None:
        if x.dtype not in (mstype.float16, mstype.float32, mstype.float64):
            const_utils.raise_type_error("If None is used as value, the original Tensor's dtype must be float.")
        value = nan_tensor
        return F.tile(value, x.shape).astype(x.dtype)
    if not isinstance(value, (int, float, bool)):
        const_utils.raise_type_error("input value must be a scalar.")
    return F.fill(x.dtype, x.shape, value)


def ptp(x, axis=None, keepdims=False):
    """
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
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor
        >>> x = Tensor([[4.0, 9.0, 2.0, 10.0], [6.0, 9.0, 7.0, 12.0]]).astype("float32")
        >>> print(x.ptp(axis=1))
        [8. 6.]
        >>> print(x.ptp(axis=0))
        [2. 0. 5. 2.]
    """
    if not isinstance(keepdims, bool):
        const_utils.raise_type_error('keepdims should be boolean')
    if axis is None:
        axis = ()
    else:
        check_axis_type(axis, True, True, False)
        axis = check_axis_valid(axis, x.ndim)

    return x.max(axis, keepdims) - x.min(axis, keepdims)


def clip(x, xmin, xmax, dtype=None):
    """
    Clips (limits) the values in an array.

    Given an interval, values outside the interval are clipped to the interval edges.
    For example, if an interval of :math:`[0, 1]` is specified, values smaller than 0 become 0,
    and values larger than 1 become 1.

    Note:
        Currently, clip with `nan` is not supported.

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
        >>> from mindspore import Tensor
        >>> x = Tensor([1, 2, 3, -4, 0, 3, 2, 0]).astype("float32")
        >>> output = x.clip(0, 2)
        >>> print(output)
        [1 2 2 0 0 2 2 0]
    """
    if xmin is None and xmax is None:
        const_utils.raise_value_error("One of max or min must be given.")
    is_scalar = False
    if xmin is not None:
        xmin = const_utils.make_tensor(xmin).astype(x.dtype)
        if x.ndim == 0 and xmin.ndim == 0:
            x = F.maximum(x.reshape((1,)), xmin).squeeze()
        else:
            x = F.maximum(x, xmin)
    if xmax is not None:
        xmax = const_utils.make_tensor(xmax).astype(x.dtype)
        if x.ndim == 0 and xmax.ndim == 0:
            x = F.minimum(x.reshape((1,)), xmax).squeeze()
        else:
            x = F.minimum(x, xmax)
    if is_scalar:
        return x.squeeze()
    if dtype is not None and dtype != x.dtype:
        return x.astype(dtype)
    return x


def var(x, axis=None, ddof=0, keepdims=False):
    """
    Compute the variance along the specified axis.
    The variance is the average of the squared deviations from the mean, i.e.,
    :math:`var = mean(abs(x - x.mean())**2)`.

    Return the variance, which is computed for the flattened array by default,
    otherwise over the specified axis.

    Note:
        Numpy arguments `dtype`, `out` and `where` are not supported.

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
        >>> print(input_x.var())
        1.25
    """
    if 0 in x.shape:
        return nan_tensor.astype(x.dtype)
    if not isinstance(ddof, int) or not isinstance(keepdims, int):
        const_utils.raise_type_error("integer argument expected")

    if axis is None:
        axis = ()
    else:
        axis = check_and_canonicalize_axes(axis, x.ndim)
    x_mean = _mean_keepdims(x, axis)
    x_sub = F.tensor_sub(x, x_mean)
    x_pow = F.tensor_pow(x_sub, 2)
    if keepdims:
        x_sum = _reduce_sum_keepdims(x_pow, axis)
    else:
        x_sum = _reduce_sum_default(x_pow, axis)

    if axis == ():
        axis = F.make_range(x.ndim)
    nums = 1
    for ax in axis:
        nums *= x.shape[ax]
    return F.tensor_div(x_sum, nums - ddof)


def std(x, axis=None, ddof=0, keepdims=False):
    """
    Compute the standard deviation along the specified axis.
    The standard deviation is the square root of the average of the squared deviations
    from the mean, i.e., :math:`std = sqrt(mean(abs(x - x.mean())**2))`.

    Return the standard deviation, which is computed for the flattened array by default,
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
        keepdims: Default: `False`.

    Returns:
        Standard deviation tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> input_x = np.array([1., 2., 3., 4.])
        >>> print(input_x.std())
        1.118034
    """
    x_var = var(x, axis, ddof, keepdims)
    return F.tensor_pow(x_var, 0.5)


def sum(x, axis=None, dtype=None, keepdims=False, initial=None): # pylint: disable=redefined-builtin
    """
    Return sum of array elements over a given axis.

    Note:
        Numpy arguments `out`, `where`, `casting`, `order`, `subok`, `signature`, and
        `extobj` are not supported.

    Args:
        x (Union[int, float, bool, list, tuple, Tensor]): Elements to sum.
        axis (Union[None, int, tuple(int)]): Axis or axes along which a sum is performed. Default: None.
            If None, sum all of the elements of the input array.
            If axis is negative it counts from the last to the first axis.
            If axis is a tuple of ints, a sum is performed on all of the axes specified in the tuple
            instead of a single axis or all the axes as before.
        dtype (:class:`mindspore.dtype`, optional): defaults to None. Overrides the dtype of the
            output Tensor.
        keepdims (bool): If this is set to True, the axes which are reduced are left in the result as
            dimensions with size one. With this option, the result will broadcast correctly against the input array.
            If the default value is passed, then keepdims will not be passed through to the sum method of
            sub-classes of ndarray, however any non-default value will be. If the sub-class’ method does not
            implement keepdims any exceptions will be raised.
        initial (scalar): Starting value for the sum.

    Returns:
        Tensor. A tensor with the same shape as input, with the specified axis removed.
        If input tensor is a 0-d array, or if axis is None, a scalar is returned.

    Raises:
        TypeError: If input is not array_like or `axis` is not int or tuple of ints or
            `keepdims` is not integer or `initial` is not scalar.
        ValueError: If any axis is out of range or duplicate axes exist.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> input_x = np.array([-1, 0, 1]).astype('int32')
        >>> print(input_x.sum())
        0
        >>> input_x = np.arange(10).reshape(2, 5).astype('float32')
        >>> print(input_x.sum(axis=1))
        [10. 35.]
    """
    input_x = x.astype(mstype.int32) if x.dtype == mstype.bool_ else x
    dtype = input_x.dtype if dtype is None else dtype
    if not isinstance(keepdims, int):
        const_utils.raise_type_error("integer argument expected")
    if initial is not None and not isinstance(initial, (int, float, bool)):
        const_utils.raise_type_error("initial argument should be a scalar.")
    if axis is None:
        axis = ()
    else:
        axis = check_and_canonicalize_axes(axis, x.ndim)

    if not check_type_support(input_x.dtype, 'GPU', (mstype.float64, mstype.float32, mstype.float16)):
        input_x = input_x.astype(mstype.float32)
    if 0 in x.shape:
        x = const_utils.make_tensor([0], x.dtype)
    if keepdims:
        res = _reduce_sum_keepdims(input_x, axis)
    else:
        res = _reduce_sum_default(input_x, axis)
    if initial is not None:
        res += initial
    return res.astype(dtype)


def repeat(x, repeats, axis=None):
    """
    Repeat elements of an array.

    Args:
        x (Tensor): Input tensor.
        repeats (Union[int, tuple, list]): The number of repetitions for each element.
            `repeats` is broadcasted to fit the shape of the given axis.
        axis (int, optional): The axis along which to repeat values. By default,
            use the flattened input tensor, and return a flat output tensor.

    Returns:
        Tensor, has the same shape as input tensor except along the given axis.

    Raises:
        ValueError: if axis is out of range.
        TypeError: if input is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.array(3)
        >>> print(x.repeat(4))
        [3 3 3 3]
        >>> x = np.array([[1,2],[3,4]])
        >>> print(x.repeat(2))
        [1 1 2 2 3 3 4 4]
        >>> print(x.repeat(3, axis=1))
        [[1 1 1 2 2 2]
        [3 3 3 4 4 4]]
        >>> print(x.repeat([1,2], axis=0))
        [[1 2]
        [3 4]
        [3 4]]
    """
    if not isinstance(repeats, (tuple, list)):
        repeats = (repeats,)
    for element in repeats:
        if not isinstance(element, int):
            const_utils.raise_type_error("Each element should be integer")
    if axis is None:
        x = ravel(x)
        axis = 0
    if not isinstance(axis, int):
        const_utils.raise_type_error('axes should be integers')
    check_axis_in_range_const(axis, x.ndim)
    axis = axis + x.ndim if axis < 0 else axis

    if len(repeats) == 1:
        repeats = repeats[0]
        if repeats == 0:
            return empty_tensor(x.dtype)
        return repeat_elements(x, repeats, axis)
    size = x.shape[axis]
    if len(repeats) != size:
        const_utils.raise_value_error('operands could not be broadcast together')
    subs = P.Split(axis, size)(x)
    repeated_subs = []
    for sub, rep in zip(subs, repeats):
        if rep != 0:
            repeated_subs.append(repeat_elements(sub, rep, axis))
    return P.Concat(axis)(repeated_subs)


def getitem(data, index):
    """Implementation of `getitem`."""
    return data.__getitem__(index)


def setitem(data, index, value):
    """Implementation of `setitem`."""
    return data.__setitem__(index, value)


def item(data, *args):
    """Implementation of `item`."""
    return compile_utils.tensor_item(data, *args)


def itemset(data, *args):
    """Implementation of `itemset`."""
    return compile_utils.tensor_itemset(data, *args)


def ms_iter(xs):
    """Implementation of `iter`."""
    return xs.__ms_iter__()


def ms_next(it):
    """Implementation of `next`."""
    return it.__ms_next__()


def hasnext(it):
    """Implementation of `hasnext`."""
    return it.__ms_hasnext__()


def ms_len(data):
    """Implementation of `len`."""
    return data.__len__()


def floor(x):
    """Implementation of `floor`."""
    return x.__floor__()


def trunc(x):
    """Implementation of `trunc`."""
    return x.__trunc__()


def uadd(x):
    """Implementation of `uadd`."""
    return x.__pos__()


def usub(x):
    """Implementation of `usub`."""
    return x.__neg__()


def scalar_truediv(x, y):
    """Implementation of `scalar_truediv`."""
    return x.__truediv__(y)


def scalar_floordiv(x, y):
    """Implementation of `scalar_floordiv`."""
    return x.__floordiv__(y)


def bool_(x):
    """Implementation of `bool`."""
    return x.__bool__()


def enumerate_(x, start=0):
    """Enumerate list or tuple or tensor."""
    x_type = F.typeof(x)
    ret = ()
    op_name = "enumerate"
    if check_is_tuple_or_list_or_tensor(x_type, op_name, "first input") and \
            check_is_const_int(start, op_name, "start"):
        if check_is_tensor(x_type):
            for i in range(x.shape[0]):
                ret += ((start + i, x[i]),)
        else:
            ret = zip(range(start, start + len(x)), x)
    return ret


def expand_tensor_as(x, y):
    """Expand tensor"""
    broadcast_to = P.BroadcastTo(shape_(y))
    return broadcast_to(x)


def view(x, *shape):
    """Reshape tensor, if shape is -1, reshape tensor into one dimension"""
    shape = check_view_shape(shape)
    return F.reshape(x, shape)


def isinstance_(x, base_type):
    """Determine whether x is an instance of base_type."""
    x_type = F.typeof(x)
    return check_type_same(x_type, base_type)


def while_cond(x):
    """For while condition, if the condition is a tensor, the loop will not be unrolled"""
    if F.issubclass_(F.typeof(x), F.typeof(mstype.tensor)):
        is_cond = check_is_tensor_bool_cond(F.shape(x))
        if is_cond:
            return F.cast(x, mstype.bool_)
    return x


@constexpr
def empty_tensor(dtype):
    return Tensor([], dtype)


@constexpr
def check_type_same(x_type, base_type):
    """Check x_type is same as base_type."""
    pytype_to_mstype = {
        bool: mstype.Bool,
        int: mstype.Int,
        float: mstype.Float,
        str: mstype.String,
        list: mstype.List,
        tuple: mstype.Tuple,
        dict: mstype.Dict,
        Tensor: mstype.tensor_type,
        Parameter: mstype.ref_type
    }

    has_int = False
    has_tensor = False

    def to_target_type(origin_type):
        try:
            if isinstance(origin_type, type):
                ret_type = pytype_to_mstype[origin_type]
                if ret_type == mstype.Int:
                    nonlocal has_int
                    has_int = True
                if ret_type == mstype.tensor_type:
                    nonlocal has_tensor
                    has_tensor = True
                return (ret_type,)
            if isinstance(origin_type, tuple):
                return tuple(to_target_type(i) for i in origin_type)
            raise TypeError(f"The second arg of 'isinstance' must be a type or a tuple of types, "
                            f"but got a {type(origin_type).__name__}")
        except KeyError:
            raise TypeError(f"The second arg of 'isinstance' should be bool, int, float, str, list, tuple, "
                            f"Tensor, Parameter, or a tuple containing only these types, but got {origin_type}")
    target_type = to_target_type(base_type)
    if (isinstance(x_type, mstype.Bool) and has_int) or (isinstance(x_type, mstype.ref_type) and has_tensor):
        return True
    return isinstance(x_type, target_type)


@constexpr
def get_itemsize(x_type):
    """get itemsize from tensor's dtype."""
    return itemsize_map[x_type]


@constexpr
def check_is_tensor(x):
    """check whether x is tensor."""
    if isinstance(x, mstype.tensor_type):
        return True
    return False


@constexpr
def check_is_tuple_or_list_or_tensor(x, op_name, arg_name):
    """check whether x is list or tuple or tensor."""
    if isinstance(x, (mstype.List, mstype.Tuple, mstype.tensor_type)):
        return True
    raise TypeError(f"For '{op_name}', the '{arg_name}' should be tuple or list or tensor, but got {x}.")


@constexpr
def check_is_const_int(x, op_name, arg_name):
    """check whether x is const int."""
    if x is None:
        raise TypeError(f"For '{op_name}', the '{arg_name}' should be a const int number, but got not const.")
    if not isinstance(x, int):
        raise TypeError(f"For '{op_name}', the '{arg_name}' should be a const int number, but got {x}.")
    return True


@constexpr
def check_is_tensor_bool_cond(shp):
    """check if tensor is a bool condition"""
    if shp in ((), (1,)):
        return True
    raise ValueError(f"Only tensor which shape is () or (1,) can be converted to bool, but got tensor shape is {shp}")


@constexpr
def const_tensor_to_bool(x):
    """convert bool tensor to bool condition"""
    if x is None:
        raise ValueError("Only tensor which shape is () or (1,) can be converted to bool, but got None")
    x = x.asnumpy()
    if x.shape == ():
        return bool(x)
    if x.shape == (1,):
        return bool(x[0])
    raise ValueError(
        f"Only tensor which shape is () or (1,) can be converted to bool, but got tensor shape is {x.shape}")


@constexpr
def check_view_shape(x):
    """Check view function input shape"""
    if not x:
        raise ValueError("The shape variable should not be empty")
    if isinstance(x[0], tuple):
        if len(x) != 1:
            raise ValueError(f"Only one tuple is needed, but got {x}")
        x = x[0]
    return x


# convert normal param_check functions to constexpr functions
check_astype_dtype_const = constexpr(validator.check_astype_dtype)
check_transpose_axis_const = constexpr(validator.check_transpose_axis)
check_reshape_shp_const = constexpr(validator.check_reshape_shp)
check_flatten_order_const = constexpr(validator.check_flatten_order)
check_swapaxes_axis_const = constexpr(validator.check_swapaxes_axis)
prepare_shape_for_squeeze_const = constexpr(validator.prepare_shape_for_squeeze)
check_axis_in_range_const = constexpr(validator.check_axis_in_range)
check_axis_valid = constexpr(validator.check_axis_valid)
max_ = constexpr(validator.max_)
min_ = constexpr(validator.min_)
expanded_shape = constexpr(validator.expanded_shape)
tuple_slice = constexpr(validator.tuple_slice)
infer_out_shape = constexpr(validator.infer_out_shape)
get_log2_size = constexpr(validator.get_log2_size)
check_axis_type = constexpr(validator.check_axis_type)
check_and_canonicalize_axes = constexpr(validator.check_and_canonicalize_axes)
empty_compile = constexpr(validator.empty_compile)
check_type_support = constexpr(validator.check_type_support)


def tensor_bool(x):
    """tensor as condition, if is constant, return immediate bool value"""
    is_cond = check_is_tensor_bool_cond(F.shape(x))
    if is_cond and F.isconstant(x):
        return const_tensor_to_bool(x)
    return F.cast(x, mstype.bool_)


def and_(x, y):
    """Implementation of `and` (`&`)."""
    return x.__and__(y)


def or_(x, y):
    """Implementation of `or` (`|`)."""
    return x.__or__(y)


def matmul(x, y):
    """Implementation of `matmul` (`@`)."""
    return x.__matmul__(y)


def float_bool(x):
    """Implementation of `float_bool`."""
    return x != 0.0


def int_bool(x):
    """Implementation of `int_bool`."""
    return x != 0


def str_bool(x):
    """Implementation of `str_bool`."""
    if x == "":
        return False
    return True


def list_bool(x):
    """Implementation of `tuple_bool`."""
    return len(x) != 0


def tuple_bool(x):
    """Implementation of `tuple_bool`."""
    return len(x) != 0


def dict_bool(x):
    """Implementation of `dict_bool`."""
    return len(x) != 0


def none_bool(x):
    """Implementation of `none_bool`."""
    return False


def func_bool(x):
    """Implementation of `func_bool`."""
    return True


def float_floordiv(x, y):
    """Implementation of `float_floordiv`."""
    return floor(x / y)


#############
# Iteration #
#############


@dataclass(frozen=True)
class SequenceIterator:
    """
    SequenceIterator is a util dataclass for iterating sequence object.

    Iterator to use for sequences like List, Array.
    """

    idx: int
    seq: list

    @core(ignore_values=True)
    def __ms_hasnext__(self):
        """Whether the index is past the length of the sequence."""
        return self.idx < ms_len(self.seq)

    @core(ignore_values=True)
    def __ms_next__(self):
        """Return the next element and a new iterator."""
        return self.seq[self.idx], SequenceIterator(self.idx + 1, self.seq)


def list_iter(xs):
    """Iterator for List."""
    return SequenceIterator(0, xs)


def array_iter(xs):
    """Iterator for Array."""
    return SequenceIterator(0, xs)


def tuple_next(xs):
    """Next tuple."""
    return xs[0], tail(xs)


def tuple_hasnext(xs):
    """Whether the tuple is empty or not."""
    return len(xs) > 0


def list_next(xs):
    """Next list."""
    return xs[0], tail(xs)


def list_hasnext(xs):
    """Whether the list is empty or not."""
    return len(xs) > 0


# pylint: disable=redefined-outer-name
def list_append(self_, item):
    return _append(self_, item)


#################
# Array methods #
#################


def to_array(x):
    """Implementation of `to_array`."""
    return x.__ms_to_array__()
