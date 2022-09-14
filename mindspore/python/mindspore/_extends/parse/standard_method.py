# This is the Python adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
#
# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

from __future__ import absolute_import
from mindspore import Tensor, CSRTensor, COOTensor, RowTensor, ms_class
from mindspore import dtype as mstype
from mindspore._c_expression import Tensor as Tensor_
from mindspore.ops.function.sparse_func import sparse_add
from mindspore.ops.composite.base import _append, _insert, _pop, _list_clear, _reverse, \
    _count, _extend
import mindspore.common._monad as monad

from ..._checkparam import Validator as validator
from ...ops import functional as F
from ...ops import operations as P
from ...ops.composite import tail, core, MultitypeFuncGraph, env_get, hyper_add, \
    zeros_like, ones_like, repeat_elements
from ...ops.composite.base import _append, _insert, _pop, _list_clear, _reverse, \
    _count, _extend, _dict_clear, _haskey, _update, _fromkeys
from ...ops.composite.multitype_ops import _constexpr_utils as const_utils
from ...ops.composite.multitype_ops import _compile_utils as compile_utils
from ...ops.operations.math_ops import Median
from ...ops.operations._inner_ops import Format
from ...ops.operations import _csr_ops
from ...ops.primitive import constexpr
from ...common import dtype as mstype

__all__ = ['MultitypeFuncGraph', 'env_get', 'hyper_add', 'zeros_like', 'ones_like']

shape_ = P.Shape()
dtype_ = P.DType()
abs_ = P.Abs()
ndim_ = P.Rank()
cumsum_ = P.CumSum()
size_op_ = P.Size()
_format = Format()
_reduce_sum_default = P.ReduceSum()
_reduce_sum_keepdims = P.ReduceSum(True)
_mean_keepdims = P.ReduceMean(True)
_csr_mm = _csr_ops.CSRMM()
_addcdiv = P.Addcdiv()
_addcmul = P.Addcmul()

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


def prod(x, axis=(), keep_dims=False):
    """
    Reduces a dimension of a tensor by product all elements in the dimension.

    Args:
        x (Tensor): Input Tensor.
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
        >>> output = input_x.prod()
        >>> print(output)
        6.0
    """
    return F.prod(x, axis, keep_dims)


def addcdiv(input_data, x1, x2, value):
    """
    Performs the element-wise division of tensor x1 by tensor x2,
    multiply the result by the scalar value and add it to input_data.

    Args:
        input_data (Tensor): The tensor to be added.
        x1 (Tensor): The numerator tensor.
        x2 (Tensor): The denominator tensor.
        value (Tensor): The multiplier for tensor x1/x2.

    Returns:
        Tensor, has the same shape and dtype as x1/x2.
    """
    return _addcdiv(input_data, x1, x2, value)


def addcmul(input_data, x1, x2, value):
    """
    Performs the element-wise product of tensor x1 and tensor x2,
    multiply the result by the scalar value and add it to input_data.

    Args:
        input_data (Tensor): The tensor to be added.
        x1 (Tensor): The tensor to be multiplied.
        x2 (Tensor): The tensor to be multiplied.
        value (Tensor): The multiplier for tensor x1*x2.

    Returns:
        Tensor, has the same shape and dtype as x1*x2.
    """
    return _addcmul(input_data, x1, x2, value)


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


def atan2(x, y):
    r"""
    Computes the first input tensor multiplied by the logarithm of second input tensor element-wise.
    Refer to :func:`mindspore.ops.atan2` for more details.
    """
    return F.atan2(x, y)


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


def hasattr(x, attr):  # pylint: disable=redefined-builtin
    """
    Return whether an object has the attribute.

    Args:
        x (object): Input object.
        attr (string): The name of attribute

    Returns:
        Boolean value, indicates whether the object x has attribute attr.
    """
    out = getattr(x, attr, mstype._null)
    return not isinstance(out, mstype._null_type)


def astype(x, dtype, copy=True):  # pylint: disable=redefined-outer-name
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


def minimum(x, y):
    r"""
    Computes the minimum of input tensors element-wise.

    Refer to :func:`mindspore.ops.minimum` for more detail.
    """
    return F.minimum(x, y)


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


def reverse_sequence(x, seq_lengths, seq_dim, batch_dim=0):
    """
    Reverses variable length slices.

    Args:
        x (Tensor): The input to reverse, supporting all number types including bool.
        seq_lengths (Tensor): Must be a 1-D vector with int32 or int64 types.
        seq_dim (int): The dimension where reversal is performed. Required.
        batch_dim (int): The input is sliced in this dimension. Default: 0.

    Returns:
        Reversed tensor with the same shape and data type as input.

    Raises:
        TypeError: If `seq_dim` or `batch_dim` is not an int.
        ValueError: If value of `batch_dim` is equal to or greater than length of shape of input.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), mindspore.float32)
        >>> seq_lengths = Tensor(np.array([1, 2, 3]))
        >>> output = x.reverse_sequence(seq_lengths, seq_dim=1)
        >>> print(output)
        [[1. 2. 3.]
         [5. 4. 6.]
         [9. 8. 7.]]
        >>> x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), mindspore.float32)
        >>> seq_lengths = Tensor(np.array([1, 2, 3]))
        >>> output = x.reverse_sequence(seq_lengths, seq_dim=0, batch_dim=1)
        >>> print(output)
        [[1. 5. 9.]
         [4. 2. 6.]
         [7. 8. 3.]]
        >>> x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), mindspore.float32)
        >>> seq_lengths = Tensor(np.array([2, 2, 3]))
        >>> output = x.reverse_sequence(seq_lengths, seq_dim=1)
        >>> print(output)
        [[2. 1. 3.]
         [5. 4. 6.]
         [9. 8. 7.]]
        >>> x = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), mindspore.float32)
        >>> seq_lengths = Tensor(np.array([3, 2, 3]))
        >>> output = x.reverse_sequence(seq_lengths, seq_dim=1)
        >>> print(output)
        [[3. 2. 1.]
         [5. 4. 6.]
         [9. 8. 7.]]
        >>> x = Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), mindspore.float32)
        >>> seq_lengths = Tensor(np.array([4, 4]))
        >>> output = x.reverse_sequence(seq_lengths, seq_dim=1)
        >>> print(output)
        [[4. 3. 2. 1.]
         [8. 7. 6. 5.]]
    """
    return F.reverse_sequence(x, seq_lengths, seq_dim, batch_dim)


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
            Defaults to None.

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
            Defaults to None.

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


def argmax_with_value(x, axis=0, keep_dims=False):
    """Calculates the maximum value with corresponding index, and returns indices and values."""
    return F.max(x, axis, keep_dims)


def argmin_with_value(x, axis=0, keep_dims=False):
    """Calculates the minimum value with corresponding index, and returns indices and values."""
    return F.min(x, axis, keep_dims)


def median(x, global_median, axis=0, keep_dims=False):
    r"""
    Computes the median of input tensor.

    .. warning::
        When attr `global_median` is True, the second output Tensor value is meaningless.

    """
    check_axis_in_range_const(axis, x.ndim)
    median_ = Median(global_median, axis, keep_dims)
    return median_(x)


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
    if dtype is not None:
        dtype = check_astype_dtype_const(dtype)
        if original_dtype != dtype:
            return cumsum_(x, axis).astype(dtype, copy=False)
    return cumsum_(x, axis)


def cummin(x, axis):
    """
    Returns the cumulative minimum of elements and the index.
    """
    return F.cummin(x, axis)


def cummax(x, axis):
    """
    Returns the cumulative maximum of elements and the index.
    """
    return F.cummax(x, axis)


def index_fill(x, dim, index, value):
    """
    Fills the elements under the dim dimension of the input Tensor with the input value
    by selecting the indices in the order given in index.
    """
    return F.index_fill(x, dim, index, value)


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


def max(x, axis=None, keepdims=False, initial=None, where=True):  # pylint: disable=redefined-builtin
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


def min(x, axis=None, keepdims=False, initial=None, where=True):  # pylint: disable=redefined-builtin
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


def pow(x, y):  # pylint: disable=redefined-builtin
    """
    Calculate the power of Tensor.
    """
    return F.pow(x, y)


def log(x):
    """
    Calculate the logarithm of Tensor.
    """
    return F.log(x)


def round_(x):
    """
    Returns half to even of a tensor element-wise.
    """
    return F.round(x)


def unique_consecutive(x, return_idx=False, return_counts=False, axis=None):
    """
    Returns the elements that are unique in each consecutive group of equivalent elements in the input tensor.
    """
    return F.unique_consecutive(x, return_idx, return_counts, axis)


def unique_with_pad(x, pad_num):
    """
    Returns unique elements and relative indexes in 1-D tensor, filled with padding num.
    """
    return F.unique_with_pad(x, pad_num)


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
            e_right = e[..., 0:m - offset:1]
            e = P.Concat(1)((e_left, e_right)).astype(dtype)
        elif offset < 0:
            e_upper = F.fill(dtype, (-offset, m), 0)
            e_lower = e[0:n + offset:1, ...]
            e = P.Concat(0)((e_upper, e_lower)).astype(dtype)
    e = P.BroadcastTo(shape)(e)

    prod_val = F.tensor_mul(x, e)
    res = F.reduce_sum(prod_val.astype(mstype.float32), -1)

    begin = ()
    for _ in range(ndim - 2):
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


def isclose(x1, x2, rtol=1e-05, atol=1e-08, equal_nan=False):
    """
    Returns a boolean tensor where two tensors are element-wise equal within a tolerance.
    """
    return F.isclose(x1, x2, rtol, atol, equal_nan)


def inv(x):
    """
    Computes Reciprocal of input tensor element-wise.
    """
    return F.inv(x)


def invert(x):
    """
    Flips all bits of input tensor element-wise.
    """
    return F.invert(x)


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
    dtype = check_astype_dtype_const(dtype)
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
            the flattened input array is used. Defaults to None.
        mode ('raise', 'wrap', 'clip', optional): Defaults to "clip".

            - edge: Pads with the edge values of `arr`.
            - raise: Raises an error;
            - wrap: Wraps around;
            - clip: Clips to the range. 'clip' mode means that all indices that are
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
            is taken as defining the "sequence".
        mode ('raise', 'wrap', 'clip', optional): Specifies how indices outside
            ``[0, n-1]`` will be treated:

            'raise' – raise an error (default);

            'wrap' – wrap around;

            'clip' – clip to the range. 'clip' mode means that all indices that are
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
        side ('left', 'right', optional): If 'left', the index of the first suitable
            location found is given. If 'right', return the last such index. If there is
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

    sort_range = F.make_range(get_log2_size(F.shape_mul(a.shape) + 1))
    for _ in sort_range:
        mid = (i - F.neg_tensor(j)) // 2
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


def fills(x, value):
    """
    Create a tensor of the same shape and type as the input tensor and fill it with specified value.
    """
    return F.fills(x, value)


def ptp(x, axis=None, keepdims=False):
    """
    The name of the function comes from the acronym for "peak to peak".

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
        xmin = const_utils.make_tensor(xmin, x.dtype)
        if x.ndim == 0 and xmin.ndim == 0:
            x = F.maximum(x.reshape((1,)), xmin).squeeze()
        else:
            x = F.maximum(x, xmin)
    if xmax is not None:
        xmax = const_utils.make_tensor(xmax, x.dtype)
        if x.ndim == 0 and xmax.ndim == 0:
            x = F.minimum(x.reshape((1,)), xmax).squeeze()
        else:
            x = F.minimum(x, xmax)
    if is_scalar:
        return x.squeeze()
    if dtype is not None:
        dtype = check_astype_dtype_const(dtype)
        if dtype != x.dtype:
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


def gather_elements(x, dim, index):
    r"""
    Gathers elements along an axis specified by dim.

    Refer to :func:`mindspore.ops.gather_elements` for more detail.
    """
    return F.gather_elements(x, dim, index)


def sum(x, axis=None, dtype=None, keepdims=False, initial=None):  # pylint: disable=redefined-builtin
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
            sub-classes of ndarray, however any non-default value will be. If the sub-class method does not
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
    dtype = check_astype_dtype_const(dtype)
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
    for sub_item, rep in zip(subs, repeats):
        if rep != 0:
            repeated_subs.append(repeat_elements(sub_item, rep, axis))
    return P.Concat(axis)(repeated_subs)


def hardshrink(x, lambd=0.5):
    r"""
    Apply the Hard Shrink function for a tensor. Calculates the output according to the input elements.

    The formula is defined as follows:

    .. math::
        \text{HardShrink}(x) =
        \begin{cases}
        x, & \text{ if } x > \lambda \\
        x, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}

    Args:
        x (Tensor): Input tensor.
        lambd (float): The threshold :math:`\lambda` defined by the Hard Shrink formula. Default: 0.5.

    Returns:
        Tensor, has the same shape and data type as input tensor.

    Raises:
        TypeError: If `lambd` is not a float.
        TypeError: If dtype of the input tensor is neither float16 nor float32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.array([[0.5,  1,  2.0], [0.0533, 0.0776, -2.1233]])
        >>> print(x.hardshrink())
        [[ 0.      1.      2.    ]
        [ 0.      0.     -2.1233]]
    """
    return P.HShrink(lambd)(x)


def soft_shrink(x, lambd=0.5):
    """Apply the soft shrink function for a tensor. Calculates the output according to the input elements."""
    return F.SoftShrink(lambd)(x)


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


@constexpr
def constant_abs(x):
    """Returns the absolute value of the constant."""
    if x is None:
        raise ValueError("For abs(), the input should be a constant or Tensor type.")
    return abs(x)


def ms_abs(x):
    """Implementation of `abs`."""
    if isinstance(x, Tensor):
        return abs_(x)
    return constant_abs(x)


@constexpr
def constant_round(*data):
    """Returns the rounded value of the constant."""
    for x in data:
        if x is None:
            raise ValueError("For round(), the input should be a Tensor or 1-2 constants.")
    return round(*data)


def ms_round(*data):
    """Implementation of `round`."""
    len_data = len(data)
    if len_data <= 0 or len_data > 2:
        const_utils.raise_type_error("round() requires 1 or 2 arguments.")
    if len_data == 1 or data[1] is None:
        x = data[0]
        if isinstance(x, Tensor):
            return round_(x)
        return constant_round(x)
    if isinstance(data[0], Tensor) or isinstance(data[1], Tensor):
        const_utils.raise_type_error("When applying round() to tensor, only one tensor is supported as input.")
    return constant_round(*data)


@constexpr
def cast_to_str(data):
    return str(data)


def str_func(*data):
    """Implementation of `str`."""
    data_len = len(data)
    if data_len >= 2:
        const_utils.raise_type_error("str() requires 0 or 1 arguments.")
    if data_len == 0:
        return ''
    data = data[0]
    if isinstance(data, (CSRTensor, COOTensor, RowTensor)):
        const_utils.raise_type_error("str() does not support sparse tensor input.")
    if not F.isconstant(data):
        const_utils.raise_type_error("str() does not support non-constant input.")
    return cast_to_str(data)


@constexpr
def cast_to_bool(data):
    return bool(data)


def bool_func(*data):
    """Implementation of `bool`."""
    data_len = len(data)
    if data_len >= 2:
        const_utils.raise_type_error("bool() requires 0 or 1 arguments.")
    if data_len == 0:
        return False
    data = data[0]
    if isinstance(data, (CSRTensor, COOTensor, RowTensor)):
        const_utils.raise_type_error("bool() does not support sparse tensor input.")
    if isinstance(data, (Tensor, Tensor_)):
        tensor_shape = F.shape(data)
        tensor_shape_len = len(tensor_shape)
        if tensor_shape_len == 0 or (tensor_shape_len == 1 and tensor_shape[0] == 1):
            return data != 0
        const_utils.raise_value_error("The truth value of an array with several elements is ambiguous.")
    if not F.isconstant(data):
        return len(data) != 0
    return cast_to_bool(data)


@constexpr
def cast_to_int(*data):
    target = data[0]
    if isinstance(target, Tensor_):
        target = Tensor(target, internal=True)
    if len(data) == 1:
        return int(target)
    return int(target, data[1])


def int_func(*data):
    """Implementation of `int`."""
    data_len = len(data)
    if data_len >= 3:
        const_utils.raise_type_error("int() requires 0, 1 or 2 arguments.")
    if data_len == 0:
        return 0
    target = data[0]
    if not F.isconstant(target):
        const_utils.raise_type_error("int() does not support non-constant input.")
    if isinstance(target, (CSRTensor, COOTensor, RowTensor)):
        const_utils.raise_type_error("int() does not support sparse tensor input.")
    return cast_to_int(*data)


@constexpr
def cast_to_float(data):
    if isinstance(data, Tensor_):
        data = Tensor(data, internal=True)
    return float(data)


def float_func(*data):
    """Implementation of `float`."""
    data_len = len(data)
    if data_len >= 2:
        const_utils.raise_type_error("float() requires 0 or 1 arguments.")
    if data_len == 0:
        return 0.0
    data = data[0]
    if not F.isconstant(data):
        const_utils.raise_type_error("float() does not support non-constant input.")
    if isinstance(data, (CSRTensor, COOTensor, RowTensor)):
        const_utils.raise_type_error("float() does not support sparse tensor input.")
    return cast_to_float(data)


def list_func(*data):
    """Implementation of `list`."""
    data_len = len(data)
    if data_len >= 2:
        const_utils.raise_type_error("list() requires 0 or 1 arguments.")
    if data_len == 0:
        return F.make_list()
    data = data[0]
    if isinstance(data, (CSRTensor, COOTensor, RowTensor)):
        const_utils.raise_type_error("list() does not support single sparse tensor input.")
    if not isinstance(data, Tensor) and not hasattr(data, "__ms_iter__"):
        data_type = F.typeof(data)
        const_utils.raise_type_error(str(data_type) + " object is not iterable.")
    if isinstance(data, dict):
        data = data.keys()
    ret = F.make_list()
    for i in range(len(data)):
        ret = ret + F.make_list(data[i])
    return ret


def tuple_func(*data):
    """Implementation of `tuple`."""
    data_len = len(data)
    if data_len >= 2:
        const_utils.raise_type_error("tuple() requires 0 or 1 arguments.")
    if data_len == 0:
        return F.make_tuple()
    data = data[0]
    if isinstance(data, (CSRTensor, COOTensor, RowTensor)):
        const_utils.raise_type_error("tuple() does not support single sparse tensor input.")
    if not isinstance(data, Tensor) and not hasattr(data, "__ms_iter__"):
        data_type = F.typeof(data)
        const_utils.raise_type_error(str(data_type) + " object is not iterable.")
    if isinstance(data, dict):
        data = data.keys()
    ret = F.make_tuple()
    for i in range(len(data)):
        ret = ret + F.make_tuple(data[i])
    return ret


def max_tensor(*data):
    """Get the max of tensor inputs."""
    if len(data) == 1:
        data = data[0]
    max_tensor_data = data[0]
    for input_data in data:
        max_tensor_data = P.Maximum()(max_tensor_data, input_data)
    return max_tensor_data


def get_max_min_data_len(*data):
    """Get the real length of data."""
    len_data = 0
    if isinstance(data, tuple) and len(data) == 1 and isinstance(data[0], (dict, list, tuple)):
        data = data[0]
        if isinstance(data, dict):
            data = iter(data)
    if isinstance(data, (dict, list, tuple)):
        len_data = len(data)
    else:
        const_utils.raise_type_error("max() or min() does not support the data type.")
    return len_data


def get_tensor_num(data):
    """Get the number of tensor in data."""
    tensor_num = 0
    for input_data in data:
        if isinstance(input_data, Tensor):
            tensor_num = tensor_num + 1
    return tensor_num


def exist_tensor(data):
    """Check if tensor exist in sequence."""
    for input_data in data:
        if isinstance(input_data, Tensor):
            return True
        if isinstance(input_data, (list, tuple)):
            if exist_tensor(input_data):
                return True
    return False


def ms_max(*data):
    """Implementation of `max`."""
    len_data = get_max_min_data_len(data)
    if len_data <= 0:
        const_utils.raise_type_error("max() requires 1 argument at least.")
    elif len_data == 1:
        x = data[0]
        if isinstance(x, Tensor):
            return x.max()
        # Deal with Tensor in tuple or list
        if isinstance(x, (list, tuple)):
            tensor_num = get_tensor_num(x)
            if tensor_num == len(x):
                return max_tensor(x)
            if tensor_num != 0:
                const_utils.raise_type_error("max() cannot contain both tensor and non-tensor type.")
            if exist_tensor(x):
                const_utils.raise_type_error("max() cannot support tensor in list or tuple nested now.")
        return max_(x)
    elif len_data >= 2:
        tensor_num = get_tensor_num(data)
        # All inputs is Tensor
        if tensor_num == len_data:
            return max_tensor(*data)
        if tensor_num != 0:
            const_utils.raise_type_error("max() cannot contain both tensor and non-tensor type.")
    return max_(*data)


def min_tensor(*data):
    """Get the min of tensor inputs."""
    if len(data) == 1:
        data = data[0]
    min_tensor_data = data[0]
    for input_data in data:
        min_tensor_data = P.Minimum()(min_tensor_data, input_data)
    return min_tensor_data


def min_list_tuple(seq1, seq2):
    """Get the min of two sequence."""
    if len(seq1) == 0:
        return seq1
    if len(seq2) == 0:
        return seq2
    min_len = min(len(seq1), len(seq2))
    for i in min_len:
        if seq1[i] == seq2[i]:
            continue
        iter_min = ms_min([seq1[i], seq2[i]])
        if iter_min == seq1[i]:
            return seq1
        return seq2
    return seq1


def ms_min(*data):
    """Implementation of `min`."""
    len_data = get_max_min_data_len(data)
    if len_data <= 0:
        const_utils.raise_type_error("min() requires 1 argument at least.")
    elif len_data == 1:
        x = data[0]
        if isinstance(x, Tensor):
            return x.min()
        # Deal with Tensor in tuple or list
        if isinstance(x, (list, tuple)):
            tensor_num = get_tensor_num(x)
            if tensor_num == len(x):
                return min_tensor(x)
            if tensor_num != 0:
                const_utils.raise_type_error("min() cannot contain both tensor and non-tensor type.")
            if exist_tensor(x):
                const_utils.raise_type_error("min() cannot support tensor in list or tuple nested now.")
        return min_(x)
    elif len_data >= 2:
        tensor_num = get_tensor_num(data)
        # All inputs is Tensor
        if tensor_num == len_data:
            return min_tensor(*data)
        if tensor_num != 0:
            const_utils.raise_type_error("min() cannot contain both tensor and non-tensor type.")
    return min_(*data)


def ms_sum(*data):
    """Implementation of `sum`."""
    len_data = len(data)
    if len_data <= 0 or len_data > 2:
        const_utils.raise_type_error("sum() requires 1 or 2 arguments.")
    x = data[0]
    if not isinstance(x, Tensor) and not hasattr(x, "__ms_iter__"):
        data_type = F.typeof(x)
        const_utils.raise_type_error(str(data_type) + " object is not iterable.")
    if isinstance(x, dict):
        x = x.keys()
    result = 0
    if len_data == 2:
        result = data[1]
    if isinstance(x, Tensor):
        result += x.sum(0)
    else:
        for element in x:
            result += element
    return result


@constexpr
def python_len(data):
    """Return the result of python built-in len function"""
    return len(data)


def ms_len(data):
    """Implementation of `len`."""
    if not isinstance(data, Tensor) and F.isconstant(data):
        return python_len(data)
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
    return P.BroadcastTo(shape_(y))(x)


def broadcast_to(x, shape):
    """Broadcasts tensor to a given shape."""
    return P.BroadcastTo(shape)(x)


def expand_dims(x, axis):
    """
    Insert a dimension of shape 1 at the specified axis of Tensor
    """
    check_is_int(axis, 'axis')
    return P.ExpandDims()(x, axis)


def masked_fill(x, mask, value):
    """
    Fills elements of Tensor with value where mask is True.
    """
    check_is_tensor(mask)
    check_type_name('mask', mask.dtype, [mstype.bool_], "Tensor")
    return F.masked_fill(x, mask, value)


def col2im(*inputs):
    """
    inputs: input_x, output_size, kernel_size, dilation, padding_value, stride
    Combines an array of sliding local blocks into a large containing tensor.
    """
    return F.col2im(*inputs)


def narrow(x, axis, start, length):
    """
    Returns a narrowed tensor from input tensor.
    The dimension axis is input from start to start + length.
    """
    return F.narrow(x, axis, start, length)


def to_csr(x):
    """
    Convert a Tensor to CSRTensor.
    Please refer to tensor.py Tensor::to_csr(self) for more details.
    """
    return F.dense_to_sparse_csr(x)


def to_coo(x):
    """
    Convert a Tensor to COOTensor.
    Please refer to tensor.py Tensor::to_coo(self) for more details.
    """
    return F.dense_to_sparse_coo(x)


@constexpr
def check_select_condition(cond_type):
    """
    Check select input condition.
    """
    if isinstance(cond_type, mstype.tensor_type):
        return
    raise TypeError(f"For select, the argument condition should be Tensor, but got {cond_type}.")


@constexpr
def check_select_input(y, x_dtype):
    """
    Check select input x and y.
    """
    if not isinstance(y, (int, float)):
        raise TypeError(f"For 'Tensor.select', the argument 'y' should be Tensor, int or float,"
                        f" but got {type(y)}.")
    if isinstance(y, int) and x_dtype != mstype.int32:
        raise TypeError(f"For 'Tensor.select', if the argument 'y' is int,"
                        f" then the tensor type should be int32 but got {x_dtype}")
    if isinstance(y, float) and x_dtype != mstype.float32:
        raise TypeError(f"For 'Tensor.select', if the argument 'y' is float,"
                        f" then the tensor type should be float32 but got {x_dtype}")


def select(x, condition, y):
    """Returns the selected elements for tensor 'x' and input 'y' according to input 'condition'"""
    check_select_condition(F.typeof(condition))
    if not isinstance(y, Tensor):
        check_select_input(y, x.dtype)
    input_y = y
    if isinstance(y, (int, float)):
        input_y = F.zeros_like(x) + y
        if isinstance(y, int):
            input_y = F.cast(input_y, mstype.int32)
        else:
            input_y = F.cast(input_y, mstype.float32)
    return F.select(condition, x, y)


def view(x, *shape):
    """Reshape tensor, if shape is -1, reshape tensor into one dimension"""
    shape = check_view_shape(shape)
    return F.reshape(x, shape)


def bitwise_and(x, y):
    """Returns bitwise `and` of two tensors element-wise."""
    return F.bitwise_and(x, y)


def bitwise_or(x, y):
    """Returns bitwise `or` of two tensors element-wise."""
    return F.bitwise_or(x, y)


def bitwise_xor(x, y):
    """Returns bitwise `xor` of two tensors element-wise."""
    return F.bitwise_xor(x, y)


def exp(x):
    """Returns exponential of a tensor element-wise."""
    return F.exp(x)


def sqrt(x):
    """Returns sqrt of a tensor element-wise."""
    return F.sqrt(x)


def square(x):
    """Returns square of a tensor element-wise."""
    return F.square(x)


def sub(x, y):
    """Returns sub of a tensor element-wise."""
    return F.sub(x, y)


def tan(x):
    """Returns tangent of `x`."""
    return F.tan(x)


def tanh(x):
    """Returns hyperbolic tangent of `x`."""
    return F.tanh(x)


def cosh(x):
    """
    Computes hyperbolic cosine of `x` element-wise.
    """
    return F.cosh(x)


def ger(x, y):
    """Ger product of `x1` and `x2`.."""
    return F.ger(x, y)


def while_cond(x):
    """For while condition, if the condition is a tensor, the loop will not be unrolled"""
    if F.issubclass_(F.typeof(x), F.typeof(mstype.tensor)):
        is_cond = check_is_tensor_bool_cond(F.shape(x))
        if is_cond:
            return F.cast(x, mstype.bool_)
    return x


def tensor_scatter_add(x, indices, updates):
    """
    Creates a new tensor by adding the values from the positions in `x` indicated by
    `indices`, with values from `updates`. When multiple values are given for the same
    index, the updated result will be the sum of all values.
    """
    return F.tensor_scatter_add(x, indices, updates)


def tensor_scatter_sub(x, indices, updates):
    """
    Creates a new tensor by subtracting the values from the positions in `x` indicated by
    `indices`, with values from `updates`. When multiple values are given for the same
    index, the updated result will be the sum of all values.
    """
    return F.tensor_scatter_sub(x, indices, updates)


def tensor_scatter_mul(input_x, indices, updates):
    """
    Create a new tensor by multiplying the values from the positions in `input_x` indicated by
    `indices`, with values from `updates`. When multiple value are given for the same index,
    the output result will be the division of values.
    """
    return F.tensor_sactter_mul(input_x, indices, updates)


def tensor_sactter_div(input_x, indices, updates):
    """
    Create a new tensor by division the values from the positions in `input_x` indicated by
    `indices`, with values from `updates`. When multiple value are given for the same index,
    the output result will be the division of values.
    """
    return F.tensor_scatter_div(input_x, indices, updates)


def tensor_scatter_max(x, indices, updates):
    """
    By comparing the value at the position indicated by `indices` in `x` with the value in the `updates`,
    the value at the index will eventually be equal to the largest one to create a new tensor.
    """
    return F.tensor_scatter_max(x, indices, updates)


def tensor_scatter_min(x, indices, updates):
    """
    By comparing the value at the position indicated by `indices` in `x` with the value in the `updates`,
    the value at the index will eventually be equal to the smallest one to create a new tensor.
    """
    return F.tensor_scatter_min(x, indices, updates)


def unsorted_segment_min(x, segment_ids, num_segments):
    """Apply the unsorted segment min function for a tensor. Calculates the output according to the input elements."""
    return F.unsorted_segment_min(x, segment_ids, num_segments)


def unsorted_segment_max(x, segment_ids, num_segments):
    """Apply the unsorted segment max function for a tensor. Calculates the output according to the input elements."""
    return F.unsorted_segment_max(x, segment_ids, num_segments)


def unsorted_segment_prod(x, segment_ids, num_segments):
    """Apply the unsorted segment prod function for a tensor. Calculates the output according to the input elements."""
    return F.unsorted_segment_prod(x, segment_ids, num_segments)


def nonzero(x):
    """
    Return a Tensor of the positions of all non-zero values.
    """
    return F.nonzero(x)


def diag(x):
    """
    Constructs a diagonal tensor with a given diagonal values.
    """
    return F.diag(x)


def masked_select(x, mask):
    """
    Returns a new 1-D Tensor which indexes the input tensor according to the boolean mask.
    """
    return F.masked_select(x, mask)


def inplace_update(x, v, indices):
    """
    Update specified rows of x with values in v according to indices.
    """
    return F.inplace_update(x, v, indices)


def coo_to_csr(x):
    """convert coo to csr."""
    row_indices = x.indices[:, 0]
    col_indices = x.indices[:, 1]
    idx_dtype = x.indices.dtype
    row_indices, sort_idx = F.sort(row_indices.astype(mstype.float32))
    row_indices = row_indices.astype(idx_dtype)
    col_indices = col_indices[sort_idx]
    values = x.values[sort_idx]
    indptr = F.coo2csr(row_indices, x.shape[0])
    return CSRTensor(indptr, col_indices, values, x.shape)


def coo_to_dense(x):
    """convert coo to dense."""
    zeros_tensor = F.zeros(x.shape, x.values.dtype)
    return F.tensor_scatter_update(zeros_tensor, x.indices, x.values)


def coo_coalesce(x):
    """Returns the coalesced sparse tensor of the input."""
    shape = const_utils.make_tensor(x.shape)
    res_indices, res_values, _ = P.Coalesce()(x.indices.transpose(), x.values, shape)
    return COOTensor(res_indices.transpose(), res_values, x.shape)


def csr_to_coo(x):
    """convert csr to coo."""
    if x.ndim != 2:
        const_utils.raise_value_error("Currently only support 2-D CSRTensor when converting to COOTensor.")
    row_indices = F.csr2coo(x.indptr, x.values.shape[0])
    coo_indices = P.Stack(1)((row_indices, x.indices))
    return COOTensor(coo_indices, x.values, x.shape)


def csr_to_dense(x):
    """convert csr to dense."""
    return F.csr_to_dense(x)


def random_categorical_(x, num_sample, seed=0, dtype=mstype.int64):
    r"""
    Generates random samples from a given categorical distribution tensor.
    Refer to :func:`mindspore.ops.random_categorical` for more detail.
    """
    validator.check_is_int(num_sample, 'num_sample')
    validator.check_is_int(seed, 'seed')
    return F.random_categorical(x, num_sample, seed, dtype)


@constexpr
def empty_tensor(dtype):
    """Return empty tensor"""
    return Tensor([], dtype)


@constexpr
def get_itemsize(x_type):
    """get itemsize from tensor's dtype."""
    return itemsize_map[x_type]


@constexpr(check=False)
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
check_is_int = constexpr(validator.check_is_int)
check_type_name = constexpr(validator.check_type_name)
check_value_type = constexpr(validator.check_value_type)
check_int = constexpr(validator.check_int)
check_bool = constexpr(validator.check_bool)


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
    return F.matmul(x, y)


def float_bool(x):
    """Implementation of `float_bool`."""
    return x != 0.0


def xdivy(x, y):
    r"""
    Divides the first input tensor by the second input tensor element-wise. Returns zero when `x` is zero.
    """
    return F.xdivy(x, y)


def int_bool(x):
    """Implementation of `int_bool`."""
    return x != 0


def str_bool(x):
    """Implementation of `str_bool`."""
    if x == "":
        return False
    return True


def matrix_determinant(x):
    """Computes the determinant of one or more square matrices."""
    return F.matrix_determinant(x)


def log1p(x):
    r"""
    Returns the natural logarithm of one plus the input tensor element-wise.
    Refer to :func:`mindspore.ops.log1p` for more detail.
    """
    return F.log1p(x)


def logit(x, eps=None):
    r"""
    Calculate the logit of a tensor element-wise. When eps is not None, element in 'x' is clamped to [eps, 1-eps].
    When eps is None, input 'x' is not clamped.

    `x` refer to self tensor.

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
    """

    if eps is None:
        eps = -1.0
    check_value_type('eps', eps, (float,), 'Tensor.logit')
    return F.logit(x, eps)


def log_matrix_determinant(x):
    """Computes the sign and the log of the absolute value of the determinant of one or more square matrices."""
    return F.log_matrix_determinant(x)


def lerp(start, end, weight):
    """Does a linear interpolation of two tensors start and end based on a float or tensor weight."""
    return F.lerp(start, end, weight)


def norm(input_x, axis, p=2, keep_dims=False, epsilon=1e-12):
    """Returns the matrix norm or vector norm of a given tensor."""
    return F.norm(input_x, axis, p, keep_dims, epsilon)


def renorm(input_x, p, dim, maxnorm):
    """
    Renormalizes the sub-tensors along dimension `dim`, and each sub-tensor's p-norm should not exceed the
    'maxnorm'. The values of current sub-tensor don't need change if the p-norm of the sub-tensor is less than
    `maxnorm`. Otherwise the sub-tensor needs to be modified to the original value of the corresponding position
    divided by the p-norm of the substensor and then multiplied by `maxnorm`.
    """
    return F.renorm(input_x, p, dim, maxnorm)


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


def ceil(x):
    """
    Rounds a tensor up to the closest integer element-wise.
    """
    return F.ceil(x)


def top_k(input_x, k, sorted=True):
    """
    Finds values and indices of the `k` largest entries along the last dimension.
    """
    check_is_int(k, 'k')
    check_bool(sorted, 'sorted')
    return F.top_k(input_x, k, sorted)


#############
# Iteration #
#############


@ms_class
class SequenceIterator:
    """
    SequenceIterator is a util class for iterating sequence object.

    Iterator to use for sequences like List, Array.
    """

    def __init__(self, idx, seq):
        self.idx = idx
        self.seq = seq

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


def list_append(self_, list_item):
    """Append into list"""
    return _append(self_, list_item)


def list_insert(self_, index, obj):
    """Insert into list"""
    return _insert(self_, index, obj)


def list_pop(self_, index=-1):
    """Pop from list"""
    self_, pop_val = _pop(self_, index)
    return self_, pop_val


def list_clear(self_):
    """Clear the list"""
    return _list_clear(self_)


def list_reverse(self_):
    """Reverse the list"""
    return _reverse(self_)


def list_extend(self_, obj):
    """Append another list to the end of the list"""
    return _extend(self_, obj)


def list_count(self_, value):
    """"Count the number of times an element appears in list"""
    return _count(self_, value)


def dict_get(self_, key_index, default_value=None):
    """Get value by key from dict"""
    if not _haskey(self_, key_index):
        return default_value
    return F.dict_getitem(self_, key_index)


def dict_clear(self_):
    """Clear the dict"""
    return _dict_clear(self_)


def dict_haskey(self_, key_index):
    """Check if key is in dict"""
    return _haskey(self_, key_index)


def dict_update(self_, dict_obj):
    """Update the dict"""
    return _update(self_, dict_obj)


def dict_fromkeys(self_, seq, value=None):
    """Check if key is in dict"""
    return _fromkeys(self_, seq, value)


#################
# Array methods #
#################


def to_array(x):
    """Implementation of `to_array`."""
    return x.__ms_to_array__()


def filter_(fun, iter_):
    """Support the use of built-in function filter."""
    result = []
    for elem in iter_:
        if fun(elem):
            result.append(elem)
    return result


##################
# Sparse methods #
##################


def csr_softmax(logits, dtype):
    """Implementation of `sum` for CSRTensor."""
    return F.sparse_matrix_softmax(logits, dtype)


def csr_add(a, b, alpha, beta):
    """Implementation of "csr_add" for CSRTensor."""
    return F.csr_add(a, b, alpha, beta)


def csr_astype(x, dtype):
    """Implementation of `astype` for CSRTensor."""
    data = x.values.astype(dtype)
    return F.make_csr_tensor(x.indptr, x.indices, data, x.shape)


def csr_sum(x, axis):
    """Implementation of `sum` for CSRTensor."""
    return F.csr_reduce_sum(x, axis)


def csr_abs(x):
    """Implementation of `abs` for CSRTensor."""
    data = F.absolute(x.values)
    return F.make_csr_tensor(x.indptr, x.indices, data, x.shape)


def csr_mv(x, dense_vector):
    """Implementation of `mv` for CSRTensor."""
    return F.csr_mv(x, dense_vector)


def csr_mm(x, dense):
    """Implementation of `mm` for CSRTensor."""
    return _csr_mm(x.indptr, x.indices, x.values, x.shape, dense)


def csr_to_tuple(x):
    """Implementation of `to_tuple` for CSRTensor."""
    res = (x.indptr, x.indices, x.values, x.shape)
    return res


def coo_astype(x, dtype):
    """Implementation of `astype` for COOTensor."""
    data = x.values.astype(dtype)
    return F.make_coo_tensor(x.indices, data, x.shape)


def coo_to_tuple(x):
    """Implementation of `to_tuple` for COOTensor."""
    return x.indices, x.values, x.shape


def coo_abs(x):
    """Implementation of `abs` for COOTensor."""
    data = F.absolute(x.values)
    return F.make_coo_tensor(x.indices, data, x.shape)


def coo_add(x, y, thresh):
    """Implementation of `add` for COOTensor."""
    return sparse_add(x, y, thresh)


################
# Sparse Attrs #
################


def sparse_size_(x):
    """
    Return the size of SparseTensor.values. That is the number of non-zero values in SparseTensor.
    """
    return size_(x.values)


def sparse_ndim_(x):
    """
    Return the ndim of SparseTensor, according to its dense shape.
    """
    return F.tuple_len(x.shape)


def bernoulli(x, p=0.5, seed=-1):
    """
    Randomly draws binary numbers from a Bernoulli distribution.
    """
    check_is_int(seed, 'bernoulli', 'seed')
    return F.bernoulli(x, p, seed)


def gather_nd(input_x, indices):
    r"""
    Gathers slices from a tensor by indices.
    Refer to :func:`mindspore.ops.gather_nd` for more detail.
    """
    return F.gather_nd(input_x, indices)


def gather(input_x, input_indices, axis):
    r"""
    Returns the slice of the input tensor corresponding to the elements of `input_indices` on the specified `axis`.
    Refer to :func:`mindspore.ops.gather` for more detail.
    """
    return F.gather(input_x, input_indices, axis)


def split(input_x, axis=0, output_num=1):
    """
    Splits the input tensor into output_num of tensors along the given axis and output numbers.
    Refer to :func:`mindspore.ops.split` for more detail.
    """
    return F.split(input_x, axis, output_num)


def xlogy(x, y):
    r"""
    Computes the first input tensor multiplied by the logarithm of second input tensor element-wise.
    Refer to :func:`mindspore.ops.xlogy` for more details.
    """
    return F.xlogy(x, y)


def erf(x):
    r"""
    Computes the Gauss error function of `x` element-wise.
    Refer to :func:`mindspore.ops.erf` for more detail.
    """
    return F.erf(x)


def erfc(x):
    r"""
    Computes the complementary error function of `x` element-wise.
    Refer to :func:`mindspore.ops.erfc` for more details.
    """
    return F.erfc(x)


def isfinite(x):
    r"""
    Determines which elements are finite for each position.
    Refer to :func:`mindspore.ops.isfinite` for more details.
    """
    return F.isfinite(x)


def cos(x):
    r"""
    Computes cosine of input element-wise.
    """
    return F.cos(x)


def acos(x):
    r"""
    Computes arccosine of input tensors element-wise.
    """
    return F.acos(x)


def asin(x):
    r"""
    Computes arcsine of input tensors element-wise.
    """
    return F.asin(x)


def acosh(x):
    r"""
    Computes inverse hyperbolic cosine of the inputs element-wise.
    """
    return F.acosh(x)


def add(x, y):
    r"""
    Computes the element-wise addition of input tensors.
    """
    return F.add(x, y)


def addr(x, vec1, vec2, beta=1, alpha=1):
    r"""
    Computes the outer-product of `vec1` and `vec2` and adds it to `x`.
    """
    return F.addr(x, vec1, vec2, beta=1, alpha=1)


def addmv(x, mat, vec, beta=1, alpha=1):
    r"""
    Multiplies matrix `mat` and vector `vec`. The vector `x` is added to the final result.
    """
    return F.addmv(x, mat, vec, beta, alpha)


def asinh(x):
    r"""
    Computes inverse hyperbolic sine of the input element-wise.
    """
    return F.asinh(x)


def atan(x):
    r"""
    Computes inverse tangent of the input element-wise.
    """
    return F.atan(x)


def atanh(x):
    r"""
    Computes inverse hyperbolic tangent of the input element-wise.
    """
    return F.atanh(x)


def bmm(input_x, mat2):
    r"""
    Computes  matrix multiplication between two tensors by batch.
    """
    return F.bmm(input_x, mat2)


def value_(x):
    r"""
    Get the value of Parameter or Tensor x. If x is Parameter, will change the type from RefTensor to Tensor.
    """
    return P.Load()(x, monad.U)
