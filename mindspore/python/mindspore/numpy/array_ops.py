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
"""array operations, the function docs are adapted from Numpy API."""
from __future__ import absolute_import

import operator

from mindspore.common import dtype as mstype
from mindspore.common import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.primitive import constexpr, _primexpr
from mindspore.nn import Cell
from mindspore import ops

from mindspore.numpy.utils import _convert_list_tensor_to_tuple_tensor, _expand, _broadcast_to_shape, \
    _check_input_tensor, _broadcast_to, _to_tensor, _callable
from mindspore.numpy.utils_const import _check_axes_range, _check_start_normalize, \
    _raise_type_error, _raise_value_error, _infer_out_shape, _empty, _promote, \
    _check_same_type, _check_axis_valid, _add_unit_axes, _broadcast_tuples, \
    _check_is_float, _check_axis_in_range, _check_axis_type, _canonicalize_axis, \
    _list_comprehensions, _check_element_int, _is_shape_empty, _type_convert, \
    _tuple_slice, _expanded_shape, _seq_prod, _tuple_setitem, _iota, \
    _raise_unimplemented_error, _cumprod, _get_device, _check_is_int


# According to official numpy reference, the dimension of a numpy array must be less
# than 32
MAX_NUMPY_DIMS = 32


def expand_dims(a, axis):
    """
    Expands the shape of a tensor.

    Inserts a new axis that will appear at the axis position in the expanded tensor shape.

    Args:
        a (Tensor): Input tensor array.
        axis (Union[int, list(int), tuple(int)]): Position in the expanded axes where
            the new axis is placed,

    Returns:
        Tensor, with the number of dimensions increased at specified axis.

    Raises:
        TypeError: If input arguments have types not specified above.
        ValueError: If axis exceeds a.ndim.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.ones((2,2))
        >>> x = np.expand_dims(x,0)
        >>> print(x.shape)
        (1, 2, 2)
    """
    _check_input_tensor(a)
    if not isinstance(axis, (int, tuple, list)):
        _raise_type_error("axis must be tuple, list or int, but got ", axis)
    if isinstance(axis, int):
        return F.expand_dims(a, axis)
    ndim = a.ndim + len(axis)
    axis = _canonicalize_axis(axis, ndim)
    for ax in axis:
        a = F.expand_dims(a, ax)
    return a


def squeeze(a, axis=None):
    """
    Removes single-dimensional entries from the shape of a tensor.

    Args:
        a (Tensor): Input tensor array.
        axis (Union[None, int, list(int), tuple(list)]): The axis(axes) to squeeze,
            default is None.

    Returns:
        Tensor, with all or a subset of the dimensions of length :math:`1` removed.

    Raises:
        TypeError: If input arguments have types not specified above.
        ValueError: If specified axis has shape entry :math:`> 1`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.ones((1,2,2,1))
        >>> x = np.squeeze(x)
        >>> print(x.shape)
        (2, 2)
    """
    _check_input_tensor(a)
    return a.squeeze(axis)


def transpose(a, axes=None):
    """
    Reverses or permutes the axes of a tensor; returns the modified tensor.

    Args:
        a (Tensor): a tensor to be transposed
        axes (Union[None, tuple, list]): the axes order, if `axes` is `None`, transpose
            the entire tensor. Default is `None`.

    Returns:
        Tensor, the transposed tensor array.

    Raises:
        TypeError: If input arguments have types not specified above.
        ValueError: If the number of `axes` is not equal to a.ndim.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.ones((1,2,3))
        >>> x = np.transpose(x)
        >>> print(x.shape)
        (3, 2, 1)
    """
    _check_input_tensor(a)
    return a.transpose(axes)


def rollaxis(x, axis, start=0):
    """
    Rolls the specified axis backwards, until it lies in the given position.
    The positions of the other axes do not change relative to one another.

    Args:
        x (Tensor): A Tensor to be transposed.
        axis (int): The axis to be rolled.
        start (int): Default: 0.
            If :math:`start <= axis`, the axis is rolled back until it lies in this position (`start`).
            If :math:`start > axis`: the axis is rolled until it lies before this position (`start`).
            If :math:`start < 0`, the start will be normalized as a non-negative number (more details
            can be seen in the source code.)

            .. table
                +===========+=================+
                |start      |Normalized start |
                +===========+=================+
                |-(x.ndim+1)| raise ValueError|
                +-----------+-----------------+
                |-x.ndim    |0                |
                +-----------+-----------------+
                |...        |...              |
                +-----------+-----------------+
                |-1         |x.ndim-1         |
                +-----------+-----------------+
                |...        |...              |
                +-----------+-----------------+
                |x.ndim     |x.ndim           |
                +-----------+-----------------+
                |x.ndim+1   |raise ValueError |
                +===========+=================+
            ..

    Returns:
        Transposed Tensor. Has the same data type as the original tensor `x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Raises:
        TypeError: If `axis` or `start` is not integer, or `x` is not tensor.
        ValueError: If `axis` is not in the range of :math:`[-ndim, ndim-1]` or
            `start` is not in the range of :math:`[-ndim, ndim]`.

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.ones((2,3,4))
        >>> output = np.rollaxis(x, 0, 2)
        >>> print(output.shape)
        (3, 2, 4)
    """
    _check_input_tensor(x)
    if not isinstance(axis, int):
        _raise_type_error("integer argument expected, but got ", axis)
    if not isinstance(start, int):
        _raise_type_error("integer argument expected, but got ", start)

    shape = F.shape(x)
    ndim = F.tuple_len(shape)

    axis = _check_axes_range(axis, ndim)
    start = _check_start_normalize(start, ndim)
    if start - axis >= 0 and start - axis <= 1:
        return x
    perm = F.make_range(0, ndim)
    new_perm = None
    if start < axis:
        if axis + 1 < ndim:
            new_perm = perm[0:start] + perm[axis:axis+1] + \
                perm[start:axis] + perm[axis+1:]
        else:
            new_perm = perm[0:start] + perm[axis:axis+1] + perm[start:axis]
    if start > axis:
        if start < ndim:
            new_perm = perm[0:axis] + perm[axis+1:start] + \
                perm[axis:axis+1] + perm[start:]
        else:
            new_perm = perm[0:axis] + perm[axis+1:start] + \
                perm[axis:axis+1]

    return F.transpose(x, new_perm)


def swapaxes(x, axis1, axis2):
    """
    Interchanges two axes of a tensor.

    Args:
        x (Tensor): A tensor to be transposed.
        axis1 (int): First axis.
        axis2 (int): Second axis.

    Returns:
        Transposed tensor, has the same data type as the original tensor `x`.

    Raises:
        TypeError: If `axis1` or `axis2` is not integer, or `x` is not tensor.
        ValueError: If `axis1` or `axis2` is not in the range of :math:`[-ndim, ndim-1]`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.ones((2,3,4))
        >>> output = np.swapaxes(x, 0, 2)
        >>> print(output.shape)
        (4,3,2)
    """
    _check_input_tensor(x)
    return x.swapaxes(axis1, axis2)


def reshape(x, new_shape):
    """
    Reshapes a tensor without changing its data.

    Args:
        x (Tensor): A tensor to be reshaped.
        new_shape (Union[int, list(int), tuple(int)]): The new shape should be
            compatible with the original shape. If the tuple has only one element,
            the result will be a 1-D tensor of that length. One shape dimension
            can be :math:`-1`. In this case, the value is inferred from the length of
            the tensor and remaining dimensions.

    Returns:
        Reshaped Tensor. Has the same data type as the original tensor `x`.

    Raises:
        TypeError: If new_shape is not integer, list or tuple, or `x` is not tensor.
        ValueError: If new_shape is not compatible with the original shape.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.asarray([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]])
        >>> output = np.reshape(x, (3, 2))
        >>> print(output)
        [[-0.1  0.3]
         [ 3.6  0.4]
         [ 0.5 -3.2]]
        >>> output = np.reshape(x, (3, -1))
        >>> print(output)
        [[-0.1  0.3]
         [ 3.6  0.4]
         [ 0.5 -3.2]]
        >>> output = np.reshape(x, (6, ))
        >>> print(output)
        [-0.1  0.3  3.6  0.4  0.5 -3.2]
    """
    _check_input_tensor(x)
    return x.reshape(new_shape)


def ravel(x):
    """
    Returns a contiguous flattened tensor.

    A 1-D tensor, containing the elements of the input, is returned.

    Args:
        x (Tensor): A tensor to be flattened.

    Returns:
        Flattened tensor, has the same data type as the original tensor `x`.

    Raises:
        TypeError: If `x` is not tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.ones((2,3,4))
        >>> output = np.ravel(x)
        >>> print(output.shape)
        (24,)
    """
    _check_input_tensor(x)
    return x.ravel()


@_primexpr
def _move_axes_for_concatenate(arr_shape, axis):
    """
    Moves axis 0 to the desiganated position, while keeps other axes' relative
    positions unchanged, only used if a single tensor is concatenated.
    """

    original_axes = tuple(range(len(arr_shape)))
    new_axes = original_axes[1:axis+1] + (0,) + original_axes[axis+1:]
    new_shape = arr_shape[1:axis+1] + (arr_shape[0] * arr_shape[axis+1],) + \
        arr_shape[axis+2:]
    return new_axes, new_shape


def _promote_type_for_concatenate(tuple_of_tensors):
    """
    Checks dtype for all tensors in the tuple. If dtypes are not the same, promote
    them to the `highest` dtype in the tuple, so that they are ready for the concat
    operator.

    Args:
        tuple_of_tensors(tuple(tensor)): A tuple of tensors

    Returns:
        tuple of tensors, with each tensor promoted to ths same dtype.
    """
    need_cast = False
    final_type = tuple_of_tensors[0].dtype

    for tensor in tuple_of_tensors:
        if not _check_same_type(final_type, tensor.dtype):
            need_cast = True
        final_type = _promote(final_type, tensor.dtype)

    if not need_cast:
        return tuple_of_tensors
    tuple_of_casted_tensors = ()
    for tensor in tuple_of_tensors:
        tuple_of_casted_tensors += (tensor.astype(final_type, False),)
    return tuple_of_casted_tensors


def concatenate(arrays, axis=0):
    """
    Joins a sequence of tensors along an existing axis.

    Note:
        To match Numpy behaviour, :math:`axis >= 32` will not cause value error, the
        `axis` will be treated as :class:`None` instead.

    Args:
        arrays (Union[Tensor, tuple(Tensor), list(Tensor)]): a tensor or a list
            of tensors to be concatenated.
        axis (Union[None, int], optional): The axis along which the tensors will be joined,
            if `axis` is :class:`None`, tensors are flattened before use. Default is 0.

    Returns:
        A tensor concatenated from a tensor or a list of tensors.

    Raises:
        TypeError: If input arguments have types not specified above.
        ValueError: If `axis` is not in the range of :math:`[-ndim, ndim-1]`, and less than 32.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x1 = np.ones((1,2,3))
        >>> x2 = np.ones((1,2,1))
        >>> x = np.concatenate((x1, x2), axis=-1)
        >>> print(x.shape)
        (1, 2, 4)
    """
    if isinstance(arrays, Tensor):
        # if only one tensor is provided, it is treated as a tuple along the
        # first dimension. For example, a tensor of shape (3,4,5) will be treated
        # as: tuple(tensor_1(4,5), tensor_2(4,5), tensor_3(4,5))
        if axis is None or axis >= MAX_NUMPY_DIMS:
            return ravel(arrays)
        arr_shape = F.shape(arrays)
        _check_axes_range((axis,), len(arr_shape))
        # move axis 0 to the disiganated position, while keep other axes' relative
        # positions unchanged
        new_axes, new_shape = _move_axes_for_concatenate(arr_shape, axis)
        arrays = transpose(arrays, new_axes)
        arrays = reshape(arrays, new_shape)
        return arrays

    flattened_arrays = ()
    if axis is None or axis >= MAX_NUMPY_DIMS:
        for arr in arrays:
            flattened_arrays += (ravel(arr),)
        axis = -1
        flattened_arrays = _promote_type_for_concatenate(flattened_arrays)
        return P.Concat(axis)(flattened_arrays)

    # convert a list of tensor to a tuple of tensor
    arrays = _convert_list_tensor_to_tuple_tensor(arrays)

    arr_shape = F.shape(arrays[0])
    _check_axes_range((axis,), len(arr_shape))

    # if only one tensor in the tuple/list, return the tensor itself
    if len(arrays) == 1:
        return arrays[0]

    arrays = _promote_type_for_concatenate(arrays)
    return P.Concat(axis)(arrays)


def append(arr, values, axis=None):
    """
    Appends values to the end of a tensor.

    Args:
        arr (Tensor): Values are appended to a copy of this tensor.
        values (Tensor): These values are appended to a copy of `arr`. It must be of
            the correct shape (the same shape as `arr`, excluding `axis`). If `axis` is
            not specified, `values` can be any shape and will be flattened before use.
        axis (None, int, optional): The `axis` along which values are appended. If `axis` is not
            given, both `arr` and `values` are flattened before use, default is :class:`None`.

    Returns:
        Tensor, a copy of tensor with values appended to axis.

    Raises:
        TypeError: If input arguments have types not specified above.
        ValueError: If specified axis exceeds `arr.ndim`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.ones((2, 3))
        >>> b = np.ones((2, 1))
        >>> print(np.append(a, b, axis=1).shape)
        (2, 4)
    """
    _check_input_tensor(arr)
    _check_input_tensor(values)
    if axis is None:
        arr = arr.ravel()
        values = values.ravel()
    else:
        _check_axis_in_range(axis, arr.ndim)
    if F.rank(arr) != F.rank(values):
        _raise_value_error("all tensors must have same number of dimensions")
    return concatenate((arr, values), axis)


def column_stack(tup):
    """
    Stacks 1-D tensors as columns into a 2-D tensor. 2-D tensors are stacked as-is,
    like np.hstack.

    Args:
        tup (Union[Tensor, tuple, list]): A sequence of 1-D or 2-D tensors. All
            of them must have the same shape except the axis to be concatenated.

    Returns:
        2-D Tensor, formed by stacking the given tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Raises:
        TypeError: If `tup` is not Tensor, list or tuple.
        ValueError: If `tup` is empty.

    Examples:
        >>> import mindspore.numpy as np
        >>> x1 = np.array([1, 2, 3]).astype('int32')
        >>> x2 = np.array([4, 5, 6]).astype('int32')
        >>> output = np.column_stack((x1, x2))
        >>> print(output)
        [[1 4]
         [2 5]
         [3 6]]
    """
    if isinstance(tup, Tensor):
        return tup
    if not isinstance(tup, (list, tuple)):
        _raise_type_error("Tensor or, list or tuple of tensors are required, but got ", tup)

    trans_tup = ()
    for tensor in tup:
        if tensor.ndim < 1:
            tensor = F.expand_dims(tensor, 0)
        if tensor.ndim == 1:
            tensor = F.expand_dims(tensor, 1)
        trans_tup += (tensor,)
    if not trans_tup:
        _raise_value_error("Need at least one tensor to concatenate.")
    return P.Concat(1)(trans_tup)


def vstack(tup):
    """
    Stacks tensors in sequence vertically.
    This is equivalent to concatenation along the first axis. 1-D tensors should firstly be reshaped to `(1, N)`,
    and then be concatenated along the first axis.

    Args:
        tup (Union[Tensor, tuple, list]): A sequence of 1-D or 2-D tensors. The tensors must have the same shape
            along all but the first axis. 1-D tensors must have the same shape.

    Returns:
        Stacked Tensor, formed by stacking the given tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Raises:
        TypeError: If `tup` is not Tensor, list or tuple.
        ValueError: If `tup` is empty.

    Examples:
        >>> import mindspore.numpy as np
        >>> x1 = np.array([1, 2, 3]).astype('int32')
        >>> x2 = np.array([4, 5, 6]).astype('int32')
        >>> output = np.vstack((x1, x2))
        >>> print(output)
        [[1 2 3]
         [4 5 6]]
    """
    if isinstance(tup, Tensor):
        return tup
    if not isinstance(tup, (list, tuple)):
        _raise_type_error("Tensor or, list or tuple of tensors are required, but got", tup)

    trans_tup = ()
    for tensor in tup:
        if tensor.ndim <= 1:
            tensor = _expand(tensor, 2, 0)
        trans_tup += (tensor,)
    if not trans_tup:
        _raise_value_error("Need at least one tensor to concatenate.")
    return P.Concat(0)(trans_tup)


def hstack(tup):
    """
    Stacks tensors in sequence horizontally.
    This is equivalent to concatenation along the second axis, except for 1-D tensors
    where it concatenates along the first axis.

    Args:
        tup (Union[Tensor, tuple, list]): A sequence of 1-D or 2-D tensors. The
            tensors must have the same shape along all but the second axis, except
            1-D tensors which can be any length.

    Returns:
        Stacked Tensor, formed by stacking the given tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Raises:
        TypeError: If `tup` is not Tensor, list or tuple.
        ValueError: If `tup` is empty.

    Examples:
        >>> import mindspore.numpy as np
        >>> x1 = np.array([1, 2, 3]).astype('float32')
        >>> x2 = np.array([4, 5, 6]).astype('float32')
        >>> output = np.hstack((x1, x2))
        >>> print(output)
        [1. 2. 3. 4. 5. 6.]
    """
    if isinstance(tup, Tensor):
        return tup
    if not isinstance(tup, (list, tuple)):
        _raise_type_error("Tensor or, list or tuple of tensors are required, but got", tup)

    tuple_of_tensor = ()
    for tensor in tup:
        if tensor.ndim < 1:
            tensor = F.expand_dims(tensor, 0)
        tuple_of_tensor += (tensor,)
    if not tuple_of_tensor:
        _raise_value_error("Need at least one tensor to concatenate.")
    if tuple_of_tensor[0].ndim <= 1:
        return P.Concat(0)(tuple_of_tensor)
    return P.Concat(1)(tuple_of_tensor)


def dstack(tup):
    """
    Stacks tensors in sequence depth wise (along the third axis).
    This is equivalent to concatenation along the third axis. 1-D tensors :math:`(N,)` should be
    reshaped to :math:`(1,N,1)`.
    2-D tensors :math:`(M,N)` should be reshaped to :math:`(M,N,1)` before concatenation.

    Args:
        tup (Union[Tensor, tuple, list]): A sequence of tensors. The tensors must have the same shape along all but
            the third axis. 1-D or 2-D tensors must have the same shape.

    Returns:
        Stacked Tensor, formed by stacking the given tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Raises:
        TypeError: If `tup` is not Tensor, list or tuple.
        ValueError: If `tup` is empty.

    Examples:
        >>> import mindspore.numpy as np
        >>> x1 = np.array([1, 2, 3]).astype('float32')
        >>> x2 = np.array([4, 5, 6]).astype('float32')
        >>> output = np.dstack((x1, x2))
        >>> print(output)
        [[[1. 4.]
          [2. 5.]
          [3. 6.]]]
    """
    if isinstance(tup, Tensor):
        return tup
    if not isinstance(tup, (list, tuple)):
        _raise_type_error("Tensor or list or tuple of tensors are required, but got", tup)

    trans_tup = ()
    for tensor in tup:
        if tensor.ndim <= 1:
            tensor = _expand(tensor, 2, 0)
        if tensor.ndim == 2:
            tensor = F.expand_dims(tensor, 2)
        trans_tup += (tensor,)
    if not trans_tup:
        _raise_value_error("Need at least one tensor to concatenate.")
    return P.Concat(2)(trans_tup)


def where(condition, x=None, y=None):
    """
    Returns elements chosen from `x` or `y` depending on `condition`.

    Note:
        As nonzero is not supported, both `x` and `y` must be provided Tensor
    input.

    Args:
        condition (Tensor): where True, yield `x`, otherwise yield `y`.
        x (Tensor): Values from which to choose. Defaults to None.
        y (Tensor): Values from which to choose. `x`, `y` and `condition` need
            to be broadcastable to some shape. Defaults to None.

    Returns:
        Tensor or scalar, with elements from `x` where `condition` is True, and
        elements from `y` elsewhere.

    Raises:
        ValueError: If operands cannot be broadcast.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> condition = np.full((1, 1, 2), [False, True])
        >>> x = np.full((1, 3, 2), 5)
        >>> y = np.full((2, 1, 1), 7)
        >>> output = np.where(condition, x, y)
        >>> print(output)
        [[[7 5]
        [7 5]
        [7 5]]
        [[7 5]
        [7 5]
        [7 5]]]
    """
    condition, x, y = _to_tensor(condition, x, y)
    # type promotes input tensors
    dtype1 = F.dtype(x)
    dtype2 = F.dtype(y)
    dtype = _promote(dtype1, dtype2)
    if not _check_same_type(dtype1, dtype):
        x = F.cast(x, dtype)
    if not _check_same_type(dtype2, dtype):
        y = F.cast(y, dtype)
    is_bool = _check_same_type(dtype1, mstype.bool_) and _check_same_type(dtype2, mstype.bool_)
    if is_bool:
        # select does not support bool type for x or y
        x = F.cast(x, mstype.float32)
        y = F.cast(y, mstype.float32)

    if not _check_same_type(F.dtype(condition), mstype.float32):
        # tiling with bool is not supported on GPU
        condition = F.cast(condition, mstype.float32)
    dynamic = F.is_sequence_value_unknown(F.shape(condition)) or F.is_sequence_value_unknown(F.shape(x))\
              or F.is_sequence_value_unknown(F.shape(y))
    # As select op currently does not support broadcast, broadcasts input tensors
    if not dynamic:
        shape_out = _infer_out_shape(F.shape(condition),
                                     F.shape(x), F.shape(y))
        condition = _broadcast_to_shape(condition, shape_out)
        x = _broadcast_to_shape(x, shape_out)
        y = _broadcast_to_shape(y, shape_out)
    else:
        # Get the broadcast shape through broadcast calculation
        add_x_y = x + y
        add_out = condition + F.cast(add_x_y, condition.dtype)
        shape_out = P.TensorShape()(add_out)
        condition = ops.broadcast_to(condition, shape_out)
        x = ops.broadcast_to(x, shape_out)
        y = ops.broadcast_to(y, shape_out)

    if not _check_same_type(F.dtype(condition), mstype.bool_):
        condition = F.cast(condition, mstype.bool_)
    res = F.select(condition, x, y)
    if is_bool:
        res = F.cast(res, mstype.bool_)
    return res


def _atleast_xd(ndim, arys):
    """Returns arys with at least ndim."""
    _check_input_tensor(*arys)
    res = []
    for arr in arys:
        arr = _expand(arr, ndim)
        res.append(arr)
    if len(res) == 1:
        return res[0]
    return res


def atleast_1d(*arys):
    """
    Converts inputs to arrays with at least one dimension.

    Scalar inputs are converted to 1-dimensional arrays, whilst
    higher-dimensional inputs are preserved.

    Note:
        In graph mode, returns a tuple of tensor instead of a list of
        tensors.

    Args:
        *arys (Tensor): one or more input tensors.

    Returns:
        Tensor, or list of tensors, each with ``a.ndim >= 1``.

    Raises:
        TypeError: If the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.ones((2, 3))
        >>> b = np.ones(())
        >>> c = np.ones(5)
        >>> output = np.atleast_1d(a, b, c)
        >>> print(output)
            [Tensor(shape=[2, 3], dtype=Float32, value=
            [[1.00000000e+00, 1.00000000e+00, 1.00000000e+00],
            [1.00000000e+00, 1.00000000e+00, 1.00000000e+00]]),
            Tensor(shape=[1], dtype=Float32, value= [1.00000000e+00]),
            Tensor(shape=[5], dtype=Float32,
            value= [1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
            1.00000000e+00, 1.00000000e+00])]
    """
    return _atleast_xd(1, arys)


def atleast_2d(*arys):
    """
    Reshapes inputs as arrays with at least two dimensions.

    Note:
        In graph mode, returns a tuple of tensor instead of a list of
        tensors.
    Args:
        *arys (Tensor): one or more input tensors.

    Returns:
        Tensor, or list of tensors, each with ``a.ndim >= 2``.

    Raises:
        TypeError: If the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.ones((2, 3))
        >>> b = np.ones(())
        >>> c = np.ones(5)
        >>> output = np.atleast_2d(a, b, c)
        >>> print(output)
            [Tensor(shape=[2, 3], dtype=Float32, value=
            [[1.00000000e+00, 1.00000000e+00, 1.00000000e+00],
            [1.00000000e+00, 1.00000000e+00, 1.00000000e+00]]),
            Tensor(shape=[1, 1], dtype=Float32, value= [[1.00000000e+00]]),
            Tensor(shape=[1, 5], dtype=Float32,
            value= [[1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
            1.00000000e+00, 1.00000000e+00]])]
    """
    return _atleast_xd(2, arys)


def atleast_3d(*arys):
    """
    Reshapes inputs as arrays with at least three dimensions.

    Note:
        In graph mode, returns a tuple of tensor instead of a list of
        tensors.

    Args:
        *arys (Tensor): one or more input tensors.

    Returns:
        Tensor, or list of tensors, each with ``a.ndim >= 3``. For example,
        a 1-D array of shape `(N,)` becomes a tensor of shape `(1, N, 1)`, and
        a 2-D array of shape `(M, N)` becomes a tensor of shape `(M, N, 1)`.

    Raises:
        TypeError: If the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.ones((2, 3))
        >>> b = np.ones(())
        >>> c = np.ones(5)
        >>> output = np.atleast_3d(a, b, c)
        >>> print(output)
            [Tensor(shape=[2, 3, 1], dtype=Float32, value=
            [[[1.00000000e+00], [1.00000000e+00], [1.00000000e+00]],
            [[1.00000000e+00], [1.00000000e+00], [1.00000000e+00]]]),
            Tensor(shape=[1, 1, 1], dtype=Float32, value= [[[1.00000000e+00]]]),
            Tensor(shape=[1, 5, 1], dtype=Float32,
            value= [[[1.00000000e+00], [1.00000000e+00], [1.00000000e+00],
            [1.00000000e+00], [1.00000000e+00]]])]
    """
    res = []
    for arr in arys:
        ndim = F.rank(arr)
        if ndim == 0:
            arr = F.reshape(arr, (1, 1, 1))
        elif ndim == 1:
            arr = F.reshape(arr, (1, F.size(arr), 1))
        elif ndim == 2:
            arr = F.reshape(arr, F.shape(arr) + (1,))
        res.append(arr)
    if len(res) == 1:
        return res[0]
    return res


def stack(arrays, axis=0):
    """
    Joins a sequence of arrays along a new axis.

    The `axis` parameter specifies the index of the new axis in the
    dimensions of the result. For example, if ``axis=0`` it will be the
    first dimension and if ``axis=-1`` it will be the last dimension.

    Note:
        Numpy argument out is not supported.

    Args:
        arrays (sequence of Tensor): Each array must have the same shape.
        axis (int, optional): The axis in the result array along which the
            input arrays are stacked. Default: 0.

    Returns:
        Tensor, The stacked array has one more dimension than the input
        arrays.

    Raises:
        ValueError: If input is not Tensor, tuple, or list.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> arrays = [np.ones((3, 4)) for _ in range(10)]
        >>> output = np.stack(arrays, axis=0)
        >>> print(output.shape)
        (10, 3, 4)
        >>> output = np.stack(arrays, axis=1)
        >>> print(output.shape)
        (3, 10, 4)
        >>> output = np.stack(arrays, axis=2)
        >>> print(output.shape)
        (3, 4, 10)
    """

    if isinstance(arrays, Tensor):
        shape = F.shape(arrays)
        ndim = F.rank(arrays)
        axis = axis % ndim
        axes = F.make_range(ndim)
        perm = axes[1:axis+1] + (0,) + axes[axis+1:]
        if _is_shape_empty(shape):
            return _empty(mstype.float32, shape[1:axis+1] + (shape[0],) + shape[axis+1:])
        return transpose(arrays, perm)

    if isinstance(arrays, (list, tuple)):
        shape = (len(arrays),) + F.shape(arrays[0])
        ndim = len(shape)
        axis = axis % ndim
        if _is_shape_empty(shape):
            return _empty(mstype.float32, shape[1:axis+1] + (shape[0],) + shape[axis+1:])
        seq = ()
        for arr in arrays:
            seq += (F.expand_dims(arr, axis),)
        return concatenate(seq, axis)
    return _raise_value_error('input arrays must be Tensor, tuple, or list')


class UniqueNet(Cell):
    """The operation is wrapped inside a model. """

    def __init__(self):
        super(UniqueNet, self).__init__()
        self.unique = P.Unique()

    def construct(self, x):
        return self.unique(x)


def unique(x, return_inverse=False):
    """
    Finds the unique elements of a tensor. The input tensor will be flattened first
    when it has more than one dimension.

    Note:
        Numpy arguments `axis`, `return_index` and `return_counts` are not supported.
        On CPU, this operator must be executed in graph mode.

    Args:
        x (Tensor): The input tensor to be processed.
        return_inverse (bool): If `True`, also return the indices of the unique tensor.
            Default: `False`.

    Returns:
        Tensor or tuple of Tensors.
        If `return_inverse` is `False`, return the unique tensor, otherwise return tuple of tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Raises:
        TypeError: If `x` is not tensor.

    Examples:
        >>> import mindspore.numpy as np
        >>> import mindspore as ms
        >>> ms.set_context(mode=ms.GRAPH_MODE)
        >>> input_x = np.asarray([1, 2, 2, 2, 3, 4, 5]).astype('int32')
        >>> output_x = np.unique(input_x)
        >>> print(output_x)
        [1 2 3 4 5]
        >>> output_x = np.unique(input_x, return_inverse=True)
        >>> print(output_x)
        (Tensor(shape=[5], dtype=Int32, value= [ 1, 2, 3, 4, 5]), Tensor(shape=[7], dtype=Int32,
            value= [0, 1, 1, 1, 2, 3, 4]))
    """
    _check_input_tensor(x)
    if F.tuple_len(F.shape(x)) > 1:
        x = ravel(x)
    uniq = UniqueNet()
    res = uniq(x)
    if not return_inverse:
        return res[0]
    return res


def roll_along_axis(a, shift, axis):
    """
    Rolls a tensor along a given axis. This is a helper function of np.roll.

    Args:
        a (Tensor): Input tensor.
        shift (int): The number of places the tensor is shifted.
        axis (int): The designated axis for shifting.

    Returns:
        Shifted tensor.
    """
    _check_axis_in_range(axis, a.ndim)
    _check_element_int((shift, axis))
    if axis < 0:
        axis += a.ndim
    shift = -(shift % a.shape[axis])
    # if shift is 0, we do not need to roll at all
    if shift == 0:
        return a
    begin1 = ()
    begin2 = ()
    end1 = ()
    end2 = ()
    stride = _list_comprehensions(a.ndim, 1, True)
    for i in F.make_range(a.ndim):
        if i != axis:
            begin1 += (0,)
            end1 += (a.shape[i],)
            begin2 += (0,)
            end2 += (a.shape[i],)
        else:
            begin1 += (shift,)
            end1 += (a.shape[i],)
            begin2 += (0,)
            end2 += (shift,)
    return append(F.strided_slice(a, begin1, end1, stride),
                  F.strided_slice(a, begin2, end2, stride), axis=axis)


def roll(a, shift, axis=None):
    """
    Rolls a tensor along given axes.

    Elements that rolls beyond the last position are re-introduced at the first.

    Args:
        a (Tensor): Input tensor.
        shift (Union[int, tuple(int)]): The number of places by which elements are
            shifted. If a tuple, then axis must be a tuple of the same size, and
            each of the given axes is shifted by the corresponding number. If shift
            is an int while axis is a tuple of integers, then the same value is used
            for all given axes.
        axis (Union[int, tuple(int)], optional): Axis or axes along which elements
            are shifted. By default, the array is flattened before shifting, after
            which the original shape is restored. Default: None.

    Returns:
        Tensor, with the same shape as a.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Raises:
        TypeError: If input arguments have types not specified above.
        ValueError: If axis exceeds `a.ndim`, or `shift` and `axis` cannot broadcast.

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.reshape(np.arange(12), (3, 4))
        >>> print(np.roll(a, [2,-3], [0,-1]))
            [[ 7  4  5  6]
             [11  8  9 10]
             [ 3  0  1  2]]
    """
    _check_input_tensor(a)
    original_shape = a.shape
    original_dtype = a.dtype
    restore_shape = False
    # F.strided_slice only supports float on cpu, this will change once more supports
    # are added.
    if not _check_is_float(original_dtype):
        if not original_dtype in (mstype.complex64, mstype.complex128):
            a = a.astype(mstype.float32)
    if axis is None:
        restore_shape = True
        axis = 0
        a = a.ravel()
    # Broadcast shift and axis to the same length
    shift, axis = _broadcast_tuples(shift, axis)
    for shift_each, axis_each in zip(shift, axis):
        a = roll_along_axis(a, shift_each, axis_each)
    if restore_shape:
        a = a.reshape(original_shape)
    if not _check_is_float(original_dtype):
        if not original_dtype in (mstype.complex64, mstype.complex128):
            a = a.astype(original_dtype)
    return a


@constexpr
def _get_moved_perm(ndim, source, destination):
    """
    Helper function for moveaxis, returns permutation after moving axes
    from source to destination.
    """
    dest_sorted_idx = [i for i, _ in sorted(enumerate(destination), key=operator.itemgetter(1))]
    axes_orig = [i for i in range(ndim) if i not in source]

    k = 0
    m = 0
    perm = []
    for i in dest_sorted_idx:
        # inserts an axis that has been moved, denoted by n, and axes that remain
        # in their original position, indexed from k to k + n - m, into index m in
        # the list of permuted axes
        n = destination[i]
        j = k + n - m
        perm += axes_orig[k:j]
        perm.append(source[i])
        k += n - m
        m = n + 1
    perm += axes_orig[k:]
    return tuple(perm)


@_primexpr
def _get_moved_shape(shape, perm):
    """
    Helper function for moveaxis, returns the permuated shape after
    applying perm.
    """
    return tuple([shape[i] for i in perm])


def moveaxis(a, source, destination):
    """
    Moves axes of an array to new positions.

    Other axes remain in their original order.

    Args:
        a (Tensor): The array whose axes should be reordered.
        source (int or sequence of ints): Original positions of the
            axes to move. These must be unique.
        destination (int or sequence of ints): Destination positions
            for each of the original axes. These must also be unique.

    Returns:
        Tensor, array with moved axes.

    Raises:
        ValueError: If axes are out of the range of ``[-a.ndim, a.ndim)``, or
            if the axes contain duplicates.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.zeros((3, 4, 5))
        >>> output = np.moveaxis(x, 0, -1)
        >>> print(output.shape)
        (4, 5, 3)
        >>> output = np.moveaxis(x, -1, 0)
        >>> print(output.shape)
        (5, 3, 4)
        >>> output = np.moveaxis(x, [0, 1, 2], [-1, -2, -3])
        >>> print(output.shape)
        (5, 4, 3)
    """
    ndim = F.rank(a)
    source = _check_axis_valid(source, ndim)
    destination = _check_axis_valid(destination, ndim)
    if len(source) != len(destination):
        _raise_value_error('`source` and `destination` arguments must have the same number of elements')
    perm = _get_moved_perm(ndim, source, destination)

    shape = F.shape(a)
    if _is_shape_empty(shape):
        return _empty(F.dtype(a), _get_moved_shape(shape, perm))

    return F.transpose(a, perm)


def tile(a, reps):
    """
    Constructs an array by repeating `a` the number of times given by `reps`.

    If `reps` has length `d`, the result will have dimension of ``max(d, a.ndim)``.
    If ``a.ndim < d``, `a` is promoted to be d-dimensional by prepending new axes.
    So a shape (3,) array is promoted to (1, 3) for 2-D replication, or
    shape (1, 1, 3) for 3-D replication. If this is not the desired behavior,
    promote `a` to d-dimensions manually before calling this function.
    If ``a.ndim > d``, `reps` is promoted to ``a.ndim`` by pre-pending 1's to it. Thus
    for an `a` of shape (2, 3, 4, 5), a `reps` of (2, 2) is treated as (1, 1, 2, 2).

    Args:
        a (Tensor): The input array.
        reps (int or sequence of ints): The number of repetitions of `a` along
            each axis.

    Returns:
        Tensor, the tiled output array.

    Raises:
        TypeError: If the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.array([0, 1, 2])
        >>> output = np.tile(a, 2)
        >>> print(output)
        [0 1 2 0 1 2]
        >>> output = np.tile(a, (2, 2))
        >>> print(output)
        [[0 1 2 0 1 2]
        [0 1 2 0 1 2]]
        >>> output = np.tile(a, (2, 1, 2))
        >>> print(output)
        [[[0 1 2 0 1 2]]
        [[0 1 2 0 1 2]]]
    """
    _check_input_tensor(a)
    ndim = F.rank(a)
    shape = F.shape(a)
    reps = _add_unit_axes(reps, ndim)
    if _is_shape_empty(shape) or _is_shape_empty(reps):
        shape = _add_unit_axes(shape, len(reps))
        return _empty(F.dtype(a), _seq_prod(shape, reps))
    return F.tile(a, reps)


def broadcast_to(array, shape):
    """
    Broadcasts an array to a new shape.

    Args:
        array (Tensor): The array to broadcast.
        shape (tuple): The shape of the desired array.

    Returns:
        Tensor, original array broadcast to the given shape.

    Raises:
        ValueError: If array cannot be broadcast to shape.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Example:
        >>> import mindspore.numpy as np
        >>> x = np.array([1, 2, 3])
        >>> output = np.broadcast_to(x, (3, 3))
        >>> print(output)
        [[1 2 3]
        [1 2 3]
        [1 2 3]]
    """
    return _broadcast_to_shape(array, shape)


def broadcast_arrays(*args):
    """
    Broadcasts any number of arrays against each other.

    Note:
        Numpy argument `subok` is not supported.
        In graph mode, returns a tuple of Tensor instead of a list
        of Tensor.

    Args:
        *args (Tensor): The arrays to broadcast.

    Returns:
        List of Tensor.

    Raises:
        ValueError: If arrays cannot be broadcast.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Example:
        >>> import mindspore.numpy as np
        >>> x = np.array([[1,2,3]])
        >>> y = np.array([[4],[5]])
        >>> output = np.broadcast_arrays(x, y)
        >>> print(output)
        [Tensor(shape=[2, 3], dtype=Int32, value=
        [[1, 2, 3],
        [1, 2, 3]]), Tensor(shape=[2, 3], dtype=Int32, value=
        [[4, 4, 4],
        [5, 5, 5]])]
    """
    shapes = map(F.shape, args)
    out_shape = _infer_out_shape(*shapes)
    res = []
    for arr in args:
        res.append(broadcast_to(arr, out_shape))
    return res


def array_split(x, indices_or_sections, axis=0):
    """
    Splits a tensor into multiple sub-tensors.

    Note:
        Currently, array_split only supports :class:`mindspore.float32` on ``CPU``.

    The only difference between ``np.split`` and ``np.array_split`` is that
    ``np.array_split`` allows indices_or_sections to be an integer that does not
    equally divide the axis. For a tensor of length l that should be split into
    n sections, it returns :math:`l % n` sub-arrays of size :math:`l//n + 1` and
    the rest of size :math:`l//n`.

    Args:
        x (Tensor): A Tensor to be divided.
        indices_or_sections (Union[int, tuple(int), list(int)]):
            If integer, :math:`N`, the tensor will be divided into
            :math:`N` tensors along axis.
            If tuple(int), list(int) or of sorted integers,
            the entries indicate where along axis the array is split.
            For example, :math:`[2, 3]` would, for :math:`axis=0`, result in
            three sub-tensors :math:`x[:2]`, :math:`x[2:3]`and :math:`x[3:]`.
            If an index exceeds the dimension of the array along axis,
            an empty sub-array is returned correspondingly.
        axis (int): The axis along which to split. Default: 0.

    Returns:
        A list of sub-tensors.

    Raises:
        TypeError: If argument `indices_or_sections` is not integer,
            tuple(int) or list(int) or argument `axis` is not integer.
        ValueError: If argument `axis` is out of range of :math:`[-x.ndim, x.ndim)`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> input_x = np.arange(9).astype("float32")
        >>> output = np.array_split(input_x, 4)
        >>> print(output)
        (Tensor(shape=[3], dtype=Float32,
            value= [ 0.00000000e+00,  1.00000000e+00,  2.00000000e+00]),
        Tensor(shape=[2], dtype=Float32,
            value= [ 3.00000000e+00,  4.00000000e+00]),
        Tensor(shape=[2], dtype=Float32,
            value= [ 5.00000000e+00,  6.00000000e+00]),
        Tensor(shape=[2], dtype=Float32,
            value= [ 7.00000000e+00,  8.00000000e+00]))
    """
    return _split(x, indices_or_sections, opname="array_split", axis=axis)


def split(x, indices_or_sections, axis=0):
    """
    Splits a tensor into multiple sub-tensors along the given axis.

    Args:
        x (Tensor): A Tensor to be divided.
        indices_or_sections (Union[int, tuple(int), list(int)]):
            If integer, :math:`N`, the tensor will be divided into
            :math:`N` equal tensors along axis.
            If tuple(int), list(int) or of sorted integers,
            the entries indicate where along axis the array is split.
            For example, :math:`[2, 3]` would, for :math:`axis=0`, result in
            three sub-tensors :math:`x[:2]`, :math:`x[2:3]`and :math:`x[3:]`.
            If an index exceeds the dimension of the array along axis,
            an empty sub-array is returned correspondingly.
        axis (int): The axis along which to split. Default: 0.

    Returns:
        A tuple of sub-tensors.

    Raises:
        TypeError: If argument `indices_or_sections` is not integer,
            tuple(int) or list(int) or argument `axis` is not integer.
        ValueError: If argument `axis` is out of range of :math:`[-x.ndim, x.ndim)`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> input_x = np.arange(9).astype("float32")
        >>> output = np.split(input_x, 3)
        >>> print(output)
        (Tensor(shape=[3], dtype=Float32,
          value= [ 0.00000000e+00,  1.00000000e+00,  2.00000000e+00]),
         Tensor(shape=[3], dtype=Float32,
          value= [ 3.00000000e+00,  4.00000000e+00,  5.00000000e+00]),
         Tensor(shape=[3], dtype=Float32,
          value= [ 6.00000000e+00,  7.00000000e+00,  8.00000000e+00]))
    """
    return _split(x, indices_or_sections, opname="split", axis=axis)


def _split(x, indices_or_sections, opname, axis=0):
    """Splits a tensor based on ``np.split`` or ``np.array_split``."""
    _check_input_tensor(x)
    _ = _check_axis_type(axis, True, False, False)
    axis = _canonicalize_axis(axis, x.ndim)
    res = None
    arr_shape = x.shape
    length_along_dim = arr_shape[axis]
    if isinstance(indices_or_sections, int):
        if indices_or_sections > length_along_dim:
            _raise_value_error("empty tensor encountered.")
        if opname == "split" or length_along_dim % indices_or_sections == 0:
            res = P.Split(axis, indices_or_sections)(x)
        else:
            num_long_tensor = length_along_dim % indices_or_sections
            num_short_tensor = indices_or_sections - num_long_tensor
            length1 = num_long_tensor * (length_along_dim // indices_or_sections + 1)
            length2 = length_along_dim - length1
            start1 = _list_comprehensions(F.rank(x), 0, True)
            size1 = _tuple_setitem(arr_shape, axis, length1)
            start2 = _tuple_setitem(start1, axis, length1)
            size2 = _tuple_setitem(arr_shape, axis, length2)
            res = P.Split(axis, num_long_tensor)(F.tensor_slice(x, start1, size1)) + \
                P.Split(axis, num_short_tensor)(F.tensor_slice(x, start2, size2))

    elif isinstance(indices_or_sections, (list, tuple)) and _check_element_int(indices_or_sections):
        res = _split_sub_tensors(x, indices_or_sections, axis)
    else:
        _raise_type_error("Argument `indices_or_sections` in `mindspore.numpy.split`\
            should be integer, tuple(int) or list(int), but got", indices_or_sections)
    return res


@constexpr
def convert_neg_indices(indices, ndim):
    """converts negative values in tuple/list indices"""
    def canonicalizer(ax):
        return ax + ndim if ax < 0 else ax
    indices = tuple([canonicalizer(axis) for axis in indices])
    return indices


def _split_sub_tensors(x, indices, axis):
    """
    Splits the input tensor `x` into multiple sub-tensors
    along the axis according to the given indices.
    """
    length_along_dim = x.shape[axis]
    indices = convert_neg_indices(indices, length_along_dim)
    indices += (length_along_dim,)

    sub_tensors = []
    strides = _list_comprehensions(x.ndim, 1, True)
    begin = _list_comprehensions(x.ndim, 0)
    end = _list_comprehensions(x.shape)
    for i, idx in enumerate(indices):
        begin[axis] = 0 if i == 0 else indices[i-1]
        end[axis] = idx
        if end[axis] <= begin[axis]:
            _raise_value_error("empty sub-tensor encountered.")
        sliced_tensor = F.strided_slice(x, _type_convert(tuple, begin), _type_convert(tuple, end), strides)
        sub_tensors.append(sliced_tensor)
    return sub_tensors


def vsplit(x, indices_or_sections):
    """
    Splits a tensor into multiple sub-tensors vertically (row-wise).
    It is equivalent to split with :math:`axis=0` (default), the array is always
    split along the first axis regardless of the array dimension.

    Args:
        x (Tensor): A Tensor to be divided.
        indices_or_sections (Union[int, tuple(int), list(int)]):
            If integer, :math:`N`, the tensor will be divided into
            :math:`N` equal tensors along axis.
            If tuple(int), list(int) or of sorted integers,
            the entries indicate where along axis the array is split.
            For example, :math:`[2, 3]` would, for :math:`axis=0`, result in
            three sub-tensors :math:`x[:2]`, :math:`x[2:3]`and :math:`x[3:]`.
            If an index exceeds the dimension of the array along axis,
            an empty sub-array is returned correspondingly.

    Returns:
        A list of sub-tensors.

    Raises:
        TypeError: If argument `indices_or_sections` is not integer.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> input_x = np.arange(9).reshape((3, 3)).astype('float32')
        >>> output = np.vsplit(input_x, 3)
        >>> print(output)
        (Tensor(shape=[1, 3], dtype=Float32,
          value=[[ 0.00000000e+00,  1.00000000e+00,  2.00000000e+00]]),
         Tensor(shape=[1, 3], dtype=Float32,
          value=[[ 3.00000000e+00,  4.00000000e+00,  5.00000000e+00]]),
         Tensor(shape=[1, 3], dtype=Float32,
          value=[[ 6.00000000e+00,  7.00000000e+00,  8.00000000e+00]]))
    """
    return split(x, indices_or_sections, 0)


def hsplit(x, indices_or_sections):
    """
    Splits a tensor into multiple sub-tensors horizontally (column-wise).
    It is equivalent to split with :math:`axis=1` (default), the array is always
    split along the second axis regardless of the array dimension.

    Args:
        x (Tensor): A Tensor to be divided.
        indices_or_sections (Union[int, tuple(int), list(int)]):
            If integer, :math:`N`, the tensor will be divided into
            :math:`N` equal tensors along axis.
            If tuple(int), list(int) or of sorted integers,
            the entries indicate where along axis the array is split.
            For example, :math:`[2, 3]` would, for :math:`axis=0`, result in
            three sub-tensors :math:`x[:2]`, :math:`x[2:3]`and :math:`x[3:]`.
            If an index exceeds the dimension of the array along axis,
            an empty sub-array is returned correspondingly.

    Returns:
        A list of sub-tensors.

    Raises:
        TypeError: If argument `indices_or_sections` is not integer.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> input_x = np.arange(6).reshape((2, 3)).astype('float32')
        >>> output = np.hsplit(input_x, 3)
        >>> print(output)
        (Tensor(shape=[2, 1], dtype=Float32,
        value=[[ 0.00000000e+00],
               [ 3.00000000e+00]]),
        Tensor(shape=[2, 1], dtype=Float32,
        value=[[ 1.00000000e+00],
               [ 4.00000000e+00]]),
        Tensor(shape=[2, 1], dtype=Float32,
        value=[[ 2.00000000e+00],
               [ 5.00000000e+00]]))
    """
    return split(x, indices_or_sections, 1)


def dsplit(x, indices_or_sections):
    """
    Splits a tensor into multiple sub-tensors along the 3rd axis (depth).
    It is equivalent to split with :math:`axis=2` (default), the array is always
    split along the third axis regardless of the array dimension.

    Args:
        x (Tensor): A Tensor to be divided.
        indices_or_sections (Union[int, tuple(int), list(int)]):
            If integer, :math:`N`, the tensor will be divided into
            :math:`N` equal tensors along axis.
            If tuple(int), list(int) or of sorted integers,
            the entries indicate where along axis the array is split.
            For example, :math:`[2, 3]` would, for :math:`axis=0`, result in
            three sub-tensors :math:`x[:2]`, :math:`x[2:3]`and :math:`x[3:]`.
            If an index exceeds the dimension of the array along axis,
            an empty sub-array is returned correspondingly.

    Returns:
        A list of sub-tensors.

    Raises:
        TypeError: If argument `indices_or_sections` is not integer.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> input_x = np.arange(6).reshape((1, 2, 3)).astype('float32')
        >>> output = np.dsplit(input_x, 3)
        >>> print(output)
        (Tensor(shape=[1, 2, 1], dtype=Float32,
        value=[[[ 0.00000000e+00],
                [ 3.00000000e+00]]]),
        Tensor(shape=[1, 2, 1], dtype=Float32,
        value=[[[ 1.00000000e+00],
                [ 4.00000000e+00]]]),
        Tensor(shape=[1, 2, 1], dtype=Float32,
        value=[[[ 2.00000000e+00],
                [ 5.00000000e+00]]]))
    """
    return split(x, indices_or_sections, 2)


@_primexpr
def _get_flip_start(ndim, shape, axes):
    return tuple([shape[i] - 1 if i in axes else 0 for i in range(ndim)])


@_primexpr
def _get_flip_end(ndim, shape, axes):
    return tuple([-shape[i] - 1 if i in axes else shape[i] + 1 for i in range(ndim)])


@_primexpr
def _get_flip_strides(ndim, axes):
    return tuple([-1 if i in axes else 1 for i in range(ndim)])


def flip(m, axis=None):
    """
    Reverses the order of elements in an array along the given axis.

    The shape of the array is preserved, but the elements are reordered.

    Args:
        m (Tensor): Input array.
        axis (None or int or tuple of integers, optional): Axis or axes along which
            to flip over. The default, ``axis=None``, will flip over all of the axes
            of the input array. If `axis` is negative it counts from the last to
            the first axis. If `axis` is a tuple of integers, flipping is performed on
            all of the axes specified in the tuple.

    Returns:
        Tensor, with the entries of `axis` reversed.

    Raises:
        TypeError: If the input is not a tensor.

    Supported Platforms:
        ``GPU`` ``CPU``

    Example:
        >>> import mindspore.numpy as np
        >>> A = np.arange(8.0).reshape((2,2,2))
        >>> output = np.flip(A)
        >>> print(output)
        [[[7. 6.]
        [5. 4.]]
        [[3. 2.]
        [1. 0.]]]
        >>> output = np.flip(A, (0, 2))
        >>> print(output)
        [[[5. 4.]
        [7. 6.]]
        [[1. 0.]
        [3. 2.]]]
    """
    _check_input_tensor(m)
    ndim = F.rank(m)
    axes = _check_axis_valid(axis, ndim)
    shape = F.shape(m)
    dtype = F.dtype(m)
    if _is_shape_empty(shape):
        return m
    if not _check_is_float(dtype):
        m = m.astype(mstype.float32)
    start = _get_flip_start(ndim, shape, axes)
    end = _get_flip_end(ndim, shape, axes)
    strides = _get_flip_strides(ndim, axes)
    res = F.strided_slice(m, start, end, strides)
    if not _check_same_type(F.dtype(res), dtype):
        res = F.cast(res, dtype)
    return res


def flipud(m):
    """
    Flips the entries in each column in the up/down direction.
    Rows are preserved, but appear in a different order than before.

    Args:
        m (Tensor): Input array.

    Returns:
        Tensor.

    Raises:
        TypeError: If the input is not a tensor.

    Supported Platforms:
        ``GPU`` ``CPU``

    Example:
        >>> import mindspore.numpy as np
        >>> A = np.arange(8.0).reshape((2,2,2))
        >>> output = np.flipud(A)
        >>> print(output)
        [[[4. 5.]
        [6. 7.]]
        [[0. 1.]
        [2. 3.]]]
    """
    return flip(m, 0)


def fliplr(m):
    """
    Flips the entries in each row in the left/right direction.
    Columns are preserved, but appear in a different order than before.

    Args:
        m (Tensor): Input array.

    Returns:
        Tensor.

    Raises:
        TypeError: If the input is not a tensor.

    Supported Platforms:
        ``GPU`` ``CPU``

    Example:
        >>> import mindspore.numpy as np
        >>> A = np.arange(8.0).reshape((2,2,2))
        >>> output = np.fliplr(A)
        >>> print(output)
        [[[2. 3.]
        [0. 1.]]
        [[6. 7.]
        [4. 5.]]]
    """
    return flip(m, 1)


def take_along_axis(arr, indices, axis):
    """
    Takes values from the input array by matching 1d index and data slices.

    This iterates over matching 1d slices oriented along the specified axis in the
    index and data arrays, and uses the former to look up values in the latter.
    These slices can be different lengths.

    Args:
        arr (Tensor): Source array with shape `(Ni…, M, Nk…)`.
        indices (Tensor): Indices with shape `(Ni…, J, Nk…)` to take along each 1d
            slice of `arr`. This must match the dimension of `arr`, but dimensions `Ni`
            and `Nj` only need to broadcast against `arr`.
        axis (int): The axis to take 1d slices along. If `axis` is None, the input
            array is treated as if it had first been flattened to 1d.

    Returns:
        Tensor, the indexed result, with shape `(Ni…, J, Nk…)`.

    Raises:
        ValueError: If input array and indices have different number of dimensions.
        TypeError: If the input is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Example:
        >>> import mindspore.numpy as np
        >>> x = np.arange(12).reshape(3, 4)
        >>> indices = np.arange(3).reshape(1, 3)
        >>> output = np.take_along_axis(x, indices, 1)
        >>> print(output)
        [[ 0  1  2]
        [ 4  5  6]
        [ 8  9 10]]
    """
    _check_input_tensor(arr, indices)
    if axis is None:
        arr = ravel(arr)
        axis = 0
    ndim = F.rank(arr)
    if ndim != F.rank(indices):
        _raise_value_error('`indices` and `arr` must have the same number of dimensions')
    axis = _check_axis_in_range(axis, ndim)

    shape_arr = F.shape(arr)
    shape_indices = F.shape(indices)
    # broadcasts indices against the shape of arr except at axis
    indices = _broadcast_to(indices, _tuple_slice(shape_indices, None, axis),
                            _tuple_slice(shape_arr, None, axis), ndim)
    indices = _broadcast_to(indices, _tuple_slice(shape_arr, None, axis + 1) +
                            _tuple_slice(shape_indices, axis + 1, None), shape_arr, ndim)
    arr = _broadcast_to(arr, shape_arr, indices.shape, ndim)
    return F.gather_d(arr, axis, indices)


def _mod(x, y):
    """Computes x mod y."""
    quotient = F.tensor_floordiv(x, y)
    prod = F.tensor_mul(y, quotient)
    return F.tensor_sub(x, prod)


def _check_indices(dims, indices, mode, allow_negative_index=True):
    """Checks whether indices are out of bounds."""
    shape = F.shape(indices)
    dtype = F.dtype(indices)
    if not allow_negative_index:
        lowerbounds = F.fill(dtype, shape, 0)
    else:
        lowerbounds = F.fill(dtype, shape, -dims)
    upperbounds = F.fill(dtype, shape, dims - 1)
    out_of_lowerbounds = F.tensor_lt(indices, lowerbounds)
    out_of_upperbounds = F.tensor_gt(indices, upperbounds)
    if mode == 'raise':
        _raise_unimplemented_error('"raise" mode is not implemented')
    if mode == 'wrap':
        return _mod(indices, F.fill(mstype.float32, shape, dims)).astype(dtype)
    if mode != 'clip':
        _raise_value_error('invalid mode. Expected "raise", "wrap", or "clip"')
    zeros = F.fill(dtype, shape, 0)
    clipped = F.select(out_of_lowerbounds, zeros, indices)
    clipped = F.select(out_of_upperbounds, upperbounds, clipped)
    return clipped


def take(a, indices, axis=None, mode='clip'):
    """
    Takes elements from an array along an axis.

    When axis is not None, this function does the same thing as "fancy" indexing
    (indexing arrays using arrays); however, it can be easier to use if you need
    elements along a given axis. A call such as ``np.take(arr, indices, axis=3)`` is
    equivalent to ``arr[:,:,:,indices,...]``.

    Note:
        Numpy argument out is not supported.
        ``mode = 'raise'`` is not supported, and the default mode is 'clip' instead.

    Args:
        a (Tensor): Source array with shape `(Ni…, M, Nk…)`.
        indices (Tensor): The indices with shape `(Nj...)` of the values to extract.
        axis (int, optional): The axis over which to select values. By default,
            the flattened input array is used. Defaults to None.
        mode ('raise', 'wrap', 'clip', optional): Specifies how out-of-bounds
            indices will behave. Defaults to "clip".

            'raise' – raise an error;

            'wrap' – wrap around;

            'clip' – clip to the range. 'clip' mode means that all indices that are
            too large are replaced by the index that addresses the last element
            along that axis. Note that this disables indexing with negative numbers.

    Returns:
        Tensor, the indexed result.

    Raises:
        ValueError: If axis is out of range.
        TypeError: If the input is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.array([4, 3, 5, 7, 6, 8])
        >>> indices = np.array([0, 1, 4])
        >>> output = np.take(a, indices)
        >>> print(output)
        [4 3 6]
        >>> indices = np.array([[0, 1], [2, 3]])
        >>> output = np.take(a, indices)
        >>> print(output)
        [[4 3]
        [5 7]]
    """
    _check_input_tensor(a, indices)
    return a.take(indices, axis, mode)


def repeat(a, repeats, axis=None):
    """
    Repeats elements of an array.

    Args:
        a (Tensor): Input array.
        repeats (int or sequence of ints): The number of repetitions for each element.
            `repeats` is broadcasted to fit the shape of the given axis.
        axis (int, optional): The axis along which to repeat values. By default,
            use the flattened input array, and return a flat output array. Defaults to None.

    Returns:
        Tensor, output array which has the same shape as `a`, except along the given
        axis.

    Raises:
        ValueError: If axis is out of range.
        TypeError: If input `a` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> output = np.repeat(np.array(3), 4)
        >>> print(output)
        [3 3 3 3]
        >>> x = np.array([[1,2],[3,4]])
        >>> output = np.repeat(x, 2)
        >>> print(output)
        [1 1 2 2 3 3 4 4]
        >>> output = np.repeat(x, 3, axis=1)
        >>> print(output)
        [[1 1 1 2 2 2]
        [3 3 3 4 4 4]]
        >>> output = np.repeat(x, [1, 2], axis=0)
        >>> print(output)
        [[1 2]
        [3 4]
        [3 4]]
    """
    a = _to_tensor(a)
    return a.repeat(repeats, axis)


def rot90(a, k=1, axes=(0, 1)):
    """
    Rotates a tensor by 90 degrees in the plane specified by axes.
    Rotation direction is from the first towards the second axis.

    Args:
        a (Tensor): Input tensor of two or more dimensions.
        k (int): Number of times the tensor is rotated by 90 degrees. Default: 1.
        axes (Union[tuple(int), list(int)]): The tensor is rotated in the plane
            defined by the axes. Default: `(0, 1)`.
            Axes must be different and with the shape of `(2,)`.

    Returns:
        Tensor.

    Raises:
        TypeError: If input `a` is not a Tensor or
            the argument `k` is not integer or
            the argument `axes` is not tuple of integers or list of ints.
        ValueError: If any axis is out of range or
            the length of `axes` is not `2`.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.arange(24).reshape((2, 3, 4))
        >>> output = np.rot90(a)
        >>> print(output)
        [[[ 8  9 10 11]
          [20 21 22 23]]
         [[ 4  5  6  7]
          [16 17 18 19]]
         [[ 0  1  2  3]
          [12 13 14 15]]]
        >>> output = np.rot90(a, 3, (1, 2))
        >>> print(output)
        [[[ 8  4  0]
          [ 9  5  1]
          [10  6  2]
          [11  7  3]]
         [[20 16 12]
          [21 17 13]
          [22 18 14]
          [23 19 15]]]
    """
    _check_input_tensor(a)

    if not isinstance(k, int):
        _raise_type_error("integer argument expected, but got ", k)
    k = k % 4 if k >= 0 else 4 - (-k % 4)

    if not isinstance(axes, (tuple, list)):
        _raise_type_error("tuple(ints) or list(ints) expected, but got ", axes)
    if len(axes) != 2:
        _raise_value_error("len(axes) must be 2.")
    axis1_tmp, axis2_tmp = axes[0], axes[1]
    axis1 = _canonicalize_axis(axis1_tmp, a.ndim)
    axis2 = _canonicalize_axis(axis2_tmp, a.ndim)
    if axis1 == axis2:
        _raise_value_error('Axes must be different.')

    if k == 0:
        return a
    if k == 2:
        return flip(flip(a, axis1), axis2)
    perm = _list_comprehensions(a.ndim)
    perm[axis1], perm[axis2] = perm[axis2], perm[axis1]
    if k == 1:
        return flip(transpose(a, perm), axis1)
    return flip(transpose(a, perm), axis2)


def select(condlist, choicelist, default=0):
    """
    Returns an array drawn from elements in `choicelist`, depending on conditions.

    Args:
        condlist (Union[int, float, bool, list, tuple, Tensor]): The list of conditions
            which determine from which array in `choicelist` the output elements are
            taken. When multiple conditions are satisfied, the first one encountered in
            `condlist` is used.
        choicelist (Union[int, float, bool, list, tuple, Tensor]): The list of arrays
            from which the output elements are taken. It has to be of the same length as
            `condlist`.
        default (scalar, optional): The element inserted in output when all conditions
            evaluate to `False`. Defaults to 0.

    Returns:
        Tensor, the output at position `m` is the `m-th` element of the array in
        `choicelist` where the `m-th` element of the corresponding array in `condlist`
        is `True`.

    Raises:
        ValueError: If ``len(condlist) != len(choicelist)``.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> condlist = [[True, True, True, False, False], [False, False, True, False, True]]
        >>> choicelist = [[0, 1, 2, 3, 4], [0, 1, 4, 9, 16]]
        >>> output = np.select(condlist, choicelist)
        >>> print(output)
        [ 0  1  2  0 16]
    """
    condlist, choicelist = _to_tensor(condlist, choicelist)
    shape_cond = F.shape(condlist)
    shape_choice = F.shape(choicelist)
    if F.rank(condlist) == 0 or F.rank(choicelist) == 0:
        _raise_value_error('input cannot be scalars')
    case_num = shape_cond[0]
    if shape_choice[0] != case_num:
        _raise_value_error('list of cases must be same length as list of conditions')

    case_size_cond = _tuple_slice(shape_cond, 1, None)
    case_size_choice = _tuple_slice(shape_choice, 1, None)
    # performs broadcast over the cases in condlist and choicelist
    case_size = _infer_out_shape(case_size_cond, case_size_choice)
    shape_broadcasted = (case_num,) + case_size
    ndim = len(shape_broadcasted)
    shape_cond_expanded = ((case_num,) + _list_comprehensions(ndim - F.rank(condlist), 1, True) +
                           case_size_cond)
    condlist = _broadcast_to_shape(F.reshape(condlist, shape_cond_expanded), shape_broadcasted)
    shape_choice_expanded = ((case_num,) + _list_comprehensions(ndim - F.rank(choicelist), 1, True) +
                             case_size_choice)
    choicelist = _broadcast_to_shape(F.reshape(choicelist, shape_choice_expanded), shape_broadcasted)

    slice_start = _list_comprehensions(ndim - 1, 0, True)
    slice_size = (1,) + case_size
    dtype = F.dtype(choicelist)
    if isinstance(default, Tensor):
        default_slice = default.astype(F.dtype(choicelist)).reshape(slice_size)
    else:
        default_slice = F.fill(F.dtype(choicelist), slice_size, default)
    for i in range(case_num - 1, -1, -1):
        cond_slice = F.tensor_slice(condlist.astype(mstype.float32), (i,) + slice_start, slice_size)
        choice_slice = F.tensor_slice(choicelist, (i,) + slice_start, slice_size)
        default_slice = F.select(cond_slice.astype(mstype.bool_), choice_slice, default_slice)
    return F.reshape(default_slice, (case_size)).astype(dtype)


@_primexpr
def _get_grid(shape):
    """Returns a grid representing all the indices for an array with the given shape."""
    grids = []
    ndim = len(shape)
    for i in range(ndim):
        dim_grid = _iota(mstype.int32, shape[i])
        dim_shape = _expanded_shape(ndim, shape[i], i)
        dim_grid = _broadcast_to_shape(dim_grid.reshape(dim_shape), shape)
        grids.append(dim_grid)
    return stack(grids, -1)


def choose(a, choices, mode='clip'):
    """
    Construct an array from an index array and a list of arrays to choose from.
    Given an "index" array `a` of integers and a sequence of n arrays (choices),
    `a` and each choice array are first broadcast, as necessary, to arrays of a
    common shape; calling these `Ba` and `Bchoices[i], i = 0,…,n-1` we have that,
    necessarily, ``Ba.shape == Bchoices[i].shape`` for each `i`. Then, a new array
    with ``shape Ba.shape`` is created as follows:

    - if ``mode='raise'`` (the default), then, first of all, each element of `a`
      (and thus `Ba`) must be in the range `[0, n-1]`; now, suppose that `i`
      (in that range) is the value at the `(j0, j1, ..., jm)` position in
      `Ba` - then the value at the same position in the new array is the
      value in ``Bchoices[i]`` at that same position;

    - if ``mode='wrap'``, values in `a` (and thus `Ba`) may be any (signed)
      integer; modular arithmetic is used to map integers outside the
      range ``[0, n-1]`` back into that range; and then the new array is
      constructed as above;

    - if ``mode='clip'``, values in `a` (and thus `Ba`) may be any (signed) integer;
      negative integers are mapped to 0; values greater than `n-1` are mapped to
      `n-1`; and then the new array is constructed as above.

    Note:
        Numpy argument `out` is not supported.
        ``mode = 'raise'`` is not supported, and the default mode is 'clip' instead.

    Args:
        a (int array): This array must contain integers in ``[0, n-1]``, where `n` is
            the number of choices, unless ``mode=wrap`` or ``mode=clip``, in which
            cases any integers are permissible.
        choices (sequence of arrays): Choice arrays. `a` and all of the `choices` must
            be broadcastable to the same shape. If `choices` is itself an array, then
            its outermost dimension (i.e., the one corresponding to ``choices.shape[0]``)
            is taken as defining the "sequence".
        mode ('raise', 'wrap', 'clip', optional): Specifies how indices outside
            ``[0, n-1]`` will be treated:

            'raise' – raise an error;

            'wrap' – wrap around;

            'clip' – clip to the range. 'clip' mode means that all indices that are
            too large are replaced by the index that addresses the last element
            along that axis. Note that this disables indexing with negative numbers.

    Returns:
        Tensor, the merged result.

    Raises:
        ValueError: If `a` and any of the `choices` cannot be broadcast.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> choices = [[0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]]
        >>> print(np.choose([2, 3, 1, 0], choices))
        [20 31 12  3]
        >>> print(np.choose([2, 4, 1, 0], choices, mode='clip'))
        [20 31 12  3]
        >>> print(np.choose([2, 4, 1, 0], choices, mode='wrap'))
        [20  1 12  3]
        >>> a = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
        >>> choices = [-10, 10]
        >>> print(np.choose(a, choices))
        [[ 10 -10  10]
         [-10  10 -10]
         [ 10 -10  10]]
    """
    a = _to_tensor(a)
    if not _check_is_int(F.dtype(a)):
        _raise_value_error('`a` should be an int array')
    if isinstance(choices, (tuple, list)):
        # broadcasts choices to the same shape if choices is a sequence
        choices = _to_tensor(*choices)
        shapes = ()
        for choice in choices:
            shapes += (F.shape(choice),)
        shape_choice = _infer_out_shape(F.shape(a), *shapes)
        tmp = []
        for choice in choices:
            tmp.append(broadcast_to(choice, shape_choice))
        choices = stack(tmp)
    else:
        choices = _to_tensor(choices)
        shape_choice = _infer_out_shape(F.shape(a), F.shape(choices)[1:])
        choices = F.reshape(choices, choices.shape[:1] + _add_unit_axes(choices.shape[1:], len(shape_choice)))
        choices = broadcast_to(choices, (F.shape(choices)[0],) + shape_choice)

    if F.rank(a) == 0 or F.rank(choices) == 0:
        _raise_value_error('input cannot be scalars')
    a = broadcast_to(a, shape_choice)
    a = _check_indices(F.shape(choices)[0], a, mode, allow_negative_index=False)
    grid = _get_grid(F.shape(a))
    indices = concatenate((a.reshape(F.shape(a) + (1,)), grid), -1)
    return F.gather_nd(choices, indices)


def size(a, axis=None):
    """
    Returns the number of elements along a given axis.

    Args:
        a (Union[int, float, bool, list, tuple, Tensor]): Input data.
        axis (int): Axis along which the elements are counted. Default: None.
            If None, give the total number of elements.

    Returns:
        Number of elements along the specified axis.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Raises:
        TypeError: If input is not array_like or `axis` is not int.
        ValueError: If any axis is out of range or duplicate axes exist.

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.arange(10).reshape(2, 5).astype('float32')
        >>> print(np.size(x))
        10
        >>> print(np.size(x, axis=1))
        5
    """
    a = _to_tensor(a)
    if axis is None:
        return a.size
    if not isinstance(axis, int):
        _raise_type_error("axis argument should be integer.")
    axis = _canonicalize_axis(axis, a.ndim)
    return a.shape[axis]


def array_str(a):
    """
    Returns a string representation of the data in an array.

    The data in the array is returned as a single string.
    This function is similar to array_repr, the difference being that array_repr also
    returns information on the kind of array and its data type.

    Note:
        Numpy argument `max_line_width`, `precision` and `suppress_small` are not supported.
        Graph mode does not support the function.

    Args:
        a (Tensor): Input data.

    Returns:
        String.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Raises:
        TypeError: If input is not tensor.

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.arange(5)
        >>> np.array_str(x)
        '[0 1 2 3 4]'
    """
    if not isinstance(a, Tensor):
        _raise_type_error("Expect input to be tensor.")
    return a.__str__()


def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    Applies a function to 1-D slices along the given axis.
    Executes ``func1d(a, *args, **kwargs)`` where `func1d` operates on 1-D arrays and `a` is a
    1-D slice of arr along axis.

    Args:
        func1d (function): Maps `(M,) -> (Nj…)`. This function should accept 1-D arrays. It is
            applied to 1-D slices of arr along the specified axis.
        axis (int): Axis along which arr is sliced.
        arr (Tensor): Input array with shape `(Ni…, M, Nk…)`.
        args (any): Additional arguments to `func1d`.
        kwargs (any): Additional named arguments to `func1d`.

    Returns:
        Tensor with shape `(Ni…, Nj…, Nk…)`, the output array. Its shape is identical to the
        shape of `arr`, except along the `axis` dimension. This axis is removed, and replaced
        with new dimensions equal to the shape of the return value of `func1d`. So if `func1d`
        returns a scalar, the output will have one fewer dimensions than `arr`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Raises:
        ValueError: If axis is out of the range.

    Examples:
        >>> import mindspore.numpy as np
        >>> b = np.array([[1,2,3], [4,5,6], [7,8,9]])
        >>> print(np.apply_along_axis(np.diag, -1, b))
        [[[1 0 0]
        [0 2 0]
        [0 0 3]]
        [[4 0 0]
        [0 5 0]
        [0 0 6]]
        [[7 0 0]
        [0 8 0]
        [0 0 9]]]
    """
    ndim = F.rank(arr)
    shape = F.shape(arr)
    axis = _check_axis_in_range(axis, ndim)
    arr = moveaxis(arr, axis, -1)
    arr = F.reshape(arr, (-1, F.shape(arr)[-1]))
    slices = []
    for i in range(F.shape(arr)[0]):
        slices.append(func1d(arr[i], *args, **kwargs))
    stacked_slices = stack(slices)
    shape_stacked = (_tuple_slice(shape, None, axis) + _tuple_slice(shape, axis + 1, None) +
                     _tuple_slice(F.shape(stacked_slices), 1, None))
    res = F.reshape(stacked_slices, shape_stacked)

    # moves the dimensions returned by `func1d` back to `axis`
    ndim_func = F.rank(res) - ndim + 1
    if ndim_func >= 1:
        res = moveaxis(res, F.make_range(ndim - 1, F.rank(res)),
                       F.make_range(axis, axis + ndim_func))
    return res


def _stack_arrays(arrs):
    """Stacks a sequence of Tensor"""
    if isinstance(arrs, (tuple, list)):
        tensor_list = []
        for arr in arrs:
            tensor_list.append(_to_tensor(arr))
        return stack(tensor_list)
    return atleast_1d(_to_tensor(arrs))


def piecewise(x, condlist, funclist, *args, **kw):
    """
    Evaluates a piecewise-defined function.
    Given a set of conditions and corresponding functions, evaluate each function on the input
    data wherever its condition is true.

    Args:
        x (Union[int, float, bool, list, tuple, Tensor]): The input domain.
        condlist (Union[bool, list of bool Tensor]): Each boolean array corresponds to a
            function in `funclist`. Wherever `condlist[i]` is True, `funclist[i](x)` is used as
            the output value. Each boolean array in `condlist` selects a piece of `x`, and
            should therefore be of the same shape as `x`. The length of `condlist` must
            correspond to that of `funclist`. If one extra function is given, i.e. if
            ``len(funclist) == len(condlist) + 1``, then that extra function is the default
            value, used wherever all conditions are false.
        funclist (Union[list of callables, list of scalars]): Each function is evaluated over
            `x` wherever its corresponding condition is True. It should take a 1d array as input
            and give an 1d array or a scalar value as output. If, instead of a callable, a scalar
            is provided then a constant function ``(lambda x: scalar)`` is assumed.
        args (any): Any further arguments given to `piecewise` are passed to the functions upon
            execution, i.e., if called ``piecewise(..., ..., 1, 'a')``, then each function is
            called as ``f(x, 1, 'a')``.
        kw (any): Keyword arguments used in calling `piecewise` are passed to the functions upon
            execution, i.e., if called ``piecewise(..., ..., alpha=1)``, then each function is
            called as ``f(x, alpha=1)``.

    Returns:
        Tensor, the output is the same shape and type as `x` and is found by calling the
        functions in `funclist` on the appropriate portions of `x`, as defined by the boolean
        arrays in `condlist`. Portions not covered by any condition have a default value of 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Raises:
        ValueError: If length of `funclist` is not in ``(len(condlist), len(condlist) + 1)``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.linspace(-2.5, 2.5, 6)
        >>> print(np.piecewise(x, [x < 0, x >= 0], [-1, 1]))
        [-1 -1 -1  1  1  1]
    """
    x = _to_tensor(x)
    choicelist = funclist
    if isinstance(funclist, (tuple, list)):
        if _callable(x, funclist[0]):
            choicelist = []
            for func in funclist:
                choicelist.append(func(x, *args, **kw))
    condlist = _stack_arrays(condlist)
    choicelist = _stack_arrays(choicelist)

    default = 0
    n1 = len(condlist)
    n2 = len(funclist)
    if n1 + 1 == n2:
        default = choicelist[-1]
        choicelist = choicelist[:-1]
    elif n1 != n2:
        _raise_value_error('the number of choices should be either equal to conditions or ', n1 + 1)
    return select(condlist, choicelist, default=default)


def unravel_index(indices, shape, order='C'):
    """
    Converts a flat index or array of flat indices into a tuple of coordinate arrays.

    Note:
        Out-of-bound indices are clipped by the boundaries of `shape` instead of raising
        an error.

    Args:
        indices (Union[int, float, bool, list, tuple, Tensor]): An integer array whose elements
            are indices into the flattened version of an array of dimensions shape.
        shape (tuple of integers): The shape of the array to use for unraveling indices.
        order (Union['C', 'F'], optional): Determines whether the indices should be viewed as
            indexing in row-major (C-style) or column-major (Fortran-style) order. Defaults to "C".

    Returns:
        Tensor, each array in the tuple has the same shape as the indices array.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Raises:
        ValueError: If `order` is not 'C' or 'F'.

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.unravel_index([22, 41, 37], (7,6)))
        (Tensor(shape=[3], dtype=Int32, value= [3, 6, 6]),
        Tensor(shape=[3], dtype=Int32, value= [4, 5, 1]))
        >>> print(np.unravel_index([31, 41, 13], (7,6), order='F'))
        (Tensor(shape=[3], dtype=Int32, value= [3, 6, 6]),
        Tensor(shape=[3], dtype=Int32, value= [4, 5, 1]))
    """
    indices = _to_tensor(indices)
    if order not in ('C', 'F'):
        _raise_value_error('invalid order. Expected "C" or "F"')
    if isinstance(shape, int):
        shape = (shape,)
    ndim = F.rank(indices)
    if order == 'F':
        sizes = _cumprod(shape)
    else:
        sizes = _cumprod(shape[::-1])
    sizes = _to_tensor(sizes[::-1] + (1,))
    sizes = F.reshape(sizes, (-1,) + _list_comprehensions(ndim, 1, True))
    total_size = sizes[0]
    indices = where(indices > total_size - 1, total_size - 1, indices)
    if _get_device() == 'GPU':
        dtype = F.dtype(total_size)
        lowerbounds = (-(total_size.astype(mstype.float32))).astype(dtype)
    else:
        lowerbounds = -total_size
    indices = where(indices < lowerbounds, lowerbounds, indices)
    res = _mod(indices, sizes[:-1])//sizes[1:]

    num = len(res)
    if ndim == 0 and num == 1:
        return res.ravel()
    if order == 'F':
        r = range(num - 1, -1, -1)
    else:
        r = range(num)
    subs = ()
    for i in r:
        subs += (res[i],)
    return subs


def apply_over_axes(func, a, axes):
    """
    Applies a function repeatedly over multiple axes.

    `func` is called as `res = func(a, axis)`, where `axis` is the first element of `axes`.
    The result `res` of the function call must have either the same dimensions as `a` or
    one less dimension. If `res` has one less dimension than `a`, a dimension is inserted before `axis`.
    The call to `func` is then repeated for each axis in `axes`, with `res` as the first argument.

    Args:
        func (function): This function must take two arguments, `func(a, axis)`.
        a (Union[int, float, bool, list, tuple, Tensor]): Input tensor.
        axes (Union[int, list, tuple]): Axes over which `func` is applied; the elements must be integers.

    Returns:
        Tensor. The number of dimensions is the same as `a`, but the shape can be different.
        This depends on whether `func` changes the shape of its output with respect to its input.

    Raises:
        TypeError: If input `a` is not array_like or `axes` is not int or sequence of ints.
        ValueError: If any axis is out of range or duplicate axes exist.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.arange(10).reshape(2, 5).astype('float32')
        >>> print(x)
        [[0. 1. 2. 3. 4.]
         [5. 6. 7. 8. 9.]]
        >>> print(np.apply_over_axes(np.sum, x, axes=0))
        [[ 5.  7.  9. 11. 13.]]
    """
    a = _to_tensor(a)
    if isinstance(axes, int):
        axes = (axes,)
    res = a
    for axis in axes:
        res = func(res, axis=axis)
        res = F.expand_dims(res, axis) if res.ndim != a.ndim else res
        if res.ndim != a.ndim:
            _raise_value_error("function is not returning a tensor of the correct shape")
    return res
