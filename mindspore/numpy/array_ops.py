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

from ..common import dtype as mstype
from ..ops import operations as P
from ..ops import functional as F
from ..ops.primitive import constexpr
from ..nn import Cell

from .utils import _convert_list_tensor_to_tuple_tensor, _expand, _broadcast_to, \
    _is_empty
from .utils_const import _check_is_int, _check_axes_range, _check_start_normalize, \
    _check_is_tensor, _check_is_tuple, _check_is_list, _raise_type_error, _raise_value_error, \
    _infer_out_shape, _empty, _promote, _check_same_type, _check_input_tensor

# According to official numpy reference, the dimension of a numpy array must be less
# than 32
MAX_NUMPY_DIMS = 32


@constexpr
def _prepare_shape_for_expand_dims(shape, axes):
    """
    Creates the expanded new shape based on the shape and given axes

    Args:
        shape (tuple): the shape of the tensor
        axes Union(int, tuple(int), list(int)): the axes with dimensions expanded.

    Returns:
        new_shape(tuple): the shape with dimensions expanded.
    """

    new_shape = []
    shape_idx = 0
    new_shape_length = len(shape)

    # Convert to set
    if isinstance(axes, int):
        new_shape_length += 1
        if axes >= new_shape_length or axes < -new_shape_length:
            raise ValueError(f"axis {axes} is out of bounds for tensor of dimension {new_shape_length}")
        axes = {axes}

    elif isinstance(axes, (list, tuple)):
        new_shape_length += len(axes)
        for axis in axes:
            if axis >= new_shape_length or axis < -new_shape_length:
                raise ValueError(f"axis {axis} is out of bounds for tensor of dimension {new_shape_length}")
        axes = set(axes)

    else:
        raise TypeError(f"only int, tuple and list are allowed for axes, but got {type(axes)}")

    for new_shape_idx in range(new_shape_length):
        if new_shape_idx in axes or new_shape_idx - new_shape_length in axes:
            new_shape.append(1)
        else:
            new_shape.append(shape[shape_idx])
            shape_idx += 1
    return tuple(new_shape)


def expand_dims(a, axis):
    """
    Expands the shape of a tensor.

    Inserts a new axis that will appear at the axis position in the expanded tensor shape.

    Args:
        a (Tensor): Input tensor array.
        axis Union[int, list(int), tuple(int)]: Position in the expanded axes where
        the new axis is placed,

    Returns:
        Tensor, view of a tensor with the number of dimensions increased.

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
    if not _check_is_tensor(F.typeof(a)):
        _raise_type_error("Input is not Tensor.")
    shape = F.shape(a)
    # yield expanded shape based on the axes
    new_shape = _prepare_shape_for_expand_dims(shape, axis)
    return F.reshape(a, new_shape)


def squeeze(a, axis=None):
    """
    Removes single-dimensional entries from the shape of an tensor.

    Args:
        a (Tensor): Input tensor array.
        axis: Union[None, int, list(int), tuple(list)]. Default is None.

    Returns:
        Tensor, with all or a subset of the dimensions of length 1 removed.

    Raises:
        TypeError: If input arguments have types not specified above.
        ValueError: If specified axis has shape entry > 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.ones((1,2,2,1))
        >>> x = np.squeeze(x)
        >>> print(x.shape)
        (2, 2)
    """
    if not _check_is_tensor(F.typeof(a)):
        _raise_type_error("Input is not Tensor.")
    return a.squeeze(axis)


def transpose(a, axes=None):
    """
    Reverses or permutes the axes of a tensor; returns the modified tensor.

    Args:
        a (Tensor): a tensor to be transposed
        axes (Union[None, tuple, list]): the axes order, if axes is None, transpose
        the entire tensor. Default is None.

    Returns:
        Tensor, the transposed tensor array.

    Raises:
        TypeError: If input arguments have types not specified above.
        ValueError: If the number of axes is not euqal to a.ndim.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.ones((1,2,3))
        >>> x = np.transpose(x)
        >>> print(x.shape)
        (3, 2, 1)
    """
    if not _check_is_tensor(F.typeof(a)):
        _raise_type_error("Input is not Tensor.")
    return a.transpose(axes)


def rollaxis(x, axis, start=0):
    """
    Rolls the specified axis backwards, until it lies in the given position.
    The positions of the other axes do not change relative to one another.

    Args:
        x (Tensor): A Tensor to be transposed.
        axis (int): The axis to be rolled.
        start (int):
            - When start >= 0:
                - When start <= axis: the axis is rolled back until it lies in
                  this position (start).
                - When start > axis: the axis is rolled until it lies before this
                  position (start).
            - When start < 0: the start will be normalized as follows:
                start ........... Normalized start
                -(x.ndim+1)       raise ValueError
                -x.ndim           0
                ...               ...
                -1                x.ndim-1
                0                 0
                ...               ...
                x.ndim            x.ndim
                x.ndim+1          raise ValueError

    Returns:
        Transposed Tensor. Has the same data type as the original tensor x.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Raises:
        TypeError: If axis or start is not integer, or x is not tensor.
        ValueError: If axis is not in the range from -ndim to ndim-1 or
            start is not in the range from -ndim to ndim.

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.ones((2,3,4))
        >>> output = np.rollaxis(x, 0, 2)
        >>> print(output.shape)
        (3, 2, 4)
    """
    if not _check_is_tensor(F.typeof(x)):
        _raise_type_error("Input is not Tensor.")
    if not _check_is_int(axis):
        _raise_type_error("integer argument expected, but got ", axis)
    if not _check_is_int(start):
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
        Transposed tensor, has the same data type as the original tensor x.

    Raises:
        TypeError: If axis1 or axis2 is not integer, or x is not tensor.
        ValueError: If axis1 or axis2 is not in the range from -ndim to ndim-1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.ones((2,3,4))
        >>> output = np.swapaxes(x, 0, 2)
        >>> print(output.shape)
        (4,3,2)
    """
    if not _check_is_tensor(F.typeof(x)):
        _raise_type_error("Input is not Tensor.")
    return x.swapaxes(axis1, axis2)


def reshape(x, new_shape):
    """
    Reshapes a tensor without changing its data.

    Args:
        x (Tensor): A tensor to be reshaped.
        new_shape (Union[int, list(int), tuple(int)]): The new shape should be
            compatible with the original shape. If the tuple has only one element,
            the result will be a 1-D tensor of that length. One shape dimension
            can be -1. In this case, the value is inferred from the length of
            the tensor and remaining dimensions.

    Returns:
        Reshaped Tensor. Has the same data type as the original tensor x.

    Raises:
        TypeError: If new_shape is not integer, list or tuple, or x is not tensor.
        ValueError: If new_shape does not compatible with the original shape.

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
    if not _check_is_tensor(F.typeof(x)):
        _raise_type_error("Input is not Tensor.")
    return x.reshape(new_shape)


def ravel(x):
    """
    Returns a contiguous flattened tensor.

    A 1-D tensor, containing the elements of the input, is returned.

    Args:
        x (Tensor): A tensor to be flattened.

    Returns:
        Flattened tensor, has the same data type as the original tensor x.

    Raises:
        TypeError: If x is not tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.ones((2,3,4))
        >>> output = np.ravel(x)
        >>> print(output.shape)
        (24,)
    """
    if not _check_is_tensor(F.typeof(x)):
        _raise_type_error("Input is not Tensor.")
    return x.ravel()


@constexpr
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
        tuple_of_casted_tensors += (tensor.astype(final_type, copy=False),)
    return tuple_of_casted_tensors


def concatenate(arrays, axis=0):
    """
    Joins a sequence of tensors along an existing axis.

    Args:
        arrays: Union[Tensor, tuple(Tensor), list(Tensor)], a tensor or a list
        of tensors to be concatenated.

        axis (int, optional): The axis along which the tensors will be joined,
            if axis is None, tensors are flattened before use. Default is 0.

    Returns:
        Tensor, a tensor concatenated from a tensor or a list of tensors.

    Raises:
        TypeError: If input arguments have types not specified above.
        ValueError: If specified axis < 0, and exceeds tensor.ndim.

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
    array_type = F.typeof(arrays)
    if _check_is_tensor(array_type):
        # if the input is a single tensor
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
        TypeError: If tup is not Tensor, list or tuple.
        ValueError: If tup is empty.

    Examples:
        >>> import mindspore.numpy as mnp
        >>> import numpy as onp
        >>> from mindspore import Tensor
        >>> x1 = Tensor(onp.array([1, 2, 3]).astype('int32'))
        >>> x2 = Tensor(onp.array([4, 5, 6]).astype('int32'))
        >>> output = mnp.column_stack((x1, x2))
        >>> print(output)
        [[1, 4],
         [2, 5],
         [3, 6]]
    """
    if _check_is_tensor(F.typeof(tup)):
        return tup
    if not _check_is_list(tup) and not _check_is_tuple(tup):
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
    This is equivalent to concatenation along the first axis. 1-D tensors should firstly be reshaped to (1, N),
        and then be concatenated along the first axis.

    Args:
        tup (Union[Tensor, tuple, list]): A sequence of 1-D or 2-D tensors. The tensors must have the same shape
            along all but the first axis. 1-D tensors must have the same shape.

    Returns:
        Stacked Tensor, formed by stacking the given tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Raises:
        TypeError: If tup is not Tensor, list or tuple.
        ValueError: If tup is empty.

    Examples:
        >>> import mindspore.numpy as mnp
        >>> import numpy as onp
        >>> from mindspore import Tensor
        >>> x1 = Tensor(onp.array([1, 2, 3]).astype('int32'))
        >>> x2 = Tensor(onp.array([4, 5, 6]).astype('int32'))
        >>> output = mnp.vstack((x1, x2))
        >>> print(output)
        [[1, 2, 3],
         [4, 5, 6]]
    """
    if _check_is_tensor(F.typeof(tup)):
        return tup
    if not _check_is_list(tup) and not _check_is_tuple(tup):
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
        TypeError: If tup is not Tensor, list or tuple.
        ValueError: If tup is empty.

    Examples:
        >>> import mindspore.numpy as mnp
        >>> import numpy as onp
        >>> from mindspore import Tensor
        >>> x1 = Tensor(onp.array([1, 2, 3]).astype('int32'))
        >>> x2 = Tensor(onp.array([4, 5, 6]).astype('int32'))
        >>> output = mnp.hstack((x1, x2))
        >>> print(output)
        [1, 2, 3, 4, 5, 6]
    """
    if _check_is_tensor(F.typeof(tup)):
        return tup
    if not _check_is_list(tup) and not _check_is_tuple(tup):
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
    This is equivalent to concatenation along the third axis. 1-D tensors (N,) should be reshaped to (1,N,1).
        2-D tensors (M,N) should be reshaped to (M,N,1) before concatenation.

    Args:
        tup (Union[Tensor, tuple, list]): A sequence of tensors. The tensors must have the same shape along all but
            the third axis. 1-D or 2-D tensors must have the same shape.

    Returns:
        Stacked Tensor, formed by stacking the given tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Raises:
        TypeError: If tup is not Tensor, list or tuple.
        ValueError: If tup is empty.

    Examples:
        >>> import mindspore.numpy as mnp
        >>> import numpy as onp
        >>> from mindspore import Tensor
        >>> x1 = Tensor(onp.array([1, 2, 3]).astype('int32'))
        >>> x2 = Tensor(onp.array([4, 5, 6]).astype('int32'))
        >>> output = mnp.dstack((x1, x2))
        >>> print(output)
        [[[1, 4],
           [2, 5],
           [3, 6]]]
    """
    if _check_is_tensor(F.typeof(tup)):
        return tup
    if not _check_is_list(tup) and not _check_is_tuple(tup):
        _raise_type_error("Tensor or, list or tuple of tensors are required, but got", tup)

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
    Returns elements chosen from x or y depending on condition.

    Note:
        As nonzero is not supported, neither x or y can be None.

    Args:
        condition (Tensor): where True, yield x, otherwise yield y.
        x, y (Tensor): Values from which to choose. x, y and condition need
                        to be broadcastable to some shape.

    Returns:
        Tensor or scalar, with elements from x where condition is True, and
        elements from y elsewhere.

    Raises:
        ValueError: if operands cannot be broadcast.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> condition = np.full((1, 1, 2), [False, True])
        >>> x = np.full((1, 3, 2), 5)
        >>> y = np.full((2, 1, 1), 7)
        >>> output = np.where(condition, x, y)
        >>> print(output)
        [[[7, 5],
        [7, 5],
        [7, 5]],

       [[7, 5],
        [7, 5],
        [7, 5]]]
    """
    # type promotes input tensors
    dtype1 = F.dtype(x)
    dtype2 = F.dtype(y)
    dtype = _promote(dtype1, dtype2)
    if not _check_same_type(dtype1, dtype):
        x = F.cast(x, dtype)
    if not _check_same_type(dtype2, dtype):
        y = F.cast(y, dtype)
    is_bool = _check_same_type(dtype1, mstype.bool_) and _check_same_type(
        dtype2, mstype.bool_)
    if is_bool:
        # select does not support bool type for x or y
        x = F.cast(x, mstype.float32)
        y = F.cast(y, mstype.float32)

    # broadcasts input tensors
    shape_out = _infer_out_shape(F.shape(condition),
                                 F.shape(x), F.shape(y))
    ndim_out = len(shape_out)
    if not _check_same_type(F.dtype(condition), mstype.float32):
        # tiling with bool is not supported on GPU
        condition = F.cast(condition, mstype.float32)
    condition = _expand(condition, ndim_out)
    x = _expand(x, ndim_out)
    y = _expand(y, ndim_out)
    condition = _broadcast_to(
        condition, F.shape(condition), shape_out, ndim_out)
    x = _broadcast_to(x, F.shape(x), shape_out, ndim_out)
    y = _broadcast_to(y, F.shape(y), shape_out, ndim_out)
    if not _check_same_type(F.dtype(condition), mstype.bool_):
        condition = F.cast(condition, mstype.bool_)
    res = F.select(condition, x, y)
    if is_bool:
        res = F.cast(res, mstype.bool_)
    return res


def _atleast_xd(ndim, arys):
    """Returns arys with at least ndim."""
    for arr in arys:
        _check_input_tensor(F.typeof(arr))
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
        arys1, arys2, … (Tensor): one or more input tensors.

    Returns:
        Tensor, or list of tensors, each with a.ndim >= 1.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> a = np.ones((2, 3))
        >>> b = np.ones(())
        >>> c = np.ones(5)
        >>> output = np.atleast_1d(a, b, c)
        >>> print(output)
            (Tensor(shape=[2, 3], dtype=Float32, value=
            [[1.00000000e+000, 1.00000000e+000, 1.00000000e+000],
            [1.00000000e+000, 1.00000000e+000, 1.00000000e+000]]),
            Tensor(shape=[1], dtype=Float32, value= [1.00000000e+000]),
            Tensor(shape=[5], dtype=Float32,
            value= [1.00000000e+000, 1.00000000e+000, 1.00000000e+000,
            1.00000000e+000, 1.00000000e+000]))
    """
    return _atleast_xd(1, arys)


def atleast_2d(*arys):
    """
    Views inputs as arrays with at least two dimensions.

    Note:
        In graph mode, returns a tuple of tensor instead of a list of
        tensors.
    Args:
        arys1, arys2, … (Tensor): one or more input tensors.

    Returns:
        Tensor, or list of tensors, each with a.ndim >= 2.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> a = np.ones((2, 3))
        >>> b = np.ones(())
        >>> c = np.ones(5)
        >>> output = np.atleast_2d(a, b, c)
        >>> print(output)
            (Tensor(shape=[2, 3], dtype=Float32, value=
            [[1.00000000e+000, 1.00000000e+000, 1.00000000e+000],
            [1.00000000e+000, 1.00000000e+000, 1.00000000e+000]]),
            Tensor(shape=[1, 1], dtype=Float32, value= [[1.00000000e+000]]),
            Tensor(shape=[1, 5], dtype=Float32,
            value= [[1.00000000e+000, 1.00000000e+000, 1.00000000e+000,
            1.00000000e+000, 1.00000000e+000]]))
    """
    return _atleast_xd(2, arys)


def atleast_3d(*arys):
    """
    Views inputs as arrays with at least three dimensions.

    Note:
        In graph mode, returns a tuple of tensor instead of a list of
        tensors.

    Args:
        arys1, arys2, … (Tensor): one or more input tensors.

    Returns:
        Tensor, or list of tensors, each with a.ndim >= 3. For example,
        a 1-D array of shape (N,) becomes a view of shape (1, N, 1), and
        a 2-D array of shape (M, N) becomes a view of shape (M, N, 1).

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> a = np.ones((2, 3))
        >>> b = np.ones(())
        >>> c = np.ones(5)
        >>> output = np.atleast_3d(a, b, c)
        >>> print(output)
            (Tensor(shape=[2, 3, 1], dtype=Float32, value=
            [[[1.00000000e+000], [1.00000000e+000], [1.00000000e+000]],
            [[1.00000000e+000], [1.00000000e+000], [1.00000000e+000]]]),
            Tensor(shape=[1, 1, 1], dtype=Float32, value= [[[1.00000000e+000]]]),
            Tensor(shape=[1, 5, 1], dtype=Float32,
            value= [[[1.00000000e+000], [1.00000000e+000], [1.00000000e+000],
            [1.00000000e+000], [1.00000000e+000]]]))
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

    The axis parameter specifies the index of the new axis in the
    dimensions of the result. For example, if axis=0 it will be the
    first dimension and if axis=-1 it will be the last dimension.


    Note:
        Numpy argument out is not supported.

    Args:
        arrays (sequence of Tensor): Each array must have the same shape.
        axis (int): optional. The axis in the result array along which the
            input arrays are stacked.

    Returns:
        Tensor, The stacked array has one more dimension than the input
        arrays.

    Raises:
        ValueError: if input is not Tensor, tuple, or list.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
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
    arr_type = F.typeof(arrays)

    if _check_is_tensor(arr_type):
        shape = F.shape(arrays)
        ndim = F.rank(arrays)
        axis = axis % ndim
        axes = F.make_range(ndim)
        perm = axes[1:axis+1] + (0,) + axes[axis+1:]
        if _is_empty(shape):
            return _empty(mstype.float32, shape[1:axis+1] + (shape[0],) + shape[axis+1:])
        return transpose(arrays, perm)

    if _check_is_tuple(arr_type) or _check_is_list(arr_type):
        shape = (len(arrays),) + F.shape(arrays[0])
        ndim = len(shape)
        axis = axis % ndim
        if _is_empty(shape):
            return _empty(mstype.float32, shape[1:axis+1] + (shape[0],) + shape[axis+1:])
        seq = ()
        for arr in arrays:
            seq += (F.expand_dims(arr, axis),)
        return concatenate(seq, axis)
    return _raise_value_error('input arrays must be Tensor, tuple, or list')


class UniqueNet(Cell):
    """The operation `mindspore.ops.Unique` must be wrapped inside a model and executed in graph mode. """

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
        This operator must be executed in graph mode.

    Args:
        x (Tensor): The input tensor to be processed.
        return_inverse (bool): If True, also return the indices of the unique tensor.
            Default: False.

    Returns:
        Tensor or tuple of Tensors.
        - If `return_inverse` is False, just return the unique tensor.
        - If `return_inverse` is True, return tuple of tensors.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Raises:
        TypeError: If x is not tensor.

    Examples:
        >>> import mindspore.numpy as mnp
        >>> import numpy as onp
        >>> from mindspore import context
        >>> context.set_context(mode=context.GRAPH_MODE)
        >>> input_x = mnp.asarray(onp.array([1, 2, 2, 2, 3, 4, 5]).astype('int32'))
        >>> output_x = mnp.unique(input_x)
        >>> print(output_x)
        [1, 2, 3, 4, 5]
        >>> output_x = mnp.unique(input_x, return_inverse=True)
        >>> print(output_x)
        (Tensor(shape=[5], dtype=Int32, value= [ 1, 2, 3, 4, 5]), Tensor(shape=[7], dtype=Int32,
            value= [0, 1, 1, 1, 2, 3, 4]))
    """
    if not _check_is_tensor(F.typeof(x)):
        _raise_type_error("Tensor is expected, but got", x)
    if F.tuple_len(F.shape(x)) > 1:
        x = ravel(x)
    uniq = UniqueNet()
    res = uniq(x)
    if not return_inverse:
        return res[0]
    return res
