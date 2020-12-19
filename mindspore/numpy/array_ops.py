# Copyright 2020 Huawei Technologies Co., Ltd
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
from copy import copy as py_copy

import numpy as onp

from ..common import Tensor
from ..common import dtype as mstype
from ..ops import operations as P
from ..ops import functional as F
from ..ops.primitive import constexpr

from .utils import _check_shape, _check_shape_compile, _check_dtype, _check_is_int, \
    _check_axes_range, _check_start_normalize, _check_shape_contain_zero, _check_is_tensor, \
    _check_input_for_asarray, _deep_list, _deep_tensor_to_nparray, _check_is_list, \
    _covert_list_tensor_to_tuple_tensor

DEFAULT_FLOAT_DTYPE = mstype.float32
DEFAULT_INT_DTYPE = mstype.int32
# According to official numpy reference, the dimension of a numpy array must be less
# than 32
MAX_NUMPY_DIMS = 32

def array(obj, dtype=None, copy=True, ndmin=0):
    """
    Creates a tensor.

    This function creates tensors from an array-like object.

    Args:
        obj (Union[int, float, bool, list, tuple, numpy.ndarray]): Input data, in
        any form that can be converted to a tensor. This includes lists, lists of
        tuples, tuples, tuples of tuples, tuples of lists and numpy.ndarray.
        dtype (Union[mstype.dtype, str], optional): Designated tensor dtype, can
            be in format of np.int32, or `int32`. If dtype is None, the data type
            of the new tensor will be inferred from obj. Default is None.
        copy (bool): If true, then the object is copied. Otherwise, a copy will
            only be made if necessary. Default: True.
        ndmin (int): Specifies the minimum number of dimensions that the resulting
            tensor should have. Ones will be pre-pended to the shape as needed to
            meet this requirement. Default: 0

    Returns:
        Tensor, generated tensor with the specified dtype.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.array([1,2,3]))
        [1 2 3]
    """
    if ndmin > 0:
        # Fall back to original numpy creation.
        if isinstance(obj, Tensor):
            obj = obj.asnumpy()
        return asarray(onp.array(obj, dtype, copy=copy, ndmin=ndmin))

    if not copy:
        return asarray(obj, dtype=dtype)

    obj = py_copy(obj)
    return asarray(obj, dtype=dtype)


def asarray(a, dtype=None):
    """
    Converts the input to tensor.

    This function converts tensors from an array-like object.

    Args:
        a (Union[int, float, bool, list, tuple, numpy.ndarray]): Input data, in
        any form that can be converted to a tensor. This includes lists, lists of
        tuples, tuples, tuples of tuples, tuples of lists and ndarrays.
        dtype (Union[mstype.dtype, str], optional): Designated tensor dtype, can
            be in format of np.int32, or `int32`. If dtype is None, the data type
            of the new tensor will be inferred from a. Default is None.

    Returns:
        Tensor, generated tensor with the specified dtype.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.asarray([1,2,3]))
        [1 2 3]
    """

    if dtype is not None:
        dtype = _check_dtype(dtype)

    _ = _check_input_for_asarray(a)

    if isinstance(a, float) and (dtype is None):
        dtype = DEFAULT_FLOAT_DTYPE

    if isinstance(a, int) and not isinstance(a, bool) and (dtype is None):
        dtype = DEFAULT_INT_DTYPE

    if isinstance(a, bool) and (dtype is None):
        dtype = mstype.bool_

    if isinstance(a, (list, tuple)):
        # Convert all tuple/nested tuples to lists
        a = _deep_list(a)
        # Convert all tensor sub-elements to numpy arrays
        a = _deep_tensor_to_nparray(a)
        a = onp.asarray(a)
        # If dtype is not specified, we keep consistent with numpy decision
        # only exceptions are: we use int/float32
        if dtype is None:
            if a.dtype is onp.dtype('int64'):
                dtype = DEFAULT_INT_DTYPE
            elif a.dtype is onp.dtype('float64'):
                dtype = DEFAULT_FLOAT_DTYPE

    if isinstance(a, onp.ndarray) and dtype is None:
        if a.dtype is onp.dtype('bool'):
            dtype = mstype.bool_
        elif a.dtype is onp.dtype('int'):
            dtype = DEFAULT_INT_DTYPE
        elif a.dtype is onp.dtype('float'):
            dtype = DEFAULT_FLOAT_DTYPE
        a = Tensor.from_numpy(a)

    # If a is already a tensor and we don't need to cast dtype, return a
    if isinstance(a, Tensor):
        if dtype is None:
            return a
        dtype = _check_dtype(dtype)
        if dtype == a.dtype:
            return a

    return Tensor(a, dtype=dtype)


def asfarray(a, dtype=DEFAULT_FLOAT_DTYPE):
    """
    Similar to asarray, converts the input to a float tensor.

    If non-float dtype is defined, this function will return a float32 tensor instead.

    Args:
        a (Union[int, float, bool, list, tuple, numpy.ndarray]): Input data, in
        any form that can be converted to a tensor. This includes lists, lists of
        tuples, tuples, tuples of tuples, tuples of lists and numpy.ndarray.
        dtype (Union[mstype.dtype, str], optional): Designated tensor dtype, can
            be in format of np.float32, or `float32`. Default is mstype.float32.

    Returns:
        Tensor, generated tensor with the specified float dtype.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.asfarray([1,2,3]))
        [1. 2. 3.]
    """
    dtype = _check_dtype(dtype)
    _ = _check_input_for_asarray(a)

    if dtype not in (mstype.float16, mstype.float32, mstype.float64):
        dtype = DEFAULT_FLOAT_DTYPE

    if isinstance(a, (list, tuple)):
        # Convert all tuple/nested tuples to lists
        a = _deep_list(a)
        # Convert all tensor sub-elements to numpy arrays
        a = _deep_tensor_to_nparray(a)
        a = onp.asarray(a)

    if isinstance(a, onp.ndarray):
        a = Tensor.from_numpy(a)

    return Tensor(a, dtype)


def copy_(a):
    """
    Returns a tensor copy of the given object.

    Args:
        a (Tensor): Input tensor.

    Returns:
        Tensor, has the same data as `a`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.ones((2,2))
        >>> print(np.copy(x))
        [[1. 1.]
         [1. 1.]]
    """
    return py_copy(a)


def ones(shape, dtype=DEFAULT_FLOAT_DTYPE):
    """
    Returns a new tensor of given shape and type, filled with ones.

    Args:
        shape (Union[int, tuple, list]): the shape of the new tensor.
        dtype (Union[mstype.dtype, str], optional): Designated tensor dtype, can
            be in format of np.float32, or `float32`. Default is mstype.float32.

    Returns:
        Tensor, with the designated shape and dtype, filled with ones.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.ones((2,2)))
        [[1. 1.]
        [1. 1.]]
    """
    if _check_shape_contain_zero(shape):
        return asarray(onp.ones(shape), dtype=dtype)
    shape = _check_shape(shape)
    dtype = _check_dtype(dtype)
    fill = P.Fill()
    output = fill(dtype, shape, 1)
    return output


def zeros(shape, dtype=DEFAULT_FLOAT_DTYPE):
    """
    Returns a new tensor of given shape and type, filled with zeros.

    Args:
        shape (Union[int, tuple, list]): the shape of the new tensor.
        dtype (Union[mstype.dtype, str], optional): Designated tensor dtype, can
            be in format of np.float32, or `float32`. Default is mstype.float32.

    Returns:
        Tensor, with the designated shape and dtype, filled with zeros.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.zeros((2,2)))
        [[0. 0.]
        [0. 0.]]
    """
    if _check_shape_contain_zero(shape):
        return asarray(onp.zeros(shape), dtype=dtype)
    shape = _check_shape(shape)
    dtype = _check_dtype(dtype)
    fill = P.Fill()
    output = fill(dtype, shape, 0)
    return output


def full(shape, fill_value, dtype=None):
    """
    Returns a new tensor of given shape and type, filled with fill_value.

    Args:
        shape (Union[int, tuple(int), list(int)]): Shape of the new tensor, e.g.,
            (2, 3) or 2.
        fill_value (Union[int, float, bool, list, tuple]): scalar or array_like
            fill value.
        dtype (Union[mstype.dtype, str], optional): Designated tensor dtype, can
            be in format of np.float32, or `float32`, if dtype is None, the data type
            of the new tensor will be inferred from fill_value. Default is None.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Returns:
        Tensor, with the designated shape and dtype, filled with `fill_value`.

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.full((2,2), True))
        [[True True]
        [True True]]
    """
    if dtype is None:
        dtype = array(fill_value).dtype

    shape = _check_shape(shape)
    _ = _check_input_for_asarray(fill_value)
    dtype = _check_dtype(dtype)

    if isinstance(fill_value, (int, float, bool)) and not _check_shape_contain_zero(shape):
        return P.Fill()(dtype, shape, fill_value)

    # if fill_value is array_like or shape contains zero. fall back to original
    # numpy creation
    return Tensor(onp.full(shape, fill_value, mstype.dtype_to_nptype(dtype)))


def arange(*args, **kwargs):
    """
    Returns evenly spaced values within a given interval.

    Returns `num` evenly spaced samples, calculated over the interval [`start`, `stop`].
    The endpoint of the interval can optionally be excluded.
    The current implementation is a direct wrapper on top of numpy.arange, except that
    the default dtype is float32 and int32, compare to float64 and int64 for numpy
    implementation.

    Args:
        start(Union[int, float]): Start of interval. The interval includes this value.
            When stop is provided as a position argument, start must be given, when stop
            is a normal argument, start can be optional, and default is 0.
            Please see additional examples below.
        stop(Union[int, float], optional): End of interval. The interval does not
            include this value, except in some cases where step is not an integer
            and floating point round-off affects the length of out.
        step(Union[int, float], optional): Spacing between values. For any output
            out, this is the distance between two adjacent values, out[i+1] - out[i].
            The default step size is 1. If step is specified as a position argument,
            start must also be given.
        dtype (Union[mstype.dtype, str], optional): Designated tensor dtype, can
            be in format of np.float32, or `float32`. If dtype is None, the data type
            of the new tensor will be inferred from start, stop and step. Default is None.

    Returns:
        arangend tensor of evenly spaced values.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.arange(0, 5, 1))
        [0 1 2 3 4]
        >>> print(np.arange(3))
        [0 1 2]
        >>> print(np.arange(start=0, stop=3))
        [0 1 2]
        >>> print(np.arange(0, stop=3, step=0.5))
        [0.  0.5 1.  1.5 2.  2.5]
        >>> print(np.arange(stop=3)) # This will lead to TypeError
    """
    # infer the dtype, if either of start, end, step is float, default dtype is
    # float32, else int32.
    int_flag = True
    final_dtype = None

    if args:
        for item in args:
            if isinstance(item, float):
                int_flag = False
    if kwargs:
        if ('start' in kwargs and isinstance(kwargs['start'], float)) or \
           ('stop' in kwargs and isinstance(kwargs['stop'], float)) or \
           ('step' in kwargs and isinstance(kwargs['step'], float)):
            int_flag = False

    if int_flag:
        final_dtype = onp.int32
    else:
        final_dtype = onp.float32

    if 'dtype' in kwargs and kwargs['dtype'] is not None:
        final_dtype = _check_dtype(kwargs['dtype'])
        final_dtype = mstype.dtype_to_nptype(final_dtype)
    kwargs['dtype'] = final_dtype
    out = onp.arange(*args, **kwargs)
    out = Tensor.from_numpy(out)
    return out


def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    """
    Returns evenly spaced values within a given interval.

    The current implementation is a direct wrapper on top of numpy.linspace, except
    the default dtype is float32, compare to float64 for numpy,

    Args:
        start (Union[int, list(int), tuple(int),tensor]):The starting value of the sequence.
        stop (Union[int, list(int), tuple(int),tensor]):The end value of the sequence,
            unless `endpoint` is set to False. In that case, the sequence consists
            of all but the last of ``num + 1` evenly spaced samples, so that `stop`
            is excluded.  Note that the step size changes when `endpoint` is False.
        num (int, optional): Number of samples to generate. Default is 50.
        endpoint (bool, optional): If True, `stop` is the last sample. Otherwise, it is
            not included. Default is True.
        retstep (bool, optional): If True, return (`samples`, `step`), where `step` is
            the spacing between samples.
        dtype (Union[mstype.dtype, str], optional): Designated tensor dtype, can
            be in format of np.float32, or `float32`.If `dtype` is None, infer the data
            type from other input arguments. Default is None.
        axis (int, optional): The axis in the result to store the samples. Relevant
            only if start or stop are array-like.  By default (0), the samples will
            be along a new axis inserted at the beginning. Use -1 to get an axis at the end.
            Default is 0.

    Returns:
        samples (Tensor): There are `num` equally spaced samples in the closed interval
            ``[start, stop]`` or the half-open interval ``[start, stop)``
            (depending on whether `endpoint` is True or False).

        step (float, optional): Only returned if `retstep` is True.
            Size of spacing between samples.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.linspace(0, 5, 6))
        [0. 1. 2. 3. 4. 5.]
    """

    if isinstance(start, Tensor):
        start = start.asnumpy()

    if isinstance(stop, Tensor):
        stop = stop.asnumpy()

    if not isinstance(num, int):
        raise TypeError(f"num should be an integer, but got {type(num)}")

    final_dtype = None
    if dtype is not None:
        final_dtype = _check_dtype(dtype)
        final_dtype = mstype.dtype_to_nptype(final_dtype)
    else:
        final_dtype = onp.float32

    dtype = final_dtype
    out = onp.linspace(start, stop, num, endpoint, retstep, dtype, axis)

    if retstep:
        array_out, step_out = out[0], out[1]
        tensor_out = Tensor(array_out)
        return tensor_out, step_out

    tensor_out = Tensor(out)
    return tensor_out


def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
    """
    Returns numbers spaced evenly on a log scale.

    In linear space, the sequence starts at base ** start (base to the power of
    start) and ends with base ** stop (see endpoint below).
    The current implementation is a direct wrapper on top of numpy.logspace, except
    the default dtype is float32, compare to float64 for numpy,

    Args:
        start (Union[int, list(int), tuple(int), tensor]):The starting value of the sequence.
        stop (Union[int, list(int), tuple(int), tensor]):The end value of the sequence,
            unless `endpoint` is set to False. In that case, the sequence consists
            of all but the last of ``num + 1` evenly spaced samples, so that `stop`
            is excluded.  Note that the step size changes when `endpoint` is False.
        num (int, optional): Number of samples to generate. Default is 50.
        endpoint (bool, optional): If True, `stop` is the last sample. Otherwise, it is
            not included. Default is True.
        base (Union[int, float], optional): The base of the log space. The step size
            between the elements in ln(samples) / ln(base) (or log_base(samples))
            is uniform. Default is 10.0.
        dtype (Union[mstype.dtype, str], optional): Designated tensor dtype, can
            be in format of np.float32, or `float32`.If `dtype` is None, infer the data
            type from other input arguments. Default is None.
        axis (int, optional): The axis in the result to store the samples. Relevant
            only if start or stop is array-like.  By default (0), the samples will
            be along a new axis inserted at the beginning. Use -1 to get an axis at the end.
            Default is 0.

    Returns:
        samples (Tensor): num samples, equally spaced on a log scale.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.logspace(0, 5, 6, base=2.0))
        [ 1.  2.  4.  8. 16. 32.]
    """

    if isinstance(start, Tensor):
        start = start.asnumpy()

    if isinstance(stop, Tensor):
        stop = stop.asnumpy()

    final_dtype = None
    if dtype is not None:
        final_dtype = _check_dtype(dtype)
        final_dtype = mstype.dtype_to_nptype(final_dtype)
    else:
        final_dtype = onp.float32

    dtype = final_dtype
    out = onp.logspace(start, stop, num, endpoint, base, dtype, axis)

    tensor_out = Tensor.from_numpy(out)
    return tensor_out


def eye(N, M=None, k=0, dtype=DEFAULT_FLOAT_DTYPE):
    """
    Returns a 2-D tensor with ones on the diagnoal and zeros elsewhere.

    Args:
        N (int): Number of rows in the output, must be larger than 0.
        M (int, optional): Number of columns in the output. If None, defaults to N,
            if defined, must be larger than 0. Deault is None.
        k (int, optional): Index of the diagonal: 0 (the default) refers to the main
            diagonal, a positive value refers to an upper diagonal, and a negative value
            to a lower diagonal. Default is 0.
        dtype (Union[mstype.dtype, str], optional): Designated tensor dtype, can
            be in format of np.float32, or `float32`. Default is mstype.float32.

    Returns:
        result (Tensor): A tensor of shape (N,M). A tensor where all elements
        are equal to zero, except for the k-th diagonal, whose values are equal to one.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.eye(2, 2))
        [[1. 0.]
        [0. 1.]]
    """
    dtype = _check_dtype(dtype)
    make_eye = P.Eye()
    if M is None:
        M = N
    M = int(M)
    N = int(N)
    k = int(k)
    out = None
    if k != 0 or N == 0 or M == 0:
        # Fall back to original numpy creation method
        out = onp.eye(N, M, k)
    else:
        out = make_eye(N, M, dtype)
    return asarray(out, dtype=dtype)


def identity(n, dtype=DEFAULT_FLOAT_DTYPE):
    """
    Returns the identity tensor.

    Args:
        n (int): Number of rows and columns in the output, must be larger than 0.
        dtype (Union[mstype.dtype, str], optional): Designated tensor dtype, can
            be in format of np.float32, or `float32`. Default is mstype.float32.

    Returns:
        result (Tensor): A tensor of shape (n,n). A tensor where all elements
        are equal to zero, except for the diagonal, whose values are equal to one.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.identity(2))
        [[1. 0.]
        [0. 1.]]
    """
    dtype = _check_dtype(dtype)
    return eye(n, dtype=dtype)


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
            raise ValueError(
                f"axis {axes} is out of bounds for tensor of dimension {new_shape_length}")
        axes = {axes}

    elif isinstance(axes, (list, tuple)):
        new_shape_length += len(axes)
        for axis in axes:
            if axis >= new_shape_length or axis < -new_shape_length:
                raise ValueError(
                    f"axis {axis} is out of bounds for tensor of dimension {new_shape_length}")
        axes = set(axes)

    else:
        raise TypeError(
            f"only int, tuple and list are allowed for axes, but got {type(axes)}")

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

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.ones((2,2))
        >>> x = np.expand_dims(x,0)
        >>> print(x.shape)
        (1, 2, 2)
    """
    shape = F.shape(a)
    # yield expanded shape based on the axes
    new_shape = _prepare_shape_for_expand_dims(shape, axis)
    return P.Reshape()(a, new_shape)


@constexpr
def _prepare_shape_for_squeeze(shape, axes):
    """
    Creates the squeezed new shape based on the tensor and given axes.

    Args:
        shape (tuple): the shape of the tensor
        axes Union[None, int, tuple(int), list(int)]: the axes with dimensions squeezed.

    Returns:
        new_shape(tuple): the shape with dimensions squeezed.
    """
    new_shape = []
    ndim = len(shape)

    # Convert to set
    if isinstance(axes, int):
        if axes >= ndim or axes < -ndim:
            raise ValueError(
                f"axis {axes} is out of bounds for tensor of dimension {ndim}")
        axes = {axes}

    elif isinstance(axes, (list, tuple)):
        for axis in axes:
            if axis >= ndim or axis < -ndim:
                raise ValueError(
                    f"axis {axis} is out of bounds for tensor of dimension {ndim}")
        axes = set(axes)

    elif axes is not None:
        raise TypeError(
            f"only int, tuple and list are allowed for axes, but got {type(axes)}")

    if axes is None:
        new_shape = [s for s in shape if s != 1]
    else:
        for idx, s in enumerate(shape):
            if s != 1 or (idx not in axes) and (idx - ndim not in axes):
                new_shape.append(s)
            # if an axis is selected with shape entry greater than one, an error is raised.
            if s != 1 and ((idx in axes) or (idx - ndim in axes)):
                raise ValueError(
                    f"axis {axes} has shape entry {s} > 1, cannot be squeezed.")
    return tuple(new_shape)


def squeeze(a, axis=None):
    """
    Removes single-dimensional entries from the shape of an tensor.

    This is a temporary solution to support CPU backend. Will be changed
    once CPU backend supports P.Squeeze().

    Args:
        a (Tensor): Input tensor array.
        axis: Union[None, int, list(int), tuple(list)]. Default is None.

    Returns:
        Tensor, with all or a subset of the dimensions of length 1 removed.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.ones((1,2,2,1))
        >>> x = np.squeeze(x)
        >>> print(x.shape)
        (2, 2)
    """
    shape = F.shape(a)
    # yield squeezed shape based on the axes
    new_shape = _prepare_shape_for_squeeze(shape, axis)
    return P.Reshape()(a, new_shape)


def transpose(a, axes=None):
    """
    Reverses or permutes the axes of a tensor; returns the modified tensor.

    Args:
        a (Tensor): a tensor to be transposed
        axes (Union[None, tuple, list]): the axes order, if axes is None, transpose
        the entire tensor. Default is None.

    Returns:
        Tensor, the transposed tensor array.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.ones((1,2,3))
        >>> x = np.transpose(x)
        >>> print(x.shape)
        (3, 2, 1)
    """
    if axes is None:
        shape = F.shape(a)
        length = F.tuple_len(shape)
        perm = F.make_range(0, length)
        new_order = F.tuple_reversed(perm)
        return P.Transpose()(a, new_order)

    axes = _check_shape_compile(axes)
    return P.Transpose()(a, axes)


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
        TypeError: If axis or start is not integer.
        ValueError: If axis is not in the range from -ndim to ndim-1 or
            start is not in the range from -ndim to ndim.

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.ones((2,3,4))
        >>> output = np.rollaxis(x, 0, 2)
        >>> print(output.shape)
        (3, 2, 4)
    """
    _check_is_int(axis)
    _check_is_int(start)

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

    return P.Transpose()(x, new_perm)


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
        TypeError: If axis1 or axis2 is not integer.
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
    _check_is_int(axis1)
    _check_is_int(axis2)

    shape = F.shape(x)
    ndim = F.tuple_len(shape)

    axes = _check_axes_range((axis1, axis2), ndim)
    axis1, axis2 = axes[0], axes[1]

    if axis1 == axis2:
        return x
    if axis1 > axis2:
        axis1, axis2 = axis2, axis1

    perm = F.make_range(0, ndim)
    new_perm = None
    if axis2 + 1 < ndim:
        new_perm = perm[0:axis1] + perm[axis2:axis2+1] + \
            perm[axis1+1:axis2] + perm[axis1:axis1+1] + perm[axis2+1:]
    else:
        new_perm = perm[0:axis1] + perm[axis2:axis2+1] + \
            perm[axis1+1:axis2] + perm[axis1:axis1+1]

    return P.Transpose()(x, new_perm)


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
        TypeError: If new_shape is not integer, list or tuple.
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
    new_shape = _check_shape_compile(new_shape)
    return P.Reshape()(x, new_shape)


def ravel(x):
    """
    Returns a contiguous flattened tensor.

    A 1-D tensor, containing the elements of the input, is returned.

    Args:
        x (Tensor): A tensor to be flattened.

    Returns:
        Flattened tensor, has the same data type as the original tensor x.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.ones((2,3,4))
        >>> output = np.ravel(x)
        >>> print(output.shape)
        (24,)
    """
    return reshape(x, (-1,))


@constexpr
def _move_axes_for_concatenate(arr_shape, axis):
    """
    Moves axis 0 to the disiganated position, while keeps other axes' relative
    positions unchanged, only used if a single tensor is concatenated.
    """

    original_axes = tuple(range(len(arr_shape)))
    new_axes = original_axes[1:axis+1] + (0,) + original_axes[axis+1:]
    new_shape = arr_shape[1:axis+1] + (arr_shape[0] * arr_shape[axis+1],) + \
        arr_shape[axis+2:]
    return new_axes, new_shape


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
        return P.Concat(axis)(flattened_arrays)

    # convert a list of tensor to a tuple of tensor
    if _check_is_list(array_type):
        arrays = _covert_list_tensor_to_tuple_tensor(arrays)

    arr_shape = F.shape(arrays[0])
    _check_axes_range((axis,), len(arr_shape))

    # if only one tensor in the tuple/list, return the tensor itself
    if len(arrays) == 1:
        return arrays[0]

    return P.Concat(axis)(arrays)
