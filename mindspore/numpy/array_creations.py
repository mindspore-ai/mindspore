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
from copy import copy as py_copy
from itertools import groupby

import numpy as onp

from ..common import Tensor
from ..common import dtype as mstype
from ..ops import functional as F
from ..ops.primitive import constexpr
from ..nn.layer.basic import tril as nn_tril
from ..nn.layer.basic import triu as nn_triu
from .._c_expression import Tensor as Tensor_
from .._c_expression.typing import Float

from .utils import _check_input_for_asarray, _deep_list, _deep_tensor_to_nparray, \
    _expand, _broadcast_to, _is_empty
from .utils_const import _raise_value_error, _empty, _check_axis_valid, _max, _min, _check_same_type, \
    _check_shape_contain_zero, _check_shape, _check_dtype
from .array_ops import transpose

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

    Raises:
        TypeError: If input arguments have types not specified above.
        ValueError: If input `obj` has different sizes at different dimensions.

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

    Raises:
        TypeError: If input arguments have types not specified above.
        ValueError: If input `a` has different sizes at different dimensions.

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
        dtype = mstype.float32

    if isinstance(a, int) and not isinstance(a, bool) and (dtype is None):
        dtype = mstype.int32

    if isinstance(a, bool) and (dtype is None):
        dtype = mstype.bool_

    if isinstance(a, (list, tuple)):
        # Convert all tuple/nested tuples to lists
        a = _deep_list(a)
        # Convert all tensor sub-elements to numpy arrays
        a = _deep_tensor_to_nparray(a)
        a = onp.asarray(a)
        if a.dtype is onp.dtype('object'):
            raise ValueError('Input array must have the same size across all dimensions.')
        # If dtype is not specified, we keep consistent with numpy decision
        # only exceptions are: we use int/float32
        if dtype is None:
            if a.dtype is onp.dtype('int64'):
                dtype = mstype.int32
            elif a.dtype is onp.dtype('float64'):
                dtype = mstype.float32

    if isinstance(a, onp.ndarray) and dtype is None:
        if a.dtype is onp.dtype('bool'):
            dtype = mstype.bool_
        elif a.dtype is onp.dtype('int'):
            dtype = mstype.int32
        elif a.dtype is onp.dtype('float'):
            dtype = mstype.float32
        elif a.dtype is onp.dtype('object'):
            raise TypeError(f"For Tensor conversion, the input_data is {a} that contains unsupported element.")
        a = Tensor.from_numpy(a)

    # If a is already a tensor and we don't need to cast dtype, return a
    if isinstance(a, Tensor):
        if dtype is None:
            return a
        dtype = _check_dtype(dtype)
        if dtype == a.dtype:
            return a

    return Tensor(a, dtype=dtype)


def asfarray(a, dtype=mstype.float32):
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

    Raises:
        TypeError: If input arguments have types not specified above.
        ValueError: If input `a` has different sizes at different dimensions.

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
        dtype = mstype.float32

    if isinstance(a, (list, tuple)):
        # Convert all tuple/nested tuples to lists
        a = _deep_list(a)
        # Convert all tensor sub-elements to numpy arrays
        a = _deep_tensor_to_nparray(a)
        a = onp.asarray(a)
        if a.dtype is onp.dtype('object'):
            raise TypeError(f"For Tensor conversion, the input_data is {a} that contains unsupported element.")
    if isinstance(a, onp.ndarray):
        a = Tensor.from_numpy(a)

    return Tensor(a, dtype)


def copy_(a):
    """
    Returns a tensor copy of the given object.

    Args:
        a (Union[int, float, bool, list, tuple, numpy.ndarray]): Input data, in
        any form that can be converted to a tensor. This includes lists, lists of
        tuples, tuples, tuples of tuples, tuples of lists and numpy.ndarray.

    Returns:
        Tensor, has the same data as `a`.

     Raises:
        TypeError: If input `a` has type not specified above.
        ValueError: If input `a` has different sizes at different dimensions.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.ones((2,2))
        >>> print(np.copy(x))
        [[1. 1.]
         [1. 1.]]
    """
    if not isinstance(a, Tensor):
        a = asarray(a)
    return py_copy(a)


@constexpr
def _fill(shape, value, dtype):
    """Original numpy.full function."""
    return Tensor(onp.full(shape, value), dtype)


def ones(shape, dtype=mstype.float32):
    """
    Returns a new tensor of given shape and type, filled with ones.

    Args:
        shape (Union[int, tuple, list]): the shape of the new tensor.
        dtype (Union[mstype.dtype, str], optional): Designated tensor dtype, can
            be in format of np.float32, or `float32`. Default is mstype.float32.

    Returns:
        Tensor, with the designated shape and dtype, filled with ones.

    Raises:
        TypeError: If input arguments have types not specified above.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.ones((2,2)))
        [[1. 1.]
        [1. 1.]]
    """
    shape = _check_shape(shape)
    dtype = _check_dtype(dtype)
    if _check_shape_contain_zero(shape):
        return _fill(shape, 1.0, dtype)
    output = F.fill(dtype, shape, 1)
    return output


def zeros(shape, dtype=mstype.float32):
    """
    Returns a new tensor of given shape and type, filled with zeros.

    Args:
        shape (Union[int, tuple, list]): the shape of the new tensor.
        dtype (Union[mstype.dtype, str], optional): Designated tensor dtype, can
            be in format of np.float32, or `float32`. Default is mstype.float32.

    Returns:
        Tensor, with the designated shape and dtype, filled with zeros.

    Raises:
        TypeError: If input arguments have types not specified above.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.zeros((2,2)))
        [[0. 0.]
        [0. 0.]]
    """
    shape = _check_shape(shape)
    dtype = _check_dtype(dtype)
    if _check_shape_contain_zero(shape):
        return _fill(shape, 0.0, dtype)
    output = F.fill(dtype, shape, 0)
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

    Returns:
        Tensor, with the designated shape and dtype, filled with `fill_value`.

    Raises:
        TypeError: If input arguments have types not specified above.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

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
        return F.fill(dtype, shape, fill_value)

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

    Raises:
        TypeError: If input arguments have types not specified above, or arguments are
            not given in the correct orders specified above.

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

    Raises:
        TypeError: If input arguments have types not specified above.

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

    tensor_out = Tensor.from_numpy(out)
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

    Raises:
        TypeError: If input arguments have types not specified above.

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


def eye(N, M=None, k=0, dtype=mstype.float32):
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

    Raises:
        TypeError: If input arguments have types not specified above.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.eye(2, 2))
        [[1. 0.]
        [0. 1.]]
    """
    dtype = _check_dtype(dtype)
    if M is None:
        M = N
    if not (isinstance(M, int) and isinstance(N, int) and isinstance(k, int)):
        raise TypeError("Input tensor dimensions should be integers.")
    out = None
    if k != 0 or N == 0 or M == 0:
        # Fall back to original numpy creation method
        out = onp.eye(N, M, k)
    else:
        out = F.eye(N, M, dtype)
    return asarray(out, dtype=dtype)


def identity(n, dtype=mstype.float32):
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

    Raises:
        TypeError: If input arguments have types not specified above.

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.identity(2))
        [[1. 0.]
        [0. 1.]]
    """
    dtype = _check_dtype(dtype)
    return eye(n, dtype=dtype)


def empty(shape, dtype=mstype.float32):
    """
    Returns a new array of given shape and type, without initializing
    entries.

    Note:
        Numpy argument order is not supported.
        Object arrays are not supported.

    Args:
        shape (int or tuple of int): Shape of the empty array, e.g.,
            (2, 3) or 2.
        dtype (data-type): optional. Desired output data-type for the
            array, e.g, numpy.int8. Default is numpy.float32.

    Returns:
        Tensor, array of uninitialized (arbitrary) data of the given
        shape and dtype.

    Raises:
        TypeError: if the input shape or dtype is invalid.


    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> output = np.empty((2, 3))
        >>> print(output)
        Tensor(shape=[2, 3], dtype=Float32, value=
        <uninitialized>)
    """
    shape = _check_shape(shape)
    dtype = _check_dtype(dtype)
    return Tensor_(dtype, shape)


def _shape_matched(fn, arr):
    """Returns the matched shape of elements in arr"""
    shapes_all = groupby(map(fn, arr))
    shape = next(shapes_all)[0]
    if next(shapes_all, False):
        return _raise_value_error('Input array must have the same size across a dimension.')
    return shape


def _get_shape(array_like):
    """Returns the shape of the array like object by recursion."""
    if isinstance(array_like, Tensor):
        return F.shape(array_like)
    if isinstance(array_like, onp.ndarray):
        return array_like.shape
    if isinstance(array_like, (list, tuple)):
        shape = _shape_matched(_get_shape, array_like)
        return (len(array_like),) + shape
    return ()


def _get_dtype(array_like):
    """Returns the data type of the array like object."""
    if isinstance(array_like, Tensor):
        return F.dtype(array_like)
    if isinstance(array_like, onp.ndarray):
        return mstype.pytype_to_dtype(array_like.dtype)
    if isinstance(array_like, (list, tuple)):
        return asarray(array_like).dtype
    return mstype.float32


def _x_like(prototype, dtype, shape, constructor, fill_value=None):
    """
    Returns a tensor with the same shape and type as prototype,
    using constructor.
    """
    _ = _check_input_for_asarray(prototype)
    dtype_out = dtype
    shape_out = shape
    if not dtype_out:
        dtype_out = _get_dtype(prototype)
    if not shape_out and shape_out != 0:
        shape_out = _get_shape(prototype)
    if fill_value is not None:
        return constructor(shape_out, fill_value, dtype_out)
    return constructor(shape_out, dtype_out)


def empty_like(prototype, dtype=None, shape=None):
    """
    Returns a new array with the same shape and type as a given array.

    Note:
        Since list or tuple arrays are not supported, input array
        must have the same size across a dimension.
        If prototype is not a Tensor or a numpy array, dtype is
        float32 by default if not provided.

    Args:
        prototype (array_like): The shape and data-type of prototype
            define these same attributes of the returned array.
        dtype (data-type): optional. Overrides the data type of the
            result.
        shape (int or sequence of ints): optional. Overrides the shape
            of the result.

    Returns:
        Tensor, array of uninitialized (arbitrary) data with the same
        shape and type as prototype.

    Raises:
        ValueError: if prototype does not have the same shape across each
        dimension.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = [[(1, 2)], onp.ones((1, 2)), [[2, 3]], onp.ones((1, 2))]
        >>> output = np.empty_like(a)
        >>> print(output)
        Tensor(shape=[4, 1, 2], dtype=Float32, value=
        <uninitialized>)
    """
    return _x_like(prototype, dtype, shape, empty)


def ones_like(a, dtype=None, shape=None):
    """
    Returns an array of ones with the same shape and type as a given array.

    Note:
        Since list or tuple arrays are not supported, input array
        must have the same size across a dimension.
        If a is not a Tensor or a numpy array, dtype is float32 by default
        if not provided.

    Args:
        a (array_like): The shape and data-type of a define these same
            attributes of the returned array.
        dtype (data-type): optional. Overrides the data type of the
            result.
        shape (int or sequence of ints): optional. Overrides the shape
        of the result.

    Returns:
        Tensor, array of ones with the same shape and type as a.

    Raises:
        ValueError: if prototype does not have the same shape across each
        dimension.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = [[(1, 2)], np.ones((1, 2)), [[2, 3]], np.ones((1, 2))]
        >>> output = np.ones_like(a)
        >>> print(output)
        [[[1. 1.]]

        [[1. 1.]]

        [[1. 1.]]

        [[1. 1.]]]
    """
    return _x_like(a, dtype, shape, ones)


def zeros_like(a, dtype=None, shape=None):
    """
    Returns an array of zeros with the same shape and type as a given array.

    Note:
        Since list or tuple arrays are not supported, input array
        must have the same size across a dimension.
        If a is not a Tensor or a numpy array, dtype is float32 by default
        if not provided.

    Args:
        a (array_like): The shape and data-type of a define these same
            attributes of the returned array.
        dtype (data-type): optional. Overrides the data type of the
            result.
        shape (int or sequence of ints): optional. Overrides the shape
            of the result.

    Returns:
        Tensor, array of zeros with the same shape and type as a.

    Raises:
        ValueError: if prototype does not have the same shape across each
        dimension.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = [[(1, 2)], np.ones((1, 2)), [[2, 3]], np.ones((1, 2))]
        >>> output = np.zeros_like(a)
        >>> print(output)
        [[[0. 0.]]

        [[0. 0.]]

        [[0. 0.]]

        [[0. 0.]]]
    """
    return _x_like(a, dtype, shape, zeros)


def full_like(a, fill_value, dtype=None, shape=None):
    """
    Returns a full array with the same shape and type as a given array.

    Note:
        Since list or tuple arrays are not supported, input array
        must have the same size across a dimension.
        If a is not a Tensor or a numpy array, dtype is float32 by default
        if not provided.

    Args:
        a (array_like): The shape and data-type of a define these same
            attributes of the returned array.
        fill_value (scalar): Fill value.
        dtype (data-type): optional. Overrides the data type of the
            result.
        shape (int or sequence of ints): optional. Overrides the shape
            of the result.

    Returns:
        Tensor, array of fill_value with the same shape and type as a.

    Raises:
        ValueError: if prototype does not have the same shape across each
        dimension.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = [[(1, 2)], onp.ones((1, 2)), [[2, 3]], onp.ones((1, 2))]
        >>> output = np.full_like(a, 0.5)
        >>> print(output)
        [[[0.5 0.5]]

        [[0.5 0.5]]

        [[0.5 0.5]]

        [[0.5 0.5]]]
    """
    return _x_like(a, dtype, shape, full, fill_value=fill_value)


def tri(N, M=None, k=0, dtype=mstype.float32):
    """
    Returns an array with ones at and below the given diagonal and zeros elsewhere.

    Args:
        N(int): Number of rows in the array.
        M(int, optional): Number of columns in the array. By default, M is taken
            equal to N.
        k(int, optional): The sub-diagonal at and below which the array is filled.
            k = 0 is the main diagonal, while k < 0 is below it, and k > 0 is above.
            The default is 0.
        dtype(mstype.dtype, optional): Data type of the returned array. The default
            is mstype.float32.

    Returns:
        tri(Tensor): Tensor with shape (N, M), with its lower triangle filled with
            ones and zeros elsewhere; in other words T[i,j] == 1 for j <= i + k,
            0 otherwise.

    Raises:
        TypeError: If input arguments have types not specified above.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
    >>> import mindspore.numpy as np
    >>> output = np.tri(3, 3, 1)
    >>> print(output)
    [[1. 1. 0.]
     [1. 1. 1.]
     [1. 1. 1.]]
    """
    if M is None:
        M = N
    return nn_tril((N, M), dtype, k)


def tril(m, k=0):
    """
    Returns a lower triangle of an array.

    Returns a copy of an array with elements above the k-th diagonal zeroed.

    Args:
        m(array_like): The shape and data-type of m define these same
            attributes of the returned array.
        k(int, optional): Diagonal above which to zero elements. k = 0 (the default)
            is the main diagonal, k < 0 is below it and k > 0 is above.

    Returns:
        tril(Tensor): Lower triangle of m, of same shape and data-type as m.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Raises:
        TypeError: If input arguments have types not specified above.
        ValueError: If input m's rank < 1.

    Examples:
    >>> import mindspore.numpy as np
    >>> output = np.tril(np.ones((3, 3)))
    >>> print(output)
    [[1. 0. 0.]
     [1. 1. 0.]
     [1. 1. 1.]]
    """
    m = asarray(m)
    shape = _get_shape(m)
    dtype = _get_dtype(m)
    m = m.astype(mstype.float32)
    assist = nn_tril(shape, mstype.float32, k)
    return F.tensor_mul(assist, m).astype(dtype)


def triu(m, k=0):
    """
    Returns an upper triangle of an array.

    Returns a copy of an array with elements below the k-th diagonal zeroed.

    Args:
        m(array_like): The shape and data-type of m define these same
            attributes of the returned array.
        k(int, optional): Diagonal below which to zero elements. k = 0 (the default)
            is the main diagonal, k < 0 is below it and k > 0 is above.

    Returns:
        triu(Tensor): Upper triangle of m, of same shape and data-type as m.

    Raises:
        TypeError: If input arguments have types not specified above.
        ValueError: If input m's rank < 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
    >>> import mindspore.numpy as np
    >>> output = np.triu(np.ones((3, 3)))
    >>> print(output)
    [[1. 1. 1.]
     [0. 1. 1.]
     [0. 0. 1.]]
    """
    m = asarray(m)
    shape = _get_shape(m)
    dtype = _get_dtype(m)
    m = m.astype(mstype.float32)
    assist = nn_triu(shape, mstype.float32, k)
    return F.tensor_mul(assist, m).astype(dtype)


def diagonal(a, offset=0, axis1=0, axis2=1):
    """
    Returns specified diagonals.

    If `a` is 2-D, returns the diagonal of a with the given offset, i.e., the
    collection of elements of the form a[i, i+offset]. If `a` has more than two
    dimensions, then the axes specified by axis1 and axis2 are used to determine
    the 2-D sub-array whose diagonal is returned. The shape of the resulting
    array can be determined by removing axis1 and axis2 and appending an index
    to the right equal to the size of the resulting diagonals.

    Args:
        a (Tensor): Array from which the diagonals are taken.
        offset (int): optional. Offset of the diagonal from the main diagonal.
            Can be positive or negative. Defaults to main diagonal (0).
        axis1 (int): optional. Axis to be used as the first axis of the 2-D
            sub-arrays from which the diagonals should be taken. Defaults to
            first axis (0).
        axis2 (int): optional. Axis to be used as the second axis of the 2-D
            sub-arrays from which the diagonals should be taken. Defaults to
            second axis (1).

    Returns:
        Tensor, if `a` is 2-D, then a 1-D array containing the diagonal. If
        a.ndim > 2, then the dimensions specified by axis1 and axis2 are removed,
        and a new axis inserted at the end corresponding to the diagonal.

    Raises:
        ValueError: if the input tensor has less than two dimensions.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> a = np.arange(4).reshape(2,2)
        >>> print(a)
        [[0 1]
        [2 3]]
        >>> output = np.diagonal(a)
        >>> print(output)
        [0 3]
        >>> output = np.diagonal(a, 1)
        >>> print(output)
        [1]
        >>> a = np.arange(8).reshape(2, 2, 2)
        >>> print(a)
        [[[0 1]
        [2 3]]

        [[4 5]
        [6 7]]]
        >>> output = np.diagonal(a, 0, 0, 1)
        >>> print(output)
        [[0 6]
        [1 7]]
    """
    ndim = F.rank(a)
    if ndim < 2:
        return _raise_value_error('diagonal requires an array of at least two dimensions')
    dtype = F.dtype(a)

    if _is_empty(F.shape(a)):
        return _empty(dtype, (0,))

    cast_type = dtype
    if not isinstance(dtype, Float):
        # reduce_sum only supports float types
        cast_type = mstype.float32
        a = F.cast(a, cast_type)

    axes = _check_axis_valid((axis1, axis2), ndim)
    perm = ()
    for i in range(ndim):
        if i not in axes:
            perm += (i,)
    perm += axes
    a = transpose(a, perm)

    shape = F.shape(a)
    n, m = shape[-2:]
    e = _eye(n, m, offset, cast_type)
    e = _expand(e, ndim)
    e = _broadcast_to(e, F.shape(e), F.shape(a), ndim)

    prod = F.tensor_mul(a, e)
    res = F.reduce_sum(prod, -1)

    begin = ()
    for i in range(ndim-2):
        begin += (0,)
    last_dim_begin = _max(0, -offset)
    begin += (last_dim_begin,)
    size = F.shape(res)[:-1]
    last_dim_end = _min(
        shape[-2], _max(0, shape[-1] - offset)) - last_dim_begin
    if last_dim_end <= 0:
        return _empty(dtype, size + (0,))
    size += (last_dim_end,)
    res = F.tensor_slice(res, begin, size)
    if not _check_same_type(cast_type, dtype):
        res = F.cast(res, dtype)
    return res


@constexpr
def _eye(N, M, k, dtype):
    return eye(N=N, M=M, k=k, dtype=dtype)


def trace(a, offset=0, axis1=0, axis2=1):
    """
    Returns the sum along diagonals of the array.

    If `a` is 2-D, the sum along its diagonal with the given offset is returned,
    i.e., the sum of elements a[i,i+offset] for all i.
    If `a` has more than two dimensions, then the axes specified by axis1 and
    axis2 are used to determine the 2-D sub-arrays whose traces are returned.
    The shape of the resulting array is the same as that of a with axis1 and
    axis2 removed.

    Note:
        Numpy arguments dtype and out are not supported.

    Args:
        a (Tensor): Array from which the diagonals are taken.
        offset (int): optional. Offset of the diagonal from the main diagonal.
            Can be positive or negative. Defaults to main diagonal (0).
        axis1 (int): optional. Axis to be used as the first axis of the 2-D
            sub-arrays from which the diagonals should be taken. Defaults to
            first axis (0).
        axis2 (int): optional. Axis to be used as the second axis of the 2-D
            sub-arrays from which the diagonals should be taken. Defaults to
            second axis (1).

    Returns:
        Tensor, sum_along_diagonals. If a is 2-D, the sum along the diagonal
        is returned. If a has larger dimensions, then an array of sums along
        diagonals is returned.

    Raises:
        ValueError: if the input tensor has less than two dimensions.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> output = np.trace(np.eye(3))
        >>> print(output)
        3.0
        >>> a = np.arange(8).reshape((2,2,2))
        >>> output = np.trace(a)
        >>> print(output)
        [6 8]
        >>> a = np.arange(24).reshape((2,2,2,3))
        >>> output = np.trace(a).shape
        >>> print(output)
        (2, 3)
    """
    d = diagonal(a, offset, axis1=axis1, axis2=axis2)
    shape = F.shape(d)
    dtype = F.dtype(d)
    if shape[-1] == 0:
        return _empty(dtype, shape[:-1])

    cast_type = dtype
    if not isinstance(dtype, Float):
        # reduce sum only supports float types
        cast_type = mstype.float32
        d = F.cast(d, cast_type)
    res = F.reduce_sum(d, -1)
    if not _check_same_type(cast_type, dtype):
        res = F.cast(res, dtype)
    return res
