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
import numpy as onp

from ..common import Tensor
from ..common import dtype as mstype
from ..ops import functional as F
from ..ops.primitive import constexpr
from ..nn.layer.basic import tril as nn_tril
from ..nn.layer.basic import triu as nn_triu
from .._c_expression import Tensor as Tensor_

from .utils import _check_input_for_asarray, _deep_list, _deep_tensor_to_nparray, \
    _broadcast_to_shape, _check_input_tensor, _convert_64_to_32, _get_dtype_from_scalar, \
    _expand
from .utils_const import _raise_value_error, _empty, _check_axis_valid, _max, _min, \
    _check_same_type, _is_shape_empty, _check_shape, _check_dtype, _tile_size, _abs, \
    _raise_type_error, _expanded_shape, _check_is_float, _iota, _type_convert, \
    _canonicalize_axis, _list_comprehensions, _ceil, _tuple_getitem, _tuple_slice
from .array_ops import transpose, ravel, concatenate, broadcast_arrays, reshape, broadcast_to
from .dtypes import nan

# According to official numpy reference, the dimension of a numpy array must be less
# than 32
MAX_NUMPY_DIMS = 32
# All types that can be accepted as "array_like" parameters in graph mode.
ARRAY_TYPES = (int, float, bool, list, tuple, Tensor)


def array(obj, dtype=None, copy=True, ndmin=0):
    """
    Creates a tensor.

    This function creates tensors from an array-like object.

    Args:
        obj (Union[int, float, bool, list, tuple]): Input data, in any form that
            can be converted to a `Tensor`. This includes Tensor, list, tuple and numbers.
        dtype (Union[:class:`mindspore.dtype`, str], optional): Designated tensor dtype, can
            be in format of np.int32, or \'int32\'. If dtype is :class:`None`, the data type
            of the new tensor will be inferred from obj. Default is :class:`None`.
        copy (bool): If `True`, then the object is copied. Otherwise, a copy will
            only be made if necessary. Default: `True`.
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
    res = asarray(obj, dtype)
    if ndmin > res.ndim:
        res = _expand(res, ndmin)

    if copy:
        res = copy_(res)
    elif dtype is not None and dtype != res.dtype:
        res = res.astype(dtype)

    return res


@constexpr
def asarray_const(a, dtype=None):
    """Converts the input to tensor. Note here `a` cannot be tensor itself."""
    _check_input_for_asarray(a)

    if dtype is not None:
        dtype = _check_dtype(dtype)

    if isinstance(a, (float, int, bool)) and dtype is None:
        dtype = _get_dtype_from_scalar(a)

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
            dtype = mstype.pytype_to_dtype(a.dtype)
            if dtype == mstype.float64:
                dtype = mstype.float32
            elif dtype == mstype.int64:
                dtype = mstype.int32

    if isinstance(a, onp.ndarray) and dtype is None:
        if a.dtype is onp.dtype('object'):
            raise TypeError(f"For Tensor conversion, the input_data is {a} that contains unsupported element.")
        dtype = mstype.pytype_to_dtype(a.dtype)
        a = Tensor.from_numpy(a)

    return Tensor(a, dtype=dtype)


def asarray(a, dtype=None):
    """
    Converts the input to tensor.

    This function converts tensors from an array-like object.

    Args:
        a (Union[int, float, bool, list, tuple, Tensor]): Input data, in any form that can
            be converted to a `Tensor`. This includes Tensor, list, tuple and numbers.
        dtype (Union[:class:`mindspore.dtype`, str], optional): Designated tensor dtype, can
            be in format of np.int32, or \'int32\'. If dtype is :class:`None`, the data type
            of the new tensor will be inferred from obj. Default is :class:`None`.

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
    if isinstance(a, Tensor):
        if dtype is None or dtype == a.dtype:
            return a
        return a.astype(dtype)
    return asarray_const(a, dtype)


@constexpr
def asfarray_const(a, dtype=mstype.float32):
    """Converts the input to tensor. Note here `a` cannot be tensor itself."""
    _check_input_for_asarray(a)
    if isinstance(a, (list, tuple)):
        # Convert all tuple/nested tuples to lists
        a = _deep_list(a)
        # Convert all tensor sub-elements to numpy arrays
        a = _deep_tensor_to_nparray(a)
        a = onp.asarray(a)
        if a.dtype is onp.dtype('object'):
            raise ValueError(f"For Tensor conversion, the input_data is {a} that contains unsupported element.")
        a = Tensor.from_numpy(a)

    return Tensor(a, dtype)


def asfarray(a, dtype=mstype.float32):
    """
    Similar to asarray, converts the input to a float tensor.

    If non-float dtype is defined, this function will return a float32 tensor instead.

    Args:
       a (Union[int, float, bool, list, tuple, Tensor]): Input data, in any form that can
            be converted to a `Tensor`. This includes Tensor, list, tuple and numbers.
        dtype (Union[:class:`mindspore.dtype`, str], optional): Designated tensor dtype, can
            be in format of np.int32, or \'int32\'. If dtype is :class:`None`, the data type
            of the new tensor will be inferred from `a`. Default is :class:`mindspore.float32`.


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
    if dtype is None:
        return asarray(a)

    dtype = _check_dtype(dtype)
    # pylint: disable=consider-using-in
    if dtype != mstype.float16 and dtype != mstype.float32 and dtype != mstype.float64:
        dtype = mstype.float32

    if isinstance(a, Tensor):
        return a.astype(dtype)

    return asfarray_const(a, dtype)


def copy_(a):
    """
    Returns a tensor copy of the given object.

    Args:
        a (Union[int, float, bool, list, tuple, Tensor]): Input data, in any form that can
            be converted to a `Tensor`. This includes Tensor, list, tuple and numbers.

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
        a = asarray_const(a)
    # The current implementation registers a new memory location for copied tensor by
    # doing some reduandent operations.
    origin_dtype = a.dtype
    if origin_dtype == mstype.bool_:
        return F.logical_not(F.logical_not(a))
    if origin_dtype != mstype.float64:
        a = a.astype("float32")
    a = a / ones_like(a)
    a = a.astype(origin_dtype)
    return a

def ones(shape, dtype=mstype.float32):
    """
    Returns a new tensor of given shape and type, filled with ones.

    Args:
        shape (Union[int, tuple, list]): the shape of the new tensor.
        dtype (Union[:class:`mindspore.dtype`, str], optional): Designated tensor dtype.
            Default is :class:`mstype.float32`.

    Returns:
        Tensor, with the designated `shape` and `dtype`, filled with ones.

    Raises:
        TypeError: If input arguments have types not specified above.
        ValueError: If `shape` entries have values :math:`< 0`.

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
    if _is_shape_empty(shape):
        return full(shape, 1.0, dtype)
    output = F.fill(dtype, shape, 1)
    return output


def zeros(shape, dtype=mstype.float32):
    """
    Returns a new tensor of given shape and type, filled with zeros.

    Args:
        shape (Union[int, tuple, list]): the shape of the new tensor.
        dtype (Union[:class:`mindspore.dtype`, str], optional): Designated tensor dtype.
            Default is :class:`mstype.float32`.

    Returns:
        Tensor, with the designated `shape` and `dtype`, filled with zeros.

    Raises:
        TypeError: If input arguments have types not specified above.
        ValueError: If `shape` entries have values :math:`< 0`.

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
    if _is_shape_empty(shape):
        return full(shape, 0.0, dtype)
    output = F.fill(dtype, shape, 0)
    return output


def full(shape, fill_value, dtype=None):
    """
    Returns a new tensor of given shape and type, filled with `fill_value`.

    Args:
        shape (Union[int, tuple(int), list(int)]): Shape of the new tensor, e.g.,
            :math:`(2, 3)` or :math:`2`.
        fill_value (Union[int, float, bool, list, tuple]): Scalar or array_like
            fill value.
        dtype (Union[:class:`mindspore.dtype`, str], optional): Designated tensor dtype,
            if `dtype` is :class:`None`, the data type of the new tensor will be inferred from
            `fill_value`. Default is :class:`None`.

    Returns:
        Tensor, with the designated shape and dtype, filled with `fill_value`.

    Raises:
        TypeError: If input arguments have types not specified above.
        ValueError: If `shape` has entries < 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.full((2,2), True))
        [[True True]
        [True True]]
    """
    shape = _check_shape(shape)
    if not isinstance(fill_value, ARRAY_TYPES):
        _raise_type_error("fill value should be int, float, bool, list, tuple, Tensor, but got", fill_value)
    if dtype is not None:
        dtype = _check_dtype(dtype)
    else:
        if isinstance(fill_value, (int, float, bool)):
            dtype = _get_dtype_from_scalar(fill_value)
        if isinstance(fill_value, Tensor):
            dtype = fill_value.dtype

    if not _is_shape_empty(shape):
        if isinstance(fill_value, (int, float, bool)):
            return F.fill(dtype, shape, fill_value)
        if isinstance(fill_value, (list, tuple)):
            fill_value = asarray_const(fill_value)
        return broadcast_to(fill_value, shape)
    # if shape contains zero, use c.Tensor()
    return _convert_64_to_32(empty_compile(dtype, shape))


def arange(start, stop=None, step=None, dtype=None):
    """
    Returns evenly spaced values within a given interval.

    Args:
        start(Union[int, float]): Start of interval. The interval includes this value.
            When `stop` is provided as a position argument, `start` must be given, when `stop`
            is a normal argument, `start` can be optional, and default is 0.
            Please see additional examples below.
        stop(Union[int, float], optional): End of interval. The interval does not
            include this value, except in some cases where `step` is not an integer
            and floating point round-off affects the length of out.
        step(Union[int, float], optional): Spacing between values. For any output
            `out`, this is the distance between two adjacent values, :math:`out[i+1] - out[i]`.
            The default step size is 1. If `step` is specified as a position argument,
            `start` must also be given.
        dtype (Union[:class:`mindspore.dtype`, str], optional): Designated tensor dtype.
            If dtype is None, the data type of the new tensor will be inferred from start,
            stop and step. Default is None.

    Returns:
        Tensor with evenly spaced values.

    Raises:
        TypeError(PyNative Mode) or RuntimeError(Graph Mode): If input arguments
            have types not specified above, or arguments are not given in the correct
            orders specified above.

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
    # This implementation was inspired by jax.numpy.arange
    # infer the dtype
    if dtype is None:
        dtype = _get_dtype_from_scalar(start, stop, step)
    if stop is None and step is None: # (start, stop, step) -> (0, start, 1)
        num = _ceil(start)
        out = _iota(mstype.float32, num)
    elif step is None:  # (start, stop, step) -> (start, stop, 1)
        num = _ceil(stop - start)
        out = _iota(mstype.float32, num) + start
    elif stop is None: # (start, stop, step) -> (0, start, step)
        num = _ceil(start / step)
        out = _iota(mstype.float32, num) * step
    else:
        num = _ceil((stop - start) / step)
        out = _iota(mstype.float32, num) * step + start
    return out.astype(dtype)


def _type_checking_for_xspace(start, stop, num, endpoint, dtype, axis):
    """utility parameter checking function for linspace, logspace, geomspace."""
    if not isinstance(start, ARRAY_TYPES):
        _raise_type_error("start should be int, float, bool, list, tuple, Tensor, but got", start)
    if not isinstance(stop, ARRAY_TYPES):
        _raise_type_error("end should be int, float, bool, list, tuple, Tensor, but got", stop)
    if not isinstance(start, Tensor):
        start = _type_convert(Tensor, start).astype(mstype.float32)
    if not isinstance(stop, Tensor):
        stop = _type_convert(Tensor, stop).astype(mstype.float32)
    if not isinstance(num, int):
        _raise_type_error("num should be an integer, but got ", num)
    if not isinstance(endpoint, bool):
        _raise_type_error("endpoint should be an boolean, but got ", endpoint)
    if dtype is not None:
        dtype = _check_dtype(dtype)
    else:
        dtype = mstype.float32
    start, stop = broadcast_arrays(start, stop)
    axis = _canonicalize_axis(axis, start.ndim+1)
    return start, stop, num, endpoint, dtype, axis


def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    """
    Returns evenly spaced values within a given interval.

    Args:
        start (Union[int, list(int), tuple(int), tensor]): The starting value of the sequence.
        stop (Union[int, list(int), tuple(int), tensor]): The end value of the sequence,
            unless `endpoint` is set to False. In that case, the sequence consists
            of all but the last of `num + 1` evenly spaced samples, so that `stop`
            is excluded.  Note that the step size changes when `endpoint` is False.
        num (int, optional): Number of samples to generate. Default is 50.
        endpoint (bool, optional): If True, `stop` is the last sample. Otherwise, it is
            not included. Default is True.
        retstep (bool, optional): If True, return (`samples`, `step`), where `step` is
            the spacing between samples.
        dtype (Union[:class:`mindspore.dtype`, str], optional): Designated tensor dtype,
            If `dtype` is None, infer the data type from other input arguments. Default is None.
        axis (int, optional): The axis in the result to store the samples. Relevant
            only if start or stop are array-like.  By default :math:`(0)`, the samples will
            be along a new axis inserted at the beginning. Use :math:`-1` to get an axis at the end.
            Default is :math:`0`.

    Returns:
        Tensor, with `num` equally spaced samples in the closed interval
        :math:`[start, stop]` or the half-open interval :math:`[start, stop)`
        (depending on whether `endpoint` is True or False).

        Step, the size of spacing between samples, only returned if `retstep` is True.

    Raises:
        TypeError: If input arguments have types not specified above.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.linspace(0, 5, 6))
        [0. 1. 2. 3. 4. 5.]
    """
    # This implementation was inspired by jax.numpy.linspace and numpy.linspace
    start, stop, num, endpoint, dtype, axis = _type_checking_for_xspace(start, stop, num, endpoint, dtype, axis)
    if not isinstance(retstep, bool):
        _raise_type_error("retstep should be an boolean, but got ", retstep)
    bounds_shape = start.shape
    bounds_shape = _tuple_slice(bounds_shape, None, axis) + (1,) + _tuple_slice(bounds_shape, axis, None)
    iota_shape = _list_comprehensions(start.ndim+1, 1, True)
    iota_shape = _tuple_slice(iota_shape, None, axis) + (num,) + _tuple_slice(iota_shape, axis+1, None)
    num_tensor = _type_convert(Tensor, num).astype(mstype.float32)
    div = (num_tensor - 1) if endpoint else num_tensor

    if num > 1:
        delta = (stop - start) / div
        # This is similar to how numpy and jax compute linspace
        start_expand = reshape(start, bounds_shape)
        incremental_expand = reshape(_iota(mstype.float32, num), iota_shape)
        delta_expand = reshape(delta, bounds_shape)
        start_expand, incremental_expand, delta_expand = broadcast_arrays(
            start_expand, incremental_expand, delta_expand)
        out = start_expand + (incremental_expand * delta_expand)
    elif num == 1:
        delta = nan if endpoint else stop - start
        out = reshape(start, bounds_shape)
    else:  # num == 0
        delta = nan
        out = _type_convert(Tensor, []).astype(dtype)
    if retstep:
        return out.astype(dtype), delta
    return out.astype(dtype)


def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
    """
    Returns numbers spaced evenly on a log scale.

    In linear space, the sequence starts at base ** start (base to the power of
    start) and ends with base ** stop (see endpoint below).

    Args:
        start (Union[int, list(int), tuple(int), tensor]): ``base ** start`` is the starting
            value of the sequence.
        stop (Union[int, list(int), tuple(int), tensor]): ``base ** stop`` is the final value of
            the sequence, unless `endpoint` is False. In that case, ``num + 1`` values are spaced
            over the interval in log-space, of which all but the last (a sequence of length num)
            are returned.
        num (int, optional): Number of samples to generate. Default is 50.
        endpoint (bool, optional): If True, `stop` is the last sample. Otherwise, it is
            not included. Default is True.
        base (Union[int, float], optional): The base of the log space. The step size
            between the elements in :math:`ln(samples) / ln(base)` (or :math:`log_{base}(samples)`)
            is uniform. Default is :math:`10.0`.
        dtype (Union[:class:`mindspore.dtype`, str], optional): Designated tensor dtype.
            If `dtype` is None, infer the data type from other input arguments. Default is None.
        axis (int, optional): The axis in the result to store the samples. Relevant
            only if start or stop is array-like.  By default (:math:`0`), the samples will
            be along a new axis inserted at the beginning. Use :math:`-1` to get an axis at the end.
            Default is :math:`0`.

    Returns:
        Tensor, equally spaced on a log scale.

    Raises:
        TypeError: If input arguments have types not specified above.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.logspace(0, 5, 6, base=2.0))
        [ 1.  2.  4.  8. 16. 32.]
    """
    # This implementation was inspired by jax.numpy.linspace and numpy.linspace
    start, stop, num, endpoint, dtype, axis = _type_checking_for_xspace(start, stop, num, endpoint, dtype, axis)
    if not isinstance(base, (int, float, bool)):
        _raise_type_error("base should be a number, but got ", base)
    linspace_res = linspace(start, stop, num, endpoint=endpoint, retstep=False, dtype=None, axis=axis)
    return F.tensor_pow(base, linspace_res).astype(dtype)


def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    """
    Returns numbers spaced evenly on a log scale (a geometric progression).

    This is similar to logspace, but with endpoints specified directly. Each output sample
    is a constant multiple of the previous.

    Args:
        start (Union[int, list(int), tuple(int), tensor]): The starting value of the sequence.
        stop (Union[int, list(int), tuple(int), tensor]): The final value of the sequence,
            unless endpoint is False. In that case, num + 1 values are spaced over the
            interval in log-space, of which all but the last (a sequence of length num) are
            returned.
        num (int, optional): Number of samples to generate. Default is 50.
        endpoint (bool, optional): If True, `stop` is the last sample. Otherwise, it is
            not included. Default is True.
        dtype (Union[:class:`mindspore.dtype`, str], optional): Designated tensor dtype, can
            be in format of np.float32, or `float32`.If `dtype` is None, infer the data
            type from other input arguments. Default is None.
        axis (int, optional): The axis in the result to store the samples. Relevant
            only if start or stop is array-like.  By default (0), the samples will
            be along a new axis inserted at the beginning. Use -1 to get an axis at the end.
            Default is 0.

    Returns:
        Tensor, with samples equally spaced on a log scale.

    Raises:
        TypeError: If input arguments have types not specified above.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> output = np.geomspace(1, 256, num=9)
        >>> print(output)
        [  1.   2.   4.   8.  16.  32.  64. 128. 256.]
        >>> output = np.geomspace(1, 256, num=8, endpoint=False)
        >>> print(output)
        [  1.   2.   4.   8.  16.  32.  64. 128.]
    """
    start, stop, num, endpoint, dtype, axis = _type_checking_for_xspace(start, stop, num, endpoint, dtype, axis)
    root = num
    if endpoint:
        root -= 1
    bases = F.tensor_pow(F.tensor_div(stop, start), asarray_const(1/(root)))
    exponents = linspace(zeros(F.shape(bases)), F.fill(F.dtype(bases), F.shape(bases), root),
                         num, endpoint=endpoint, dtype=dtype, axis=axis)
    shape = F.shape(bases)
    axis = axis + F.rank(bases) + 1 if axis < 0 else axis
    expanded_shape = _tuple_getitem(shape, axis, False) + (1,) + _tuple_getitem(shape, axis)
    bases = F.reshape(bases, expanded_shape)
    start = F.reshape(start, expanded_shape)
    res = F.tensor_mul(F.tensor_pow(bases, exponents), start)
    if dtype is not None:
        res = F.cast(res, dtype)
    return res


def eye(N, M=None, k=0, dtype=mstype.float32):
    """
    Returns a 2-D tensor with ones on the diagnoal and zeros elsewhere.

    Args:
        N (int): Number of rows in the output, must be larger than 0.
        M (int, optional): Number of columns in the output. If is :class:`None`, defaults to `N`,
            if defined, must be larger than 0. Deault is :class:`None`.
        k (int, optional): Index of the diagonal: 0 (the default) refers to the main
            diagonal, a positive value refers to an upper diagonal, and a negative value
            to a lower diagonal. Default is 0.
        dtype (Union[:class:`mindspore.dtype`, str], optional): Designated tensor dtype.
            Default is mstype.float32.

    Returns:
        A tensor of shape (N, M). A tensor where all elements are equal to zero,
        except for the k-th diagonal, whose values are equal to one.

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
        _raise_type_error("Input tensor dimensions should be integers.")
    out = None
    if N == 0 or M == 0:
        # Fill the shape with any value is fine.
        return full((N, M), 0, dtype)

    out = F.eye(N, M, dtype)

    if k >= M or k <= -N:
        return full((N, M), 0, dtype)
    if k != 0:
        out = out.astype(mstype.float32)
        if k > 0:
            out_left = full((N, k), 0, dtype)
            out_right = out[..., 0:M-k:1]
            return concatenate((out_left, out_right), 1).astype(dtype)
        if k < 0:
            out_upper = full((-k, M), 0, dtype)
            out_lower = out[0:N+k:1, ...]
            return concatenate((out_upper, out_lower), 0).astype(dtype)
    return out


def identity(n, dtype=mstype.float32):
    """
    Returns the identity tensor.

    Args:
        n (int): Number of rows and columns in the output, must be larger than 0.
        dtype (Union[:class:`mindspore.dtype`, str], optional): Designated tensor dtype,
            default is :class:`mstype.float32`.

    Returns:
        A tensor of shape `(n, n)`, where all elements are equal to zero,
        except for the diagonal, whose values are equal to one.

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
    if not isinstance(n, int):
        _raise_type_error("Input tensor dimensions should be integers.")
    dtype = _check_dtype(dtype)
    return eye(n, dtype=dtype)


@constexpr
def empty_compile(dtype, shape):
    return Tensor_(dtype, shape)


def empty(shape, dtype=mstype.float32):
    """
    Returns a new array of given shape and type, without initializing
    entries.

    Note:
        Numpy argument `order` is not supported.
        Object arrays are not supported.

    Args:
        shape (Union[int, tuple(int)]): Shape of the empty array, e.g.,
            (2, 3) or 2.
        dtype (:class:`mindspore.dtype`, optional): Desired output data-type for the
            array, e.g, mstype.int8. Default is mstype.float32.

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
        # result may vary
        Tensor(shape=[2, 3], dtype=Float32, value=
        <uninitialized>)
    """
    shape = _check_shape(shape)
    dtype = _check_dtype(dtype)
    return empty_compile(dtype, shape)


def _get_shape(array_like):
    """Returns the shape of the array like object."""
    if isinstance(array_like, Tensor):
        return array_like.shape
    return asarray_const(array_like).shape


def _get_dtype(array_like):
    """Returns the data type of the array like object."""
    if isinstance(array_like, Tensor):
        return array_like.dtype
    return asarray_const(array_like).dtype


def _x_like(prototype, dtype, shape, constructor, fill_value=None):
    """
    Returns a tensor with the same shape and type as prototype,
    using constructor.
    """
    if not isinstance(prototype, ARRAY_TYPES):
        _raise_type_error("prototype should be int, float, bool, list, tuple, Tensor, but got", prototype)
    dtype_out = dtype
    shape_out = shape
    if dtype_out is None:
        dtype_out = _get_dtype(prototype)
    if shape_out is None or isinstance(shape_out, (list, tuple)) and not shape_out:
        shape_out = _get_shape(prototype)
    if fill_value is not None:
        return constructor(shape_out, fill_value, dtype_out)
    return constructor(shape_out, dtype_out)


def empty_like(prototype, dtype=None, shape=None):
    """
    Returns a new array with the same shape and type as a given array.

    Note:
        Input array must have the same size across a dimension.
        If `prototype` is not a Tensor, dtype is float32 by default if not provided.

    Args:
        prototype (Union[Tensor, list, tuple]): The shape and data-type of `prototype`
            define these same attributes of the returned array.
        dtype (:class:`mindspore.dtype`, optional): Overrides the data type of the
            result.
        shape (int or sequence of ints, optional): Overrides the shape
            of the result.

    Returns:
        Tensor, array of uninitialized (arbitrary) data with the same
        shape and type as `prototype`.

    Raises:
        ValueError: if `prototype` is not a Tensor, list or tuple.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.ones((4,1,2))
        >>> output = np.empty_like(a)
        >>> print(output)
        # result may vary
        Tensor(shape=[4, 1, 2], dtype=Float32, value=
        <uninitialized>)
    """
    return _x_like(prototype, dtype, shape, empty)


def ones_like(a, dtype=None, shape=None):
    """
    Returns an array of ones with the same shape and type as a given array.

    Note:
        Input array must have the same size across a dimension.
        If `a` is not a Tensor, dtype is float32 by default if not provided.

    Args:
        a (Union[Tensor, list, tuple]): The shape and data-type of a define these same
            attributes of the returned array.
        dtype (:class:`mindspore.dtype`, optional): Overrides the data type of the
            result.
        shape (int or sequence of ints, optional): Overrides the shape
            of the result.

    Returns:
        Tensor, array of ones with the same shape and type as `a`.

    Raises:
        ValueError: if `a` is not a Tensor, list or tuple.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.ones((4,1,2))
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
        Input array must have the same size across a dimension.
        If `a` is not a Tensor, dtype is float32 by default if not provided.

    Args:
        a (Union[Tensor, list, tuple]): The shape and data-type of a define these same
            attributes of the returned array.
        dtype (:class:`mindspore.dtype`, optional): Overrides the data type of the
            result.
        shape (int or sequence of ints, optional): Overrides the shape
            of the result.

    Returns:
        Tensor, array of zeros with the same shape and type as `a`.

    Raises:
        ValueError: if `a` is not a Tensor, list or tuple.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.ones((4,1,2))
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
        Input array must have the same size across a dimension.
        If `a` is not a Tensor, dtype is float32 by default if not provided.

    Args:
        a (Union[Tensor, list, tuple]): The shape and data-type of `a` define these same
            attributes of the returned array.
        fill_value (scalar): Fill value.
        dtype (:class:`mindspore.dtype`, optional): Overrides the data type of the
            result.
        shape (int or sequence of ints, optional): Overrides the shape
            of the result.

    Returns:
        Tensor, array of fill_value with the same shape and type as `a`.

    Raises:
        ValueError: if `a` is not a Tensor, list or tuple.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.ones((4,1,2))
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
    Returns a tensor with ones at and below the given diagonal and zeros elsewhere.

    Args:
        N(int): Number of rows in the array.
        M(int, optional): Number of columns in the array. By default, `M` is taken
            equal to N.
        k(int, optional): The sub-diagonal at and below which the array is filled.
            :math:`k = 0` is the main diagonal, while :math:`k < 0` is below it, and :math:`k > 0` is above.
            The default is 0.
        dtype(:class:`mindspore.dtype`, optional): Data type of the returned array. The default
            is :class:`mindspore.dtype`.

    Returns:
        Tensor with shape `(N, M)`, with its lower triangle filled with
        ones and zeros elsewhere; in other words :math:`T[i,j] = 1` for :math:`j <= i + k`,
        :math:`0` otherwise.

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
    Returns a lower triangle of a tensor.

    Returns a copy of a tensor with elements above the `k-th` diagonal zeroed.

    Args:
        m (Union[Tensor, list, tuple]): The shape and data-type of `m` define these same
            attributes of the returned tensor.
        k (int, optional): Diagonal above which to zero elements. :math:`k = 0` (the default)
            is the main diagonal, :math:`k < 0` is below it and :math:`k > 0` is above.

    Returns:
        Lower triangle of `m`, of same shape and data-type as `m`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Raises:
        TypeError: If input arguments have types not specified above.
        ValueError: If input `m`\'s rank :math:`< 1`.

    Examples:
        >>> import mindspore.numpy as np
        >>> output = np.tril(np.ones((3, 3)))
        >>> print(output)
        [[1. 0. 0.]
        [1. 1. 0.]
        [1. 1. 1.]]
    """
    if not isinstance(m, Tensor):
        m = asarray_const(m)
    dtype = m.dtype
    m = m.astype(mstype.float32)
    assist = nn_tril(m.shape, mstype.float32, k)
    return F.tensor_mul(assist, m).astype(dtype)


def triu(m, k=0):
    """
    Returns an upper triangle of a tensor.

    Returns a copy of a tensor with elements below the `k-th` diagonal zeroed.

    Args:
        m (Union[Tensor, list, tuple]): The shape and data-type of `m` define these same
            attributes of the returned tensor.
        k (int, optional): Diagonal below which to zero elements. :math:`k = 0` (the default)
            is the main diagonal, :math:`k < 0` is below it and :math:`k > 0` is above.

    Returns:
        Upper triangle of `m`, of same shape and data-type as `m`.

    Raises:
        TypeError: If input arguments have types not specified above.
        ValueError: If input `m`\'s rank < 1.

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
    if not isinstance(m, Tensor):
        m = asarray_const(m)
    dtype = m.dtype
    m = m.astype(mstype.float32)
    assist = nn_triu(m.shape, mstype.float32, k)
    return F.tensor_mul(assist, m).astype(dtype)


def diagonal(a, offset=0, axis1=0, axis2=1):
    """
    Returns specified diagonals.

    If `a` is 2-D, returns the diagonal of `a` with the given offset, i.e., the
    collection of elements of the form ``a[i, i+offset]``. If `a` has more than two
    dimensions, then the axes specified by `axis1` and `axis2` are used to determine
    the 2-D sub-array whose diagonal is returned. The shape of the resulting
    array can be determined by removing `axis1` and `axis2` and appending an index
    to the right equal to the size of the resulting diagonals.

    Args:
        a (Tensor): Array from which the diagonals are taken.
        offset (int, optional): Offset of the diagonal from the main diagonal.
            Can be positive or negative. Defaults to main diagonal.
        axis1 (int, optional): Axis to be used as the first axis of the 2-D
            sub-arrays from which the diagonals should be taken. Defaults to
            first axis (0).
        axis2 (int, optional): Axis to be used as the second axis of the 2-D
            sub-arrays from which the diagonals should be taken. Defaults to
            second axis.

    Returns:
        Tensor, if `a` is 2-D, then `a` 1-D array containing the diagonal. If
        ``a.ndim > 2``, then the dimensions specified by `axis1` and `axis2` are removed,
        and a new axis inserted at the end corresponding to the diagonal.

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

    if _is_shape_empty(F.shape(a)):
        return _empty(dtype, (0,))

    cast_type = dtype
    if not _check_is_float(dtype):
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
    e = eye(n, m, offset, cast_type)
    e = _broadcast_to_shape(e, F.shape(a))

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


def trace(a, offset=0, axis1=0, axis2=1, dtype=None):
    """
    Returns the sum along diagonals of the array.

    If `a` is 2-D, the sum along its diagonal with the given offset is returned,
    i.e., the sum of elements ``a[i,i+offset]`` for all `i`.
    If `a` has more than two dimensions, then the axes specified by `axis1` and
    `axis2` are used to determine the 2-D sub-arrays whose traces are returned.
    The shape of the resulting array is the same as that of a with `axis1` and
    `axis2` removed.

    Note:
        On GPU, the supported dtypes are np.float16, and np.float32.
        On CPU, the supported dtypes are np.float16, np.float32, and np.float64.

    Args:
        a (Tensor): Array from which the diagonals are taken.
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
        Tensor, sum_along_diagonals. If `a` is 2-D, the sum along the diagonal
        is returned. If `a` has larger dimensions, then an array of sums along
        diagonals is returned.

    Raises:
        ValueError: if the input tensor has less than two dimensions.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
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
    if dtype is None:
        dtype = F.dtype(d)
    if shape[-1] == 0:
        return _empty(dtype, shape[:-1])

    cast_type = dtype
    if not _check_is_float(dtype):
        # reduce sum only supports float types
        cast_type = mstype.float32
        d = F.cast(d, cast_type)
    res = F.reduce_sum(d, -1)
    if not _check_same_type(cast_type, dtype):
        res = F.cast(res, dtype)
    return res


def _index(i, size, Cartesian=True):
    """If Cartesian=True, index 0 is swapped with index 1."""
    if Cartesian:
        if i == 1:
            return 0
        if i == 0 and size >= 2:
            return 1
    return i


def meshgrid(*xi, sparse=False, indexing='xy'):
    """
    Returns coordinate matrices from coordinate vectors.

    Make `N-D` coordinate arrays for vectorized evaluations of `N-D`
    scalar/vector fields over `N-D` grids, given one-dimensional
    coordinate arrays `x1, x2,…, xn`.

    Note:
        Numpy argument copy is not supported, and a copy is always
        returned.

    Args:
        *xi (Tensor): 1-D arrays representing the coordinates
            of a grid.
        indexing (‘xy’, ‘ij’, optional): Cartesian (‘xy’, default) or
            matrix (‘ij’) indexing of output. In the 2-D case with
            inputs of length `M` and `N`, the outputs are of shape `(N, M)`
            for ‘xy’ indexing and `(M, N)` for ‘ij’ indexing. In the 3-D
            case with inputs of length `M`, `N` and `P`, outputs are of shape
            `(N, M, P)` for ‘xy’ indexing and `(M, N, P)` for ‘ij’ indexing.
        sparse (bool, optional): If True a sparse grid is returned in
            order to conserve memory. Default is False.

    Returns:
        Tuple of tensors, for vectors `x1, x2,…, xn` with lengths
        ``Ni=len(xi)``, return `(N1, N2, N3,...Nn)` shaped arrays if
        ``indexing=’ij’`` or `(N2, N1, N3,...Nn)` shaped arrays if
        ``indexing=’xy’`` with the elements of `xi` repeated to fill the matrix
        along the first dimension for `x1`, the second for `x2` and so on.

    Raises:
        TypeError: if the input is not a tensor, or sparse is not boolean, or
            indexing is not 'xy' or 'ij'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.linspace(0, 1, 3)
        >>> y = np.linspace(0, 1, 2)
        >>> xv, yv = np.meshgrid(x, y)
        >>> print(xv)
        [[0.  0.5 1. ]
        [0.  0.5 1. ]]
        >>> print(yv)
        [[0.  0.  0.]
        [1.  1.  1.]]
        >>> xv, yv = np.meshgrid(x, y, sparse=True)
        >>> print(xv)
        [[0.  0.5  1. ]]
        >>> print(yv)
        [[0.]
        [1.]]
    """
    _check_input_tensor(*xi)
    if not isinstance(sparse, bool):
        _raise_type_error('argument sparse should be boolean')
    if indexing not in ('xy', 'ij'):
        _raise_type_error("Valid values for `indexing` are 'xy' and 'ij'.")

    shape_out = ()
    for x in xi:
        shape_out += (x.size,)
    if _is_shape_empty(shape_out):
        return ones(shape_out)

    grids = []
    for x in xi:
        if F.rank(x) == 1:
            grids.append(x)
        else:
            grids.append(ravel(x))
    ndim = len(grids)

    Cartesian = indexing == 'xy'
    shape_out = ()
    for i in range(len(grids)):
        grid_index = _index(i, ndim, Cartesian=Cartesian)
        shape_out += (F.shape(grids[grid_index])[0],)

    res = []
    for i, x in enumerate(grids):
        grid_index = _index(i, ndim, Cartesian=Cartesian)
        shape_expanded = _expanded_shape(ndim, shape_out[grid_index], grid_index)
        x = x.reshape(shape_expanded)
        if not sparse:
            x = F.tile(x, _tile_size(shape_expanded, shape_out, ndim))
        res.append(x)
    return res


class nd_grid:
    """
    Construct a multi-dimensional "meshgrid".

    ``grid = nd_grid()`` creates an instance which will return a mesh-grid
    when indexed.
    If instantiated with an argument of ``sparse=True``, the mesh-grid is
    open (or not fleshed out) so that only one-dimension of each
    returned argument is greater than 1.

    Args:
        sparse (bool): Whether the grid is sparse or not. Default is
            False.

    Returns:
        Tensor or tuple of tensor, a meshgrid. If ``sparse=False``, returns
        tensors are all of the same dimensions; and if ``sparse=True``,
        returns tensors with only one dimension not equal to `1`.
    """
    def __init__(self, sparse=False):
        self.sparse = sparse

    def __getitem__(self, keys):
        if isinstance(keys, slice):
            keys = (keys,)

        xi = []
        for k in keys:
            if not isinstance(k.start, int) or not isinstance(k.stop, int):
                _raise_type_error('slice indices must be integers')
            if k.step:
                step = k.step
            else:
                step = 1
            if isinstance(step, complex):
                v = linspace(k.start, k.stop, int(abs(step)))
            else:
                v = arange(k.start, k.stop, step)
            xi.append(v)
        grids = meshgrid(*xi, sparse=self.sparse, indexing='ij')

        if len(grids) == 1:
            return grids[0]
        if self.sparse:
            return grids

        if isinstance(grids, Tensor_):
            return grids
        expanded = []
        for grid in grids:
            expanded.append(F.expand_dims(grid, 0))
        res = concatenate(tuple(expanded))
        return res


class mGridClass(nd_grid):
    """
    mgrid is an :class:`nd_grid` instance with ``sparse=False``.

    The dimension and number of the output arrays are equal to the number
    of indexing dimensions. If the step length is not a complex number,
    then the stop is not inclusive. However, if the step length is a complex
    number (e.g. 5j), then the integer part of its magnitude is interpreted
    as specifying the number of points to create between the start and
    stop values, where the stop value is inclusive.

    Note:
        Not supported in graph mode.
        Unlike Numpy, if the step length is a complex number with a real
        component, the step length is handled as equivalent to
        ``int(abs(step))``.

    Returns:
        Tensor or tuple of tensor, a meshgrid.

    Raises:
        TypeError: if slicing indices are not integers.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore.numpy import mgrid
        >>> output = mgrid[0:5, 0:5]
        >>> print(output)
        [[[0 0 0 0 0]
        [1 1 1 1 1]
        [2 2 2 2 2]
        [3 3 3 3 3]
        [4 4 4 4 4]]
        [[0 1 2 3 4]
        [0 1 2 3 4]
        [0 1 2 3 4]
        [0 1 2 3 4]
        [0 1 2 3 4]]]
        >>> output = mgrid[-1:1:5j]
        >>> print(output)
        [-1.  -0.5  0.   0.5  1. ]
    """
    def __init__(self):
        super(mGridClass, self).__init__(sparse=False)


class oGridClass(nd_grid):
    """
    ogrid is an :class:`nd_grid` instance with ``sparse=True``.

    The dimension and number of the output arrays are equal to the number
    of indexing dimensions. If the step length is not a complex number,
    then the stop is not inclusive. However, if the step length is a complex
    number (e.g. 5j), then the integer part of its magnitude is interpreted
    as specifying the number of points to create between the start and
    stop values, where the stop value is inclusive.

    Note:
        Not supported in graph mode.
        Unlike Numpy, if the step length is a complex number with a real
        component, the step length is handled as equivalent to
        ``int(abs(step))``.

    Raises:
        TypeError: if slicing indices are not integers.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore.numpy import ogrid
        >>> output = ogrid[0:5,0:5]
        >>> print(output)
        [Tensor(shape=[5, 1], dtype=Int32, value=
        [[0],
        [1],
        [2]
        [3],
        [4]]), Tensor(shape=[1, 5], dtype=Int32, value=
        [[0, 1, 2, 3, 4]])]
        >>> output = ogrid[-1:1:5j]
        >>> print(output)
        [-1.  -0.5  0.   0.5  1. ]
    """
    def __init__(self):
        super(oGridClass, self).__init__(sparse=True)


mgrid = mGridClass()


ogrid = oGridClass()


def diag(v, k=0):
    """
    Extracts a diagonal or construct a diagonal array.

    Args:
        v (Tensor): If `v` is a 2-D array, return a copy of its `k-th` diagonal.
            If `v` is a 1-D array, return a 2-D array with v on the `k-th` diagonal.
        k (int, optional): Diagonal in question. The default is 0. Use ``k>0`` for
            diagonals above the main diagonal, and ``k<0`` for diagonals below the
            main diagonal.

    Returns:
        Tensor, the extracted diagonal or constructed diagonal array.

    Raises:
        ValueError: if input is not 1-D or 2-D.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> x = np.arange(9).reshape((3,3))
        >>> print(x)
        [[0 1 2]
        [3 4 5]
        [6 7 8]]
        >>> output = np.diag(x)
        >>> print(output)
        [0 4 8]
        >>> output = np.diag(x, k=1)
        >>> print(output)
        [1 5]
        >>> output = np.diag(x, k=-1)
        >>> print(output)
        [3 7]
    """
    ndim = F.rank(v)
    if ndim == 1:
        return diagflat(v, k=k)
    if ndim == 2:
        shape = F.shape(v)
        dtype = F.dtype(v)
        if _is_shape_empty(shape):
            return _empty(dtype, (0,))
        e = eye(shape[0], shape[1], k, dtype)
        prod = F.tensor_mul(v, e)

        cast_type = dtype
        if not _check_is_float(dtype):
            # reduce sum only supports float types
            cast_type = mstype.float32
            prod = F.cast(prod, cast_type)

        res = F.reduce_sum(prod, 1)
        res = res[_max(0, -k): _min(shape[0], _max(0, shape[1] - k))]

        if not _check_same_type(cast_type, dtype):
            res = F.cast(res, dtype)

        return res
    return _raise_value_error("Input must be 1- or 2-d.")


def diagflat(v, k=0):
    """
    Creates a two-dimensional array with the flattened input as a diagonal.

    Note:
        On GPU, the supported dtypes are np.float16, and np.float32.

    Args:
        v (Tensor): Input data, which is flattened and set as the `k-th` diagonal
            of the output.
        k (int, optional): Diagonal to set; 0, the default, corresponds to the
            “main” diagonal, a positive (negative) `k` giving the number of the
            diagonal above (below) the main.

    Returns:
        Tensor, The 2-D output array.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> output = np.diagflat(np.asarray([[1,2], [3,4]]))
        >>> print(output)
        [[1 0 0 0]
        [0 2 0 0]
        [0 0 3 0]
        [0 0 0 4]]
        >>> output = np.diagflat(np.asarray([1,2]), 1)
        >>> print(output)
        [[0 1 0]
        [0 0 2]
        [0 0 0]]
    """
    _check_input_tensor(v)
    dtype = F.dtype(v)
    k_abs = _abs(k)
    if _is_shape_empty(F.shape(v)):
        return zeros((k_abs, k_abs), dtype)

    v = ravel(v)
    size = F.shape(v)[0]
    e = eye(size, size, 0, dtype)
    res = F.tensor_mul(v, e)

    if k != 0:
        pad_y = zeros((size, k_abs), dtype)
        pad_x = zeros((k_abs, size + k_abs), dtype)
        if k < 0:
            res = concatenate((res, pad_y), axis=1)
            res = concatenate((pad_x, res), axis=0)
        else:
            res = concatenate((pad_y, res), axis=1)
            res = concatenate((res, pad_x), axis=0)
    return res


def diag_indices(n, ndim=2):
    """
    Returns the indices to access the main diagonal of an array.

    This returns a tuple of indices that can be used to access the main
    diagonal of an array a with ``a.ndim >= 2`` dimensions and shape `(n, n, …, n)`.
    For ``a.ndim = 2`` this is the usual diagonal, for ``a.ndim > 2`` this is the set
    of indices to access ``a[i, i, ..., i]`` for ``i = [0..n-1]``.

    Args:
        n (int): The size, along each dimension, of the arrays for which
            the returned indices can be used.
        ndim (int, optional): The number of dimensions.

    Returns:
        Tuple of Tensor.

    Raises:
        TypeError: if input are not integers.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> output = np.diag_indices(5, 3)
        >>> print(output)
        (Tensor(shape=[5], dtype=Int32, value= [0, 1, 2, 3, 4]),
        Tensor(shape=[5], dtype=Int32, value= [0, 1, 2, 3, 4]),
        Tensor(shape=[5], dtype=Int32, value= [0, 1, 2, 3, 4]))
    """
    if not isinstance(n, int) or not isinstance(ndim, int):
        _raise_type_error('input must be integers')
    return _list_comprehensions(ndim, arange(start=0, stop=n), True)


def ix_(*args):
    r"""
    Constructs an open mesh from multiple sequences.

    This function takes `N` 1-D sequences and returns `N` outputs with `N`
    dimensions each, such that the shape is 1 in all but one dimension
    and the dimension with the non-unit shape value cycles through all
    N dimensions.
    Using ix\_ one can quickly construct index arrays that will index
    the cross product. ``a[np.ix_([1,3],[2,5])]`` returns the array
    ``[[a[1,2] a[1,5]], [a[3,2] a[3,5]]]``.

    Note:
        Boolean masks are not supported.

    Args:
        *args (Tensor): 1-D sequences.

    Returns:
        Tuple of Tensor, `N` arrays with `N` dimensions each, with `N` the
        number of input sequences. Together these arrays form an open
        mesh.

    Raises:
        TypeError: if the input is not a tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> ixgrid = np.ix_(np.array([0, 1]), np.array([2, 4]))
        >>> print(ixgrid)
        (Tensor(shape=[2, 1], dtype=Int32, value=
        [[0],
        [1]]), Tensor(shape=[1, 2], dtype=Int32, value=
        [[2, 4]]))
    """
    # TODO boolean mask
    _check_input_tensor(*args)
    ndim = len(args)
    res = ()
    for i, arr in enumerate(args):
        if F.rank(arr) != 1:
            return _raise_value_error('Cross index must be 1 dimensional')
        res += (F.reshape(arr, _expanded_shape(ndim, arr.size, i)),)
    return res


def vander(x, N=None, increasing=False):
    """
    Generates a Vandermonde matrix.

    The columns of the output matrix are powers of the input vector. The order of
    the powers is determined by the increasing boolean argument. Specifically, when
    increasing is `False`, the i-th output column is the input vector raised element-wise
    to the power of :math:`N - i - 1`. Such a matrix with a geometric progression in each row
    is named for Alexandre-Theophile Vandermonde.

    Args:
        x (Union[list, tuple, Tensor]): 1-D input array.
        N (int, optional): Number of columns in the output. If N is not specified, a
            square array is returned (``N = len(x)``).
        increasing (bool, optional): Order of the powers of the columns. If True, the
            powers increase from left to right, if False (the default) they are reversed.

    Returns:
        Vandermonde matrix. If `increasing` is `False`, the first column is :math:`x^{(N-1)}`,
        the second :math:`x^{(N-2)}` and so forth. If `increasing` is `True`, the columns are
        :math:`x^0, x^1, ..., x^{(N-1)}`.

    Raises:
        TypeError: If inputs have types not specified above.
        ValueError: If `x` is not 1-D, or `N` < 0.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.vander([1,2,3,4,5]))
        [[  1   1   1   1   1]
         [ 16   8   4   2   1]
         [ 81  27   9   3   1]
         [256  64  16   4   1]
         [625 125  25   5   1]]
    """
    if isinstance(x, (list, tuple)):
        x = asarray_const(x)
    elif not isinstance(x, Tensor):
        _raise_type_error("Input x must be list, tuple or Tensor, but got ", x)
    if x.ndim != 1:
        _raise_value_error("Input x must be 1-D, but got dimension=", x.ndim)
    N = N or x.size
    if not isinstance(N, int):
        _raise_type_error("Input N must be an integer.")
    if N <= 0:
        _raise_value_error("Input N must > 0.")
    if not isinstance(increasing, bool):
        _raise_type_error("increasing must be a bool.")
    exponent = _iota(x.dtype, N, increasing)
    x = F.expand_dims(x, 1)
    exponent = F.expand_dims(exponent, 0)
    return F.tensor_pow(x, exponent)


def indices(dimensions, dtype=mstype.int32, sparse=False):
    """
    Returns an array representing the indices of a grid.

    Computes an array where the subarrays contain index values 0, 1, …
    varying only along the corresponding axis.

    Args:
        dimensions (tuple or list of ints): The shape of the grid.
        dtype (data type, optional): Data type of the result.
        sparse (boolean, optional): Defaults to False. Return a sparse
            representation of the grid instead of a dense representation.

    Returns:
        Tensor or tuple of Tensor, If `sparse` is False, returns one array
        of grid indices, ``grid.shape = (len(dimensions),) + tuple(dimensions)``.
        If sparse is True, returns a tuple of arrays, with
        ``grid[i].shape = (1, ..., 1, dimensions[i], 1, ..., 1)`` with
        ``dimensions[i]`` in the `ith` place

    Raises:
        TypeError: if input dimensions is not a tuple or list.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> grid = np.indices((2, 3))
        >>> print(grid)
        [Tensor(shape=[2, 3], dtype=Int32, value=
        [[0, 0, 0],
        [1, 1, 1]]), Tensor(shape=[2, 3], dtype=Int32, value=
        [[0, 1, 2],
        [0, 1, 2]])]
    """
    if not isinstance(dimensions, (tuple, list)):
        _raise_type_error('Shape of the grid must be tuple or list')
    grids = ()
    for d in dimensions:
        grids += (arange(d, dtype=dtype),)
    return meshgrid(*grids, sparse=sparse, indexing='ij')
