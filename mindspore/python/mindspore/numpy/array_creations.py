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
from __future__ import division

import math
import operator

import numpy as onp

from mindspore import context
from mindspore import ops
from mindspore.common import Tensor
from mindspore.common import dtype as mstype
from mindspore.common.seed import get_seed
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.primitive import constexpr, _primexpr
from mindspore.ops.function.random_func import _get_seed
from mindspore.nn.layer.basic import tril as nn_tril
from mindspore.nn.layer.basic import triu as nn_triu
from mindspore._c_expression import Tensor as Tensor_

from mindspore.numpy.utils import _check_input_for_asarray, _deep_list, _deep_tensor_to_nparray, \
    _check_input_tensor, _convert_64_to_32, _get_dtype_from_scalar, \
    _expand, _to_tensor, _slice_along_axis, _callable
from mindspore.numpy.utils_const import _raise_value_error, _empty, _max, _min, \
    _check_same_type, _is_shape_empty, _check_shape, _check_dtype, _tile_size, _abs, \
    _raise_type_error, _expanded_shape, _check_is_float, _iota, _type_convert, \
    _canonicalize_axis, _list_comprehensions, _ceil, _tuple_slice, _raise_unimplemented_error, \
    _tuple_setitem
from mindspore.numpy.array_ops import ravel, concatenate, broadcast_arrays, reshape, broadcast_to, flip, \
    apply_along_axis, where, moveaxis
from mindspore.numpy.dtypes import nan, pi

# According to official numpy reference, the dimension of a numpy array must be less
# than 32
MAX_NUMPY_DIMS = 32
# All types that can be accepted as "array_like" parameters in graph mode.
ARRAY_TYPES = (int, float, bool, list, tuple, Tensor)

_reduce_min_keepdims = P.ReduceMin(True)
_reduce_max_keepdims = P.ReduceMax(True)
_reduce_mean_keepdims = P.ReduceMean(True)


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
    if dtype is not None:
        dtype = _check_dtype(dtype)
    res = asarray(obj, dtype)

    if ndmin > res.ndim:
        if res.size == 0:
            _raise_value_error("Empty tensor cannot be expanded beyond the current dimension.")
        res = _expand(res, ndmin)

    if copy and isinstance(obj, Tensor):
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
    if dtype is not None:
        dtype = _check_dtype(dtype)
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
    if dtype not in (mstype.float16, mstype.float32, mstype.float64):
        dtype = mstype.float32

    if isinstance(a, Tensor):
        return a.astype(dtype)

    return asfarray_const(a, dtype)


def copy_(a):
    """
    Returns a tensor copy of the given object.

    Args:
        a (Union[int, float, bool, list, tuple, Tensor]): Input data, in any form that can
            be converted to a Tensor. This includes Tensor, list, tuple and numbers.

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
    a = asarray(a)
    return a.copy()


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


@constexpr
def _generate_shapes(shape):
    """Generate shapes for randn and rand."""
    if not shape:
        size = (1,)
    elif len(shape) == 1:
        if isinstance(shape[0], int):
            size = shape
        elif isinstance(shape[0], list):
            size = tuple(shape[0])
        elif isinstance(shape[0], tuple):
            size = shape[0]
        else:
            _raise_type_error("If the length of the argument 'shape' is 1, the type of the argument 'shape' must be "
                              "one of ['int', 'list', 'tuple'], but got ", shape[0])
    else:
        for value in shape:
            if not isinstance(value, int):
                _raise_type_error("If the length of the argument 'shape' is > 1, the type of the argument 'shape' must "
                                  "all be int, but got ", value)
        size = shape
    return size


@constexpr
def _check_rand_type(dtype):
    """Check type for randn and rand"""
    type_list = ['float', 'float16', 'float32', 'float64']
    if isinstance(dtype, str):
        if dtype not in type_list:
            _raise_value_error("If the argument 'dtype' is str, it must be one of ['float', 'float16', 'float32', "
                               "'float64'], but got ", dtype)
    elif dtype not in (mstype.float64, mstype.float32, mstype.float16):
        _raise_value_error("The argument 'dtype' must be 'mindspore.float64', 'mindspore.float32' or "
                           "'mindspore.float16', but got ", dtype)


def randn(*shape, dtype=mstype.float32):
    """
    Returns a new Tensor with given shape and dtype, filled with a sample (or samples)
    from the standard normal distribution.

    Args:
        *shape (Union[int, tuple(int), list(int)]): Shape of the new tensor, e.g.,
            :math:`(2, 3)` or :math:`2`.
        dtype (Union[:class:`mindspore.dtype`, str], optional): Designated tensor dtype, it must
            be float type. Default is :class:`mindspore.float32`.

    Returns:
        Tensor, with the designated shape and dtype, filled with a sample (or samples)
        from the "standard normal" distribution.

    Raises:
        TypeError: If input arguments have types not specified above.
        ValueError: If `dtype` is not float type.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> from mindspore import set_seed
        >>> set_seed(1)
        >>> print(np.randn((2,3)))
        [[ 0.30639967 -0.42438635 -0.20454668]
        [-0.4287376   1.3054721   0.64747655]]
    """
    _check_rand_type(dtype)
    size = _generate_shapes(shape)
    seed = get_seed()
    if seed is not None:
        seed1, seed2 = _get_seed(seed, "StandardNormal")
        stdnormal = P.StandardNormal(seed=seed1, seed2=seed2)
    else:
        stdnormal = P.StandardNormal()
    return stdnormal(size).astype(dtype)


def rand(*shape, dtype=mstype.float32):
    """
    Returns a new Tensor with given shape and dtype, filled with random numbers from the
    uniform distribution on the interval :math:`[0, 1)`.

    Args:
        *shape (Union[int, tuple(int), list(int)]): Shape of the new tensor, e.g.,
            :math:`(2, 3)` or :math:`2`.
        dtype (Union[:class:`mindspore.dtype`, str], optional): Designated tensor dtype, it must
            be float type. Default is :class:`mindspore.float32`.

    Returns:
        Tensor, with the designated shape and dtype, filled with random numbers from the
        uniform distribution on the interval :math:`[0, 1)`.

    Raises:
        TypeError: If input arguments have types not specified above.
        ValueError: If `dtype` is not float type.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> from mindspore import set_seed
        >>> set_seed(1)
        >>> print(np.rand((2,3)))
        [[4.1702199e-01 9.9718481e-01 7.2032452e-01]
        [9.3255734e-01 1.1438108e-04 1.2812445e-01]]
    """
    _check_rand_type(dtype)
    size = _generate_shapes(shape)
    seed = get_seed()
    if seed is not None:
        seed1, seed2 = _get_seed(seed, "UniformReal")
        uniformreal = P.UniformReal(seed=seed1, seed2=seed2)
    else:
        uniformreal = P.UniformReal()
    return uniformreal(size).astype(dtype)


def randint(minval, maxval=None, shape=None, dtype=mstype.int32):
    """
    Return random integers from minval (inclusive) to maxval (exclusive). Return random integers from the discrete
    uniform distribution of the specified dtype in the “half-open” interval :math:`[minval, maxval)`. If maxval is
    None (the default), the value range will be :math:`[0, minval)`, in this case, minval must be greater than 0.

    Args:
        minval(Union[int]): Start value of interval. The interval includes this value. When `maxval`
            is :class:`None`, `minval` must be greater than 0. When `maxval` is not :class:`None`,
            `minval` must be less than `maxval`.
        maxval(Union[int], optional): End value of interval. The interval does not include this value.
        shape (Union[int, tuple(int)]): Shape of the new tensor, e.g., :math:`(2, 3)` or :math:`2`.
        dtype (Union[:class:`mindspore.dtype`, str], optional): Designated tensor dtype, it must
            be int type. Default is :class:`mindspore.int32`.

    Returns:
        Tensor, with the designated shape and dtype, filled with random integers from minval (inclusive)
        to maxval (exclusive).

    Raises:
        TypeError: If input arguments have types not specified above.
        ValueError: If input arguments have values not specified above.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> from mindspore import set_seed
        >>> set_seed(1)
        >>> print(np.randint(1, 10, (2,3)))
        [[4 9 7]
        [9 1 2]]
    """
    if not isinstance(minval, int):
        _raise_type_error("For mindspore.numpy.randint, the type of the argument 'minval' must be int, "
                          "but got ", minval)
    if maxval is None:
        if minval <= 0:
            _raise_value_error("For mindspore.numpy.randint, the argument 'minval' must be > 0 when the argument "
                               "'maxval' is None, but got ", minval)
        maxval = minval
        minval = 0
    else:
        if not isinstance(maxval, int):
            _raise_type_error("For mindspore.numpy.randint, the type of the argument 'maxval' must be int, "
                              "but got ", maxval)
        if minval >= maxval:
            _raise_value_error("For mindspore.numpy.randint, the value of 'minval' must be greater than the "
                               "value of 'maxval'.")
    if isinstance(dtype, str):
        if dtype not in ('int', 'int8', 'int16', 'int32', 'int64'):
            _raise_value_error("For 'mindspore.numpy.randint', if the argument 'dtype' is str, it must be one of "
                               "['int', 'int8', 'int16', 'int32', 'int64'], but got ", dtype)
    elif dtype not in (mstype.int64, mstype.int32, mstype.int16, mstype.int8):
        _raise_value_error("For 'mindspore.numpy.randint', the argument 'dtype' must be 'mindspore.int64', "
                           "'mindspore.int32', 'mindspore.int16' or 'mindspore.int8', but got ", dtype)
    if shape is None:
        shape = (1,)
    else:
        shape = _check_shape(shape)
    seed = get_seed()
    if seed is not None:
        seed1, seed2 = _get_seed(seed, "UniformInt")
        uniformint = P.UniformInt(seed=seed1, seed2=seed2)
    else:
        uniformint = P.UniformInt()
    t_min = _type_convert(Tensor, minval).astype(dtype)
    t_max = _type_convert(Tensor, maxval).astype(dtype)
    return uniformint(shape, t_min, t_max).astype(dtype)


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
        TypeError(PyNative Mode): If input arguments have types not specified above,
            or arguments are not given in the correct orders specified above.
        RuntimeError(Graph Mode): The inputs that lead to TypeError in Pynative Mode
            will lead to RuntimeError in Graph Mode.

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
    """
    # This implementation was inspired by jax.numpy.arange
    # infer the dtype
    if dtype is None:
        dtype = _get_dtype_from_scalar(start, stop, step)
    if stop is None and step is None:  # (start, stop, step) -> (0, start, 1)
        num = _ceil(start)
        out = _iota(mstype.float32, num)
    elif step is None:  # (start, stop, step) -> (start, stop, 1)
        num = _ceil(stop - start)
        out = _iota(mstype.float32, num) + start
    elif stop is None:  # (start, stop, step) -> (0, start, step)
        num = _ceil((start + 0.0) / step)
        out = _iota(mstype.float32, num) * step
    else:
        num = _ceil((stop - start + 0.0) / step)
        out = _iota(mstype.float32, num) * step + start
    return out.astype(dtype)


def _type_checking_for_xspace(start, stop, num, endpoint, dtype):
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
    return start, stop, num, endpoint, dtype


def _compute_shapes(start, axis, num, endpoint):
    """Computes shapes for local variables for np.linspace"""
    bounds_shape = start.shape
    bounds_shape = _tuple_slice(bounds_shape, None, axis) + (1,) + _tuple_slice(bounds_shape, axis, None)
    iota_shape = _list_comprehensions(start.ndim + 1, 1, True)
    iota_shape = _tuple_slice(iota_shape, None, axis) + (num,) + _tuple_slice(iota_shape, axis + 1, None)
    num_tensor = _type_convert(Tensor, num).astype(mstype.float32)
    div = (num_tensor - 1) if endpoint else num_tensor
    return bounds_shape, iota_shape, div


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
            only if start or stop are array-like.  By default, the samples will
            be along a new axis inserted at the beginning. Use -1 to get an axis at the end.
            Default is 0.

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
    start, stop, num, endpoint, dtype = _type_checking_for_xspace(start, stop, num, endpoint, dtype)
    axis = _canonicalize_axis(axis, start.ndim + 1)
    if not isinstance(retstep, bool):
        _raise_type_error("retstep should be an boolean, but got ", retstep)
    bounds_shape, iota_shape, div = _compute_shapes(start, axis, num, endpoint)
    out = None
    delta = None
    if num > 1:
        delta = (stop - start) / div
        # This is similar to how numpy and jax compute linspace
        start_expand = reshape(start, bounds_shape)
        incremental_expand = reshape(_iota(mstype.float32, num), iota_shape)
        delta_expand = reshape(delta, bounds_shape)
        start_expand, incremental_expand, delta_expand = broadcast_arrays(
            start_expand, incremental_expand, delta_expand)
        out = start_expand + (incremental_expand * delta_expand)
        # recover endpoint
        if endpoint:
            out = moveaxis(out, axis, 0)
            out[-1] = stop
            out = moveaxis(out, 0, axis)
    elif num == 1:
        delta = nan if endpoint else stop - start
        out = reshape(start, bounds_shape)
    else:  # num == 0
        _raise_value_error("cannot support Tensor with num=0.")
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
            is uniform. Default is 10.0.
        dtype (Union[:class:`mindspore.dtype`, str], optional): Designated tensor dtype.
            If `dtype` is None, infer the data type from other input arguments. Default is None.
        axis (int, optional): The axis in the result to store the samples. Relevant
            only if start or stop is array-like.  By default, the samples will
            be along a new axis inserted at the beginning. Use -1 to get an axis at the end.
            Default is 0.

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
    start, stop, num, endpoint, dtype = _type_checking_for_xspace(start, stop, num, endpoint, dtype)
    axis = _canonicalize_axis(axis, start.ndim + 1)
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
    start, stop, num, endpoint, dtype = _type_checking_for_xspace(start, stop, num, endpoint, dtype)
    axis = _canonicalize_axis(axis, start.ndim + 1)
    root = num
    if endpoint:
        root -= 1
    bases = F.tensor_pow(F.tensor_div(stop, start), asarray_const(1. / (root)))
    exponents = linspace(zeros(F.shape(bases)), F.fill(F.dtype(bases), F.shape(bases), root),
                         num, endpoint=endpoint, dtype=dtype, axis=axis)
    shape = F.shape(bases)
    axis = axis + F.rank(bases) + 1 if axis < 0 else axis
    expanded_shape = _tuple_slice(shape, None, axis) + (1,) + _tuple_slice(shape, axis, None)
    bases = F.reshape(bases, expanded_shape)
    start = F.reshape(start, expanded_shape)
    res = F.tensor_mul(F.tensor_pow(bases, exponents), start)
    if dtype is not None:
        res = F.cast(res, dtype)
    return res


def eye(N, M=None, k=0, dtype=mstype.float32):
    """
    Returns a 2-D tensor with ones on the diagonal and zeros elsewhere.

    Args:
        N (int): Number of rows in the output, must be larger than 0.
        M (int, optional): Number of columns in the output. If is :class:`None`, defaults to `N`,
            if defined, must be larger than 0. Default is :class:`None`.
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
            out_right = out[..., 0:M - k:1]
            return concatenate((out_left, out_right), 1).astype(dtype)
        if k < 0:
            out_upper = full((-k, M), 0, dtype)
            out_lower = out[0:N + k:1, ...]
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
    """Returns an empty Tensor."""
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
        TypeError: If the input shape or dtype is invalid.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> output = np.empty((2, 3))
        >>> print(output)
        [[0. 0. 0.]
         [0. 0. 0.]]
    """
    return ops.zeros(shape, dtype)


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
        ValueError: If `prototype` is not a Tensor, list or tuple.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.ones((4,1,2))
        >>> output = np.empty_like(a)
        >>> print(output)
        [[[0. 0.]]
         [[0. 0.]]
         [[0. 0.]]
         [[0. 0.]]]
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
        ValueError: If `a` is not a Tensor, list or tuple.

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
        ValueError: If `a` is not a Tensor, list or tuple.

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
        ValueError: If `a` is not a Tensor, list or tuple.

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
            is mstype.float32.

    Returns:
        Tensor with shape `(N, M)`, with its lower triangle filled with
        ones and zeros elsewhere; in other words :math:`T[i,j] = 1` for :math:`j <= i + k`,
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


@constexpr
def _device_target():
    return context.get_context("device_target")


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
    device_target = _device_target()
    # Only Ascend hardware will reduce accuracy
    if device_target == "Ascend":
        m = m.astype(mstype.float32)
        assist = nn_tril(m.shape, mstype.float32, k)
    # MindSpore binary op do not support bool
    elif dtype == mstype.bool_:
        m = m.astype(mstype.float32)
        assist = nn_tril(m.shape, mstype.float32, k)
    else:
        assist = nn_tril(m.shape, dtype, k)
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
    device_target = _device_target()
    # Only Ascend hardware will reduce accuracy
    if device_target == "Ascend":
        m = m.astype(mstype.float32)
        assist = nn_triu(m.shape, mstype.float32, k)
    # MindSpore binary op do not support bool
    elif dtype == mstype.bool_:
        m = m.astype(mstype.float32)
        assist = nn_triu(m.shape, mstype.float32, k)
    else:
        assist = nn_triu(m.shape, dtype, k)
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
        ValueError: If the input tensor has less than two dimensions.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> a = np.arange(4).reshape(2,2).astype(np.float32)
        >>> print(a)
        [[0. 1.]
        [2. 3.]]
        >>> output = np.diagonal(a)
        >>> print(output)
        [0. 3.]
        >>> output = np.diagonal(a, 1)
        >>> print(output)
        [1.]
        >>> a = np.arange(8).reshape(2, 2, 2).astype(np.float32)
        >>> print(a)
        [[[0. 1.]
        [2. 3.]]
        [[4. 5.]
        [6. 7.]]]
        >>> output = np.diagonal(a, 0, 0, 1)
        >>> print(output)
        [[0. 6.]
        [1. 7.]]
    """
    return a.diagonal(offset=offset, axis1=axis1, axis2=axis2)


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
        dtype (:class:`mindspore.dtype`, optional): Defaults to None. Overrides the dtype of the
            output Tensor.

    Returns:
        Tensor, sum_along_diagonals. If `a` is 2-D, the sum along the diagonal
        is returned. If `a` has larger dimensions, then an array of sums along
        diagonals is returned.

    Raises:
        ValueError: If the input tensor has less than two dimensions.

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
    return a.trace(offset=offset, axis1=axis1, axis2=axis2, dtype=dtype)


def _index(i, size, cartesian=True):
    """If cartesian=True, index 0 is swapped with index 1."""
    if cartesian:
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
        indexing ('xy', 'ij', optional): Cartesian ('xy', default) or
            matrix ('ij') indexing of output. In the 2-D case with
            inputs of length `M` and `N`, the outputs are of shape `(N, M)`
            for 'xy' indexing and `(M, N)` for 'ij' indexing. In the 3-D
            case with inputs of length `M`, `N` and `P`, outputs are of shape
            `(N, M, P)` for 'xy' indexing and `(M, N, P)` for 'ij' indexing.
        sparse (bool, optional): If True a sparse grid is returned in
            order to conserve memory. Default is False.

    Returns:
        Tuple of tensors, for vectors `x1, x2,…, xn` with lengths
        ``Ni=len(xi)``, return `(N1, N2, N3,...Nn)` shaped arrays if
        ``indexing='ij'`` or `(N2, N1, N3,...Nn)` shaped arrays if
        ``indexing='xy'`` with the elements of `xi` repeated to fill the matrix
        along the first dimension for `x1`, the second for `x2` and so on.

    Raises:
        TypeError: If the input is not a tensor, or sparse is not boolean, or
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

    cartesian = indexing == 'xy'
    shape_out = ()
    for i in range(len(grids)):
        grid_index = _index(i, ndim, cartesian=cartesian)
        shape_out += (F.shape(grids[grid_index])[0],)

    res = []
    for i, x in enumerate(grids):
        grid_index = _index(i, ndim, cartesian=cartesian)
        shape_expanded = _expanded_shape(ndim, shape_out[grid_index], grid_index)
        x = x.reshape(shape_expanded)
        if not sparse:
            x = F.tile(x, _tile_size(shape_expanded, shape_out, ndim))
        res.append(x)
    return res


class NdGrid:
    """
    Construct a multi-dimensional "meshgrid".

    ``grid = NdGrid()`` creates an instance which will return a mesh-grid
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

        if isinstance(grids, (Tensor, Tensor_)):
            return grids
        expanded = []
        for grid in grids:
            expanded.append(F.expand_dims(grid, 0))
        res = concatenate(tuple(expanded))
        return res


class MGridClass(NdGrid):
    """
    mgrid is an :class:`NdGrid` instance with ``sparse=False``.

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
        TypeError: If slicing indices are not integers.

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
        super(MGridClass, self).__init__(sparse=False)


class OGridClass(NdGrid):
    """
    ogrid is an :class:`NdGrid` instance with ``sparse=True``.

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
        TypeError: If slicing indices are not integers.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore.numpy import ogrid
        >>> output = ogrid[0:5,0:5]
        >>> print(output)
        [Tensor(shape=[5, 1], dtype=Int32, value=
        [[0],
        [1],
        [2],
        [3],
        [4]]), Tensor(shape=[1, 5], dtype=Int32, value=
        [[0, 1, 2, 3, 4]])]
        >>> output = ogrid[-1:1:5j]
        >>> print(output)
        [-1.  -0.5  0.   0.5  1. ]
    """

    def __init__(self):
        super(OGridClass, self).__init__(sparse=True)


mgrid = MGridClass()

ogrid = OGridClass()


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
        ValueError: If input is not 1-D or 2-D.

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
            "main" diagonal, a positive (negative) `k` giving the number of the
            diagonal above (below) the main.

    Returns:
        Tensor, The 2-D output array.

    Raises:
        TypeError: If the input is not a tensor.

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
        TypeError: If input are not integers.

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
        TypeError: If the input is not a tensor.

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
        >>> print(np.vander([1., 2., 3., 4., 5.]))
        [[  1.   1.   1.   1.   1.]
         [ 16.   8.   4.   2.   1.]
         [ 81.  27.   9.   3.   1.]
         [256.  64.  16.   4.   1.]
         [625. 125.  25.   5.   1.]]
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
        dtype (:class:`mindspore.dtype`, optional): Data type of the result.
        sparse (boolean, optional): Defaults to False. Return a sparse
            representation of the grid instead of a dense representation.

    Returns:
        Tensor or tuple of Tensor, If `sparse` is False, returns one array
        of grid indices, ``grid.shape = (len(dimensions),) + tuple(dimensions)``.
        If sparse is True, returns a tuple of arrays, with
        ``grid[i].shape = (1, ..., 1, dimensions[i], 1, ..., 1)`` with
        ``dimensions[i]`` in the `ith` place

    Raises:
        TypeError: If input dimensions is not a tuple or list.

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


def _check_window_size(x):
    """Returns True if window size is greater than 1."""
    if not isinstance(x, int):
        _raise_type_error('the number fo points should be an int')
    return x > 1


def bartlett(M):
    """
    Returns the Bartlett window.
    The Bartlett window is very similar to a triangular window, except that the
    end points are at zero. It is often used in signal processing for tapering a
    signal, without generating too much ripple in the frequency domain.

    Args:
        M (int): Number of points in the output window. If zero or less, an empty
            array is returned.

    Returns:
        Tensor, the triangular window, with the maximum value normalized to one
        (the value one appears only if the number of samples is odd), with the
        first and last samples equal to zero.

    Raises:
        TypeError: If `M` is not an int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.bartlett(12))
        [0.         0.18181819 0.36363637 0.5454545  0.72727275 0.9090909
        0.9090909  0.72727275 0.5454545  0.36363637 0.18181819 0.        ]
    """
    if not _check_window_size(M):
        return ones(_max(0, M))
    n = _iota(mstype.float32, M)
    m_minus_one = _to_tensor(M - 1)
    return _to_tensor(1) - F.absolute(_to_tensor(2) * n - m_minus_one) / m_minus_one


def blackman(M):
    """
    Returns the Blackman window.
    The Blackman window is a taper formed by using the first three terms of a
    summation of cosines. It was designed to have close to the minimal leakage
    possible. It is close to optimal, only slightly worse than a Kaiser window.

    Args:
        M (int): Number of points in the output window. If zero or less, an empty
            array is returned.

    Returns:
        Tensor, the window, with the maximum value normalized to one (the value
        one appears only if the number of samples is odd).

    Raises:
        TypeError: If `M` is not an int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.blackman(12))
        [-1.4901161e-08  3.2606430e-02  1.5990365e-01  4.1439798e-01
        7.3604518e-01  9.6704674e-01  9.6704674e-01  7.3604518e-01
        4.1439798e-01  1.5990365e-01  3.2606430e-02 -1.4901161e-08]
    """
    if not _check_window_size(M):
        return ones(_max(0, M))
    n_doubled = arange(1 - M, M, 2, dtype=mstype.float32)
    return (_to_tensor(0.42) + _to_tensor(0.5) * F.cos(_to_tensor(pi / (M - 1)) * n_doubled) +
            _to_tensor(0.08) * F.cos(_to_tensor(2 * pi / (M - 1)) * n_doubled))


def hamming(M):
    """
    Returns the Hamming window.
    The Hamming window is a taper formed by using a weighted cosine.

    Args:
        M (int): Number of points in the output window. If zero or less, an empty
            array is returned.

    Returns:
        Tensor, the window, with the maximum value normalized to one (the value
        one appears only if the number of samples is odd).

    Raises:
        TypeError: If `M` is not an int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.hamming(12))
        [0.08000001 0.15302339 0.34890914 0.6054648  0.841236   0.9813669
        0.9813668  0.8412359  0.6054647  0.34890908 0.15302327 0.08000001]
    """
    if not _check_window_size(M):
        return ones(_max(0, M))
    n = _iota(mstype.float32, M)
    return _to_tensor(0.54) - _to_tensor(0.46) * F.cos(_to_tensor(2 * pi / (M - 1)) * n)


def hanning(M):
    """
    Returns the Hanning window.
    The Hanning window is a taper formed by using a weighted cosine.

    Args:
        M (int): Number of points in the output window. If zero or less, an empty
            array is returned.

    Returns:
        Tensor, the window, with the maximum value normalized to one (the value
        one appears only if the number of samples is odd).

    Raises:
        TypeError: If `M` is not an int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.hanning(12))
        [0.         0.07937324 0.29229254 0.5711574  0.8274304  0.9797465
        0.97974646 0.82743025 0.5711573  0.29229245 0.07937312 0.        ]
    """
    if not _check_window_size(M):
        return ones(_max(0, M))
    n = _iota(mstype.float32, M)
    return _to_tensor(0.5) - _to_tensor(0.5) * F.cos(_to_tensor(2 * pi / (M - 1)) * n)


@constexpr
def tri_indices(n, k=0, m=None, upper=True):
    """Returns triu/tril indices in o(nm) time."""
    if not isinstance(n, (int, float, bool)):
        raise TypeError("Input n must be a number.")
    if not isinstance(k, (int, float, bool)):
        raise TypeError("Input k must be a number.")
    if m is None:
        m = n
    elif not isinstance(m, (int, float, bool)):
        raise TypeError("Input m must be a number.")
    if upper:
        compare = operator.ge
    else:
        compare = operator.le
    x_coordinate = []
    y_coordinate = []
    # math.ceil is used to match numpy's behaviour
    for i in range(math.ceil(n)):
        curr_limit = i + k
        for j in range(math.ceil(m)):
            if compare(j, curr_limit):
                x_coordinate.append(i)
                y_coordinate.append(j)
    return asarray_const(x_coordinate), asarray_const(y_coordinate)


def triu_indices(n, k=0, m=None):
    """
    Returns the indices for the upper-triangle of an (n, m) array.

    Args:
        n (int): The size of the arrays for which the returned indices will be valid.
        k (int, optional): Diagonal offset, default is 0.
        m (int, optional): The column dimension of the arrays for which the returned
            arrays will be valid. By default `m` is taken equal to `n`.

    Returns:
        The indices for the triangle. The returned tuple contains two tensors, each
        with the indices along one dimension of the tensor.

    Raises:
        TypeError: If `n`, `k`, `m` are not numbers.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.triu_indices(3))
        (Tensor(shape=[6], dtype=Int32, value= [0, 0, 0, 1, 1, 2]),
         Tensor(shape=[6], dtype=Int32, value= [0, 1, 2, 1, 2, 2]))
    """
    return tri_indices(n, k, m, True)


def tril_indices(n, k=0, m=None):
    """
    Returns the indices for the lower-triangle of an (n, m) array.

    Args:
        n (int): The size of the arrays for which the returned indices will be valid.
        k (int, optional): Diagonal offset, default is 0.
        m (int, optional): The column dimension of the arrays for which the returned
            arrays will be valid. By default `m` is taken equal to `n`.

    Returns:
        The indices for the triangle. The returned tuple contains two tensors, each
        with the indices along one dimension of the tensor.

    Raises:
        TypeError: If `n`, `k`, `m` are not numbers.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> print(np.tril_indices(3))
        (Tensor(shape=[6], dtype=Int32, value= [0, 1, 1, 2, 2, 2]),
        Tensor(shape=[6], dtype=Int32, value= [0, 0, 1, 0, 1, 2]))
    """
    return tri_indices(n, k, m, False)


def triu_indices_from(arr, k=0):
    """
    Returns the indices for the upper-triangle of `arr`.

    Args:
        arr (Union[Tensor, list, tuple]): 2-dimensional array.
        k (int, optional): Diagonal offset, default is 0.

    Returns:
        triu_indices_from, tuple of 2 tensor, shape(N)
        Indices for the upper-triangle of `arr`.

    Raises:
        TypeError: If `arr` cannot be converted to tensor, or `k` is not a number.
        ValueError: If `arr` cannot be converted to a 2-dimensional tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> tensor = np.ones((3,3))
        >>> print(np.triu_indices_from(tensor))
        (Tensor(shape=[6], dtype=Int32, value= [0, 0, 0, 1, 1, 2]),
        Tensor(shape=[6], dtype=Int32, value= [0, 1, 2, 1, 2, 2]))
    """
    arr = asarray(arr)
    if arr.ndim != 2:
        _raise_value_error("input array must be 2-d")
    return triu_indices(arr.shape[-2], k=k, m=arr.shape[-1])


def tril_indices_from(arr, k=0):
    """
    Returns the indices for the lower-triangle of `arr`.

    Args:
        arr (Union[Tensor, list, tuple]): 2-dimensional array.
        k (int, optional): Diagonal offset, default is 0.

    Returns:
        triu_indices_from, tuple of 2 tensor, shape(N)
        Indices for the upper-triangle of `arr`.

    Raises:
        TypeError: If `arr` cannot be converted to tensor, or `k` is not a number.
        ValueError: If `arr` cannot be converted to a 2-dimensional tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> tensor = np.ones((3,3))
        >>> print(np.tril_indices_from(tensor))
        (Tensor(shape=[6], dtype=Int32, value= [0, 1, 1, 2, 2, 2]),
         Tensor(shape=[6], dtype=Int32, value= [0, 0, 1, 0, 1, 2]))
    """
    arr = asarray(arr)
    if arr.ndim != 2:
        _raise_value_error("input array must be 2-d")
    return tril_indices(arr.shape[-2], k=k, m=arr.shape[-1])


def histogram_bin_edges(a, bins=10, range=None, weights=None):  # pylint: disable=redefined-builtin
    """
    Function to calculate only the edges of the bins used by the histogram function.

    Note:
        String values for `bins` is not supported.

    Args:
        a (Union[int, float, bool, list, tuple, Tensor]): Input data. The histogram
            is computed over the flattened array.
        bins ((Union[int, tuple, list, Tensor])): If `bins` is an int, it defines the number
            of equal-width bins in the given range (10, by default). If `bins` is a
            sequence, it defines the bin edges, including the rightmost edge,
            allowing for non-uniform bin widths.
        range((float, float), optional): The lower and upper range of the bins. If
            not provided, `range` is simply ``(a.min(), a.max())``. Values outside
            the range are ignored. The first element of the range must be less than
            or equal to the second. Default is None.
        weights(Union[int, float, bool, list, tuple, Tensor], optional):  An array of weights,
            of the same shape as `a`. Each value in `a` only contributes its associated weight
            towards the bin count (instead of 1). This is currently not used by any of the bin
            estimators, but may be in the future. Default is None.

    Returns:
        Tensor, the edges to pass into `histogram`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Raises:
        TypeError: If `bins` is an array and not one-dimensional.

    Examples:
        >>> import mindspore.numpy as np
        >>> arr = np.array([0, 0, 0, 1, 2, 3, 3, 4, 5])
        >>> print(np.histogram_bin_edges(arr, bins=2))
        [0.  2.5 5. ]
    """
    a = _to_tensor(a)
    if weights is not None:
        weights = _to_tensor(weights)
        if F.shape(a) != F.shape(weights):
            _raise_value_error('weights should have the same shape as a')
    if isinstance(bins, (tuple, list, Tensor)):
        bins = _to_tensor(bins)
        if F.rank(bins) != 1:
            _raise_value_error('`bins` must be 1d, when an array')
        return bins
    if isinstance(bins, str):
        # linspace does not support Tensor for num
        _raise_unimplemented_error('string value for `bins` not implemented')
    a = a.ravel().astype(mstype.float32)
    if range is None:
        start = F.reduce_min(a)
        end = F.reduce_max(a)
    else:
        if not isinstance(range, (list, tuple)) or len(range) != 2:
            _raise_value_error('`range` should take the form (start, end)')
        start, end = range
        if start > end:
            _raise_value_error('max must be larger than min in range parameter')
        start, end = _to_tensor(start, end)
    no_range = (end - start) == 0
    start = where(no_range, start - 0.5, start)
    end = where(no_range, end + 0.5, end)
    return linspace(start, end, bins + 1)


def _pad_empty(arr, pad_width):
    """
    pads the array with constant values, used in mode: "empty"
    """
    dtype = arr.dtype
    for i in range(arr.ndim):
        shape = arr.shape
        pad_before = ()
        pad_after = ()
        # To avoid any memory issues, we don't make tensor with 0s in their shapes
        if pad_width[i][0] > 0:
            pad_before += (empty(_tuple_setitem(shape, i, pad_width[i][0]), dtype=dtype),)
        if pad_width[i][1] > 0:
            pad_after += (empty(_tuple_setitem(shape, i, pad_width[i][1]), dtype=dtype),)
        tensor_with_pad = pad_before + (arr,) + pad_after
        arr = concatenate(tensor_with_pad, axis=i)
    return arr


def _pad_constant(arr, pad_width, value):
    """
    pads the array with constant values, used in mode: "constant"
    """
    dtype = arr.dtype
    for i in range(arr.ndim):
        shape = arr.shape
        pad_before = ()
        pad_after = ()
        # To avoid any memory issues, we don't make tensor with 0s in their shapes
        if pad_width[i][0] > 0:
            pad_before += (full(_tuple_setitem(shape, i, pad_width[i][0]), value[i][0], dtype=dtype),)
        if pad_width[i][1] > 0:
            pad_after += (full(_tuple_setitem(shape, i, pad_width[i][1]), value[i][1], dtype=dtype),)
        tensor_with_pad = pad_before + (arr,) + pad_after
        arr = concatenate(tensor_with_pad, axis=i)
    return arr


def _pad_statistic(arr, pad_width, stat_length, stat_op):
    """
    pads the array with values calculated along the given axis, used in mode: "maximum",
    "minimum", "mean"
    """
    ndim = arr.ndim
    shape = arr.shape
    if stat_length is None:
        stat_length = _make_stat_length(shape)
    else:
        stat_length = _convert_pad_to_nd(stat_length, ndim)
    stat_length = _limit_stat_length(stat_length, shape)
    for i in range(ndim):
        pad_before = stat_op(_slice_along_axis(arr, i, 0, stat_length[i][0]), i)
        pad_before = (F.tile(pad_before, _tuple_setitem((1,) * ndim, i, pad_width[i][0])),)
        pad_after = stat_op(_slice_along_axis(arr, i, shape[i] - stat_length[i][1], shape[i]), i)
        pad_after = (F.tile(pad_after, _tuple_setitem((1,) * ndim, i, pad_width[i][1])),)
        tensor_with_pad = pad_before + (arr,) + pad_after
        arr = concatenate(tensor_with_pad, axis=i)
    return arr


def _pad_edge(arr, pad_width):
    """pad_edge is equivalent to pad_statistic with stat_lenght=1, used in mode:"edge"."""

    def identity_op(arr, axis):
        return arr

    return _pad_statistic(arr, pad_width, 1, identity_op)


def _pad_wrap(arr, pad_width):
    """The behaviour of wrap mode is consistent with jax.numpy, used in mode:"wrap"."""
    ndim = arr.ndim
    shape = arr.shape
    for i in range(ndim):
        padsize_before = pad_width[i][0] % shape[i]
        padsize_after = pad_width[i][1] % shape[i]
        total_repeats = pad_width[i][0] // shape[i] + 1 + pad_width[i][1] // shape[i]
        tensor_with_pad = ()
        # To avoid any memory issues, we don't make tensor with 0s in their shapes
        if padsize_before > 0:
            tensor_with_pad += (_slice_along_axis(arr, i, shape[i] - padsize_before, shape[i]),)
        tensor_with_pad += (F.tile(arr, _tuple_setitem((1,) * ndim, i, total_repeats)),)
        if padsize_after > 0:
            tensor_with_pad += (_slice_along_axis(arr, i, 0, padsize_after),)
        arr = concatenate(tensor_with_pad, axis=i)
    return arr


def _pad_linear(arr, pad_width, end_values):
    """Pads the arr with linear range values, used in mode: "linear_ramp"."""
    ndim = arr.ndim
    shape = arr.shape
    dtype = arr.dtype
    end_values = _convert_pad_to_nd(end_values, ndim)
    for i in range(ndim):
        left_value = _slice_along_axis(arr, i, 0, 1)
        right_value = _slice_along_axis(arr, i, shape[i] - 1, shape[i])
        pad_before = ()
        pad_after = ()
        if pad_width[i][0] > 0:
            pad_before = (linspace(end_values[i][0], left_value, num=pad_width[i][0],
                                   endpoint=False, dtype=dtype, axis=i).squeeze(i + 1),)
        if pad_width[i][1] > 0:
            pad_after = linspace(right_value, end_values[i][1], num=pad_width[i][1] + 1,
                                 endpoint=True, dtype=dtype, axis=i).squeeze(i + 1)
            pad_after = (_slice_along_axis(pad_after, i, 1, pad_width[i][1] + 1),)
        tensor_with_pad = pad_before + (arr,) + pad_after
        arr = concatenate(tensor_with_pad, axis=i)
    return arr


def _add_pads_before(arr, pad_args, mode):
    """handle pads before the array"""
    idx, array_length, times_to_pad_before, additional_pad_before, reflect_type = pad_args
    curr_pad = None
    endpoint_adder = None
    edge_before = _slice_along_axis(arr, idx, 0, 1)
    if mode == "reflect":
        endpoint_adder = 1
    else:
        endpoint_adder = 0
    # Deal with paddings before the original array
    for times in range(times_to_pad_before):
        if times < times_to_pad_before - 1:
            endpoint = array_length
        else:
            endpoint = additional_pad_before + endpoint_adder
        if endpoint != endpoint_adder:
            curr_pad = _slice_along_axis(arr, idx, endpoint_adder, endpoint)
            curr_pad = flip(curr_pad, axis=idx)
            if reflect_type == "odd":
                curr_pad = 2 * edge_before - curr_pad
            arr = P.Concat(idx)((curr_pad, arr))
            edge_before = _slice_along_axis(arr, idx, 0, 1)
    return arr


def _add_pads_after(arr, pad_args, mode):
    """handle pads after the array"""
    idx, array_length, times_to_pad_after, additional_pad_after, reflect_type = pad_args
    curr_pad = None
    endpoint_adder = None
    edge_end = _slice_along_axis(arr, idx, arr.shape[idx] - 1, arr.shape[idx])
    if mode == "reflect":
        endpoint_adder = 1
    else:
        endpoint_adder = 0
    # Deal with paddings after the original array
    for times in range(times_to_pad_after):
        if times < times_to_pad_after - 1:
            startpoint = arr.shape[idx] - array_length
        else:
            startpoint = arr.shape[idx] - additional_pad_after - endpoint_adder
        if startpoint != arr.shape[idx] - endpoint_adder:
            curr_pad = _slice_along_axis(arr, idx, startpoint, arr.shape[idx] - endpoint_adder)
            curr_pad = flip(curr_pad, axis=idx)
            if reflect_type == "odd":
                curr_pad = 2 * edge_end - curr_pad
            arr = P.Concat(idx)((arr, curr_pad))
            edge_end = _slice_along_axis(arr, idx, arr.shape[idx] - 1, arr.shape[idx])
    return arr


def _pad_symmetric(arr, pad_width, reflect_type):
    """pad the array with symmetric paddings"""
    for i in range(arr.ndim):
        array_length = arr.shape[i]

        has_pad_before = (pad_width[i][0] > 0)
        has_pad_after = (pad_width[i][1] > 0)

        times_to_pad_before = pad_width[i][0] // array_length + 1
        additional_pad_before = pad_width[i][0] % array_length
        times_to_pad_after = pad_width[i][1] // array_length + 1
        additional_pad_after = pad_width[i][1] % array_length
        if has_pad_before:
            # Deal with paddings before the original array
            pad_args = (i, array_length, times_to_pad_before, additional_pad_before, reflect_type)
            arr = _add_pads_before(arr, pad_args, "symmetric")
        if has_pad_after:
            # Deal with paddings after the original array
            pad_args = (i, array_length, times_to_pad_after, additional_pad_after, reflect_type)
            arr = _add_pads_after(arr, pad_args, "symmetric")
    return arr


def _pad_reflect(arr, pad_width, reflect_type):
    """
    pad the array with reflect paddings, this is very similar to symmetric paddings,
    but differs at how edges are selected.
    """
    for i in range(arr.ndim):
        array_length = arr.shape[i]
        if array_length == 1:
            total_repeats = pad_width[i][0] + pad_width[i][1] + 1
            arr = F.tile(arr, _tuple_setitem((1,) * arr.ndim, i, total_repeats))
        else:
            has_pad_before = (pad_width[i][0] > 0)
            has_pad_after = (pad_width[i][1] > 0)

            pad_size = array_length - 1
            times_to_pad_before = pad_width[i][0] // pad_size + 1
            additional_pad_before = pad_width[i][0] % pad_size
            times_to_pad_after = pad_width[i][1] // pad_size + 1
            additional_pad_after = pad_width[i][1] % pad_size
            if has_pad_before:
                # Deal with paddings before the original array
                pad_args = (i, array_length, times_to_pad_before, additional_pad_before, reflect_type)
                arr = _add_pads_before(arr, pad_args, "reflect")
            if has_pad_after:
                # Deal with paddings after the original array
                pad_args = (i, array_length, times_to_pad_after, additional_pad_after, reflect_type)
                arr = _add_pads_after(arr, pad_args, "reflect")
    return arr


def _pad_func(arr, pad_width, func, **kwargs):
    """applies padding function over different axis."""
    # first creates a padded array with fixed length.
    arr_dim = arr.ndim
    pad_width = _convert_pad_to_nd(pad_width, arr_dim)
    arr = _pad_empty(arr, pad_width)
    for i in range(arr_dim):
        # function signature: padding_func(tensor, iaxis_pad_width, iaxis, kwargs)
        arr = apply_along_axis(func, i, arr, pad_width[i], i, kwargs)
    return arr


@_primexpr
def _make_stat_length(shape):
    """converts the stat_length values."""
    return tuple((shape[i], shape[i]) for i, _ in enumerate(shape))


@_primexpr
def _limit_stat_length(stat_length, shape):
    """limits the stat_length to current array length along given dimension."""
    return tuple((min(stat_pair[0], shape[i]), min(stat_pair[1], shape[i])) for i, stat_pair in enumerate(stat_length))


@constexpr
def _convert_pad_to_nd(pad_values, ndim):
    """broadcasts the pad_values to (ndim * 2)"""
    if not isinstance(pad_values, (int, list, tuple, Tensor)):
        raise TypeError(
            "pad_width, stat_length, constant_values or end_values should only be int, list, tuple or tensor")
    pad_tensor = _to_tensor(pad_values)
    pad_shape = pad_tensor.shape
    if not pad_shape:
        pad_values = tuple((((pad_values,) * 2) for i in range(ndim)))
    elif pad_shape == (1,):
        pad_values = tuple((tuple(pad_values) * 2) for i in range(ndim))
    elif pad_shape == (2,):
        pad_values = tuple(tuple(pad_values) for i in range(ndim))
    elif pad_shape == (1, 2):
        pad_values = tuple(tuple(pad_values[0]) for i in range(ndim))
    elif pad_shape == (ndim, 2):
        pad_values = tuple(tuple(pad_pair) for pad_pair in pad_values)
    else:
        raise ValueError(f"input values must be able to broadcast to {(ndim, 2)}")
    return pad_values


def pad(arr, pad_width, mode="constant", stat_length=None, constant_values=0,
        end_values=0, reflect_type="even", **kwargs):
    """
    Pads an array.

    Note:
        Currently, `median` mode is not supported. `reflect` and `symmetric` mode
        only supports GPU backend.

    Args:
        arr (Union[list, tuple, Tensor]): The array to pad.
        pad_width (Union[int, tuple, list]): Number of values padded to the edges of
            each axis. :class:`((before_1, after_1), ... (before_N, after_N))` creates
            unique pad widths for each axis. :class:`((before, after),)` yields same
            before and after pad for each axis. :class:`(pad,)` or int is a shortcut
            for :class:`before = after = pad width` for all axes.
        mode (string, optional):
            One of the following string values:

            - constant (default): Pads with a constant value.
            - edge: Pads with the edge values of `arr`.
            - linear_ramp: Pads with the linear ramp between end_value and the `arr` edge value.
            - maximum: Pads with the maximum value of all or part of the vector along each axis.
            - mean: Pads with the mean value of all or part of the vector along each axis.
            - median: Pads with the median value of all or part of the vector along each axis.
            - minimum: Pads with the minimum value of all or part of the vector along each axis.
            - reflect: Pads with the reflection of the vector mirrored on the first
              and last values of the vector along each axis.
            - symmetric: Pads with the reflection of the vector mirrored along the edge
              of the `arr`.
            - wrap: Pads with the wrap of the vector along the axis. The first values
              are used to pad the end and the end values are used to pad the beginning.
            - empty: Pads with undefined values.
            - <function>: The padding function, if used, should modify and return a new 1-d tensor.
              It has the following signature: :class:`padding_func(tensor, iaxis_pad_width, iaxis, kwargs)`
        stat_length (Union[tuple, list, int], optional): Used in \'maximum\', \'mean\',
            \'median\', and \'minimum\'.  Number of values at edge of each axis used
            to calculate the statistic value. :class:`((before_1, after_1), ... (before_N, after_N))`
            creates unique statistic lengths for each axis. :class:`((before, after),)`
            yields same before and after statistic lengths for each axis. :class:`(stat_length,)`
            or int is a shortcut for :class:`before = after = statistic length` for all
            axes. Default is :class:`None`, to use the entire axis.
        constant_values (Union[tuple, list, int], optional):
            Used in :class:`constant mode`. The values to set the padded values for each
            axis. :class:`((before_1, after_1), ... (before_N, after_N))` creates unique pad
            constants for each axis. :class:`((before, after),)` yields same before and
            after constants for each axis. :class:`(constant,)` or :class:`constant` is
            a shortcut for :class:`before = after = constant` for all axes. Default is 0.
        end_values (Union[tuple, list, int], optional): Used in 'linear_ramp'.  The values
            used for the ending value of the linear_ramp and that will form the edge of
            the padded `arr`. :class:`((before_1, after_1), ... (before_N, after_N))`
            unique end values for each axis. :class:`((before, after),)` yields same before
            and after end values for each axis. :class:`(constant,)` or :class:`constant`
            is a shortcut for :class:`before = after = constant` for all axes. Default is 0.
        reflect_type(string, optional) can choose between \'even\' and \'odd\'. Used in
            \'reflect\', and \'symmetric\'. The \'even\' style is the default with an
            unaltered reflection around the edge value. For the \'odd\' style, the extended
            part of the `arr` is created by subtracting the reflected values from two times
            the edge value.
        kwargs (anytype, optional): Any keyword arguments that will be used only in <function>
            mode.

    Returns:
        Padded tensor of rank equal to `arr` with shape increased according to `pad_width`.

    Raises:
        TypeError: If `arr`, `pad_width`, `stat_length`, `constant_values` or `end_values`
            have types not specified above.
        ValueError: If `mode` cannot be recognized, or if `pad_width`, `stat_length`,
            `constant_values`, `end_values` cannot broadcast to :class:`(arr.ndim, 2)`,
            or if keyword arguments got unexpected inputs.
        NotImplementedError: If mode is function or \'median\'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore.numpy as np
        >>> tensor = np.array([1., 2., 3., 4., 5.])
        >>> print(np.pad(tensor, (3, 4)))
        [0. 0. 0. 1. 2. 3. 4. 5. 0. 0. 0. 0.]
        >>> print(np.pad(tensor, (3, 4), mode="wrap"))
        [3. 4. 5. 1. 2. 3. 4. 5. 1. 2. 3. 4.]
        >>> print(np.pad(tensor, (3, 4), mode="linear_ramp", end_values=(10, 10)))
        [10.    7.    4.    1.    2.    3.    4.    5.    6.25  7.5   8.75 10.  ]
    """
    arr = _to_tensor(arr)
    if arr.ndim == 0:
        return arr
    pad_width = _convert_pad_to_nd(pad_width, arr.ndim)
    stat_func = {"maximum": _reduce_max_keepdims,
                 "minimum": _reduce_min_keepdims,
                 "mean": _reduce_mean_keepdims,
                 "median": "not implemented"}

    if mode not in ("constant", "maximum", "minimum", "mean", "median", "edge",
                    "wrap", "linear_ramp", "symmetric", "reflect", "empty") and \
            not _callable(arr, mode):
        _raise_value_error("Input mode not supported.")

    if mode == "constant":
        constant_values = _convert_pad_to_nd(constant_values, arr.ndim)
        return _pad_constant(arr, pad_width, constant_values)
    if mode in ("maximum", "minimum", "mean", "median"):
        # support median mode once P.Sort/P.Median is supported on GPU/CPU
        if mode == "median":
            _raise_unimplemented_error("median mode is not supported yet")
        return _pad_statistic(arr, pad_width, stat_length, stat_func[mode])
    if mode == "edge":
        return _pad_edge(arr, pad_width)
    if mode == "wrap":
        return _pad_wrap(arr, pad_width)
    if mode == "linear_ramp":
        return _pad_linear(arr, pad_width, end_values)
    if mode == "symmetric":
        return _pad_symmetric(arr, pad_width, reflect_type)
    if mode == "reflect":
        return _pad_reflect(arr, pad_width, reflect_type)
    if mode == 'empty':
        return _pad_empty(arr, pad_width)
    return _pad_func(arr, pad_width, mode, **kwargs)
