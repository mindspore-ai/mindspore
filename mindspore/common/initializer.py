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
"""Initializer for cell parameters."""
import numbers
import math

from functools import reduce
import numpy as np
from scipy.stats import truncnorm
from mindspore import log as logger

from . import dtype as mstype
from .tensor import Tensor

_INITIALIZER_ALIAS = dict()


class Initializer:
    """
    The base class of the initializer.

    Args:
        kwargs (dict): Keyword arguments for Initializer.

    Returns:
        Array, assigned array.
    """
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self.shape = None
        self.dtype = None
        self._seed = None

    def _initialize(self, *kwargs):
        raise NotImplementedError('Must be overridden!')

    def __call__(self, arr):
        return self._initialize(arr)

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed_):
        """set the random seed."""
        self._seed = seed_

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        self._dtype = dtype

    def to_tensor(self):
        """Get the tensor format data of this Initializer."""
        arr = None
        try:
            arr = np.ndarray(self.shape)
        except ValueError:
            msg = "Error shape={}".format(self.shape)
            logger.error(msg)
            raise ValueError(msg)
        if self._seed is not None:
            np.random.seed(self.seed)
        self.__call__(arr)
        self._seed = None
        return Tensor(arr, dtype=self.dtype)

def _register(*aliases):
    """Return the alias register."""
    def alias_reg(cls):
        name = cls.__name__
        name = name.lower()
        if name not in _INITIALIZER_ALIAS:
            _INITIALIZER_ALIAS[name] = cls

        for alias in aliases:
            if alias not in _INITIALIZER_ALIAS:
                _INITIALIZER_ALIAS[alias] = cls

        return cls

    return alias_reg


def _assignment(arr, num):
    """Assign the value of `num` to `arr`."""
    if arr.shape == ():
        arr = arr.reshape((1))
        arr[:] = num
        arr = arr.reshape(())
    else:
        if isinstance(num, np.ndarray):
            arr[:] = num[:]
        else:
            arr[:] = num
    return arr


@_register('zeros')
class Zero(Initializer):
    """
    Initialize the array to zero.

    Args:
        arr (Array): The array to be assigned.

    Returns:
        Array, assigned array.
    """
    def _initialize(self, arr):
        _assignment(arr, 0)


@_register('ones')
class One(Initializer):
    """
    Initialize the array to one.

    Args:
        arr (Array): The array to be assigned.

    Returns:
        Array, assigned array.
    """
    def _initialize(self, arr):
        _assignment(arr, 1)


def _calculate_in_and_out(arr):
    """
    Calculate n_in and n_out.

    Args:
        arr (Array): Input array.

    Returns:
        Tuple, a tuple with two elements, the first element is `n_in` and the second element is `n_out`.
    """
    dim = len(arr.shape)
    if dim < 2:
        raise ValueError("If initialize data with xavier uniform, the dimension of data must greater than 1.")

    n_in = arr.shape[1]
    n_out = arr.shape[0]

    if dim > 2:
        counter = reduce(lambda x, y: x * y, arr.shape[2:])
        n_in *= counter
        n_out *= counter
    return n_in, n_out


@_register('xavier_uniform')
class XavierUniform(Initializer):
    r"""
    Initialize the array with xavier uniform algorithm, and from a uniform distribution collect samples within
    U[-boundary, boundary] where :math:`boundary = gain * \sqrt{\frac{6}{n_{in} + n_{out}}}`.

    Args:
        gain (Array): The array to be assigned. Default: 1.

    Returns:
        Array, assigned array.
    """
    def __init__(self, gain=1):
        super(XavierUniform, self).__init__(gain=gain)
        self.gain = gain

    def _initialize(self, arr):
        n_in, n_out = _calculate_in_and_out(arr)

        boundary = self.gain * math.sqrt(6.0 / (n_in + n_out))
        data = np.random.uniform(-boundary, boundary, arr.shape)

        _assignment(arr, data)


@_register('he_uniform')
class HeUniform(Initializer):
    r"""
    Initialize the array with He kaiming uniform algorithm, and from a uniform distribution collect samples within
    U[-boundary, boundary] where :math:`boundary = \sqrt{\frac{6}{n_{in}}}` where :math:`n_{in}` is the number of
    input units in the weight tensor.

    Args:
        arr (Array): The array to be assigned.

    Returns:
        Array, assigned array.
    """

    def _initialize(self, arr):
        n_in, _ = _calculate_in_and_out(arr)

        boundary = math.sqrt(6.0 / n_in)
        data = np.random.uniform(-boundary, boundary, arr.shape)

        _assignment(arr, data)


class Constant(Initializer):
    """
    Initialize a constant.

    Args:
        value (Union[int, numpy.ndarray]): The value to initialize.

    Returns:
        Array, initialize array.
    """
    def __init__(self, value):
        super(Constant, self).__init__(value=value)
        self.value = value

    def _initialize(self, arr):
        _assignment(arr, self.value)


@_register()
class Uniform(Initializer):
    """
    Initialize a uniform array, and obtain values U(-scale, scale) from the uniform distribution
    to fill the input tensor.

    Args:
        scale (float): The scale of the array. Default: 0.07.

    Returns:
        Array, uniform array.
    """
    def __init__(self, scale=0.07):
        super(Uniform, self).__init__(scale=scale)
        self.scale = scale

    def _initialize(self, arr):
        tmp = np.random.uniform(-self.scale, self.scale, arr.shape)
        _assignment(arr, tmp)


@_register()
class Normal(Initializer):
    """
    Initialize a normal array, and obtain values N(0, sigma) from the uniform distribution
    to fill the input tensor.

    Args:
        sigma (float): The sigma of the array. Default: 0.01.

    Returns:
        Array, normal array.
    """
    def __init__(self, sigma=0.01):
        super(Normal, self).__init__(sigma=sigma)
        self.sigma = sigma

    def _initialize(self, arr):
        tmp = np.random.normal(0, self.sigma, arr.shape)
        _assignment(arr, tmp)


@_register()
class TruncatedNormal(Initializer):
    """
    Initialize a truncated normal distribution which is a bounded normal distribution within N(low, high).

    Args:
        sigma (float): The sigma of the array. Default: 0.01.

    Returns:
        Array, truncated normal array.
    """
    def __init__(self, sigma=0.01):
        super(TruncatedNormal, self).__init__(sigma=sigma)
        self.sigma = sigma

    def _initialize(self, arr):
        tmp = truncnorm.rvs(-2, 2, loc=0, scale=self.sigma, size=arr.shape, random_state=None)
        _assignment(arr, tmp)


def initializer(init, shape=None, dtype=mstype.float32):
    """
    Create and initialize a tensor.

    Args:
        init (Union[Tensor, str, Initializer, numbers.Number]): Initialize value.

            - `str`: The `init` should be the alias of the class inheriting from `Initializer` and the corresponding
              class will be called.

            - `Initializer`: The `init` should be the class inheriting from `Initializer` to initialize tensor.

            - `numbers.Number`: The `Constant` will be called to initialize tensor.

        shape (Union[tuple, list, int]): A list of integers, a tuple of integers or an integer as the shape of
            output. Default: None.
        dtype (:class:`mindspore.dtype`): The type of data in initialized tensor. Default: mindspore.float32.

    Returns:
        Union[Tensor, Initializer], When `init` is Tensor, the return is Tensor object,
        otherwise the return is Initialize object.

    Examples:
        >>> tensor = initializer('ones', [1, 2, 3], mindspore.float32)
    """
    if not isinstance(init, (Tensor, numbers.Number, str, Initializer)):
        raise TypeError("Unsupported init type '{}'.".format(type(init)))

    if isinstance(init, Tensor):
        init_shape = init.shape()
        shape = shape if isinstance(shape, (tuple, list)) else [shape]
        if shape is not None and init_shape != tuple(shape):
            raise ValueError("The shape of init should be same as variable shape, but got the shape of init {} and "
                             "the variable shape {}.".format(list(init.shape()), shape))
        return init

    if isinstance(shape, list):
        shape = tuple(shape)
    elif isinstance(shape, numbers.Number):
        shape = (shape,)
    try:
        np.ndarray(shape)
    except ValueError:
        raise ValueError("Error shape={}".format(shape))

    if isinstance(init, Initializer):
        init.shape = init.shape if init.shape is not None else shape
        init.dtype = init.dtype if init.dtype is not None else dtype
        return init

    if isinstance(init, str):
        init_obj = _INITIALIZER_ALIAS[init.lower()]()
        if init_obj is None:
            raise ValueError("The class corresponding to '{}' was not found.".format(init))
        init = init_obj
        init.shape = shape
        init.dtype = dtype
        return init

    if isinstance(init, numbers.Number):
        init_obj = Constant(init)
        init_obj.shape = shape
        init_obj.dtype = dtype
        return init_obj
    raise TypeError("Unsupported init type '{}'.".format(type(init)))

__all__ = [
    'Initializer',
    'initializer',
    'TruncatedNormal',
    'Normal',
    'Uniform',
    'HeUniform',
    'XavierUniform',
    'One',
    'Zero',
    'Constant']
