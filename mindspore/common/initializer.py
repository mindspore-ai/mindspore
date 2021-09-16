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
"""Initializer for cell parameters."""
import numbers
import math

from functools import reduce
import numpy as np
from scipy.stats import truncnorm
from .seed import get_seed, _get_graph_seed
from . import dtype as mstype
from .tensor import Tensor
from .._c_expression import random_normal

_INITIALIZER_ALIAS = dict()


class Initializer:
    """
    The base class of the initializer.
    Initialization of tensor basic attributes and model weight values.

    Args:
        kwargs (dict): Keyword arguments for Initializer.
    """
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._seed = None

    @property
    def seed(self):
        if self._seed is None:
            seed, seed2 = _get_graph_seed(get_seed(), "init")
        else:
            seed, seed2 = self._seed + 1, 0
        return seed, seed2

    @seed.setter
    def seed(self, value):
        self._seed = value

    def _initialize(self, *kwargs):
        raise NotImplementedError('Must be overridden!')

    def __call__(self, arr):
        return self._initialize(arr)

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
        arr = arr.reshape(1)
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
    Fills the input array with the values zero.

    Args:
        arr (Array): The array to be assigned.


    Examples:
        >>> import mindspore
        >>> from mindspore.common.initializer import initializer, Zero
        >>> tensor1 = initializer(Zero(), [1, 2, 3], mindspore.float32)
        >>> tensor2 = initializer('zeros', [1, 2, 3], mindspore.float32)
    """
    def _initialize(self, arr):
        _assignment(arr, 0)


@_register('ones')
class One(Initializer):
    """
    Fills the input array with the values one.

    Args:
        arr (Array): The array to be assigned.


    Examples:
        >>> import mindspore
        >>> from mindspore.common.initializer import initializer, One
        >>> tensor1 = initializer(One(), [1, 2, 3], mindspore.float32)
        >>> tensor2 = initializer('ones', [1, 2, 3], mindspore.float32)
    """
    def _initialize(self, arr):
        _assignment(arr, 1)


def _calculate_fan_in_and_fan_out(shape):
    """
    calculate fan_in and fan_out

    Args:
        shape (tuple): input shape.

    Returns:
        Tuple, a tuple with two elements, the first element is `n_in` and the second element is `n_out`.
    """
    dimensions = len(shape)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    if dimensions == 2:  # Linear
        fan_in = shape[1]
        fan_out = shape[0]
    else:
        num_input_fmaps = shape[1]
        num_output_fmaps = shape[0]
        receptive_field_size = 1
        for i in range(2, dimensions):
            receptive_field_size *= shape[i]
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def _calculate_correct_fan(shape, mode):
    """
    Calculate fan.

    Args:
        shape (tuple): input shape.
        mode (str): only support fan_in and fan_out.

    Returns:
        fan_in or fan_out.
    """
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))
    fan_in, fan_out = _calculate_fan_in_and_fan_out(shape)
    return fan_in if mode == 'fan_in' else fan_out


def _calculate_gain(nonlinearity, param=None):
    """
    Calculate gain.

    Args:
        nonlinearity (str): nonlinearity function.
        param (str): used to calculate negative_slope.

    Returns:
        number.
    """
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        res = 1
    elif nonlinearity == 'tanh':
        res = 5.0 / 3
    elif nonlinearity == 'relu':
        res = math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} is not a valid number. "
                             "It should be bool, int, or float type.".format(param))
        res = math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}, the argument 'nonlinearity' should be one of "
                         "'sigmoid', 'tanh', 'relu' or 'leaky_relu'.".format(nonlinearity))
    return res


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
        raise ValueError("If initialize data with xavier uniform, the dimension of data must be greater than 1.")

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
    :math:`{U}(-\text{boundary}, \text{boundary})` where:

    .. math::
        boundary = gain * \sqrt{\frac{6}{n_{in} + n_{out}}}

    - where :math:`gain` is an optional scaling factor.
    - where :math:`n_{in}` is the number of input units in the weight tensor.
    - where :math:`n_{out}` is the number of output units in the weight tensor.

    For details of XavierUniform algorithm, please check
    `<http://proceedings.mlr.press/v9/glorot10a.html>`_.

    Args:
        gain (float): An optional scaling factor. Default: 1.


    Examples:
        >>> import mindspore
        >>> from mindspore.common.initializer import initializer, XavierUniform
        >>> tensor1 = initializer(XavierUniform(), [1, 2, 3], mindspore.float32)
        >>> tensor2 = initializer('xavier_uniform', [1, 2, 3], mindspore.float32)
    """
    def __init__(self, gain=1):
        super(XavierUniform, self).__init__(gain=gain)
        self.gain = gain

    def _initialize(self, arr):
        n_in, n_out = _calculate_fan_in_and_fan_out(arr.shape)

        boundary = self.gain * math.sqrt(6.0 / (n_in + n_out))
        data = np.random.uniform(-boundary, boundary, arr.shape)

        _assignment(arr, data)


@_register('he_uniform')
class HeUniform(Initializer):
    r"""
    Initialize the array with HeKaiming Uniform algorithm, and from a uniform distribution collect samples within
    :math:`{U}(-\text{boundary}, \text{boundary})` where

    .. math::
        boundary = \sqrt{\frac{6}{(1 + a^2) \times \text{fan_in}}}

    - where :math:`-boundary` the lower bound of the HeUniform distribution.
    - where :math:`boundary` the upper bound of the HeUniform distribution.

    For details of HeUniform algorithm, please check
    `<https://arxiv.org/abs/1502.01852>`_.

    Args:
        negative_slope (int, float, bool): The negative slope of the rectifier used after this layer
            (only used when `nonlinearity` is 'leaky_relu'). Default: 0.
        mode (str): Either 'fan_in' or 'fan_out'. Choosing 'fan_in' preserves the magnitude of the
            variance of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes
            in the backwards pass. Default: fan_in.
        nonlinearity (str): The non-linear function, recommended to use only with 'relu' or 'leaky_relu'.
            Default: leaky_relu.


    Examples:
        >>> import mindspore
        >>> from mindspore.common.initializer import initializer, HeUniform
        >>> tensor1 = initializer(HeUniform(), [1, 2, 3], mindspore.float32)
        >>> tensor2 = initializer('he_uniform', [1, 2, 3], mindspore.float32)
    """
    def __init__(self, negative_slope=0, mode='fan_in', nonlinearity='leaky_relu'):
        super(HeUniform, self).__init__(negative_slope=negative_slope, mode=mode, nonlinearity=nonlinearity)
        self.negative_slope = negative_slope
        self.mode = mode
        self.nonlinearity = nonlinearity

    def _initialize(self, arr):
        fan = _calculate_correct_fan(arr.shape, self.mode)
        gain = _calculate_gain(self.nonlinearity, self.negative_slope)
        std = gain / math.sqrt(fan)
        boundary = math.sqrt(3.0) * std
        data = np.random.uniform(-boundary, boundary, arr.shape)

        _assignment(arr, data)


@_register('he_normal')
class HeNormal(Initializer):
    r"""
    Initialize the array with HeKaiming Normal algorithm, and from a normal distribution collect samples within
    :math:`{N}(0, \text{sigma}^2)` where

    .. math::
        sigma = \frac{gain} {\sqrt{mode}}

    - where :math:`gain` is an optional scaling factor.
    - where :math:`mode` is the number of input units or output units in the weight tensor.

    For details of HeUniform algorithm, please check
    `<https://arxiv.org/abs/1502.01852>`_.

    Args:
        negative_slope (int, float, bool): The negative slope of the rectifier used after this layer
            (only used when `nonlinearity` is 'leaky_relu'). Default: 0.
        mode (str): Either 'fan_in' or 'fan_out'. Choosing 'fan_in' preserves the magnitude of the
            variance of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes
            in the backwards pass. Default: fan_in.
        nonlinearity (str): The non-linear function, recommended to use only with 'relu' or 'leaky_relu'.
            Default: leaky_relu.


    Examples:
        >>> import mindspore
        >>> from mindspore.common.initializer import initializer, HeNormal
        >>> tensor1 = initializer(HeNormal(), [1, 2, 3], mindspore.float32)
        >>> tensor2 = initializer('he_normal', [1, 2, 3], mindspore.float32)
    """
    def __init__(self, negative_slope=0, mode='fan_in', nonlinearity='leaky_relu'):
        super(HeNormal, self).__init__(negative_slope=negative_slope, mode=mode, nonlinearity=nonlinearity)
        self.negative_slope = negative_slope
        self.mode = mode
        self.nonlinearity = nonlinearity

    def _initialize(self, arr):
        fan = _calculate_correct_fan(arr.shape, self.mode)
        gain = _calculate_gain(self.nonlinearity, self.negative_slope)
        std = gain / math.sqrt(fan)
        data = np.random.normal(0, std, arr.shape)

        _assignment(arr, data)


class Constant(Initializer):
    """
    Initialize a constant.

    Args:
        value (Union[int, numpy.ndarray]): The value to initialize.


    Examples:
        >>> import mindspore
        >>> from mindspore.common.initializer import initializer
        >>> tensor1 = initializer(0, [1, 2, 3], mindspore.float32)
        >>> tensor2 = initializer(5, [1, 2, 3], mindspore.float32)
    """
    def __init__(self, value):
        super(Constant, self).__init__(value=value)
        self.value = value

    def _initialize(self, arr):
        _assignment(arr, self.value)


@_register()
class Uniform(Initializer):
    """
    Initialize a uniform array, and obtain values :math:`{U}(-\text{scale}, \text{scale})` from the uniform distribution
    to fill the input tensor.

    Args:
        scale (float): The scale of the array. Default: 0.07.


    Examples:
        >>> import mindspore
        >>> from mindspore.common.initializer import initializer, Uniform
        >>> tensor1 = initializer(Uniform(), [1, 2, 3], mindspore.float32)
        >>> tensor2 = initializer('uniform', [1, 2, 3], mindspore.float32)
    """
    def __init__(self, scale=0.07):
        super(Uniform, self).__init__(scale=scale)
        self.scale = scale

    def _initialize(self, arr):
        tmp = np.random.uniform(-self.scale, self.scale, arr.shape)
        _assignment(arr, tmp)


@_register()
class Normal(Initializer):
    r"""
    Initialize a normal array, and obtain values :math:`{N}(\text{sigma}, \text{mean})` from the normal distribution
    to fill the input tensor.

    .. math::
        f(x) =  \frac{1} {\sqrt{2*π} * sigma}exp(-\frac{(x - mean)^2} {2*{sigma}^2})

    Args:
        sigma (float): The sigma of the array. Default: 0.01.
        mean (float): The mean of the array. Default: 0.0.


    Examples:
        >>> import mindspore
        >>> from mindspore.common.initializer import initializer, Normal
        >>> tensor1 = initializer(Normal(), [1, 2, 3], mindspore.float32)
        >>> tensor2 = initializer('normal', [1, 2, 3], mindspore.float32)
    """
    def __init__(self, sigma=0.01, mean=0.0):
        super(Normal, self).__init__(sigma=sigma, mean=mean)
        self.sigma = sigma
        self.mean = mean

    def _initialize(self, arr):
        seed, seed2 = self.seed
        output_tensor = Tensor(np.zeros(arr.shape, dtype=np.float32))
        random_normal(arr.shape, seed, seed2, output_tensor)
        output_data = output_tensor.asnumpy()
        output_data = output_data * self.sigma + self.mean
        _assignment(arr, output_data)

@_register()
class TruncatedNormal(Initializer):
    """
    Initialize a truncated normal distribution which is a bounded normal distribution
    within :math:`{N}(\text{low}, \text{high})`.

    Args:
        sigma (float): The sigma of the array. Default: 0.01.


    Examples:
        >>> import mindspore
        >>> from mindspore.common.initializer import initializer, TruncatedNormal
        >>> tensor1 = initializer(TruncatedNormal(), [1, 2, 3], mindspore.float32)
        >>> tensor2 = initializer('truncatedNormal', [1, 2, 3], mindspore.float32)
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
              class will be called. The value of 'init' can be "normal", "ones" or "zeros", etc.

            - `Initializer`: The `init` should be the class inheriting from `Initializer` to initialize tensor.

            - `numbers.Number`: The `Constant` will be called to initialize tensor.

        shape (Union[tuple, list, int]): A list of integers, a tuple of integers or an integer as the shape of
            output. Default: None.
        dtype (:class:`mindspore.dtype`): The type of data in initialized tensor. Default: mindspore.float32.

    Returns:
        Union[Tensor], return is Tensor object.


    Examples:
        >>> import mindspore
        >>> from mindspore.common.initializer import initializer, One
        >>> tensor1 = initializer('ones', [1, 2, 3], mindspore.float32)
        >>> tensor2 = initializer(One(), [1, 2, 3], mindspore.float32)
        >>> tensor3 = initializer(0, [1, 2, 3], mindspore.float32)
    """
    if not isinstance(init, (Tensor, numbers.Number, str, Initializer)):
        raise TypeError("Unsupported init type '{}', init should be 'Tensor', 'number', 'str' "
                        "or 'initializer' type".format(type(init)))

    if isinstance(init, Tensor):
        init_shape = init.shape
        shape = shape if isinstance(shape, (tuple, list)) else [shape]
        if shape is not None and init_shape != tuple(shape):
            raise ValueError("The shape of init should be same as variable shape, but got the shape of init {} and "
                             "the variable shape {}.".format(list(init.shape), shape))
        return init

    if isinstance(shape, list):
        shape = tuple(shape)
    elif isinstance(shape, numbers.Number):
        shape = (shape,)

    for value in shape if shape is not None else ():
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"Shape is invalid, the value of shape must be positive integer, but got shape:{shape}")

    if isinstance(init, str):
        init = _INITIALIZER_ALIAS[init.lower()]()
        if init is None:
            raise ValueError("The class corresponding to '{}' was not found.".format(init))
    elif isinstance(init, numbers.Number):
        init = Constant(init)
    shape = shape if shape is not None else init.shape
    init_obj = Tensor(dtype=dtype, shape=shape, init=init)
    return init_obj

__all__ = [
    'Initializer',
    'initializer',
    'TruncatedNormal',
    'Normal',
    'Uniform',
    'HeUniform',
    'HeNormal',
    'XavierUniform',
    'One',
    'Zero',
    'Constant']
