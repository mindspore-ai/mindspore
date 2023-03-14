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
"""Initializer for cell parameters."""
from __future__ import absolute_import

import numbers
import math

from functools import reduce
import numpy as np
from mindspore.common.seed import get_seed, _get_graph_seed
from mindspore.common import dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore._c_expression import _random_normal, _random_uniform, _truncated_normal

_INITIALIZER_ALIAS = dict()


class Initializer:
    """
    The abstract base class of the initializer.

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


def _numpy_seed():
    # This will produce same value after call numpy.random.seed with same seed.
    return np.random.randint(low=1, high=(1 << 63), dtype=np.int64)


def _init_random_normal(mean, sigma, shape):
    if sigma < 0:
        raise ValueError("sigma < 0")
    data = np.ndarray(shape=shape, dtype=np.float32)
    _random_normal(_numpy_seed(), data, mean, sigma)
    return data


def _init_random_uniform(a, b, shape):
    data = np.ndarray(shape=shape, dtype=np.float32)
    _random_uniform(_numpy_seed(), data, a, b)
    return data


def _init_truncated_normal(a, b, mean, sigma, shape):
    if sigma < 0:
        raise ValueError("sigma < 0")
    data = np.ndarray(shape=shape, dtype=np.float32)
    _truncated_normal(_numpy_seed(), data, a, b, mean, sigma)
    return data


@_register('zeros')
class Zero(Initializer):
    """
    Generates an array with constant value of zero in order to initialize a tensor.

    Examples:
        >>> import mindspore
        >>> from mindspore.common.initializer import initializer, Zero
        >>> tensor1 = initializer(Zero(), [1, 2, 3], mindspore.float32)
        >>> tensor2 = initializer('zeros', [1, 2, 3], mindspore.float32)
    """

    def _initialize(self, arr):
        arr.fill(0)


@_register('ones')
class One(Initializer):
    """
    Generates an array with constant value of one in order to initialize a tensor.

    Examples:
        >>> import mindspore
        >>> from mindspore.common.initializer import initializer, One
        >>> tensor1 = initializer(One(), [1, 2, 3], mindspore.float32)
        >>> tensor2 = initializer('ones', [1, 2, 3], mindspore.float32)
    """

    def _initialize(self, arr):
        arr.fill(1)


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
        raise ValueError("'fan_in' and 'fan_out' can not be computed for tensor with fewer than"
                         " 2 dimensions, but got dimensions {}.".format(dimensions))
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
        raise ValueError("'mode' {} not supported, please use one of {}".format(mode, valid_modes))
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
            raise ValueError("For 'HeUniform', 'negative_slope' {} is not a valid number."
                             "When 'nonlinearity' has been set to "
                             "'leaky_relu', 'negative_slope' should be int or float type, but got "
                             "{}.".format(param, type(param)))
        res = math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("For 'HeUniform', the argument 'nonlinearity' should be one of "
                         "['sigmoid', 'tanh', 'relu' or 'leaky_relu'], "
                         "but got {}.".format(nonlinearity))
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
        raise ValueError("If initialize data with xavier uniform, the dimension of data must be greater than 1, "
                         "but got {}.".format(dim))

    n_in = arr.shape[1]
    n_out = arr.shape[0]

    if dim > 2:
        counter = reduce(lambda x, y: x * y, arr.shape[2:])
        n_in *= counter
        n_out *= counter
    return n_in, n_out


@_register('xavier_normal')
class XavierNormal(Initializer):
    r"""
    Generates an array with values sampled from Xavier normal distribution
    :math:`{N}(0, \text{sigma}^2)` in order to initialize a tensor, where

    .. math::
        sigma = gain * \sqrt{\frac{2}{n_{in} + n_{out}}}

    where :math:`gain` is an optional scaling factor, :math:`n_{in}` is the number of input units in the weight tensor,
    :math:`n_{out}` is the number of output units in the weight tensor.

    Args:
        gain (float): An optional scaling factor. Default: 1.

    Examples:
        >>> import mindspore
        >>> from mindspore.common.initializer import initializer, XavierNormal
        >>> tensor1 = initializer(XavierNormal(), [1, 2, 3], mindspore.float32)
        >>> tensor2 = initializer('xavier_normal', [1, 2, 3], mindspore.float32)
    """
    def __init__(self, gain=1):
        super().__init__(gain=gain)
        self.gain = gain

    def _initialize(self, arr):
        fan_in, fan_out = _calculate_fan_in_and_fan_out(arr.shape)

        std = self.gain * math.sqrt(2.0 / float(fan_in + fan_out))
        data = _init_random_normal(0, std, arr.shape)

        _assignment(arr, data)


@_register('xavier_uniform')
class XavierUniform(Initializer):
    r"""
    Generates an array with values sampled from Xavier uniform distribution
    :math:`{U}(-\text{boundary}, \text{boundary})` in order to initialize a tensor, where

    .. math::
        boundary = gain * \sqrt{\frac{6}{n_{in} + n_{out}}}

    where :math:`gain` is an optional scaling factor.  :math:`n_{in}` is the number of input units in the weight tensor,
    :math:`n_{out}` is the number of output units in the weight tensor.

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
        data = _init_random_uniform(-boundary, boundary, arr.shape)
        _assignment(arr, data)


@_register('he_uniform')
class HeUniform(Initializer):
    r"""
    Generates an array with values sampled from HeKaiming Uniform distribution
    :math:`{U}(-\text{boundary}, \text{boundary})` in order to initialize a tensor, where

    .. math::
        boundary = \text{gain} \times \sqrt{\frac{3}{fan\_mode}}

    where :math:`gain` is an optional scaling factor. If :math:`fan\_mode` is 'fan_in', it is the number of input units
    of the weight tensor. If :math:`fan\_mode` is 'fan_out',
    it is the number of output units of the weight tensor.

    For details of HeUniform algorithm, please check
    `<https://arxiv.org/abs/1502.01852>`_.

    Args:
        negative_slope (int, float, bool): The negative slope of the rectifier used after this layer
            (only used when `nonlinearity` is 'leaky_relu'). Default: 0.
        mode (str): Either 'fan_in' or 'fan_out'. Choosing 'fan_in' preserves the magnitude of the
            variance of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes
            in the backwards pass. Default: 'fan_in'.
        nonlinearity (str): The non-linear function, recommended to use only with 'relu' or 'leaky_relu'.
            Default: 'leaky_relu'.


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
        data = _init_random_uniform(-boundary, boundary, arr.shape)
        _assignment(arr, data)


@_register('he_normal')
class HeNormal(Initializer):
    r"""
    Generates an array with values sampled from HeKaiming Normal distribution
    :math:`{N}(0, \text{sigma}^2)` in order to initialize a tensor, where

    .. math::
        sigma = \frac{gain} {\sqrt{fan\_mode}}

    where :math:`gain` is an optional scaling factor. :math:`fan\_mode` is the number of input or output units of
    the weight tensor, depending on the `mode` is 'fan_in' or 'fan_out'.

    For details of HeNormal algorithm, please check `<https://arxiv.org/abs/1502.01852>`_.

    Args:
        negative_slope (int, float): The negative slope of the rectifier used after this layer
            (only used when `nonlinearity` is 'leaky_relu'). Default: 0.
        mode (str): Either 'fan_in' or 'fan_out'. Choosing 'fan_in' preserves the magnitude of the
            variance of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes
            in the backwards pass. Default: 'fan_in'.
        nonlinearity (str): The non-linear function, recommended to use only with 'relu' or 'leaky_relu'.
            Default: 'leaky_relu'.


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
        data = _init_random_normal(0, std, arr.shape)
        _assignment(arr, data)


class Constant(Initializer):
    """
    Generates an array with constant value in order to initialize a tensor.

    Args:
        value (Union[int, numpy.ndarray]): The value to initialize.


    Examples:
        >>> import mindspore
        >>> from mindspore.common.initializer import initializer, Constant
        >>> tensor1 = initializer(Constant(3), [1, 2, 3], mindspore.float32)
    """

    def __init__(self, value):
        super(Constant, self).__init__(value=value)
        self.value = value

    def _initialize(self, arr):
        arr.fill(self.value)


@_register()
class Identity(Initializer):
    """
    Generates a 2 dimension identity matrix array in order to initialize a tensor.

    Raises:
        ValueError: If the dimension of input tensor is not equal to 2.

    Examples:
        >>> import mindspore
        >>> from mindspore.common.initializer import initializer, Identity
        >>> tensor1 = initializer(Identity(), [2, 3], mindspore.float32)
        >>> tensor2 = initializer('identity', [2, 3], mindspore.float32)
    """

    def _initialize(self, arr):
        if len(arr.shape) != 2:
            raise ValueError('For Identity initializer, the dimension of the initialized tensor should be 2, '
                             'but got {}.'.format(len(arr.shape)))
        value = np.eye(arr.shape[0], arr.shape[1])
        _assignment(arr, value)


@_register()
class Sparse(Initializer):
    """
    Generates a 2 dimension sparse matrix array in order to initialize a tensor. The non-zero positions
    will be filled with the value sampled from the normal distribution :math:`{N}(0, 0.01)`.

    Args:
         sparsity (float): The fraction of elements being set to zero in each column.
         sigma (float): The standard deviation of the normal distribution. Default: 0.01.

    Raises:
        ValueError: If the dimension of input tensor is not equal to 2.

    Examples:
        >>> import mindspore
        >>> from mindspore.common.initializer import initializer, Sparse
        >>> tensor1 = initializer(Sparse(sparsity=0.1, sigma=0.01), [5, 8], mindspore.float32)
    """

    def __init__(self, sparsity, sigma=0.01):
        super(Sparse, self).__init__()
        self.sparsity = sparsity
        self.sigma = sigma

    def _initialize(self, arr):
        if len(arr.shape) != 2:
            raise ValueError('For Sparse initializer, the dimension of the initialized tensor should be 2, '
                             'but got {}.'.format(len(arr.shape)))
        rows, cols = arr.shape
        zero_num = int(np.ceil(self.sparsity * rows))
        data = _init_random_normal(0, self.sigma, arr.shape)
        for col_idx in range(cols):
            row_idx = np.random.permutation(list(range(rows)))[: zero_num]
            data[row_idx, col_idx] = 0.
        _assignment(arr, data)


@_register()
class Dirac(Initializer):
    """
    Generates an array with the Dirac delta function in order to initialize a tensor.
    It tries to preserves the identity of input for convolution layers.
    For group convolution, each group of channels will be preserved respectively.

    Args:
        groups (int): The number of group in convolution layer. Default: 1.

    Raises:
        ValueError: If the dimension of the initialized tensor is not in [3, 4, 5].
        ValueError: The first dimension of the initialized tensor cannot be divisible by group.

    Examples:
        >>> import mindspore
        >>> from mindspore.common.initializer import initializer, Dirac
        >>> tensor1 = initializer(Dirac(groups=2), [6, 4, 3, 3], mindspore.float32)
        >>> tensor2 = initializer("dirac", [6, 4, 3, 3], mindspore.float32)
    """

    def __init__(self, groups=1):
        super(Dirac, self).__init__()
        self.groups = groups

    def _initialize(self, arr):
        dimension = len(arr.shape)
        data = np.zeros(arr.shape)
        if dimension not in [3, 4, 5]:
            raise ValueError("For Dirac initializer, only support "
                             "to initialize tensor with dimension of 3, 4 or 5, but got {}.".format(dimension))

        shapes = arr.shape
        if shapes[0] % self.groups != 0:
            raise ValueError("For Dirac initializer, the first dimension of"
                             "the initialized tensor must be divisible by groups, "
                             "but got first dimension{}, groups{}.".format(shapes[0], self.groups))

        out_channel_per_group = shapes[0] // self.groups
        min_dim = min(out_channel_per_group, shapes[1])

        for group in range(self.groups):
            for dim in range(min_dim):
                if dimension == 3:
                    data[group * out_channel_per_group + dim, dim, shapes[2]//2] = 1
                elif dimension == 4:
                    data[group * out_channel_per_group + dim, dim, shapes[2] // 2, shapes[3] // 2] = 1
                else:
                    data[group * out_channel_per_group + dim, dim, shapes[2] // 2, shapes[3] // 2, shapes[4] // 2] = 1
        _assignment(arr, data)


@_register()
class Orthogonal(Initializer):
    r"""
    Generates a (semi) orthogonal matrix array in order to initialize a tensor.
    The dimension of input tensor must have at least 2 dimensions.
    If the dimension is greater than 2, the trailing dimensions will be flattened.

    Args:
         gain (float): An optional scaling factor. Default: 1.

    Raises:
        ValueError: If the dimension of input tensor is less than 2.

    Examples:
        >>> import mindspore
        >>> from mindspore.common.initializer import initializer, Orthogonal
        >>> tensor1 = initializer(Orthogonal(gain=2.), [2, 3, 4], mindspore.float32)
        >>> tensor2 = initializer('orthogonal', [2, 3, 4], mindspore.float32)
    """

    def __init__(self, gain=1.):
        super(Orthogonal, self).__init__(gain=gain)
        self.gain = gain

    def _initialize(self, arr):
        if len(arr.shape) < 2:
            raise ValueError('For Orthogonal initializer, the dimension of the initialized tensor should'
                             ' be no less than 2, but got {}.'.format(len(arr.shape)))
        rows = arr.shape[0]

        cols = np.prod(arr.shape) // rows
        data = _init_random_normal(0, 1, (rows, cols))

        if rows < cols:
            data = data.T

        q, r = np.linalg.qr(data)
        d = np.diag(r)
        ph = np.sign(d)
        q *= ph

        if rows < cols:
            q = q.T
        q = q * self.gain
        _assignment(arr, q.reshape(arr.shape))


@_register()
class VarianceScaling(Initializer):
    r"""
    Generates an random array with scaling in order to initialize a tensor.
    When `distribution` is 'truncated_normal' or 'untruncated_normal', the value will be sampled from truncated or
    untruncated normal distribution with a mean of 0 and a scaled standard deviation
    :math:`stddev = \sqrt{\frac{scale}{n}}`. :math:`n` will be the number of input units if `mode` is 'fan_in',
    the number of output units if `mode` is 'fan_out', the average of 'fan_in' and 'fan_out' if `mode` is 'fan_avg'.
    When `distribution` is 'uniform', the value will be sampled from a uniform distribution within the limit of
    :math:`[-\sqrt{\frac{3*scale}{n}}, \sqrt{\frac{3*scale}{n}}]`.

    Args:
        scale (float): The scaling factor. Default: 1.0.
        mode (str): Should be 'fan_in', 'fan_out' or 'fan_avg'. Default: 'fan_in'.
        distribution(str): The type of distribution chose to sample values. It should be
            'uniform', 'truncated_normal' or 'untruncated_normal'. Default: 'truncated_normal'.

    Raises:
        ValueError: If `scale` is not greater than 0.
        ValueError: If `mode` is not 'fan_in', 'fan_out' or 'fan_avg'.
        ValueError: If `distribution` is not 'uniform', 'truncated_normal' or 'untruncated_normal'.

    Examples:
        >>> import mindspore
        >>> from mindspore.common.initializer import initializer, VarianceScaling
        >>> tensor1 = initializer(VarianceScaling(scale=1.0, mode='fan_out',
        ...                                       distribution='untruncated_normal'), [2, 3], mindspore.float32)
        >>> tensor2 = initializer('varianceScaling', [2, 3], mindspore.float32)
    """

    def __init__(self, scale=1.0, mode='fan_in', distribution='truncated_normal'):
        super(VarianceScaling, self).__init__(scale=scale, mode=mode, distribution=distribution)
        if scale <= 0.:
            raise ValueError("For VarianceScaling initializer, "
                             "the argument 'scale' must be greater than 0, but got {}.".format(scale))

        if mode not in ['fan_in', 'fan_out', 'fan_avg']:
            raise ValueError("For VarianceScaling initializer, the argument 'mode' must be fan_in, "
                             "fan_out or fan_avg, but got {}.".format(mode))

        if distribution not in ['uniform', 'truncated_normal', 'untruncated_normal']:
            raise ValueError("For VarianceScaling initializer, the argument 'distribution' must be uniform, "
                             "truncated_norm or untruncated_norm, but got {}.".format(distribution))

        self.scale = scale
        self.mode = mode
        self.distribution = distribution

    def _initialize(self, arr):
        scale = self.scale
        fan_in, fan_out = _calculate_fan_in_and_fan_out(arr.shape)
        if self.mode == 'fan_in':
            scale /= max(1., fan_in)
        elif self.mode == 'fan_out':
            scale /= max(1., fan_out)
        else:
            scale /= max(1., (fan_in + fan_out) / 2.)

        if self.distribution == 'truncated_norm':
            stddev = np.sqrt(scale) / 0.87962566103423978
            data = _init_truncated_normal(-2, 2, 0, stddev, arr.shape)
        elif self.distribution == 'untruncated_normal':
            stddev = np.sqrt(scale)
            data = _init_random_normal(0, stddev, arr.shape)
        else:
            limit = np.sqrt(3.0 * scale)
            data = _init_random_uniform(-limit, limit, arr.shape)
        _assignment(arr, data)


@_register()
class Uniform(Initializer):
    r"""
    Generates an array with values sampled from Uniform distribution :math:`{U}(-\text{scale}, \text{scale})` in order
    to initialize a tensor.

    Args:
        scale (float): The bound of the Uniform distribution. Default: 0.07.


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
        tmp = _init_random_uniform(-self.scale, self.scale, arr.shape)
        _assignment(arr, tmp)


@_register()
class Normal(Initializer):
    r"""
    Generates an array with values sampled from Normal distribution :math:`{N}(\text{sigma}, \text{mean})` in order to
    initialize a tensor.

    .. math::
        f(x) =  \frac{1} {\sqrt{2*π} * sigma}exp(-\frac{(x - mean)^2} {2*{sigma}^2})

    Args:
        sigma (float): The standard deviation of Normal distribution. Default: 0.01.
        mean (float): The mean of Normal distribution. Default: 0.0.

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
        data = _init_random_normal(self.mean, self.sigma, arr.shape)
        _assignment(arr, data)


@_register()
class TruncatedNormal(Initializer):
    r"""
    Generates an array with values sampled from Truncated Normal distribution in order to initialize a tensor.

    Args:
        sigma (float): The standard deviation of Truncated Normal distribution. Default: 0.01.


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
        tmp = _init_truncated_normal(-2, 2, 0, self.sigma, arr.shape)
        _assignment(arr, tmp)


def initializer(init, shape=None, dtype=mstype.float32):
    """
    Create and initialize a tensor.

    Args:
        init (Union[Tensor, str, Initializer, numbers.Number]): Initialize value.

            - `str`: The `init` should be the alias of the class inheriting from `Initializer` and the corresponding
              class will be called in practice. The value of 'init' can be "normal", "ones" or "zeros", etc.

            - `Initializer`: The `init` should be the class inheriting from `Initializer` to initialize tensor.

            - `numbers.Number`: The `Constant` will be called to initialize tensor.

            - `Tensor`: The tensor will be called to initialize tensor.

        shape (Union[tuple, list, int]): The shape of the initialized tensor. Default: None.
        dtype (:class:`mindspore.dtype`): The type of data in initialized tensor. Default: mindspore.float32.

    Returns:
        Tensor, return is Tensor object.

    Raises:
        TypeError: The type of the argument 'init' is not correct.
        ValueError: The shape of the tensor which is passed through 'init' is not the same as that passed by 'shape'.


    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindspore.common.initializer import initializer, One
        >>> data = Tensor(np.zeros([1, 2, 3]), mindspore.float32)
        >>> tensor1 = initializer(data, [1, 2, 3], mindspore.float32)
        >>> tensor2 = initializer('ones', [1, 2, 3], mindspore.float32)
        >>> tensor3 = initializer(One(), [1, 2, 3], mindspore.float32)
        >>> tensor4 = initializer(0, [1, 2, 3], mindspore.float32)
    """
    if not isinstance(init, (Tensor, numbers.Number, str, Initializer)):
        raise TypeError("For 'initializer', the type of the 'init' argument should be 'Tensor', 'number', 'string' "
                        "or 'initializer', but got {}.".format(type(init)))

    if isinstance(init, Tensor):
        init_shape = init.shape
        shape = shape if isinstance(shape, (tuple, list)) else [shape]
        if shape is not None and init_shape != tuple(shape):
            raise ValueError("For 'initializer', the shape of the 'init' argument should be same as "
                             "the argument 'shape', but got the "
                             "'init' shape {} and the 'shape' {}.".format(list(init.shape), shape))
        return init

    if isinstance(shape, list):
        shape = tuple(shape)
    elif isinstance(shape, numbers.Number):
        shape = (shape,)

    for value in shape if shape is not None else ():
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"For 'initializer', the argument 'shape' is invalid, the value of 'shape' "
                             f"must be positive integer, "
                             f"but got {shape}")

    if isinstance(init, str):
        class_name = _INITIALIZER_ALIAS.get(init.lower())
        if class_name is None:
            raise ValueError(f"For 'initializer', the class corresponding to '{init}' was not found.")
        init = class_name()
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
    'XavierNormal',
    'One',
    'Zero',
    'Constant',
    'Identity',
    'Sparse',
    'Dirac',
    'Orthogonal',
    'VarianceScaling']
