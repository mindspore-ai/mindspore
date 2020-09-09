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
"""
Initialize.
"""
import math
from functools import reduce
import numpy as np
import mindspore.nn as nn
from mindspore.common import initializer as init

def _calculate_gain(nonlinearity, param=None):
    r"""
    Return the recommended gain value for the given nonlinearity function.

    The values are as follows:
    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
    ================= ====================================================

    Args:
        nonlinearity: the non-linear function
        param: optional parameter for the non-linear function

    Examples:
        >>> gain = calculate_gain('leaky_relu', 0.2)  # leaky_relu with negative_slope=0.2
    """
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    if nonlinearity == 'tanh':
        return 5.0 / 3
    if nonlinearity == 'relu':
        return math.sqrt(2.0)
    if nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))

    raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))

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

def _select_fan(array, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_in_and_out(array)
    return fan_in if mode == 'fan_in' else fan_out

class KaimingInit(init.Initializer):
    r"""
    Base Class. Initialize the array with He kaiming algorithm.

    Args:
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function, recommended to use only with
            ``'relu'`` or ``'leaky_relu'`` (default).
    """
    def __init__(self, a=0, mode='fan_in', nonlinearity='leaky_relu'):
        super(KaimingInit, self).__init__()
        self.mode = mode
        self.gain = _calculate_gain(nonlinearity, a)
    def _initialize(self, arr):
        pass


class KaimingUniform(KaimingInit):
    r"""
    Initialize the array with He kaiming uniform algorithm. The resulting tensor will
    have values sampled from :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}

    Input:
        arr (Array): The array to be assigned.

    Returns:
        Array, assigned array.

    Examples:
        >>> w = np.empty(3, 5)
        >>> KaimingUniform(w, mode='fan_in', nonlinearity='relu')
    """

    def _initialize(self, arr):
        fan = _select_fan(arr, self.mode)
        bound = math.sqrt(3.0) * self.gain / math.sqrt(fan)
        data = np.random.uniform(-bound, bound, arr.shape)

        _assignment(arr, data)


class KaimingNormal(KaimingInit):
    r"""
    Initialize the array with He kaiming normal algorithm. The resulting tensor will
    have values sampled from :math:`\mathcal{N}(0, \text{std}^2)` where

    .. math::
        \text{std} = \frac{\text{gain}}{\sqrt{\text{fan\_mode}}}

    Input:
        arr (Array): The array to be assigned.

    Returns:
        Array, assigned array.

    Examples:
        >>> w = np.empty(3, 5)
        >>> KaimingNormal(w, mode='fan_out', nonlinearity='relu')
    """

    def _initialize(self, arr):
        fan = _select_fan(arr, self.mode)
        std = self.gain / math.sqrt(fan)
        data = np.random.normal(0, std, arr.shape)

        _assignment(arr, data)


def default_recurisive_init(custom_cell):
    """default_recurisive_init"""
    for _, cell in custom_cell.cells_and_names():
        if isinstance(cell, nn.Conv2d):
            cell.weight.set_data(init.initializer(KaimingUniform(a=math.sqrt(5)),
                                                  cell.weight.shape,
                                                  cell.weight.dtype))
            if cell.bias is not None:
                fan_in, _ = _calculate_in_and_out(cell.weight)
                bound = 1 / math.sqrt(fan_in)
                cell.bias.set_data(init.initializer(init.Uniform(bound),
                                                    cell.bias.shape,
                                                    cell.bias.dtype))
        elif isinstance(cell, nn.Dense):
            cell.weight.set_data(init.initializer(KaimingUniform(a=math.sqrt(5)),
                                                  cell.weight.shape,
                                                  cell.weight.dtype))
            if cell.bias is not None:
                fan_in, _ = _calculate_in_and_out(cell.weight)
                bound = 1 / math.sqrt(fan_in)
                cell.bias.set_data(init.initializer(init.Uniform(bound),
                                                    cell.bias.shape,
                                                    cell.bias.dtype))
        elif isinstance(cell, (nn.BatchNorm2d, nn.BatchNorm1d)):
            pass
