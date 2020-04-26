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
"""weight initial"""
import math
import numpy as np
from mindspore.common import initializer as init
import mindspore.nn as nn
from mindspore import Tensor

 
def calculate_gain(nonlinearity, param=None):
    r"""Return the recommended gain value for the given nonlinearity function.
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
        nonlinearity: the non-linear function (`nn.functional` name)
        param: optional parameter for the non-linear function
 
    """
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity)) 

def _calculate_correct_fan(array, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))
 
    fan_in, fan_out = _calculate_fan_in_and_fan_out(array)
    return fan_in if mode == 'fan_in' else fan_out 
 

def kaiming_uniform_(array, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where
 
    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}
 
    Also known as He initialization.
 
    Args:
        array: an n-dimensional `tensor`
        a: the negative slope of the rectifier used after this layer (only
        used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).
    """
 
    fan = _calculate_correct_fan(array, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return np.random.uniform(-bound, bound, array.shape)
 

def kaiming_normal_(array, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    normal distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std}^2)` where
 
    .. math::
        \text{std} = \frac{\text{gain}}{\sqrt{\text{fan\_mode}}}

    Also known as He initialization.
 
    Args:
        array: an n-dimensional `tensor`
        a: the negative slope of the rectifier used after this layer (only
        used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).
    """
    fan = _calculate_correct_fan(array, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return np.random.normal(0, std, array.shape)
 
def _calculate_fan_in_and_fan_out(array):
    dimensions = len(array.shape)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for array with fewer than 2 dimensions")
 
    num_input_fmaps = array.shape[1]
    num_output_fmaps = array.shape[0]
    receptive_field_size = 1
    if dimensions > 2:
        receptive_field_size = array[0][0].size
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
 
    return fan_in, fan_out
 
class KaimingUniform(init.Initializer):
    def __init__(self, a=0, mode='fan_in', nonlinearity='leaky_relu'):
        super(KaimingUniform, self).__init__()
        self.a = a
        self.mode = mode
        self.nonlinearity = nonlinearity
 
    def _initialize(self, arr):
        tmp = kaiming_uniform_(arr, self.a, self.mode, self.nonlinearity)
        init._assignment(arr, tmp) 

class KaimingNormal(init.Initializer):
    def __init__(self, a=0, mode='fan_in', nonlinearity='leaky_relu'):
        super(KaimingNormal, self).__init__()
        self.a = a
        self.mode = mode
        self.nonlinearity = nonlinearity
 
    def _initialize(self, arr):
        tmp = kaiming_normal_(arr, self.a, self.mode, self.nonlinearity)
        init._assignment(arr, tmp)

def default_recurisive_init(custom_cell):
    for name, cell in custom_cell.cells_and_names():
        if isinstance(cell, nn.Conv2d):
            cell.weight.default_input = init.initializer(KaimingUniform(a=math.sqrt(5)), cell.weight.default_input.shape(), cell.weight.default_input.dtype())
            if cell.bias is not None:
                fan_in, _ = _calculate_fan_in_and_fan_out(cell.weight.default_input.asnumpy())
                bound = 1 / math.sqrt(fan_in)
                cell.bias.default_input = Tensor(np.random.uniform(-bound, bound, cell.bias.default_input.shape()), cell.bias.default_input.dtype())
        elif isinstance(cell, nn.Dense):
            cell.weight.default_input = init.initializer(KaimingUniform(a=math.sqrt(5)), cell.weight.default_input.shape(), cell.weight.default_input.dtype())
            if cell.bias is not None:
                fan_in, _ = _calculate_fan_in_and_fan_out(cell.weight.default_input.asnumpy())
                bound = 1 / math.sqrt(fan_in)
                cell.bias.default_input = Tensor(np.random.uniform(-bound, bound, cell.bias.default_input.shape()), cell.bias.default_input.dtype())
        elif isinstance(cell, nn.BatchNorm2d) or isinstance(cell, nn.BatchNorm1d):
            pass
