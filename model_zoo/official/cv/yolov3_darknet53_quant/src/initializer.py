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
"""Parameter init."""
import math
import numpy as np
from mindspore.common import initializer as init
from mindspore.common.initializer import Initializer as MeInitializer
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net


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

    Examples:
        >>> gain = nn.init.calculate_gain('leaky_relu', 0.2)  # leaky_relu with negative_slope=0.2
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
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))

    raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def _assignment(arr, num):
    """Assign the value of 'num' and 'arr'."""
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


def _calculate_correct_fan(array, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(array)
    return fan_in if mode == 'fan_in' else fan_out


def kaiming_uniform_(arr, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `Tensor`
        a: the negative slope of the rectifier used after this layer (only
        used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).

    Examples:
        >>> w = np.empty(3, 5)
        >>> nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
    """
    fan = _calculate_correct_fan(arr, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return np.random.uniform(-bound, bound, arr.shape)


def _calculate_fan_in_and_fan_out(arr):
    """Calculate fan in and fan out."""
    dimensions = len(arr.shape)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for array with fewer than 2 dimensions")

    num_input_fmaps = arr.shape[1]
    num_output_fmaps = arr.shape[0]
    receptive_field_size = 1
    if dimensions > 2:
        receptive_field_size = arr[0][0].size
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


class KaimingUniform(MeInitializer):
    """Kaiming uniform initializer."""
    def __init__(self, a=0, mode='fan_in', nonlinearity='leaky_relu'):
        super(KaimingUniform, self).__init__()
        self.a = a
        self.mode = mode
        self.nonlinearity = nonlinearity

    def _initialize(self, arr):
        tmp = kaiming_uniform_(arr, self.a, self.mode, self.nonlinearity)
        _assignment(arr, tmp)


def default_recurisive_init(custom_cell):
    """Initialize parameter."""
    for _, cell in custom_cell.cells_and_names():
        if isinstance(cell, nn.Conv2d):
            cell.weight.set_data(init.initializer(KaimingUniform(a=math.sqrt(5)),
                                                  cell.weight.data.shape,
                                                  cell.weight.data.dtype))
            if cell.bias is not None:
                fan_in, _ = _calculate_fan_in_and_fan_out(cell.weight.data.asnumpy())
                bound = 1 / math.sqrt(fan_in)
                cell.bias.set_data(Tensor(np.random.uniform(-bound, bound, cell.bias.data.shape),
                                          cell.bias.data.dtype))
        elif isinstance(cell, nn.Dense):
            cell.weight.set_data(init.initializer(KaimingUniform(a=math.sqrt(5)),
                                                  cell.weight.data.shape,
                                                  cell.weight.data.dtype))
            if cell.bias is not None:
                fan_in, _ = _calculate_fan_in_and_fan_out(cell.weight.data.asnumpy())
                bound = 1 / math.sqrt(fan_in)
                cell.bias.set_data(Tensor(np.random.uniform(-bound, bound, cell.bias.data.shape),
                                          cell.bias.data.dtype))
        elif isinstance(cell, (nn.BatchNorm2d, nn.BatchNorm1d)):
            pass

def load_yolov3_quant_params(args, network):
    """Load quant yolov3 darknet parameter from checkpoint."""
    if args.resume_yolov3:
        param_dict = load_checkpoint(args.resume_yolov3)
        param_dict_new = {}
        for key, values in param_dict.items():
            args.logger.info('ckpt param name = {}'.format(key))
            if key.startswith('moments.') or key.startswith('global_') or \
               key.startswith('learning_rate') or key.startswith('momentum'):
                continue
            elif key.startswith('yolo_network.'):
                key_new = key[13:]

                if key_new.endswith('1.beta'):
                    key_new = key_new.replace('1.beta', 'batchnorm.beta')

                if key_new.endswith('1.gamma'):
                    key_new = key_new.replace('1.gamma', 'batchnorm.gamma')

                if key_new.endswith('1.moving_mean'):
                    key_new = key_new.replace('1.moving_mean', 'batchnorm.moving_mean')

                if key_new.endswith('1.moving_variance'):
                    key_new = key_new.replace('1.moving_variance', 'batchnorm.moving_variance')

                if key_new.endswith('.weight'):
                    if key_new.endswith('0.weight'):
                        key_new = key_new.replace('0.weight', 'conv.weight')
                    else:
                        key_new = key_new.replace('.weight', '.conv.weight')

                if key_new.endswith('.bias'):
                    key_new = key_new.replace('.bias', '.conv.bias')
                param_dict_new[key_new] = values

                args.logger.info('in resume {}'.format(key_new))
            else:
                param_dict_new[key] = values
                args.logger.info('in resume {}'.format(key))

        args.logger.info('resume finished')
        for _, param in network.parameters_and_names():
            args.logger.info('network param name = {}'.format(param.name))
            if param.name not in param_dict_new:
                args.logger.info('not match param name = {}'.format(param.name))
        load_param_into_net(network, param_dict_new)
        args.logger.info('load_model {} success'.format(args.resume_yolov3))
