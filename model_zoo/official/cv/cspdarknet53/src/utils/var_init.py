# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Initialize."""
import os
import math
from functools import reduce
import numpy as np
import mindspore.nn as nn
from mindspore.common import initializer as init
from mindspore.train.serialization import load_checkpoint, load_param_into_net
import mindspore as ms
from mindspore import Tensor

def _calculate_gain(nonlinearity, param=None):
    """calculate_gain"""
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    res = 0
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
            raise ValueError("negative_slope {} not a valid number".format(param))
        res = math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
    return res

def _assignment(arr, num):
    """Assign the value of `num` to `arr`."""
    if arr.shape == ():
        arr = arr.reshape((1))
        arr[:] = num
        arr.reshape(())
    else:
        if isinstance(num, np.ndarray):
            arr[:] = num[:]
        else:
            arr[:] = num
    return arr

def _calculate_in_and_out(tensor):
    """_calculate_fan_in_and_fan_out"""
    dimensions = len(tensor.shape)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    fan_in = tensor.shape[1]
    fan_out = tensor.shape[0]

    if dimensions > 2:
        counter = reduce(lambda x, y: x * y, tensor.shape[2:])
        fan_in *= counter
        fan_out *= counter

    return fan_in, fan_out

def _select_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))
    fan_in, fan_out = _calculate_in_and_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


class KaimingInit(init.Initializer):
    """Base class. Initialize the array with HeKaiming init algorithm."""
    def __init__(self, a=0., mode='fan_in', nonlinearity='leaky_relu'):
        super(KaimingInit, self).__init__()
        self.mode = mode
        self.gain = _calculate_gain(nonlinearity, a)

    def _initialize(self, arr):
        raise NotImplementedError("Init algorithm not-implemented.")

class KaimingUniform(KaimingInit):
    """KaimingUniform init algorithm."""

    def _initialize(self, arr):
        fan = _select_fan(arr, self.mode)
        bound = math.sqrt(3.0) * self.gain / math.sqrt(fan)
        data = np.random.uniform(-bound, bound, arr.shape)

        _assignment(arr, data)


class KaimingNormal(KaimingInit):
    """KaimingNormal init algorithm."""

    def _initialize(self, arr):
        fan = _select_fan(arr, self.mode)
        std = self.gain / math.sqrt(fan)
        data = np.random.normal(0, std, arr.shape)

        _assignment(arr, data)


def default_recurisive_init(custom_cell):
    ms.common.set_seed(0)
    for _, cell in custom_cell.cells_and_names():
        if isinstance(cell, nn.Conv2d):
            cell.weight.set_data(init.initializer(KaimingUniform(a=math.sqrt(5.0)), cell.weight.data.shape,
                                                  cell.weight.data.dtype).to_tensor())
            if cell.bias is not None:
                fan_in, _ = _calculate_in_and_out(cell.weight.data.asnumpy())
                bound = 1 / math.sqrt(fan_in)
                cell.bias.set_data(Tensor(np.random.uniform(-bound, bound, cell.bias.data.shape), cell.bias.data.dtype))
        elif isinstance(cell, nn.Dense):
            cell.weight.set_data(init.initializer(KaimingUniform(a=math.sqrt(5)), cell.weight.data.shape,
                                                  cell.weight.data.dtype).to_tensor())
            if cell.bias is not None:
                fan_in, _ = _calculate_in_and_out(cell.weight.data.asnumpy())
                bound = 1 / math.sqrt(fan_in)
                cell.bias.set_data(Tensor(np.random.uniform(-bound, bound, cell.bias.data.shape), cell.bias.data.dtype))
        elif isinstance(cell, (nn.BatchNorm2d, nn.BatchNorm1d)):
            pass


def load_pretrain_model(ckpt_file, network, args):
    """load pretrain model."""
    if os.path.isfile(ckpt_file):
        param_dict = load_checkpoint(ckpt_file)
        param_dict_new = {}
        for k, v in param_dict.items():
            if k.startswith('moments.'):
                continue
            elif k.startswith('network.'):
                param_dict_new[k[8:]] = v
            else:
                param_dict_new[k] = v
        load_param_into_net(network, param_dict_new)
        args.logger.info("Load pretrained {:s} success".format(ckpt_file))
    else:
        args.logger.info("Do not load pretrained.")
