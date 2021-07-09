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
"""Network Component"""

from mindspore import nn

def add_normalization_1d(layers, fn, n_out, mode='test'):
    if fn == "none":
        pass
    elif fn == "batchnorm":
        layers.append(nn.BatchNorm1d(n_out, use_batch_statistics=(mode == 'train')))
    elif fn == "instancenorm":
        layers.append(nn.GroupNorm(n_out, n_out, affine=True))
    else:
        raise Exception('Unsupported normalization: ' + str(fn))
    return layers


def add_normalization_2d(layers, fn, n_out, mode='test'):
    if fn == 'none':
        pass
    elif fn == 'batchnorm':
        layers.append(nn.BatchNorm2d(n_out, use_batch_statistics=(mode == 'train')))
    elif fn == "instancenorm":
        layers.append(nn.GroupNorm(n_out, n_out, affine=True))
    else:
        raise Exception('Unsupported normalization: ' + str(fn))
    return layers


def add_activation(layers, fn):
    """Add Activation"""
    if fn == "none":
        pass
    elif fn == "relu":
        layers.append(nn.ReLU())
    elif fn == "lrelu":
        layers.append(nn.LeakyReLU(alpha=0.01))
    elif fn == "sigmoid":
        layers.append(nn.Sigmoid())
    elif fn == "tanh":
        layers.append(nn.Tanh())
    else:
        raise Exception('Unsupported activation function: ' + str(fn))
    return layers


class LinearBlock(nn.Cell):
    """Linear Block"""
    def __init__(self, n_in, n_out, norm_fn="none", acti_fn="none", mode='test'):
        super().__init__()
        layers = [nn.Dense(n_in, n_out, has_bias=(norm_fn == 'none'))]
        layers = add_normalization_1d(layers, norm_fn, n_out, mode)
        layers = add_activation(layers, acti_fn)
        self.layers = nn.SequentialCell(layers)

    def construct(self, x):
        return self.layers(x)


class Conv2dBlock(nn.Cell):
    """Convolution Block"""
    def __init__(self, n_in, n_out, kernel_size, stride=1, padding=0,
                 norm_fn=None, acti_fn=None, mode='test'):
        super().__init__()
        layers = [nn.Conv2d(n_in, n_out, kernel_size, stride=stride, padding=padding, pad_mode='pad',
                            has_bias=(norm_fn == 'none'))]
        layers = add_normalization_2d(layers, norm_fn, n_out, mode)
        layers = add_activation(layers, acti_fn)
        self.layers = nn.SequentialCell(layers)

    def construct(self, x):
        return self.layers(x)


class ConvTranspose2dBlock(nn.Cell):
    """Transpose Convolution Block"""
    def __init__(self, n_in, n_out, kernel_size, stride=1, padding=0,
                 norm_fn=None, acti_fn=None, mode='test'):
        super().__init__()
        layers = [nn.Conv2dTranspose(n_in, n_out, kernel_size, stride=stride, padding=padding, pad_mode='pad',
                                     has_bias=(norm_fn == 'none'))]
        layers = add_normalization_2d(layers, norm_fn, n_out, mode)
        layers = add_activation(layers, acti_fn)
        self.layers = nn.SequentialCell(layers)

    def construct(self, x):
        return self.layers(x)
