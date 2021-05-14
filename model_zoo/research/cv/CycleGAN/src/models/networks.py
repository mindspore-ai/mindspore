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

"""Cycle GAN network."""

import mindspore.nn as nn
from mindspore.common import initializer as init


def init_weights(net, init_type='normal', init_gain=0.02):
    """
    Initialize network weights.
    Parameters:
        net (Cell): Network to be initialized
        init_type (str): The name of an initialization method: normal | xavier.
        init_gain (float): Gain factor for normal and xavier.
    """
    for _, cell in net.cells_and_names():
        if isinstance(cell, (nn.Conv2d, nn.Conv2dTranspose)):
            if init_type == 'normal':
                cell.weight.set_data(init.initializer(init.Normal(init_gain), cell.weight.shape))
            elif init_type == 'xavier':
                cell.weight.set_data(init.initializer(init.XavierUniform(init_gain), cell.weight.shape))
            elif init_type == 'constant':
                cell.weight.set_data(init.initializer(0.001, cell.weight.shape))
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif isinstance(cell, nn.BatchNorm2d):
            cell.gamma.set_data(init.initializer('ones', cell.gamma.shape))
            cell.beta.set_data(init.initializer('zeros', cell.beta.shape))


class ConvNormReLU(nn.Cell):
    """
    Convolution fused with BatchNorm/InstanceNorm and ReLU/LackyReLU block definition.
    Args:
        in_planes (int): Input channel.
        out_planes (int): Output channel.
        kernel_size (int): Input kernel size. Default: 4.
        stride (int): Stride size for the first convolutional layer. Default: 2.
        alpha (float): Slope of LackyReLU. Default: 0.2.
        norm_mode (str): Specifies norm method. The optional values are "batch", "instance".
        pad_mode (str): Specifies padding mode. The optional values are "CONSTANT", "REFLECT", "SYMMETRIC".
            Default: "CONSTANT".
        use_relu (bool): Use relu or not. Default: True.
        padding (int): Pad size, if it is None, it will calculate by kernel_size. Default: None.
    Returns:
        Tensor, output tensor.
    """
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=4,
                 stride=2,
                 alpha=0.2,
                 norm_mode='batch',
                 pad_mode='CONSTANT',
                 use_relu=True,
                 padding=None):
        super(ConvNormReLU, self).__init__()
        norm = nn.BatchNorm2d(out_planes)
        if norm_mode == 'instance':
            # Use BatchNorm2d with batchsize=1, affine=False, training=True instead of InstanceNorm2d
            norm = nn.BatchNorm2d(out_planes, affine=False)
        has_bias = (norm_mode == 'instance')
        if padding is None:
            padding = (kernel_size - 1) // 2
        if pad_mode == 'CONSTANT':
            conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, pad_mode='pad',
                             has_bias=has_bias, padding=padding)
            layers = [conv, norm]
        else:
            paddings = ((0, 0), (0, 0), (padding, padding), (padding, padding))
            pad = nn.Pad(paddings=paddings, mode=pad_mode)
            conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, pad_mode='pad', has_bias=has_bias)
            layers = [pad, conv, norm]
        if use_relu:
            relu = nn.ReLU()
            if alpha > 0:
                relu = nn.LeakyReLU(alpha)
            layers.append(relu)
        self.features = nn.SequentialCell(layers)

    def construct(self, x):
        output = self.features(x)
        return output


class ConvTransposeNormReLU(nn.Cell):
    """
    ConvTranspose2d fused with BatchNorm/InstanceNorm and ReLU/LackyReLU block definition.
    Args:
        in_planes (int): Input channel.
        out_planes (int): Output channel.
        kernel_size (int): Input kernel size. Default: 4.
        stride (int): Stride size for the first convolutional layer. Default: 2.
        alpha (float): Slope of LackyReLU. Default: 0.2.
        norm_mode (str): Specifies norm method. The optional values are "batch", "instance".
        pad_mode (str): Specifies padding mode. The optional values are "CONSTANT", "REFLECT", "SYMMETRIC".
                        Default: "CONSTANT".
        use_relu (bool): use relu or not. Default: True.
        padding (int): pad size, if it is None, it will calculate by kernel_size. Default: None.
    Returns:
        Tensor, output tensor.
    """
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=4,
                 stride=2,
                 alpha=0.2,
                 norm_mode='batch',
                 pad_mode='CONSTANT',
                 use_relu=True,
                 padding=None):
        super(ConvTransposeNormReLU, self).__init__()
        conv = nn.Conv2dTranspose(in_planes, out_planes, kernel_size, stride=stride, pad_mode='same')
        norm = nn.BatchNorm2d(out_planes)
        if norm_mode == 'instance':
            # Use BatchNorm2d with batchsize=1, affine=False, training=True instead of InstanceNorm2d
            norm = nn.BatchNorm2d(out_planes, affine=False)
        has_bias = (norm_mode == 'instance')
        if padding is None:
            padding = (kernel_size - 1) // 2
        if pad_mode == 'CONSTANT':
            conv = nn.Conv2dTranspose(in_planes, out_planes, kernel_size, stride, pad_mode='same', has_bias=has_bias)
            layers = [conv, norm]
        else:
            paddings = ((0, 0), (0, 0), (padding, padding), (padding, padding))
            pad = nn.Pad(paddings=paddings, mode=pad_mode)
            conv = nn.Conv2dTranspose(in_planes, out_planes, kernel_size, stride, pad_mode='pad', has_bias=has_bias)
            layers = [pad, conv, norm]
        if use_relu:
            relu = nn.ReLU()
            if alpha > 0:
                relu = nn.LeakyReLU(alpha)
            layers.append(relu)
        self.features = nn.SequentialCell(layers)

    def construct(self, x):
        output = self.features(x)
        return output
