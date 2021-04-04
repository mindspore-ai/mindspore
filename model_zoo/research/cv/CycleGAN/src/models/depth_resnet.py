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

"""ResNet Generator."""

import mindspore.nn as nn
from .networks import ConvNormReLU, ConvTransposeNormReLU


class ResidualBlock(nn.Cell):
    """
    ResNet residual block definition.
    We construct a conv block with build_conv_block function,
    and implement skip connections in <forward> function..
    Args:
        dim (int): Input and output channel.
        norm_mode (str): Specifies norm method. The optional values are "batch", "instance".
        dropout (bool): Use dropout or not. Default: False.
        pad_mode (str): Specifies padding mode. The optional values are "CONSTANT", "REFLECT", "SYMMETRIC".
            Default: "CONSTANT".
    Returns:
        Tensor, output tensor.
    """
    def __init__(self, dim, norm_mode='batch', dropout=False, pad_mode="CONSTANT"):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvNormReLU(dim, dim, 3, 1, 0.2, norm_mode, pad_mode)
        self.conv2 = ConvNormReLU(dim, dim, 3, 1, 0.2, norm_mode, pad_mode)
        self.dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(0.5)

    def construct(self, x):
        out = self.conv1(x)
        if self.dropout:
            out = self.dropout(out)
        out = self.conv2(out)
        return x + out


class DepthResNetGenerator(nn.Cell):
    """
    ResNet Generator of GAN.
    Args:
        in_planes (int): Input channel.
        ngf (int): Output channel.
        n_layers (int): The number of ConvNormReLU blocks.
        alpha (float): LeakyRelu slope. Default: 0.2.
        norm_mode (str): Specifies norm method. The optional values are "batch", "instance".
        dropout (bool): Use dropout or not. Default: False.
        pad_mode (str): Specifies padding mode. The optional values are "CONSTANT", "REFLECT", "SYMMETRIC".
            Default: "CONSTANT".
    Returns:
        Tensor, output tensor.
    """
    def __init__(self, in_planes=3, ngf=64, n_layers=9, alpha=0.2, norm_mode='batch', dropout=False,
                 pad_mode="CONSTANT"):
        super(DepthResNetGenerator, self).__init__()
        conv_in1 = nn.Conv2d(in_planes, ngf, kernel_size=3, stride=1, has_bias=True)
        conv_in2 = ConvNormReLU(ngf, ngf, 7, 1, alpha, norm_mode, pad_mode=pad_mode)
        self.conv_in = nn.SequentialCell([conv_in1, conv_in2])
        down_1 = ConvNormReLU(ngf, ngf * 2, 3, 2, alpha, norm_mode)
        Res1 = ResidualBlock(ngf * 2, norm_mode, dropout=dropout, pad_mode=pad_mode)
        self.down_1 = nn.SequentialCell([down_1, Res1])
        down_2 = ConvNormReLU(ngf * 2, ngf * 3, 3, 2, alpha, norm_mode)
        Res2 = ResidualBlock(ngf * 3, norm_mode, dropout=dropout, pad_mode=pad_mode)
        self.down_2 = nn.SequentialCell([down_2, Res2])
        self.down_3 = ConvNormReLU(ngf * 3, ngf * 4, 3, 2, alpha, norm_mode)
        layers = [ResidualBlock(ngf * 4, norm_mode, dropout=dropout, pad_mode=pad_mode)] * (n_layers-5)
        self.residuals = nn.SequentialCell(layers)
        up_3 = ConvTransposeNormReLU(ngf * 4, ngf * 3, 3, 2, alpha, norm_mode)
        Res3 = ResidualBlock(ngf * 3, norm_mode, dropout=dropout, pad_mode=pad_mode)
        self.up_3 = nn.SequentialCell([up_3, Res3])
        up_2 = ConvTransposeNormReLU(ngf * 3, ngf * 2, 3, 2, alpha, norm_mode)
        Res4 = ResidualBlock(ngf * 2, norm_mode, dropout=dropout, pad_mode=pad_mode)
        self.up_2 = nn.SequentialCell([up_2, Res4])
        up_1 = ConvTransposeNormReLU(ngf * 2, ngf, 3, 2, alpha, norm_mode)
        Res5 = ResidualBlock(ngf, norm_mode, dropout=dropout, pad_mode=pad_mode)
        self.up_1 = nn.SequentialCell([up_1, Res5])
        tanh = nn.Tanh()
        if pad_mode == "CONSTANT":
            conv_out1 = nn.Conv2d(ngf, 3, kernel_size=7, stride=1, has_bias=True, pad_mode='pad', padding=3)
            conv_out2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, has_bias=True)
            self.conv_out = nn.SequentialCell([conv_out1, tanh, conv_out2, tanh])
        else:
            pad = nn.Pad(paddings=((0, 0), (0, 0), (3, 3), (3, 3)), mode=pad_mode)
            conv = nn.Conv2d(ngf, 3, kernel_size=7, stride=1, pad_mode='pad')
            self.conv_out = nn.SequentialCell([pad, conv, tanh])

    def construct(self, x):
        """ construct network """
        x = self.conv_in(x)
        x = self.down_1(x)
        x = self.down_2(x)
        x = self.down_3(x)
        x = self.residuals(x)
        x = self.up_3(x)
        x = self.up_2(x)
        x = self.up_1(x)
        output = self.conv_out(x)
        return output
     