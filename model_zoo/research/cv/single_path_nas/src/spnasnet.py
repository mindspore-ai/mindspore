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
"""
    Single-Path NASNet for ImageNet-1K, implemented in Mindspore.
    Original paper: 'Single-Path NAS: Designing Hardware-Efficient ConvNets in less than 4 Hours,'
    https://arxiv.org/abs/1904.02877.
"""

import mindspore.nn as nn
import mindspore.ops as ops

from src.utils import conv1x1_block, conv3x3_block, dwconv3x3_block, dwconv5x5_block


class SPNASUnit(nn.Cell):
    """
    Single-Path NASNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the second convolution layer.
    use_kernel3 : bool
        Whether to use 3x3 (instead of 5x5) kernel.
    exp_factor : int
        Expansion factor for each unit.
    use_skip : bool, default True
        Whether to use skip connection.
    activation : str, default 'relu'
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 use_kernel3,
                 exp_factor,
                 use_skip=True,
                 activation="relu"):
        super(SPNASUnit, self).__init__()

        self.residual = (in_channels == out_channels) and (stride == 1) and use_skip
        self.use_exp_conv = exp_factor > 1
        mid_channels = exp_factor * in_channels

        if self.use_exp_conv:
            self.exp_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                activation=activation)
        if use_kernel3:
            self.conv1 = dwconv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                stride=stride,
                activation=activation)
        else:
            self.conv1 = dwconv5x5_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                stride=stride,
                activation=activation)
        self.conv2 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            activation=None)
        if self.residual:
            self.add = ops.Add()

    def construct(self, x):
        """
        Args:
            x: Tensor of shape :math:`(N, in_channels, W_{in}, H_{in})

        Returns:
            y: Tensor of shape :math:`(N, out_channels, W_{in}, H_{in})
        """

        identity = x
        if self.use_exp_conv:
            y = self.exp_conv(x)
            y = self.conv1(y)
            y = self.conv2(y)
        else:
            y = self.conv1(x)
            y = self.conv2(y)
        if self.residual:
            y = self.add(y, identity)
        return y


class SPNASInitBlock(nn.Cell):
    """
    Single-Path NASNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of middle channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels):
        super(SPNASInitBlock, self).__init__()
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            stride=2)
        self.conv2 = SPNASUnit(
            in_channels=mid_channels,
            out_channels=out_channels,
            stride=1,
            use_kernel3=True,
            exp_factor=1,
            use_skip=False)

    def construct(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SPNASFinalBlock(nn.Cell):
    """
    Single-Path NASNet specific final block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of middle channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels):
        super(SPNASFinalBlock, self).__init__()
        self.conv1 = SPNASUnit(
            in_channels=in_channels,
            out_channels=mid_channels,
            stride=1,
            use_kernel3=True,
            exp_factor=6,
            use_skip=False)
        self.conv2 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels)

    def construct(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SPNASNet(nn.Cell):
    """
    Single-Path NASNet model from 'Single-Path NAS: Designing Hardware-Efficient ConvNets in less than 4 Hours,'
    https://arxiv.org/abs/1904.02877.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : list of 2 int
        Number of output channels for the initial unit.
    final_block_channels : list of 2 int
        Number of output channels for the final block of the feature extractor.
    kernels3 : list of list of int/bool
        Using 3x3 (instead of 5x5) kernel for each unit.
    exp_factors : list of list of int
        Expansion factor for each unit.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 final_block_channels,
                 kernels3,
                 exp_factors,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(SPNASNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.SequentialCell()
        self.features.append(SPNASInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels[1],
            mid_channels=init_block_channels[0]))
        in_channels = init_block_channels[1]
        for i, channels_per_stage in enumerate(channels):
            stage = nn.SequentialCell()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if ((j == 0) and (i != 3)) or ((j == len(channels_per_stage) // 2) and (i == 3)) else 1
                use_kernel3 = kernels3[i][j] == 1
                exp_factor = exp_factors[i][j]
                stage.append(SPNASUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    use_kernel3=use_kernel3,
                    exp_factor=exp_factor))
                in_channels = out_channels
            self.features.append(stage)
        self.features.append(SPNASFinalBlock(
            in_channels=in_channels,
            out_channels=final_block_channels[1],
            mid_channels=final_block_channels[0]))
        in_channels = final_block_channels[1]
        self.features.append(nn.AvgPool2d(
            kernel_size=7,
            stride=1))

        self.output = nn.Dense(
            in_channels=in_channels,
            out_channels=num_classes,
            weight_init='HeUniform')
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.output(x)
        return x


def get_spnasnet(**kwargs):
    """
    Create Single-Path NASNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.mindspore/models'
        Location for keeping the model parameters.
    """
    init_block_channels = [32, 16]
    final_block_channels = [320, 1280]
    channels = [[24, 24, 24], [40, 40, 40, 40], [80, 80, 80, 80], [96, 96, 96, 96, 192, 192, 192, 192]]
    kernels3 = [[1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0]]
    exp_factors = [[3, 3, 3], [6, 3, 3, 3], [6, 3, 3, 3], [6, 3, 3, 3, 6, 6, 6, 6]]

    net = SPNASNet(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        kernels3=kernels3,
        exp_factors=exp_factors,
        **kwargs)

    return net


def spnasnet(**kwargs):
    """
    Single-Path NASNet model from 'Single-Path NAS: Designing Hardware-Efficient ConvNets in less than 4 Hours,'
    https://arxiv.org/abs/1904.02877.
    """

    return get_spnasnet(**kwargs)
