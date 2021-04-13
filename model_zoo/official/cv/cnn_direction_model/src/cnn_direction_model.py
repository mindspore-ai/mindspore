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
"""CNN direction model."""
import math

import mindspore.nn as nn
from mindspore.common.initializer import Uniform
from mindspore.ops import operations as P


class NetAddN(nn.Cell):
    """
    Computes addition of all input tensors element-wise.
    """

    def __init__(self):
        super(NetAddN, self).__init__()
        self.addN = P.AddN()

    def construct(self, *z):
        return self.addN(z)


class Conv(nn.Cell):
    """
    A convolution layer

    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        kernel (tuple): Size of the kernel. Default: (3, 3).
        dilate (bool): If set to true a second convolution layer is added. Default: True.
        act (string): The activation function. Default: 'relu'.
        mp (int): Size of max pooling layer. Default: None.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> Conv(3, 64)
    """

    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel=(3, 3),
                 dilate=True,
                 act='relu',
                 mp=None):
        super(Conv, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel = kernel
        self.dilate = dilate
        self.act = act
        self.mp = mp

        self.conv1 = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=self.kernel, pad_mode="same",
                               weight_init='he_normal')

        self.batch_norm1 = nn.BatchNorm2d(self.out_channel, eps=1e-3, momentum=0.99,
                                          gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)

        if self.dilate:
            self.dilate_relu = P.ReLU()
            self.dilate_conv = nn.Conv2d(self.out_channel, self.out_channel, kernel_size=self.kernel,
                                         dilation=(2, 2), pad_mode='same', weight_init='he_normal')

            self.dilate_batch_norm = nn.BatchNorm2d(self.out_channel, eps=1e-3, momentum=0.99,
                                                    gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)

            self.dilate_add = NetAddN()

        if self.act == 'relu':
            self.act_layer = P.ReLU()

        if self.mp is not None:
            self.mp_layer = nn.MaxPool2d(kernel_size=self.mp, stride=self.mp, pad_mode='valid')

    def construct(self, x):

        out = self.conv1(x)
        out = self.batch_norm1(out)
        out1 = out

        if self.dilate:
            out = self.dilate_relu(out)
            out = self.dilate_conv(out)
            out = self.dilate_batch_norm(out)
            out = self.dilate_add(out1, out)

        if self.act == 'relu':
            out = self.act_layer(out)

        if self.mp is not None:
            out = self.mp_layer(out)

        return out


class Block(nn.Cell):
    """
    A Block of convolution operations.

    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
    Returns:
        Tensor, output tensor.

    Examples:
        >>> Block(3, 64)
    """

    def __init__(self,
                 in_channel,
                 out_channel):
        super(Block, self).__init__()
        self.conv1 = Conv(in_channel, out_channel, act='relu')
        self.conv2 = Conv(out_channel, out_channel, act=None)
        self.add = NetAddN()
        self.relu = P.ReLU()

    def construct(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        out = self.add(x, y)
        out = self.relu(out)

        return out


class ResidualBlock(nn.Cell):
    """
    A residual block.

    Args:
        block (Block) : The building block.
        num_blocks (int): Number of blocks.
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        mp (int) : Size of the max pooling layer. Default: 2.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ResidualBlock(Block, 1, 3, 64)
    """

    def __init__(self,
                 block,
                 num_blocks,
                 in_channel,
                 out_channel,
                 mp=2):

        super(ResidualBlock, self).__init__()
        self.num_blocks = num_blocks
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.mp = mp
        self.conv1 = Conv(self.in_channel, self.out_channel, kernel=(3, 3), dilate=False)

        layers = []
        for _ in range(self.num_blocks):
            res_block = block(out_channel, out_channel)
            layers.append(res_block)

        self.layer = nn.SequentialCell(layers)

        if mp is not None:
            self.max_pool = nn.MaxPool2d(kernel_size=mp, stride=mp, pad_mode='valid')

    def construct(self, x):
        out = self.conv1(x)
        out = self.layer(out)
        if self.mp is not None:
            out = self.max_pool(out)

        return out


class CNNDirectionModel(nn.Cell):
    """
    CNN direction model.

    Args:
        in_channels (list): List of the dimesnions of the input channels. The first element is the input dimension
    of the first Conv layer, and the rest of the elements are the input dimensions of the residual blocks,
    in order.
        out_channels (list): List of the dimesnions of the output channels. The first element is the ourpur dimension
    of the first Conv layer, and the rest of the elements are the output dimensions of the residual blocks, in order.
        dense_layers (list): Dimensions of the dense layers, inorder.
        image_size (list): Size of the input images.
        num_classes (int): Number of classes. Default: 2 for binary classification.

    Returns: Tensor, output tensor.

    Examples:
        >>> CNNDirectionModel([3, 64, 48, 48, 64], [64, 48, 48, 64, 64], [256, 64], [64, 512] )
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 dense_layers,
                 image_size,
                 num_classes=2
                 ):
        super(CNNDirectionModel, self).__init__()
        self.num_classes = num_classes
        self.image_h = image_size[0]
        self.image_w = image_size[1]
        self.conv1 = Conv(in_channels[0], out_channels[0], kernel=(7, 7), dilate=False, mp=2)
        self.residual_block1 = ResidualBlock(Block, 1, in_channels[1], out_channels[1])
        self.residual_block2 = ResidualBlock(Block, 1, in_channels[2], out_channels[2])
        self.residual_block3 = ResidualBlock(Block, 2, in_channels[3], out_channels[3])
        self.residual_block4 = ResidualBlock(Block, 1, in_channels[4], out_channels[4])

        # 5 previous layers have mp=2. Height and width of the image would become 1/32.
        self.avg_pool = nn.AvgPool2d(kernel_size=(int(self.image_h / 32), int(self.image_w / 32)))

        scale = math.sqrt(6 / (out_channels[-1] + dense_layers[0]))
        self.dense1 = nn.Dense(out_channels[-1], dense_layers[0], weight_init=Uniform(scale=scale), activation='relu')

        scale = math.sqrt(6 / (dense_layers[0] + dense_layers[1]))
        self.dense2 = nn.Dense(dense_layers[0], dense_layers[1], weight_init=Uniform(scale=scale), activation='relu')

        scale = math.sqrt(6 / (dense_layers[1] + num_classes))
        self.dense3 = nn.Dense(dense_layers[1], num_classes, weight_init=Uniform(scale=scale), activation='softmax')

    def construct(self, x):
        out = self.conv1(x)

        out = self.residual_block1(out)

        out = self.residual_block2(out)
        out = self.residual_block3(out)
        out = self.residual_block4(out)

        out = self.avg_pool(out)

        out = P.Reshape()(out, (out.shape[0], out.shape[1]))

        out = self.dense1(out)
        out = self.dense2(out)
        out = self.dense3(out)

        return out
