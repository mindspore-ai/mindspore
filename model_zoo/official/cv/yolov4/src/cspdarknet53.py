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
"""DarkNet model."""
import mindspore.nn as nn
from mindspore.ops import operations as P


class Mish(nn.Cell):
    """Mish activation method"""
    def __init__(self):
        super(Mish, self).__init__()
        self.mul = P.Mul()
        self.tanh = P.Tanh()
        self.softplus = P.Softplus()

    def construct(self, input_x):
        res1 = self.softplus(input_x)
        tanh = self.tanh(res1)
        output = self.mul(input_x, tanh)

        return output

def conv_block(in_channels,
               out_channels,
               kernel_size,
               stride,
               dilation=1):
    """Get a conv2d batchnorm and relu layer"""
    pad_mode = 'same'
    padding = 0

    return nn.SequentialCell(
        [nn.Conv2d(in_channels,
                   out_channels,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding,
                   dilation=dilation,
                   pad_mode=pad_mode),
         nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5),
         Mish()
         ]
    )


class ResidualBlock(nn.Cell):
    """
    DarkNet V1 residual block definition.

    Args:
        in_channels: Integer. Input channel.
        out_channels: Integer. Output channel.

    Returns:
        Tensor, output tensor.
    Examples:
        ResidualBlock(3, 208)
    """
    def __init__(self,
                 in_channels,
                 out_channels):

        super(ResidualBlock, self).__init__()
        out_chls = out_channels
        self.conv1 = conv_block(in_channels, out_chls, kernel_size=1, stride=1)
        self.conv2 = conv_block(out_chls, out_channels, kernel_size=3, stride=1)
        self.add = P.Add()

    def construct(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.add(out, identity)

        return out

class CspDarkNet53(nn.Cell):
    """
    DarkNet V1 network.

    Args:
        block: Cell. Block for network.
        layer_nums: List. Numbers of different layers.
        in_channels: Integer. Input channel.
        out_channels: Integer. Output channel.
        num_classes: Integer. Class number. Default:100.

    Returns:
        Tuple, tuple of output tensor,(f1,f2,f3,f4,f5).

    Examples:
        DarkNet(ResidualBlock)
    """
    def __init__(self,
                 block,
                 detect=False):
        super(CspDarkNet53, self).__init__()

        self.outchannel = 1024
        self.detect = detect
        self.concat = P.Concat(axis=1)
        self.add = P.Add()

        self.conv0 = conv_block(3, 32, kernel_size=3, stride=1)
        self.conv1 = conv_block(32, 64, kernel_size=3, stride=2)
        self.conv2 = conv_block(64, 64, kernel_size=1, stride=1)
        self.conv3 = conv_block(64, 32, kernel_size=1, stride=1)
        self.conv4 = conv_block(32, 64, kernel_size=3, stride=1)
        self.conv5 = conv_block(64, 64, kernel_size=1, stride=1)
        self.conv6 = conv_block(64, 64, kernel_size=1, stride=1)
        self.conv7 = conv_block(128, 64, kernel_size=1, stride=1)
        self.conv8 = conv_block(64, 128, kernel_size=3, stride=2)
        self.conv9 = conv_block(128, 64, kernel_size=1, stride=1)
        self.conv10 = conv_block(64, 64, kernel_size=1, stride=1)
        self.conv11 = conv_block(128, 64, kernel_size=1, stride=1)
        self.conv12 = conv_block(128, 128, kernel_size=1, stride=1)
        self.conv13 = conv_block(128, 256, kernel_size=3, stride=2)
        self.conv14 = conv_block(256, 128, kernel_size=1, stride=1)
        self.conv15 = conv_block(128, 128, kernel_size=1, stride=1)
        self.conv16 = conv_block(256, 128, kernel_size=1, stride=1)
        self.conv17 = conv_block(256, 256, kernel_size=1, stride=1)
        self.conv18 = conv_block(256, 512, kernel_size=3, stride=2)
        self.conv19 = conv_block(512, 256, kernel_size=1, stride=1)
        self.conv20 = conv_block(256, 256, kernel_size=1, stride=1)
        self.conv21 = conv_block(512, 256, kernel_size=1, stride=1)
        self.conv22 = conv_block(512, 512, kernel_size=1, stride=1)
        self.conv23 = conv_block(512, 1024, kernel_size=3, stride=2)
        self.conv24 = conv_block(1024, 512, kernel_size=1, stride=1)
        self.conv25 = conv_block(512, 512, kernel_size=1, stride=1)
        self.conv26 = conv_block(1024, 512, kernel_size=1, stride=1)
        self.conv27 = conv_block(1024, 1024, kernel_size=1, stride=1)

        self.layer2 = self._make_layer(block, 2, in_channel=64, out_channel=64)
        self.layer3 = self._make_layer(block, 8, in_channel=128, out_channel=128)
        self.layer4 = self._make_layer(block, 8, in_channel=256, out_channel=256)
        self.layer5 = self._make_layer(block, 4, in_channel=512, out_channel=512)

    def _make_layer(self, block, layer_num, in_channel, out_channel):
        """
        Make Layer for DarkNet.

        :param block: Cell. DarkNet block.
        :param layer_num: Integer. Layer number.
        :param in_channel: Integer. Input channel.
        :param out_channel: Integer. Output channel.
        :return: SequentialCell, the output layer.

        Examples:
            _make_layer(ConvBlock, 1, 128, 256)
        """
        layers = []
        darkblk = block(in_channel, out_channel)
        layers.append(darkblk)

        for _ in range(1, layer_num):
            darkblk = block(out_channel, out_channel)
            layers.append(darkblk)

        return nn.SequentialCell(layers)

    def construct(self, x):
        """construct method"""
        c1 = self.conv0(x)
        c2 = self.conv1(c1)  #route
        c3 = self.conv2(c2)
        c4 = self.conv3(c3)
        c5 = self.conv4(c4)
        c6 = self.add(c3, c5)
        c7 = self.conv5(c6)
        c8 = self.conv6(c2)
        c9 = self.concat((c7, c8))
        c10 = self.conv7(c9)
        c11 = self.conv8(c10)   #route
        c12 = self.conv9(c11)
        c13 = self.layer2(c12)
        c14 = self.conv10(c13)
        c15 = self.conv11(c11)
        c16 = self.concat((c14, c15))
        c17 = self.conv12(c16)
        c18 = self.conv13(c17)  #route
        c19 = self.conv14(c18)
        c20 = self.layer3(c19)
        c21 = self.conv15(c20)
        c22 = self.conv16(c18)
        c23 = self.concat((c21, c22))
        c24 = self.conv17(c23)  #output1
        c25 = self.conv18(c24)  #route
        c26 = self.conv19(c25)
        c27 = self.layer4(c26)
        c28 = self.conv20(c27)
        c29 = self.conv21(c25)
        c30 = self.concat((c28, c29))
        c31 = self.conv22(c30)  #output2
        c32 = self.conv23(c31)  #route
        c33 = self.conv24(c32)
        c34 = self.layer5(c33)
        c35 = self.conv25(c34)
        c36 = self.conv26(c32)
        c37 = self.concat((c35, c36))
        c38 = self.conv27(c37)  #output3

        if self.detect:
            return c24, c31, c38

        return c38

    def get_out_channels(self):
        return self.outchannel
