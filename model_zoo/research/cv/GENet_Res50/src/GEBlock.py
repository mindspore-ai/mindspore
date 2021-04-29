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
""" GEBlock."""
import mindspore.nn as nn
import mindspore as ms
from mindspore.ops import operations as P

class GEBlock(nn.Cell):
    """
    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        stride (int): Stride size for the first convolutional layer. Default: 1.
        spatial(int) : output_size of block
        extra_params(bool)  : Whether to use DW Conv to down-sample
        mlp(bool)      : Whether to combine SENet (using 1*1 conv)
    Returns:
        Tensor, output tensor.
    Examples:
        >>> GEBlock(3, 128, 2, 56, True, True)
    """

    def __init__(self, in_channel, out_channel, stride, spatial, extra_params, mlp):
        super().__init__()
        expansion = 4

        self.mlp = mlp
        self.extra_params = extra_params

        # middle channel num
        channel = out_channel // expansion
        self.conv1 = nn.Conv2dBnAct(in_channel, channel, kernel_size=1, stride=1,
                                    has_bn=True, pad_mode="same", activation='relu')

        self.conv2 = nn.Conv2dBnAct(channel, channel, kernel_size=3, stride=stride,
                                    has_bn=True, pad_mode="same", activation='relu')

        self.conv3 = nn.Conv2dBnAct(channel, out_channel, kernel_size=1, stride=1, pad_mode='same',
                                    has_bn=True)

        # whether down-sample identity
        self.down_sample = False
        if stride != 1 or in_channel != out_channel:
            self.down_sample = True

        self.down_layer = None
        if self.down_sample:
            self.down_layer = nn.Conv2dBnAct(in_channel, out_channel,
                                             kernel_size=1, stride=stride,
                                             pad_mode='same', has_bn=True)

        if extra_params:
            cellList = []
            # implementation of DW Conv has some bug while kernel_size is too big, so down sample
            if spatial >= 56:
                cellList.extend([nn.Conv2d(in_channels=out_channel,
                                           out_channels=out_channel,
                                           kernel_size=3,
                                           stride=2,
                                           pad_mode="same"),
                                 nn.BatchNorm2d(out_channel)])
                spatial //= 2

            cellList.extend([nn.Conv2d(in_channels=out_channel,
                                       out_channels=out_channel,
                                       kernel_size=spatial,
                                       group=out_channel,
                                       stride=1,
                                       padding=0,
                                       pad_mode="pad"),
                             nn.BatchNorm2d(out_channel)])

            self.downop = nn.SequentialCell(cellList)

        else:

            self.downop = P.ReduceMean(keep_dims=True)

        if mlp:
            mlpLayer = []
            mlpLayer.append(nn.Conv2d(in_channels=out_channel,
                                      out_channels=out_channel//16,
                                      kernel_size=1))
            mlpLayer.append(nn.ReLU())
            mlpLayer.append(nn.Conv2d(in_channels=out_channel//16,
                                      out_channels=out_channel,
                                      kernel_size=1))
            self.mlpLayer = nn.SequentialCell(mlpLayer)

        self.sigmoid = nn.Sigmoid()
        self.add = ms.ops.Add()
        self.relu = nn.ReLU()
        self.mul = ms.ops.Mul()


    def construct(self, x):
        """
        Args:
            x : input Tensor.
        """
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.down_sample:
            identity = self.down_layer(identity)

        if self.extra_params:
            out_ge = self.downop(out)
        else:
            out_ge = self.downop(out, (2, 3))

        if self.mlp:
            out_ge = self.mlpLayer(out_ge)
        out_ge = self.sigmoid(out_ge)
        out = self.mul(out, out_ge)
        out = self.add(out, identity)
        out = self.relu(out)

        return out
