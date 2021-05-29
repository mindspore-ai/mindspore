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
"""Squeezenet."""
import mindspore.nn as nn
from mindspore.common import initializer as weight_init
from mindspore.ops import operations as P


class Fire(nn.Cell):
    """
    Fire network definition.
    """

    def __init__(self, inplanes, squeeze_planes, expand1x1_planes,
                 expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes,
                                 squeeze_planes,
                                 kernel_size=1,
                                 has_bias=True)
        self.squeeze_activation = nn.ReLU()
        self.expand1x1 = nn.Conv2d(squeeze_planes,
                                   expand1x1_planes,
                                   kernel_size=1,
                                   has_bias=True)
        self.expand1x1_activation = nn.ReLU()
        self.expand3x3 = nn.Conv2d(squeeze_planes,
                                   expand3x3_planes,
                                   kernel_size=3,
                                   pad_mode='same',
                                   has_bias=True)  # padding 0 or 1, padding=1, different from PyTorch version
        self.expand3x3_activation = nn.ReLU()
        self.concat = P.Concat(axis=1)

    def construct(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return self.concat((self.expand1x1_activation(self.expand1x1(x)),
                            self.expand3x3_activation(self.expand3x3(x))))


# Figure_2 Left
class SqueezeNet(nn.Cell):
    r"""SqueezeNet model architecture from the `"SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size"
    <https://arxiv.org/abs/1602.07360>`_ paper.

    Get SqueezeNet neural network.

    Args:
        num_classes (int): Class number.

    Returns:
        Cell, cell instance of SqueezeNet neural network.

    Examples:
        >>> net = SqueezeNet(10)
    """

    def __init__(self, num_classes=10):
        super(SqueezeNet, self).__init__()

        self.features = nn.SequentialCell([
            nn.Conv2d(3,
                      64,
                      kernel_size=3,
                      stride=2,
                      pad_mode='pad',
                      padding=1,
                      has_bias=True),  # In PyTorch version, padding=1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Fire(64, 16, 64, 64),  # inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes
            Fire(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
        ])

        # Final convolution is initialized differently from the rest
        self.final_conv = nn.Conv2d(512,
                                    num_classes,
                                    kernel_size=1,
                                    has_bias=True)
        self.dropout = nn.Dropout(keep_prob=0.5)
        self.relu = nn.ReLU()
        self.mean = P.ReduceMean(
            keep_dims=True)  # Equal to AvgPool(kernel_size=13, stride=3). It's better than AvgPool, because we don't know the size of representation exactly.
        self.flatten = nn.Flatten()  # Flattens a tensor without changing dimension of batch size on the 0-th axis.
        self.custom_init_weight()

    def custom_init_weight(self):
        """
        Init the weight of Conv2d in the net.
        """
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                if cell is self.final_conv:
                    cell.weight.set_data(
                        weight_init.initializer('normal', cell.weight.shape,
                                                cell.weight.dtype))
                else:
                    cell.weight.set_data(
                        weight_init.initializer('he_uniform',
                                                cell.weight.shape,
                                                cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(
                        weight_init.initializer('zeros', cell.bias.shape,
                                                cell.bias.dtype))

    def construct(self, x):
        x = self.features(x)
        x = self.dropout(x)
        x = self.final_conv(x)
        x = self.relu(x)
        x = self.mean(x, (2, 3))
        x = self.flatten(x)

        return x
