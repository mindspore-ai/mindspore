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
"""LightCNN network"""
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.initializer import Normal
from mindspore.common.initializer import XavierUniform


class Mfm(nn.Cell):
    """Mfn module"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad_mode='valid', mode=1):
        super(Mfm, self).__init__()
        self.out_channels = out_channels
        if mode == 1:
            self.filter = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=kernel_size, stride=stride,
                                    pad_mode=pad_mode, weight_init=XavierUniform(), has_bias=True)
        elif mode == 0:
            self.filter = nn.Dense(in_channels, 2 * out_channels, weight_init=Normal(0.02))
        self.maximum = P.Maximum()
        self.split = P.Split(axis=1, output_num=2)

    def construct(self, x):
        """Mfn construct"""
        x = self.filter(x)
        out = self.split(x)
        out = self.maximum(out[0], out[1])
        return out


class Group(nn.Cell):
    """group module"""

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Group, self).__init__()
        self.conv_a = Mfm(in_channels, in_channels, 1, 1, pad_mode='same')
        self.conv = Mfm(in_channels, out_channels, kernel_size, stride, pad_mode='same')

    def construct(self, x):
        """Group construct"""
        x = self.conv_a(x)
        x = self.conv(x)
        return x


class Resblock(nn.Cell):
    """res block"""

    def __init__(self, in_channels, out_channels):
        super(Resblock, self).__init__()
        self.conv1 = Mfm(in_channels, out_channels, kernel_size=3, stride=1, pad_mode='same')
        self.conv2 = Mfm(out_channels, out_channels, kernel_size=3, stride=1, pad_mode='same')
        self.add = P.Add()

    def construct(self, x):
        """Resblock construct"""
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        out = self.add(x, res)
        return out


def clip_gradient(dx):
    """clip gradient"""
    ret = dx
    if ret > 5.0:
        ret = 5.0
    if ret < 0.05:
        ret = 0.05
    return ret


class Network9Layers(nn.Cell):
    """9layer LightCNN network for train"""

    def __init__(self, num_classes):
        super(Network9Layers, self).__init__()
        self.features = nn.SequentialCell([
            Mfm(1, 48, kernel_size=5, stride=1, pad_mode='same'),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Group(48, 96, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Group(96, 192, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Group(192, 128, 3, 1),
            Group(128, 128, 3, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])
        self.fc1 = Mfm(8 * 8 * 128, 256, mode=0)
        self.fc2 = nn.Dense(256, num_classes, weight_init=Normal(0.02))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(keep_prob=0.5)

    def construct(self, x):
        """network construct"""
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        out = self.fc2(x)
        return out


class Network9Layers4Test(Network9Layers):
    """9layer LightCNN network for test"""

    def construct(self, x):
        """network construct"""
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        out = self.fc2(x)
        return out, x


def lightCNN_9Layers(num_classes):
    """get 9layers model for train"""
    model = Network9Layers(num_classes)
    return model


def lightCNN_9Layers4Test(num_classes):
    """get 9layers model for test"""
    model = Network9Layers4Test(num_classes)
    return model
