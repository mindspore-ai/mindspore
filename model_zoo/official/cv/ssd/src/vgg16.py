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

"""VGG16 backbone for SSD"""

from mindspore import nn
from .config_ssd_vgg16 import config

pretrain_vgg_bn = config.pretrain_vgg_bn
ssd_vgg_bn = config.ssd_vgg_bn


def _get_key_mapper():
    vgg_key_num = [1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]
    size = len(vgg_key_num)

    pretrain_vgg_bn_false = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
    pretrain_vgg_bn_true = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]
    ssd_vgg_bn_false = [0, 2, 0, 2, 0, 2, 4, 0, 2, 4, 0, 2, 4]
    ssd_vgg_bn_true = [0, 3, 0, 3, 0, 3, 6, 0, 3, 6, 0, 3, 6]

    pretrain_vgg_keys = pretrain_vgg_bn_true if pretrain_vgg_bn else pretrain_vgg_bn_false
    ssd_vgg_keys = ssd_vgg_bn_true if ssd_vgg_bn else ssd_vgg_bn_false

    pretrain_vgg_keys = ['layers.' + str(pretrain_vgg_keys[i]) for i in range(size)]
    ssd_vgg_keys = ['b' + str(vgg_key_num[i]) + '.' + str(ssd_vgg_keys[i]) for i in range(size)]

    return {pretrain_vgg_keys[i]: ssd_vgg_keys[i] for i in range(size)}


ssd_vgg_key_mapper = _get_key_mapper()


def _make_layer(channels):
    in_channels = channels[0]
    layers = []
    for out_channels in channels[1:]:
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3))
        if ssd_vgg_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        in_channels = out_channels
    return nn.SequentialCell(layers)


class VGG16(nn.Cell):
    def __init__(self):
        super(VGG16, self).__init__()
        self.b1 = _make_layer([3, 64, 64])
        self.b2 = _make_layer([64, 128, 128])
        self.b3 = _make_layer([128, 256, 256, 256])
        self.b4 = _make_layer([256, 512, 512, 512])
        self.b5 = _make_layer([512, 512, 512, 512])

        self.m1 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='SAME')
        self.m2 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='SAME')
        self.m3 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='SAME')
        self.m4 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='SAME')
        self.m5 = nn.MaxPool2d(kernel_size=3, stride=1, pad_mode='SAME')

    def construct(self, x):
        # block1
        x = self.b1(x)
        x = self.m1(x)

        # block2
        x = self.b2(x)
        x = self.m2(x)

        # block3
        x = self.b3(x)
        x = self.m3(x)

        # block4
        x = self.b4(x)
        block4 = x
        x = self.m4(x)

        # block5
        x = self.b5(x)
        x = self.m5(x)

        return block4, x


def vgg16():
    return VGG16()
