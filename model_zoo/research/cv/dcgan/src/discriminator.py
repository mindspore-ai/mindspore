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
"""dcgan discriminator"""
from mindspore import nn

from src.cell import Normal
from src.config import dcgan_imagenet_cfg as cfg


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, pad_mode="pad"):
    weight_init = Normal(mean=0, sigma=0.02)
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight_init, has_bias=False, pad_mode=pad_mode)


def bm(num_features):
    gamma_init = Normal(mean=1, sigma=0.02)
    return nn.BatchNorm2d(num_features=num_features, gamma_init=gamma_init)


class Discriminator(nn.Cell):
    """
    DCGAN Discriminator
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = nn.SequentialCell()
        # input is 3 x 32 x 32
        self.discriminator.append(conv(cfg.channel_size, cfg.feature_size * 2, 4, 2, 1))
        self.discriminator.append(nn.LeakyReLU(0.2))
        # state size. 128 x 16 x 16
        self.discriminator.append(conv(cfg.feature_size * 2, cfg.feature_size * 4, 4, 2, 1))
        self.discriminator.append(bm(cfg.feature_size * 4))
        self.discriminator.append(nn.LeakyReLU(0.2))
        # state size. 256 x 8 x 8
        self.discriminator.append(conv(cfg.feature_size * 4, cfg.feature_size * 8, 4, 2, 1))
        self.discriminator.append(bm(cfg.feature_size * 8))
        self.discriminator.append(nn.LeakyReLU(0.2))
        # state size. 512 x 4 x 4
        self.discriminator.append(conv(cfg.feature_size * 8, 1, 4, 1))
        self.discriminator.append(nn.Sigmoid())

    def construct(self, x):
        return self.discriminator(x)
