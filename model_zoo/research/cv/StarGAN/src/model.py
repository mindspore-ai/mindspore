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
"""Define Generator and Discriminator for StarGAN"""
import math
import numpy as np

import mindspore.nn as nn
import mindspore.ops as P
from mindspore import set_seed, Tensor
from mindspore.common import initializer as init


set_seed(1)
np.random.seed(1)


class ResidualBlock(nn.Cell):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.SequentialCell(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, pad_mode='pad', has_bias=False),
            nn.GroupNorm(num_groups=dim_out, num_channels=dim_out),
            nn.ReLU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, pad_mode='pad', has_bias=False),
            nn.GroupNorm(num_groups=dim_out, num_channels=dim_out)
        )

    def construct(self, x):
        return x + self.main(x)


class Generator(nn.Cell):
    """Generator network."""

    def __init__(self, conv_dim=64, c_dim=4, repeat_num=6):
        super(Generator, self).__init__()

        layers = []
        layers.append((nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1,
                                 padding=3, pad_mode='pad', has_bias=False)))
        layers.append(nn.GroupNorm(num_groups=conv_dim, num_channels=conv_dim))
        layers.append(nn.ReLU())

        # Down-sampling layers.
        curr_dim = conv_dim
        for _ in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2,
                                    padding=1, pad_mode='pad', has_bias=False))
            layers.append(nn.GroupNorm(num_groups=curr_dim*2, num_channels=curr_dim*2))
            layers.append(nn.ReLU())
            curr_dim = curr_dim*2

        # Bottleneck layers.
        for _ in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for _ in range(2):
            layers.append(nn.Conv2dTranspose(curr_dim, int(curr_dim/2), kernel_size=4, stride=2,
                                             padding=1, pad_mode='pad', has_bias=False))
            layers.append(nn.GroupNorm(num_groups=int(curr_dim/2), num_channels=int(curr_dim/2)))
            layers.append(nn.ReLU())
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, pad_mode='pad', has_bias=False))
        layers.append(nn.Tanh())
        self.main = nn.SequentialCell(*layers)

    def construct(self, x, c):
        reshape = P.Reshape()
        c = reshape(c, (c.shape[0], c.shape[1], 1, 1))
        c = P.functional.reshape(c, (c.shape[0], c.shape[1], 1, 1))
        tile = P.Tile()
        c = tile(c, (1, 1, x.shape[2], x.shape[3]))
        op = P.Concat(1)
        x = op((x, c))
        return self.main(x)


class ResidualBlock_2(nn.Cell):
    """Residual Block with instance normalization."""

    def __init__(self, weight, dim_in, dim_out):
        super(ResidualBlock_2, self).__init__()
        self.main = nn.SequentialCell(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1,
                      pad_mode='pad', has_bias=False, weight_init=Tensor(weight[0])),
            nn.GroupNorm(num_groups=dim_out, num_channels=dim_out),
            nn.ReLU(),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1,
                      pad_mode='pad', has_bias=False, weight_init=Tensor(weight[3])),
            nn.GroupNorm(num_groups=dim_out, num_channels=dim_out)
        )

    def construct(self, x):
        return x + self.main(x)


class Discriminator(nn.Cell):
    """Discriminator network with PatchGAN."""

    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1, has_bias=True,
                                pad_mode='pad', bias_init=init.Uniform(1 / math.sqrt(3))))
        layers.append(nn.LeakyReLU(alpha=0.01))

        curr_dim = conv_dim
        for _ in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, has_bias=True,
                                    pad_mode='pad', bias_init=init.Uniform(1 / math.sqrt(curr_dim))))
            layers.append(nn.LeakyReLU(alpha=0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.SequentialCell(*layers)
        # Patch GAN输出结果
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, pad_mode='pad', has_bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, has_bias=False, pad_mode='valid')

    def construct(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        reshape = P.Reshape()
        out_cls = reshape(out_cls, (out_cls.shape[0], out_cls.shape[1]))
        return out_src, out_cls
