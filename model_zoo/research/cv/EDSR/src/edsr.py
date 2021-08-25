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
"""EDSR"""
import numpy as np
from mindspore import Parameter
from mindspore import nn, ops
from mindspore.common.initializer import TruncatedNormal


class RgbNormal(nn.Cell):
    """
    "MeanShift" in EDSR paper pytorch-code:
    https://github.com/sanghyun-son/EDSR-PyTorch/blob/master/src/model/common.py

    it is not unreasonable in the case below
    if std != 1 and sign = -1: y = x * rgb_std - rgb_range * rgb_mean
    if std != 1 and sign =  1: y = x * rgb_std + rgb_range * rgb_mean
    they are not inverse operation for each other!

    so use "RgbNormal" instead, it runs as below:
    if inverse = False: y = (x / rgb_range - mean) / std
    if inverse = True : x = (y * std + mean) * rgb_range
    """
    def __init__(self, rgb_range, rgb_mean, rgb_std, inverse=False):
        super(RgbNormal, self).__init__()
        self.rgb_range = rgb_range
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        self.inverse = inverse
        std = np.array(self.rgb_std, dtype=np.float32)
        mean = np.array(self.rgb_mean, dtype=np.float32)
        if not inverse:
            # y: (x / rgb_range - mean) / std <=> x * (1.0 / rgb_range / std) + (-mean) / std
            weight = (1.0 / self.rgb_range / std).reshape((1, -1, 1, 1))
            bias = (-mean / std).reshape((1, -1, 1, 1))
        else:
            # x: (y * std + mean) * rgb_range <=> y * (std * rgb_range) + mean * rgb_range
            weight = (self.rgb_range * std).reshape((1, -1, 1, 1))
            bias = (mean * rgb_range).reshape((1, -1, 1, 1))
        self.weight = Parameter(weight, requires_grad=False)
        self.bias = Parameter(bias, requires_grad=False)

    def construct(self, x):
        return x * self.weight + self.bias

    def extend_repr(self):
        s = 'rgb_range={}, rgb_mean={}, rgb_std={}, inverse = {}' \
            .format(
                self.rgb_range,
                self.rgb_mean,
                self.rgb_std,
                self.inverse,
            )
        return s


def make_conv2d(in_channels, out_channels, kernel_size, has_bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        pad_mode="same", has_bias=has_bias, weight_init=TruncatedNormal(0.02))


class ResBlock(nn.Cell):
    """
    Resnet Block
    """
    def __init__(
            self, in_channels, out_channels, kernel_size=1, has_bias=True, res_scale=1):
        super(ResBlock, self).__init__()
        self.conv1 = make_conv2d(in_channels, in_channels, kernel_size, has_bias)
        self.relu = nn.ReLU()
        self.conv2 = make_conv2d(in_channels, out_channels, kernel_size, has_bias)
        self.res_scale = res_scale

    def construct(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        res = res * self.res_scale
        x = x + res
        return x


class PixelShuffle(nn.Cell):
    """
    PixelShuffle using ops.DepthToSpace
    """
    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor
        self.upper = ops.DepthToSpace(self.upscale_factor)

    def construct(self, x):
        return self.upper(x)

    def extend_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)


def UpsamplerBlockList(upscale_factor, n_feats, has_bias=True):
    """
    make Upsampler Block List
    """
    if upscale_factor == 1:
        return []
    allow_sub_upscale_factor = [2, 3, None]
    for sub in allow_sub_upscale_factor:
        if sub is None:
            raise NotImplementedError(
                f"Only support \"scales\" that can be divisibled by {allow_sub_upscale_factor[:-1]}")
        if upscale_factor % sub == 0:
            break
    sub_block_list = [
        make_conv2d(n_feats, sub*sub*n_feats, 3, has_bias),
        PixelShuffle(sub),
    ]
    return sub_block_list + UpsamplerBlockList(upscale_factor // sub, n_feats, has_bias)


class Upsampler(nn.Cell):

    def __init__(self, scale, n_feats, has_bias=True):
        super(Upsampler, self).__init__()
        up = UpsamplerBlockList(scale, n_feats, has_bias)
        self.up = nn.SequentialCell(*up)

    def construct(self, x):
        x = self.up(x)
        return x


class EDSR(nn.Cell):
    """
    EDSR network
    """
    def __init__(self, scale, n_feats, kernel_size, n_resblocks,
                 n_colors=3,
                 res_scale=0.1,
                 rgb_range=255,
                 rgb_mean=(0.0, 0.0, 0.0),
                 rgb_std=(1.0, 1.0, 1.0)):
        super(EDSR, self).__init__()

        self.norm = RgbNormal(rgb_range, rgb_mean, rgb_std, inverse=False)
        self.de_norm = RgbNormal(rgb_range, rgb_mean, rgb_std, inverse=True)

        m_head = [make_conv2d(n_colors, n_feats, kernel_size)]

        m_body = [
            ResBlock(n_feats, n_feats, kernel_size, res_scale=res_scale)
            for _ in range(n_resblocks)
        ]
        m_body.append(make_conv2d(n_feats, n_feats, kernel_size))

        m_tail = [
            Upsampler(scale, n_feats),
            make_conv2d(n_feats, n_colors, kernel_size)
        ]

        self.head = nn.SequentialCell(m_head)
        self.body = nn.SequentialCell(m_body)
        self.tail = nn.SequentialCell(m_tail)

    def construct(self, x):
        x = self.norm(x)
        x = self.head(x)
        x = x + self.body(x)
        x = self.tail(x)
        x = self.de_norm(x)
        return x

    def load_pre_trained_param_dict(self, new_param_dict, strict=True):
        """
        load pre_trained param dict from edsr_x2
        """
        own_param = self.parameters_dict()
        for name, new_param in new_param_dict.items():
            if len(name) >= 4 and name[:4] == "net.":
                name = name[4:]
            if name in own_param:
                if isinstance(new_param, Parameter):
                    param = own_param[name]
                    if tuple(param.data.shape) == tuple(new_param.data.shape):
                        param.set_data(type(param.data)(new_param.data))
                    elif name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_param[name].shape, new_param.shape))
                elif strict:
                    if name.find('tail') == -1:
                        raise KeyError('unexpected key "{}" in parameters_dict()'
                                       .format(name))
