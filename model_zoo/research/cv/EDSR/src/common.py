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
"""common"""
import math
import numpy as np
import mindspore
import mindspore.nn as nn


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        pad_mode='pad',
        padding=(kernel_size//2), has_bias=bias)


class MeanShift(mindspore.nn.Conv2d):
    """MeanShift"""
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1, dtype=mindspore.float32):

        std = mindspore.Tensor(rgb_std, dtype)
        weight = mindspore.Tensor(np.eye(3), dtype).reshape(
            3, 3, 1, 1) / std.reshape(3, 1, 1, 1)
        bias = sign * rgb_range * mindspore.Tensor(rgb_mean, dtype) / std

        super(MeanShift, self).__init__(3, 3, kernel_size=1,
                                        has_bias=True, weight_init=weight, bias_init=bias)

        for p in self.get_parameters():
            p.requires_grad = False


class ResBlock(nn.Cell):
    """ResBlock"""
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, act=nn.ReLU(), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if i == 0:
                m.append(act)

        self.body = nn.SequentialCell(m)
        self.res_scale = res_scale
        self.mul = mindspore.ops.Mul()

    def construct(self, x):
        res = self.body(x)
        res = self.mul(res, self.res_scale)
        res += x

        return res


class PixelShuffle(nn.Cell):
    """PixelShuffle"""
    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.DepthToSpace = mindspore.ops.DepthToSpace(upscale_factor)

    def construct(self, x):
        return self.DepthToSpace(x)


def Upsampler(conv, scale, n_feats, bias=True):
    """Upsampler"""
    m = []

    if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
        for _ in range(int(math.log(scale, 2))):
            m.append(conv(n_feats, 4 * n_feats, 3, bias))
            m.append(PixelShuffle(2))
    elif scale == 3:
        m.append(conv(n_feats, 9 * n_feats, 3, bias))
        m.append(PixelShuffle(3))
    else:
        raise NotImplementedError

    return m
