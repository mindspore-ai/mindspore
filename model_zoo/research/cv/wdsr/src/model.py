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
"""main model of wdsr"""

import mindspore
import mindspore.nn as nn
import numpy as np


class MeanShift(mindspore.nn.Conv2d):
    """add or sub means of input data"""
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1, dtype=mindspore.float32):
        std = mindspore.Tensor(rgb_std, dtype)
        weight = mindspore.Tensor(np.eye(3), dtype).reshape(3, 3, 1, 1) / std.reshape(3, 1, 1, 1)
        bias = sign * rgb_range * mindspore.Tensor(rgb_mean, dtype) / std
        super(MeanShift, self).__init__(3, 3, kernel_size=1, has_bias=True, weight_init=weight, bias_init=bias)
        for p in self.get_parameters():
            p.requires_grad = False


class Block(nn.Cell):
    """residual block"""
    def __init__(self):
        super(Block, self).__init__()
        act = nn.ReLU()
        self.res_scale = 1
        body = []
        expand = 6
        linear = 0.8
        body.append(nn.Conv2d(64, 64 * expand, 1, padding=(1 // 2), pad_mode='pad', has_bias=True))
        body.append(act)
        body.append(nn.Conv2d(64 * expand, int(64 * linear), 1, padding=1 // 2, pad_mode='pad', has_bias=True))
        body.append(nn.Conv2d(int(64 * linear), 64, 3, padding=3 // 2, pad_mode='pad', has_bias=True))
        self.body = nn.SequentialCell(body)

    def construct(self, x):
        res = self.body(x)
        res += x
        return res


class PixelShuffle(nn.Cell):
    """perform pixel shuffle"""
    def __init__(self, upscale_factor):
        super().__init__()
        self.DepthToSpace = mindspore.ops.DepthToSpace(upscale_factor)

    def construct(self, x):
        return self.DepthToSpace(x)


class WDSR(nn.Cell):
    """main structure of wdsr"""
    def __init__(self):
        super(WDSR, self).__init__()
        scale = 2
        n_resblocks = 8
        n_feats = 64
        self.sub_mean = MeanShift(255)
        self.add_mean = MeanShift(255, sign=1)
        # define head module
        head = []
        head.append(
            nn.Conv2d(
                3, n_feats, 3,
                pad_mode='pad',
                padding=(3 // 2), has_bias=True))

        # define body module
        body = []
        for _ in range(n_resblocks):
            body.append(Block())
        # define tail module
        tail = []
        out_feats = scale * scale * 3
        tail.append(
            nn.Conv2d(n_feats, out_feats, 3, padding=3 // 2, pad_mode='pad', has_bias=True))
        self.depth_to_space = mindspore.ops.DepthToSpace(2)
        skip = []
        skip.append(
            nn.Conv2d(3, out_feats, 5, padding=5 // 2, pad_mode='pad', has_bias=True))
        self.head = nn.SequentialCell(head)
        self.body = nn.SequentialCell(body)
        self.tail = nn.SequentialCell(tail)
        self.skip = nn.SequentialCell(skip)

    def construct(self, x):
        x = self.sub_mean(x) / 127.5
        s = self.skip(x)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x += s
        x = self.depth_to_space(x)
        x = self.add_mean(x * 127.5)
        return x
