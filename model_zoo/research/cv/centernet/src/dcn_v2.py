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
"""
Deformable Convolution operator V2
"""

import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.dtype as mstype

from .utils import ClipByValue


class GetOffsetPosition(nn.Cell):
    """
    Get position index after deformable shift of each kernel element.

    Args:
        begin(int): The begging position index of convolutional kernel center.
        stride (int): The distance of kernel moving.

    Returns:
        Tensor, new position index of each kernel element.
    """
    def __init__(self, begin, stride):
        super(GetOffsetPosition, self).__init__()
        self.begin = begin
        self.stride = stride
        self.meshgrid = ops.Meshgrid()
        self.shape = ops.Shape()
        self.reshape = ops.Reshape()
        self.cat_a0 = ops.Concat(axis=0)
        self.cat_a1 = ops.Concat(axis=1)
        self.tile = ops.Tile()
        self.dtype = ops.DType()
        self.range = nn.Range(-self.begin, self.begin + 1)
        self.cast = ops.Cast()

    def construct(self, offset):
        """get target position"""
        offset_shape = self.shape(offset) # b * 2N * h * w
        N, h, w = offset_shape[1] // 2, offset_shape[2], offset_shape[3]
        # get p_n
        range_pn = self.range()
        p_n_x, p_n_y = self.meshgrid((range_pn, range_pn))
        # (2N, 1)
        p_n = self.cat_a0((self.reshape(p_n_x, (N, 1)), self.reshape(p_n_y, (N, 1))))
        p_n = self.reshape(p_n, (1, 2 * N, 1, 1))

        # get p_0
        range_h = nn.Range(self.begin, h*self.stride + 1, self.stride)()
        range_w = nn.Range(self.begin, w*self.stride + 1, self.stride)()
        p_0_x, p_0_y = self.meshgrid((range_h, range_w))
        p_0_x = self.reshape(p_0_x, (1, 1, h, w))
        p_0_x = self.tile(p_0_x, (1, N, 1, 1))
        p_0_y = self.reshape(p_0_y, (1, 1, h, w))
        p_0_y = self.tile(p_0_y, (1, N, 1, 1))
        p_0 = self.cat_a1((p_0_x, p_0_y))

        # get p
        dtype = self.dtype(offset)
        p = self.cast(p_0, dtype) + self.cast(p_n, dtype) + offset
        return p


class GetSurroundFeature(nn.Cell):
    """
    Get feature after deformable shift of each kernel element.

    Args: None

    Returns:
        Tensor, feature map after deformable shift.
    """
    def __init__(self):
        super(GetSurroundFeature, self).__init__()
        self.shape = ops.Shape()
        self.concat = ops.Concat(axis=1)
        self.reshape = ops.Reshape()
        self.half = ops.Split(axis=-1, output_num=2)
        self.tile = ops.Tile()
        self.gather_nd = ops.GatherNd()
        self.transpose = ops.Transpose()
        self.perm_list = (0, 2, 3, 1)
        self.order_list = (0, 3, 1, 2)
        self.expand_dims = ops.ExpandDims()

    def construct(self, x, q_h, q_w):
        """gather feature by specified index"""
        b, c, _, w_p = self.shape(x)
        _, h, w, N = self.shape(q_h)
        hwn = h * w * N
        # (b * hw * c)
        x = self.transpose(x, self.perm_list)
        x = self.reshape(x, (b, -1, c))

        # (b * hwN)
        q = q_h * w_p + q_w
        q = self.reshape(q, (-1, 1))
        ind_b = nn.Range(0, b, 1)()
        ind_b = self.reshape(ind_b, (-1, 1))
        ind_b = self.tile(ind_b, (1, hwn))
        ind_b = self.reshape(ind_b, (-1, 1))
        index = self.concat((ind_b, q))
        # (b, hwn, 2)
        index = self.reshape(index, (b, hwn, -1))
        # (b, hwn, c)
        x_offset = self.gather_nd(x, index)
        # (b, c, h, w, N)
        x_offset = self.reshape(x_offset, (b, h * w, N, c))
        x_offset = self.transpose(x_offset, self.order_list)
        x_offset = self.reshape(x_offset, (b, c, h, w, N))

        return x_offset


class RegenerateFeatureMap(nn.Cell):
    """
    Get rescaled feature map which was enlarged by ks**2 time.

    Args:
        ks: Kernel size of convolution.
    Returns:
        Tensor, rescaled feature map.
    """
    def __init__(self, ks):
        super(RegenerateFeatureMap, self).__init__()
        self.ks = ks
        self.shape = ops.Shape()
        self.reshape = ops.Reshape()
        self.split = ops.Split(axis=-1, output_num=ks)
        self.concat = ops.Concat(axis=2)

    def construct(self, x_offset):
        b, c, h, w, _ = self.shape(x_offset)
        splits = self.split(x_offset)
        x_offset = self.concat(splits)
        ks = self.ks
        x_offset = self.reshape(x_offset, (b, c, h * ks, w * ks))
        return x_offset


class DeformConv2d(nn.Cell):
    """
    Deformable convolution opertor

    Args:
        inc(int): Input channel.
        outc(int): Output channel.
        kernel_size (int): Convolution window. Default: 3.
        stride (int): The distance of kernel moving. Default: 1.
        padding (int): Implicit paddings size on both sides of the input. Default: 1.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: False.
        modulation (bool): If True, modulated defomable convolution (Deformable ConvNets v2). Default: True.
    Returns:
        Tensor, detection of images(bboxes, score, keypoints and category id of each objects)
    """
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, has_bias=False, modulation=True):
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.Pad(((0, 0), (0, 0), (padding, padding), (padding, padding)))
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, pad_mode='valid', padding=0,
                              stride=kernel_size, has_bias=has_bias)

        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=self.kernel_size,
                                pad_mode='pad', padding=self.padding, stride=self.stride)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=self.kernel_size,
                                    pad_mode='valid', padding=0, stride=self.stride)
        if kernel_size % 2 == 0:
            raise ValueError("Only odd number is supported, but current kernel sizeis {}".format(kernel_size))
        self.N = kernel_size * kernel_size
        self.begin = kernel_size // 2
        self.sigmoid = ops.Sigmoid()
        self.dtype = ops.DType()
        self.perm_list = (0, 2, 3, 1)
        self.transpose = ops.Transpose()
        self.floor = ops.Floor()
        self.half = ops.Split(axis=-1, output_num=2)
        self.clip_value = ClipByValue()
        self.expand_dims = ops.ExpandDims()
        self.shape = ops.Shape()
        self.cast = ops.Cast()
        self._get_offset = GetOffsetPosition(self.begin, self.stride)
        self._get_surround = GetSurroundFeature()
        self._generate_fm = RegenerateFeatureMap(self.kernel_size)

    def construct(self, x):
        """deformed sampling locations with augmented offsets"""
        offset = self.p_conv(x)
        # (b, c, h, w))
        x_shape = self.shape(x)
        # (b, c, h + 2p, w + 2p)
        if self.padding > 0:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_offset(offset)

        # (b, h, w, 2N)
        p = self.transpose(p, self.perm_list)
        q_lt = self.cast(self.floor(p), mstype.int32)
        q_rb = q_lt + 1

        # (b, h, w, N)
        q_lt_h, q_lt_w = self.half(q_lt)
        q_lt_h = self.clip_value(q_lt_h, 0, x_shape[2] - 1)
        q_lt_w = self.clip_value(q_lt_w, 0, x_shape[3] - 1)
        # (b, h, w, N)
        q_rb_h, q_rb_w = self.half(q_rb)
        q_rb_h = self.clip_value(q_rb_h, 0, x_shape[2] - 1)
        q_rb_w = self.clip_value(q_rb_w, 0, x_shape[3] - 1)

        # clip p
        p_h, p_w = self.half(p)
        dtype = self.dtype(offset)
        p_h = self.clip_value(p_h, self.cast(0, dtype), self.cast(x_shape[2] - 1, dtype))
        p_w = self.clip_value(p_w, self.cast(0, dtype), self.cast(x_shape[3] - 1, dtype))

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt_h - p_h)) * (1 + (q_lt_w - p_w))
        g_rb = (1 - (q_rb_h - p_h)) * (1 - (q_rb_w - p_w))
        g_lb = (1 + (q_lt_h - p_h)) * (1 - (q_rb_w - p_w))
        g_rt = (1 - (q_rb_h - p_h)) * (1 + (q_lt_w - p_w))

        # (b, c, h, w, N)
        x_q_lt = self._get_surround(x, q_lt_h, q_lt_w)
        x_q_rb = self._get_surround(x, q_rb_h, q_rb_w)
        x_q_lb = self._get_surround(x, q_lt_h, q_rb_w)
        x_q_rt = self._get_surround(x, q_rb_h, q_lt_w)

        # (b, c, h, w, N)
        x_offset = (self.expand_dims(g_lt, 1) * x_q_lt +
                    self.expand_dims(g_rb, 1) * x_q_rb +
                    self.expand_dims(g_lb, 1) * x_q_lb +
                    self.expand_dims(g_rt, 1) * x_q_rt)

        if self.modulation:
            m = self.sigmoid(self.m_conv(x))
            m = self.transpose(m, self.perm_list)
            m = self.expand_dims(m, 1)
            x_offset = x_offset * m

        x_offset = self._generate_fm(x_offset)
        out = self.conv(x_offset)

        return out
