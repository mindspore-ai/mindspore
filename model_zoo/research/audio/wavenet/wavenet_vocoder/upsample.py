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
"""
Upsampling

"""
from __future__ import with_statement, print_function, absolute_import
import numpy as np
from mindspore import nn
from mindspore.ops import operations as P


class Resize(nn.Cell):
    """
    Resize input Tensor
    """

    def __init__(self, x_scale, y_scale, mode="nearest"):
        super(Resize, self).__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.mode = mode

    def construct(self, x):
        _, _, h, w = x.shape
        interpolate_op = P.ResizeNearestNeighbor((self.y_scale * h, self.x_scale * w))
        return interpolate_op(x)


def _get_activation(upsample_activation):
    """get activation"""
    nonlinear = getattr(nn, upsample_activation)
    return nonlinear


class UpsampleNetwork(nn.Cell):
    """UpsampleNetwork"""
    def __init__(self, upsample_scales, mode="nearest",
                 freq_axis_kernel_size=1, cin_pad=0, cin_channels=80):
        super(UpsampleNetwork, self).__init__()
        self.expand_op = P.ExpandDims()
        self.squeeze_op = P.Squeeze(1)
        up_layers = []
        total_scale = np.prod(upsample_scales)
        self.indent = cin_pad * total_scale
        for scale in upsample_scales:
            freq_axis_padding = (freq_axis_kernel_size - 1) // 2
            k_size = (freq_axis_kernel_size, scale * 2 + 1)
            padding = (freq_axis_padding, freq_axis_padding, scale, scale)
            stretch = Resize(scale, 1, mode)
            conv = nn.Conv2d(1, 1, kernel_size=k_size, has_bias=False, pad_mode='pad', padding=padding)
            up_layers.append(stretch)
            up_layers.append(conv)
        self.up_layers = nn.CellList(up_layers)

    def construct(self, c):
        """

        Args:
            c (Tensor): Local conditioning feature

        Returns:
            Tensor: Upsampling feature

        """
        # B x 1 x C x T
        c = self.expand_op(c, 1)
        for f in self.up_layers:
            c = f(c)
        # B x C x T
        c = self.squeeze_op(c)

        return c


class ConvInUpsampleNetwork(nn.Cell):
    """Upsample Network

    Args:
        upsample_scales (list): Upsample_scales list.
        upsample_activation (str): Upsample_activation.
        mode (str): Resize mode, default is NearestNeighbor.
        cin_channels (int): Local conditioning channels.
        freq_axis_kernel_size (int): Freq-axis kernel_size for the convolution layers after resize.

    """

    def __init__(self, upsample_scales, mode="nearest",
                 freq_axis_kernel_size=1, cin_pad=0,
                 cin_channels=80):
        super(ConvInUpsampleNetwork, self).__init__()
        ks = 2 * cin_pad + 1
        self.conv_in = nn.Conv1d(cin_channels, cin_channels, kernel_size=ks, has_bias=False, pad_mode='pad', padding=0)
        self.upsample = UpsampleNetwork(upsample_scales, mode, freq_axis_kernel_size, cin_pad=0,
                                        cin_channels=cin_channels)

    def construct(self, c):
        c = self.conv_in(c)
        c_up = self.upsample(c)
        return c_up
