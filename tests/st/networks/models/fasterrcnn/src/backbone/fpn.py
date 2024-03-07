# Copyright 2024 Huawei Technologies Co., Ltd
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
from mindspore import ops, nn
from .det_resnet import get_bn


class ConvModule(nn.Cell):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 norm="none",
                 act="none"):
        super(ConvModule, self).__init__()
        layers = []
        bias = (norm == "none")
        layers.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                group=groups,
                pad_mode="pad",
                padding=padding,
                has_bias=bias,
                weight_init="XavierUniform",
                bias_init="zeros"
            )
        )
        if norm == "bn":
            layers.append(get_bn()(out_channels, eps=1e-4))
        elif norm != "none":
            raise ValueError(f"not support norm: {norm}, you can set norm None or 'bn'")
        if act != "none":
            layers.append(nn.get_activation(act))
        self.conv = nn.SequentialCell(layers)

    def construct(self, x):
        return self.conv(x)


class FPN(nn.Cell):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        bottom_up (Cell): module representing the bottom up subnetwork.
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        level_index (list[int]): Index of the input backbone levels.
        norm (Cell/str): normalization layer. Default: None.
        act (Cell/str): activation layer in ConvModule. Default: None.
        upsample_mode (str): interpolate mode for upsample. Default: 'nearest'.
        frozen(bool): frozen backbone when training.
    """

    def __init__(self,
                 bottom_up,
                 in_channels,
                 out_channels,
                 num_outs,
                 level_index,
                 norm="none",
                 act="none",
                 upsample_mode="bilinear",
                 frozen=False,):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        assert isinstance(level_index, list)
        assert len(in_channels) == len(level_index)
        self.bottom_up = bottom_up
        self.in_channels = in_channels
        self.level_index = level_index
        self.num_outs = num_outs
        self.upsample_mode = upsample_mode

        self.lateral_convs = nn.CellList()
        self.fpn_convs = nn.CellList()
        self.max_pool = nn.MaxPool2d(kernel_size=1, stride=2)
        self.frozen = frozen

        for i in range(len(in_channels)):
            l_conv = ConvModule(in_channels[i], out_channels, kernel_size=1, norm=norm, act=act)
            fpn_conv = ConvModule(out_channels, out_channels, kernel_size=3, padding=1, norm=norm, act=act)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def construct(self, img):
        """Forward function."""
        inputs = self.bottom_up(img)
        laterals = ()
        for i, lateral_conv in zip(self.level_index, self.lateral_convs):
            laterals += (lateral_conv(inputs[i]),)

        # build top-down path
        used_backbone_levels = len(laterals)
        prev_feature = laterals[-1]
        ups = (laterals[-1],)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            prev_feature = laterals[i - 1] + ops.interpolate(prev_feature, size=prev_shape, mode=self.upsample_mode)
            ups = (prev_feature,) + ups

        # build outputs
        # part 1: from original levels
        outs = ()
        for i in range(used_backbone_levels):
            out = self.fpn_convs[i](ups[i])
            if self.frozen:
                out = ops.stop_gradient(out)
            outs += (out,)

        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            for i in range(self.num_outs - used_backbone_levels):
                out = self.max_pool(outs[-1])
                if self.frozen:
                    out = ops.stop_gradient(out)
                outs += (out,)
        return outs
