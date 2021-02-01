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
deep layer aggregation backbone
"""

import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from .dcn_v2 import DeformConv2d as DCN


BN_MOMENTUM = 0.9


class BasicBlock(nn.Cell):
    """
    Basic residual block for dla.

    Args:
        cin(int): Input channel.
        cout(int): Output channel.
        stride(int): Covolution stride. Default: 1.
        dilation(int): The dilation rate to be used for dilated convolution. Default: 1.

    Returns:
        Tensor, the feature after covolution.
    """
    def __init__(self, cin, cout, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv_bn_act = nn.Conv2dBnAct(cin, cout, kernel_size=3, stride=stride, pad_mode='pad',
                                          padding=dilation, has_bias=False, dilation=dilation,
                                          has_bn=True, momentum=BN_MOMENTUM,
                                          activation='relu', after_fake=False)
        self.conv_bn = nn.Conv2dBnAct(cout, cout, kernel_size=3, stride=1, pad_mode='same',
                                      has_bias=False, dilation=dilation, has_bn=True,
                                      momentum=BN_MOMENTUM, activation=None)
        self.relu = ops.ReLU()

    def construct(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv_bn_act(x)
        out = self.conv_bn(out)
        out += residual
        out = self.relu(out)
        return out


class Root(nn.Cell):
    """
    Get HDA node which play as the root of tree in each stage

    Args:
        cin(int): Input channel.
        cout(int):Output channel.
        kernel_size(int): Covolution kernel size.
        residual(bool): Add residual or not.

    Returns:
        Tensor, HDA node after aggregation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, has_bias=False,
                              pad_mode='pad', padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = ops.ReLU()
        self.residual = residual
        self.cat = ops.Concat(axis=1)

    def construct(self, x):
        children = x
        x = self.conv(self.cat(x))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)
        return x


class Tree(nn.Cell):
    """
    Construct the deep aggregation network through recurrent. Each stage can be seen as a tree with multiple children.

    Args:
        levels(list int): Tree height of each stage.
        block(Cell): Basic block of the tree.
        in_channels(list int): Input channel of each stage.
        out_channels(list int): Output channel of each stage.
        stride(int): Covolution stride. Default: 1.
        level_root(bool): Whether is the root of tree or not. Default: False.
        root_dim(int): Input channel of the root node. Default: 0.
        root_kernel_size(int): Covolution kernel size at the root. Default: 1.
        dilation(int): The dilation rate to be used for dilated convolution. Default: 1.
        root_residual(bool): Add residual or not. Default: False.

    Returns:
        Tensor, the root ida node.
    """
    def __init__(self, levels, block, in_channels, out_channels, stride=1, level_root=False,
                 root_dim=0, root_kernel_size=1, dilation=1, root_residual=False):
        super(Tree, self).__init__()
        self.levels = levels
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if self.levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1, dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels, stride, root_dim=0,
                              root_kernel_size=root_kernel_size, dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels, root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size, dilation=dilation, root_residual=root_residual)
        self.root = Root(root_dim, out_channels, root_kernel_size, root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Conv2dBnAct(in_channels, out_channels, kernel_size=1, stride=1, pad_mode='same',
                                          has_bias=False, has_bn=True, momentum=BN_MOMENTUM,
                                          activation=None, after_fake=False)

    def construct(self, x, residual=None, children=None):
        """construct each stage tree recurrently"""
        children = () if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children += (bottom,)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            ida_node = (x2, x1) + children
            x = self.root(ida_node)
        else:
            children += (x1,)
            x = self.tree2(x1, children=children)
        return x


class DLA34(nn.Cell):
    """
    Construct the downsampling deep aggregation network.

    Args:
        levels(list int): Tree height of each stage.
        channels(list int): Input channel of each stage
        block(Cell): Initial basic block. Default: BasicBlock.
        residual_root(bool): Add residual or not. Default: False

    Returns:
        tuple of Tensor, the root node of each stage.
    """
    def __init__(self, levels, channels, block=BasicBlock, residual_root=False):
        super(DLA34, self).__init__()
        self.channels = channels
        self.base_layer = nn.Conv2dBnAct(3, channels[0], kernel_size=7, stride=1, pad_mode='same',
                                         has_bias=False, has_bn=True, momentum=BN_MOMENTUM,
                                         activation='relu', after_fake=False)

        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False, root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)
        self.dla_fn = [self.level0, self.level1, self.level2, self.level3, self.level4, self.level5]

    def _make_conv_level(self, cin, cout, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.append(nn.Conv2dBnAct(cin, cout, kernel_size=3, stride=stride if i == 0 else 1,
                                          pad_mode='pad', padding=dilation, has_bias=False, dilation=dilation,
                                          has_bn=True, momentum=BN_MOMENTUM, activation='relu', after_fake=False))
            cin = cout
        return nn.SequentialCell(modules)

    def construct(self, x):
        y = ()
        x = self.base_layer(x)
        for i in range(len(self.channels)):
            x = self.dla_fn[i](x)
            y += (x,)
        return y


class DeformConv(nn.Cell):
    """
    Deformable convolution v2.

    Args:
        cin(int): Input channel
        cout(int): Output_channel

    Returns:
        Tensor, results after deformable convolution and activation
    """
    def __init__(self, cin, cout):
        super(DeformConv, self).__init__()
        self.actf = nn.SequentialCell([
            nn.BatchNorm2d(cout, momentum=BN_MOMENTUM),
            nn.ReLU()
        ])
        self.conv = DCN(cin, cout, kernel_size=3, stride=1, padding=1, modulation=True)

    def construct(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


class IDAUp(nn.Cell):
    """
    Construct the upsampling node.

    Args:
        cin(int): Input channel.
        cout(int): Output_channel.
        up_f(int): Upsampling factor. Default: 2.
        enable_dcn(bool): Use deformable convolutional operator or not. Default: False.

    Returns:
        Tensor, the upsampling node after aggregation
    """
    def __init__(self, cin, cout, up_f=2, enable_dcn=False):
        super(IDAUp, self).__init__()
        self.enable_dcn = enable_dcn
        if enable_dcn:
            self.proj = DeformConv(cin, cout)
            self.node = DeformConv(cout, cout)
        else:
            self.proj = nn.Conv2dBnAct(cin, cout, kernel_size=1, stride=1, pad_mode='same',
                                       has_bias=False, has_bn=True, momentum=BN_MOMENTUM,
                                       activation='relu', after_fake=False)
            self.node = nn.Conv2dBnAct(2 * cout, cout, kernel_size=3, stride=1, pad_mode='same',
                                       has_bias=False, has_bn=True, momentum=BN_MOMENTUM,
                                       activation='relu', after_fake=False)
        self.up = nn.Conv2dTranspose(cout, cout, up_f * 2, stride=up_f, pad_mode='pad', padding=up_f // 2)
        self.concat = ops.Concat(axis=1)

    def construct(self, down_layer, up_layer):
        project = self.proj(down_layer)
        upsample = self.up(project)
        if self.enable_dcn:
            node = self.node(upsample + up_layer)
        else:
            node = self.node(self.concat((upsample, up_layer)))
        return node


class DLAUp(nn.Cell):
    """
    Upsampling of DLA network.

    Args:
        startp(int): The beginning stage startup upsampling
        channels(list int): The channels of each stage after upsampling
        last_level(int): The ending stage of the final upsampling

    Returns:
        Tensor, output of the dla backbone after upsampling
    """
    def __init__(self, startp, channels, last_level):
        super(DLAUp, self).__init__()
        self.startp = startp
        self.channels = channels
        self.last_level = last_level
        self.num_levels = len(self.channels)
        if self.last_level > self.startp + len(self.channels) or self.last_level < self.startp:
            raise ValueError("Invalid last level value.")

        # first ida up layers
        idaup_fns = []
        for i in range(1, len(channels), 1):
            ida_up = IDAUp(channels[i], channels[i - 1])
            idaup_fns.append(ida_up)
        self.idaup_fns = nn.CellList(idaup_fns)

        # final ida up
        if self.last_level == self.startp:
            self.final_up = False
        else:
            self.final_up = True
            final_fn = []
            for i in range(1, self.last_level - self.startp):
                ida = IDAUp(channels[i], channels[0], up_f=2 ** i)
                final_fn.append(ida)
            self.final_idaup_fns = nn.CellList(final_fn)

    def construct(self, stages):
        """get upsampling ida node"""
        first_ups = (stages[self.startp],)
        for i in range(1, self.num_levels):
            ida_node = (stages[i + self.startp])
            ida_ups = (ida_node,)
            # get uplayers
            for j in range(i, 0, -1):
                ida_node = self.idaup_fns[j -1](ida_node, first_ups[i - j])
                ida_ups += (ida_node,)
            first_ups = ida_ups

        final_up = first_ups[self.num_levels - 1]
        if self.final_up:
            for i in range(self.startp + 1, self.last_level):
                final_up = self.final_idaup_fns[i - self.startp - 1](first_ups[self.num_levels + 1 - i], final_up)

        return final_up


class DLASeg(nn.Cell):
    """
    The DLA backbone network.

    Args:
        down_ratio(int): The ratio of input and output resolution
        last_level(int): The ending stage of the final upsampling
        stage_levels(list int): The tree height of each stage block
        stage_channels(list int): The feature channel of each stage

    Returns:
        Tensor, the feature map extracted by dla network
    """
    def __init__(self, down_ratio, last_level, stage_levels, stage_channels):
        super(DLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.dla = DLA34(stage_levels, stage_channels, block=BasicBlock)
        self.dla_up = DLAUp(self.first_level, stage_channels[self.first_level:], last_level)

    def construct(self, image):
        stages = self.dla(image)
        output = self.dla_up(stages)
        return output
