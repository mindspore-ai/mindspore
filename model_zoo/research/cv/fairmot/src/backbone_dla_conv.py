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
from mindspore.common.initializer import Constant

BN_MOMENTUM = 0.1


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
        """
        Basic residual block for dla.
        """
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
        """
        Get HDA node which play as the root of tree in each stage
        """
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
        if self.levels == 1:
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

    def __init__(self, levels, channels, block=None, residual_root=False):
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
        """
        Construct the downsampling deep aggregation network.
        """
        y = []
        x = self.base_layer(x)
        for i in range(len(self.channels)):
            x = self.dla_fn[i](x)
            y.append(x)
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
        self.conv = nn.Conv2d(cin, cout, kernel_size=3, stride=1, has_bias=False)

    def construct(self, x):
        """
        Deformable convolution v2.
        """
        x = self.conv(x)
        x = self.actf(x)
        return x


class IDAUp(nn.Cell):
    """IDAUp sample."""

    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        proj_list = []
        up_list = []
        node_list = []
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = DeformConv(c, o)
            node = DeformConv(o, o)
            up = nn.Conv2dTranspose(o, o, f * 2, stride=f, pad_mode='pad', padding=f // 2, group=o)
            proj_list.append(proj)
            up_list.append(up)
            node_list.append(node)
        self.proj = nn.CellList(proj_list)
        self.up = nn.CellList(up_list)
        self.node = nn.CellList(node_list)

    def construct(self, layers, startp, endp):
        """IDAUp sample."""
        for i in range(startp + 1, endp):
            upsample = self.up[i - startp - 1]
            project = self.proj[i - startp - 1]
            layers[i] = upsample(project(layers[i]))
            node = self.node[i - startp - 1]
            layers[i] = node(layers[i] + layers[i - 1])
        return layers


class DLAUp(nn.Cell):
    """DLAUp sample."""

    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        self.ida = []
        for i in range(len(channels) - 1):
            j = -i - 2
            self.ida.append(IDAUp(channels[j], in_channels[j:],
                                  scales[j:] // scales[j]))
            # setattr(self, 'ida_{}'.format(i),
            #         IDAUp(channels[j], in_channels[j:],
            #               scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]
        self.ida_nfs = nn.CellList(self.ida)

    def construct(self, layers):
        """DLAUp sample."""
        out = [layers[-1]]  # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = self.ida_nfs[i]
            layers = ida(layers, len(layers) - i - 2, len(layers))
            out.append(layers[-1])
        a = []
        i = len(out)
        while i > 0:
            a.append(out[i - 1])
            i -= 1
        return a


class DLASegConv(nn.Cell):
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

    def __init__(self, heads, down_ratio, final_kernel,
                 last_level, head_conv, out_channel=0, is_training=True):
        super(DLASegConv, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.is_training = is_training
        self.base = DLA34([1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512], block=BasicBlock)
        channels = [16, 32, 64, 128, 256, 512]
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        # self.dla_up = DLAUp(self.first_level, stage_channels[self.first_level:], last_level)
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)
        if out_channel == 0:
            out_channel = channels[self.first_level]
        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level],
                            [2 ** i for i in range(self.last_level - self.first_level)])
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                if 'hm' in head:
                    conv2d = nn.Conv2d(head_conv, classes, kernel_size=final_kernel, has_bias=True,
                                       bias_init=Constant(-2.19))
                    self.hm_fc = nn.SequentialCell(
                        [nn.Conv2d(channels[self.first_level], head_conv, kernel_size=3, has_bias=True), nn.ReLU(),
                         conv2d])
                elif 'wh' in head:
                    conv2d = nn.Conv2d(head_conv, classes, kernel_size=final_kernel, has_bias=True)
                    self.wh_fc = nn.SequentialCell(
                        [nn.Conv2d(channels[self.first_level], head_conv, kernel_size=3, has_bias=True), nn.ReLU(),
                         conv2d])
                elif 'id' in head:
                    conv2d = nn.Conv2d(head_conv, classes, kernel_size=final_kernel, has_bias=True)
                    self.id_fc = nn.SequentialCell(
                        [nn.Conv2d(channels[self.first_level], head_conv, kernel_size=3, has_bias=True), nn.ReLU(),
                         conv2d])
                else:
                    conv2d = nn.Conv2d(head_conv, classes, kernel_size=final_kernel, has_bias=True)
                    self.reg_fc = nn.SequentialCell(
                        [nn.Conv2d(channels[self.first_level], head_conv, kernel_size=3, has_bias=True), nn.ReLU(),
                         conv2d])
            else:
                if 'hm' in head:
                    self.hm_fc = nn.Conv2d(channels[self.first_level], classes, kernel_size=final_kernel, has_bias=True,
                                           bias_init=Constant(-2.19))
                elif 'wh' in head:
                    self.wh_fc = nn.Conv2d(channels[self.first_level], classes, kernel_size=final_kernel, has_bias=True)
                elif 'id' in head:
                    self.id_fc = nn.Conv2d(channels[self.first_level], classes, kernel_size=final_kernel, has_bias=True)
                else:
                    self.reg_fc = nn.Conv2d(channels[self.first_level], classes, kernel_size=final_kernel,
                                            has_bias=True)

    def construct(self, image):
        """The DLA backbone network."""
        x = self.base(image)
        x = self.dla_up(x)
        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i])
        y = self.ida_up(y, 0, len(y))
        hm = self.hm_fc(y[-1])
        wh = self.wh_fc(y[-1])
        feature_id = self.id_fc(y[-1])
        reg = self.reg_fc(y[-1])
        feature = {"hm": hm, "feature_id": feature_id, "wh": wh, "reg": reg}
        return feature
