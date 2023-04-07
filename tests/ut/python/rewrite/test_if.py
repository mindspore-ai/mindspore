# Copyright 2022 Huawei Technologies Co., Ltd
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
"""LeNet."""
from collections import OrderedDict

import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore.rewrite import SymbolTree, PatternEngine, Replacement, PatternNode, Node, ScopedValue


class IfNet(nn.Cell):
    def __init__(self, use_se=False, res_base=False):
        super(IfNet, self).__init__()

        self.use_se = use_se
        self.res_base = res_base
        self.se_block = False
        if self.use_se:
            self.se_block = True

        if self.use_se:
            self.conv1_0 = nn.Conv2d(3, 32, 3, stride=2, padding=0, pad_mode='same')
            self.bn1_0 = nn.BatchNorm2d(32)
            self.conv1_1 = nn.Conv2d(32, 32, 3, stride=1, padding=0, pad_mode='same')
            self.bn1_1 = nn.BatchNorm2d(32)
            self.conv1_2 = nn.Conv2d(32, 64, 3, stride=1, padding=0, pad_mode='same')
        else:
            self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=0, pad_mode='same')
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = P.ReLU()

        if self.res_base:
            self.pad = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid")
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")

        self.mean = P.ReduceMean(keep_dims=True)
        self.flatten = nn.Flatten()
        self.end_point = nn.Dense(in_channels=2048, out_channels=10, has_bias=True, bias_init=0)

    def construct(self, x):
        if self.use_se:
            x = self.conv1_0(x)
            x = self.bn1_0(x)
            x = self.relu(x)
            x = self.conv1_1(x)
            x = self.bn1_1(x)
            x = self.relu(x)
            x = self.conv1_2(x)
        else:
            x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.res_base:
            x = self.pad(x)
        c1 = self.maxpool(x)

        out = self.mean(c1, (2, 3))
        out = self.flatten(out)
        out = self.end_point(out)

        return out


class ConvBnReplace(Replacement):
    def build(self, pattern: PatternNode, is_chain_pattern: bool, matched: OrderedDict) -> [Node]:
        bn_node: Node = matched.get(pattern.name())
        bn: nn.BatchNorm2d = bn_node.get_instance()
        conv_p = pattern.get_inputs()[0]
        conv_node: Node = matched.get(conv_p.name())
        conv: nn.Conv2d = conv_node.get_instance()
        newconv = nn.Conv2dBnAct(conv.in_channels,
                                 conv.out_channels,
                                 conv.kernel_size,
                                 conv.stride,
                                 conv.pad_mode,
                                 conv.padding,
                                 conv.dilation,
                                 conv.group,
                                 conv.has_bias,
                                 conv.weight_init,
                                 conv.bias_init,
                                 True,
                                 bn.momentum,
                                 bn.eps)
        newconv_node = Node.create_call_cell(newconv, bn_node.get_targets(), conv_node.get_args(),
                                             conv_node.get_kwargs(), "Conv2dBnAct")
        return [newconv_node]


class ConvBnPattern(PatternEngine):
    def __init__(self):
        super().__init__([nn.Conv2d, nn.BatchNorm2d], ConvBnReplace())


def test_resnet_erase_in_if():
    """
    Feature: erase_node api and if_parser
    Description: erase a node in ast.If.
    Expectation: Success.
    """
    net = IfNet()
    stree: SymbolTree = SymbolTree.create(net)
    original_nodes_size = len(stree.get_handler()._nodes)
    for node in stree.nodes():
        node_: Node = node
        if node_.get_instance_type() == nn.Conv2d:
            input_ = node_.get_inputs()[0]
            output = node_.get_users()[0]
            output.set_arg_by_node(0, input_)
            stree.erase_node(node)
            break
    assert len(stree.get_handler()._nodes) == original_nodes_size - 1


def test_resnet_insert_in_if():
    """
    Feature: insert api and if_parser
    Description: insert a node into ast.If.
    Expectation: Success.
    """
    net = IfNet()
    stree: SymbolTree = SymbolTree.create(net)
    original_nodes_size = len(stree.get_handler()._nodes)
    for node in stree.nodes():
        node_: Node = node
        if node_.get_instance_type() == nn.Conv2d:
            pos = stree.after(node_)
            conv: nn.Conv2d = node_.get_instance()
            new_bn = Node.create_call_cell(nn.BatchNorm2d(conv.out_channels), targets=["x"],
                                           args=[ScopedValue.create_naming_value("x")], kwargs={}, name="new_bn")
            stree.insert(pos, new_bn)
            break
    assert len(stree.get_handler()._nodes) == original_nodes_size + 1


def test_resnet_replace_121_in_if():
    """
    Feature: replace api and if_parser
    Description: Replace one node by one nodes in ast.If.
    Expectation: Success.
    """
    net = IfNet()
    stree: SymbolTree = SymbolTree.create(net)
    original_nodes_size = len(stree.get_handler()._nodes)
    for node in stree.nodes():
        node_: Node = node
        if node_.get_instance_type() == nn.Conv2d:
            conv: nn.Conv2d = node_.get_instance()
            new_conv = Node.create_call_cell(nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size),
                                             targets=node_.get_targets(), args=node_.get_args(),
                                             kwargs=node.get_kwargs(), name="new_conv")
            stree.replace(node_, [new_conv])
            break
    assert len(stree.get_handler()._nodes) == original_nodes_size


def test_resnet_replace_12m_in_if():
    """
    Feature: replace api and if_parser
    Description: Replace one node by multi-nodes in ast.If.
    Expectation: Success.
    """
    net = IfNet()
    stree: SymbolTree = SymbolTree.create(net)
    original_nodes_size = len(stree.get_handler()._nodes)
    for node in stree.nodes():
        node_: Node = node
        if node_.get_instance_type() == nn.Conv2d:
            conv: nn.Conv2d = node_.get_instance()
            new_conv = Node.create_call_cell(nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size),
                                             targets=["x"], args=node_.get_args(),
                                             kwargs=node.get_kwargs(), name="new_conv")
            new_bn = Node.create_call_cell(nn.BatchNorm2d(conv.out_channels),
                                           targets=node_.get_targets(), args=[ScopedValue.create_naming_value("x")],
                                           kwargs={}, name="new_bn")
            stree.replace(node_, [new_conv, new_bn])
            break
    assert len(stree.get_handler()._nodes) == original_nodes_size + 1


def test_resnet_fusion_in_if():
    """
    Feature: PatternEngine and if_parser
    Description: Apply PatternEngine on nodes in ast.If.
    Expectation: Success.
    """
    net = IfNet()
    stree: SymbolTree = SymbolTree.create(net)
    original_nodes_size = len(stree.get_handler()._nodes)
    for node in stree.nodes():
        node_: Node = node
        if node_.get_instance_type() == nn.Conv2d:
            old_bn = node_.get_users()[0]
            pos = stree.after(node_)
            conv: nn.Conv2d = node_.get_instance()
            new_bn = Node.create_call_cell(nn.BatchNorm2d(conv.out_channels), targets=["x"],
                                           args=[node_.get_targets()[0]], kwargs={}, name="new_bn")
            stree.insert(pos, new_bn)
            old_bn.set_arg_by_node(0, new_bn)
            break
    assert len(stree.get_handler()._nodes) == original_nodes_size + 1
    ConvBnPattern().apply(stree)
    assert len(stree.get_handler()._nodes) == original_nodes_size
    assert not stree.get_node("conv1")
    assert not stree.get_node("new_bn")
    assert stree.get_node("bn1")


def test_resnet_fusion_cross_if():
    """
    Feature: PatternEngine and if_parser
    Description: Apply PatternEngine on nodes cross ast.If.
    Expectation: Success.
    """
    net = IfNet()
    stree: SymbolTree = SymbolTree.create(net)
    original_nodes_size = len(stree.get_handler()._nodes)
    for node in stree.nodes():
        node_: Node = node
        if node_.get_instance_type() == nn.Conv2d:
            pos = stree.after(node_)
            conv: nn.Conv2d = node_.get_instance()
            new_bn = Node.create_call_cell(nn.BatchNorm2d(conv.out_channels), targets=["x"],
                                           args=[ScopedValue.create_naming_value("x")], kwargs={}, name="new_bn")
            stree.insert(pos, new_bn)
            break
    assert len(stree.get_handler()._nodes) == original_nodes_size + 1
    ConvBnPattern().apply(stree)
    assert len(stree.get_handler()._nodes) == original_nodes_size
    assert not stree.get_node("conv1")
    assert stree.get_node("new_bn")
    assert not stree.get_node("bn1")
