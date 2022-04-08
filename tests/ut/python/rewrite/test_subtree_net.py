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
import numpy as np

import mindspore
from mindspore import Tensor, nn
from mindspore.ops import operations as P
from mindspore.rewrite import SymbolTree, ScopedValue, Node, NodeType, TreeNodeHelper
from mindspore.common.api import _cell_graph_executor


class SubNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 10, 3)
        self.bn = nn.BatchNorm2d(10)

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MainNet(nn.Cell):
    def __init__(self):
        super(MainNet, self).__init__()
        self.conv1 = SubNet()
        self.conv2 = SubNet()
        self.add = P.Add()

    def construct(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = self.add(x1, x2)
        return x


def add_relu_in_conv1(stree: SymbolTree):
    for node in stree.nodes():
        if node.get_node_type() != NodeType.Tree:
            continue
        if node.get_name() == "conv1":
            modify_stree: SymbolTree = TreeNodeHelper.get_sub_tree(node)
            for inner_node in modify_stree.nodes():
                if inner_node.get_node_type() != NodeType.Output:
                    continue
                position = modify_stree.before(inner_node)
                new_relu = nn.ReLU()
                new_relu_node = Node.create_call_cell(new_relu, targets=['x'], name='new_relu',
                                                      args=[ScopedValue.create_naming_value('x')])
                modify_stree.insert(position, new_relu_node)
                modify_stree.set_output(0, new_relu_node.get_targets()[0].value)
                break
            break


def replace_bn_in_conv2(stree: SymbolTree):
    for node in stree.nodes():
        if node.get_node_type() != NodeType.Tree:
            continue
        if node.get_name() == "conv2":
            modify_stree: SymbolTree = TreeNodeHelper.get_sub_tree(node)
            for inner_node in modify_stree.nodes():
                if inner_node.get_instance_type() != nn.BatchNorm2d:
                    continue
                new_relu = nn.ReLU()
                new_relu_node = Node.create_call_cell(new_relu, targets=['x'], name='new_relu',
                                                      args=inner_node.get_args(), kwargs=inner_node.get_kwargs())
                modify_stree.replace(inner_node, [new_relu_node])
                break
            break


def erase_relu_in_conv2(stree: SymbolTree):
    for node in stree.nodes():
        if node.get_node_type() != NodeType.Tree:
            continue
        if node.get_name() == "conv2":
            modify_stree: SymbolTree = TreeNodeHelper.get_sub_tree(node)
            for inner_node in modify_stree.nodes():
                if inner_node.get_instance_type() != nn.ReLU:
                    continue
                assert len(inner_node.get_args()) == 1
                arg = inner_node.get_args()[0]
                modify_stree.set_output(0, arg.value)
                modify_stree.erase_node(inner_node)
                break
            break


def transform(stree: SymbolTree):
    add_relu_in_conv1(stree)
    replace_bn_in_conv2(stree)
    erase_relu_in_conv2(stree)


def test_subtree_net():
    """
    Feature: Rewrite package api: sub-tree.
    Description: Use Rewrite to parse and transform a network with sub-network.
    Expectation: Rewrite can parse a network with sub-network and can modify node in sub-network successfully.
    """

    net = MainNet()
    stree = SymbolTree.create(net)
    transform(stree)
    print(stree.get_code())
    print(stree.get_handler().get_global_vars().keys())
    net_opt = stree.get_network()
    data_in = Tensor(np.ones([1, 1, 32, 32]), mindspore.float32)
    _cell_graph_executor.compile(net_opt, data_in)
