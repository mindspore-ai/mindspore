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
from mindspore.rewrite import SymbolTree, ScopedValue, Node, NodeType
from mindspore.common.api import _cell_graph_executor
from .comp_network import CompNet, SubNet4, SubNet2


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
            modify_stree: SymbolTree = node.get_sub_tree()
            for inner_node in modify_stree.nodes():
                if inner_node.get_node_type() != NodeType.Output:
                    continue
                position = modify_stree.before(inner_node)
                new_relu = nn.ReLU()
                new_relu_node = Node.create_call_cell(new_relu,
                                                      targets=[stree.unique_name('x')],
                                                      name='new_relu',
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
            modify_stree: SymbolTree = node.get_sub_tree()
            for inner_node in modify_stree.nodes():
                if inner_node.get_instance_type() != nn.BatchNorm2d:
                    continue
                new_relu = nn.ReLU()
                new_relu_node = Node.create_call_cell(new_relu,
                                                      targets=[stree.unique_name('x')],
                                                      name='new_relu',
                                                      args=inner_node.get_args(),
                                                      kwargs=inner_node.get_kwargs())
                modify_stree.replace(inner_node, [new_relu_node])
                break
            break


def erase_relu_in_conv2(stree: SymbolTree):
    for node in stree.nodes():
        if node.get_node_type() != NodeType.Tree:
            continue
        if node.get_name() == "conv2":
            modify_stree: SymbolTree = node.get_sub_tree()
            for inner_node in modify_stree.nodes():
                if inner_node.get_instance_type() != nn.ReLU:
                    continue
                assert len(inner_node.get_args()) == 1
                arg = inner_node.get_args()[0]
                modify_stree.set_output(0, arg.value)
                modify_stree.erase(inner_node)
                break
            break


def inset_subtree(stree: SymbolTree):
    for node in stree.nodes():
        if node.get_name() == "conv2":
            position = stree.before(node)
            subtree = SubNet()
            new_node = Node.create_call_cell(subtree,
                                             targets=[ScopedValue.create_naming_value(stree.unique_name('x'))],
                                             name='conv',
                                             args=[ScopedValue.create_naming_value('x')],
                                             kwargs={})
            stree.insert(position, new_node)
            break


def inset_subtree2(stree: SymbolTree):
    for node in stree.nodes():
        if node.get_name() == "conv2":
            position = stree.before(node)
            subtree = SubNet()
            new_node = Node.create_call_cell(subtree,
                                             targets=[ScopedValue.create_naming_value(stree.unique_name('x'))],
                                             name='conv11',
                                             args=[ScopedValue.create_naming_value('x')],
                                             kwargs={})
            stree.insert(position, new_node)
            break


def add_relu_in_conv11(stree: SymbolTree):
    for node in stree.nodes():
        if node.get_node_type() != NodeType.Tree:
            continue
        if node.get_name() == "conv11":
            _stree: SymbolTree = node.get_sub_tree()
            for inner_node in _stree.nodes():
                if inner_node.get_node_type() != NodeType.Output:
                    continue
                position = _stree.before(inner_node)
                new_relu = nn.ReLU()
                new_relu_node = Node.create_call_cell(new_relu,
                                                      targets=[stree.unique_name('x')],
                                                      name='relu1',
                                                      args=[ScopedValue.create_naming_value('x')])
                _stree.insert(position, new_relu_node)
                _stree.set_output(0, new_relu_node.get_targets()[0].value)
                break
            break


def transform(stree: SymbolTree):
    add_relu_in_conv1(stree)
    replace_bn_in_conv2(stree)
    erase_relu_in_conv2(stree)
    inset_subtree(stree)


def test_subtree_net():
    """
    Feature: Rewrite package api: sub-tree.
    Description: Use Rewrite to parse and transform a network with sub-network.
    Expectation: Rewrite can parse a network with sub-network and can modify node in sub-network successfully.
    """

    net = MainNet()
    stree = SymbolTree.create(net)
    transform(stree)
    inset_subtree2(stree)
    add_relu_in_conv11(stree)

    net_opt = stree.get_network()
    data_in = Tensor(np.ones([1, 1, 32, 32]), mindspore.float32)
    _cell_graph_executor.compile(net_opt, data_in)


def test_subtree_create_erase():
    """
    Feature: Test SymbolTree from a network with sub-tree.
    Description: Create a SymbolTree from a network with sub-network.
    Expectation: Rewrite can parse a network with sub-network and can erase node in sub-network successfully.
    """
    net = CompNet(mul_size=(16, 3, 8, 8), add_size=(16, 3, 8, 8))
    stree = SymbolTree.create(net)
    del_node = stree.get_node("mul")
    input_node = del_node.get_inputs()[0]
    output_nodes = del_node.get_users()
    for node in output_nodes:
        node.set_arg_by_node(0, input_node)
    stree.erase(del_node)
    assert stree.get_node("mul") is None
    assert 'z_1 = self.relu(x_3)' in stree.get_code()
    new_net = stree.get_network()
    data_in = Tensor(np.ones([16, 3, 8, 8]), mindspore.float32)
    _cell_graph_executor.compile(new_net, data_in)


def test_insert_replace():
    """
    Feature: Test insert api of SymbolTree.
    Description: Test insert a node after an input node.
    Expectation: Success.
    """
    net = CompNet(mul_size=(16, 3, 8, 8), add_size=(16, 3, 8, 8))
    stree = SymbolTree.create(net)
    nodes = stree.nodes()
    position = stree.after(next(nodes))
    new_node = Node.create_call_cell(P.Mul(), targets=[ScopedValue.create_naming_value(stree.unique_name("inputs"))],
                                     args=[ScopedValue.create_naming_value("inputs"),
                                           ScopedValue.create_naming_value("mul_weight", "self")],
                                     name="mulnet")
    stree.insert(position, new_node)
    for node in stree.nodes():
        if node.get_instance_type() == P.Add:
            new_node2 = Node.create_call_cell(P.Mul(), targets=[node.get_targets()[0]], args=node.get_args(),
                                              kwargs=node.get_kwargs(), name="mulnet")
            stree.replace(node, [new_node2])
            break

    new_net = stree.get_network()
    data_in = Tensor(np.ones([16, 3, 8, 8]), mindspore.float32)
    _cell_graph_executor.compile(new_net, data_in)


def test_insert_replace2():
    """
    Feature: Test insert api of SymbolTree.
    Description: Test insert a node after an input node.
    Expectation: Success.
    """
    net = CompNet(mul_size=(16, 3, 8, 8), add_size=(16, 3, 8, 8))
    stree = SymbolTree.create(net)
    for node in stree.nodes():
        if node.get_instance_type() == P.ReLU:
            position = stree.after(node)
            new_node = Node.create_call_cell(SubNet4(), targets=["subnet"],
                                             args=ScopedValue.create_name_values(["z", "z_1"]), name="subnet")
            stree.insert(position, new_node)
            break

    for node in stree.nodes():
        if node.get_instance_type() == SubNet2:
            new_node2 = Node.create_call_cell(SubNet2(), targets=node.get_targets(), args=node.get_args(),
                                              kwargs=node.get_kwargs(), name=node.get_name() + "mod", is_sub_net=True)
            stree.replace(node, [new_node2])
            break
    new_net = stree.get_network()
    data_in = Tensor(np.ones([16, 3, 8, 8]), mindspore.float32)
    _cell_graph_executor.compile(new_net, data_in)
