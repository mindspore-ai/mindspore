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
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.rewrite import SymbolTree


def external_func(x):
    x = ops.abs(x)
    return x


class OtherClass():
    def other_class_func(self, x):
        x = ops.abs(x)
        return x


class SubSubNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.abs = ops.Abs()

    def construct(self, x):
        x = self.relu(x)
        x = external_func(x)
        x = self.subsubnet_internal_func(x)
        return x

    def subsubnet_internal_func(self, x):
        x = self.abs(x)
        return x


class SubNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.s_cells = nn.SequentialCell([SubSubNet(), SubSubNet()])
        self.subsubnet = SubSubNet()
        self.abs = ops.Abs()
        self.relu = nn.ReLU()
        self.other_class = OtherClass()

    def construct(self, x):
        x = self.relu(x)
        x = self.s_cells(x)
        x = self.subsubnet(x)
        x = external_func(x)
        x = self.subnet_internal_func(x)
        x = self.other_class.other_class_func(x)
        return x

    def subnet_internal_func(self, x):
        x = self.abs(x)
        return x


class MyNet(nn.Cell):
    def __init__(self, sub_net):
        super().__init__()
        self.relu = nn.ReLU()
        self.sub_net = sub_net
        self.s_cells = nn.SequentialCell(nn.ReLU())
        self.s_cells.append(nn.ReLU())
        self.s_cells.append(SubSubNet())
        self.abs = ops.Abs()
        self.sub_net1 = SubNet()
        self.sub_net2 = SubNet()
        self.other_class = OtherClass()

    def construct(self, x):
        x = self.relu(x)
        x = self.sub_net(x)
        x = self.sub_net1(x)
        x = self.sub_net1(x)
        x = self.sub_net2(x)
        x = self.s_cells(x)
        x = external_func(x)
        x = self.internal_func(x)
        x = self.other_class.other_class_func(x)
        return x

    def internal_func(self, x):
        x = self.sub_net(self.abs(x))
        return x


def test_generate_codes_from_symboltree():
    """
    Feature: Python api get_code of Node of Rewrite.
    Description: Test rewrite generate codes from symbol tree.
    Expectation: Success.
    """
    net = MyNet(SubNet())
    stree = SymbolTree.create(net)

    codes = stree.get_code()
    assert codes.count("def external_func(x):") == 1
    assert codes.count("class SubSubNetOpt") == 1
    assert codes.count("def subsubnet_internal_func(self, x):") == 1
    assert codes.count("class SubNetOpt") == 1
    assert codes.count("def subnet_internal_func(self, x):") == 1
    assert codes.count("class MyNetOpt") == 1
    assert codes.count("def internal_func(self, x):") == 1

    subtree = stree.get_node("sub_net1").get_handler().symbol_tree
    subtree.erase_node(subtree.get_node("relu"))
    codes = stree.get_code()
    assert codes.count("class SubNetOpt") == 2

    subtree = stree.get_node("sub_net1_1").get_handler().symbol_tree
    subsubtree = subtree.get_node("subsubnet").symbol_tree
    subsubtree.erase_node(subsubtree.get_node("relu"))
    codes = stree.get_code()
    assert codes.count("class SubNetOpt") == 3
    assert codes.count("class SubSubNetOpt") == 2
