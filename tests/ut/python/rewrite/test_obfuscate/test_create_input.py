# Copyright 2023 Huawei Technologies Co., Ltd
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
"""test create input"""

import mindspore.nn as nn
from mindspore.rewrite import SymbolTree, NodeType

class MyNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 16, 3)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def construct(self, x, y):
        x = self.conv(x)
        x = self.relu1(x)
        x = self.relu2(x)
        x = self.relu3(x)
        return x

def test_create_input():
    """
    Feature: Create an input node.
    Description: Call create_input to create an input node.
    Expectation: Success.
    """
    net = MyNet()
    stree = SymbolTree.create(net)

    assert len(stree.get_handler().nodes()) == 7
    node = stree.get_node("input_y")
    assert node
    assert node.get_node_type() == NodeType.Input
    position = stree.after(node)
    assert position
    new_input_node = node.create_input("z")
    assert new_input_node
    assert new_input_node.get_node_type() == NodeType.Input

    stree.insert(position, new_input_node)
    assert stree.get_handler().get_node('input_z')
    assert len(stree.get_handler().nodes()) == 8

    codes = stree.get_code()
    assert codes.count("construct(self, x, y, z=None)") == 1
