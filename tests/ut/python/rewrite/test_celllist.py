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
"""test for CellList."""

import mindspore.nn as nn
from mindspore.rewrite import SymbolTree, NodeType, Node

class CellListSubNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(x)

class CellListNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 16, 3)
        self.blocks = nn.CellList()
        for _ in range(3):
            block = nn.ReLU()
            self.blocks.append(block)
        self.blocks.append(CellListSubNet())

    def construct(self, x):
        x = self.conv(x)
        for block in self.blocks:
            x = block(x)
        return x


def test_celllist():
    """
    Feature: for parser for nn.CellList.
    Description: parse for statement when the iterator is nn.CellList, then insert, erase and replace nodes in it.
    Expectation: Success.
    """
    net = CellListNet()
    stree = SymbolTree.create(net)

    # check for nodes
    for_node = stree.get_node("for_node")
    assert for_node.get_node_type() == NodeType.ControlFlow
    assert len(for_node.get_handler().nodes()) == 1
    # get nodes in celllist
    blocks = stree.get_node("self.blocks")
    assert len(blocks.get_handler().nodes()) == 4
    relu_node = stree.get_node("ReLU")
    relu_node_1 = stree.get_node("ReLU_1")
    relu_node_2 = stree.get_node("ReLU_2")
    subnet_node = stree.get_node("CellListSubNet")
    assert relu_node is not None
    assert relu_node_1 is not None
    assert relu_node_2 is not None
    assert subnet_node is not None
    # insert node into celllist
    new_node = Node.create_call_cell(nn.Tanhshrink(), [relu_node.get_targets()[0]], name="new_node0")
    stree.insert(stree.after(relu_node), new_node)
    assert len(blocks.get_handler().nodes()) == 5
    assert stree.get_node("new_node0") is not None
    # erase node from celllist
    stree.erase(relu_node_1)
    assert len(blocks.get_handler().nodes()) == 4
    assert stree.get_node("ReLU_1") is None
    # replace node in celllist
    new_node_1 = Node.create_call_cell(nn.Tanhshrink(), [relu_node.get_targets()[0]], name="new_node1")
    stree.replace(relu_node_2, [new_node_1])
    assert len(blocks.get_handler().nodes()) == 4
    assert stree.get_node("ReLU_2") is None
    assert stree.get_node("new_node1") is not None
    # check codes
    codes = stree.get_code()
    assert codes.count("self.blocks[3] = CellListSubNetOpt(self.blocks[3])") == 1
    assert codes.count("self.blocks.insert(1, self.new_node0)") == 1
    assert codes.count("del self.blocks[2]") == 2 # del RELU_1 & replace RELU_2
    assert codes.count("self.blocks.insert(3, self.new_node1)") == 1
    assert codes.count("self_blocks = self.blocks") == 1
    assert codes.count("for block in self_blocks:") == 1
    assert codes.count("x = block(x)") == 1

class ListOfCellSubNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(x)

class ListOfCellNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 16, 3)
        self.blocks = []
        for _ in range(3):
            block = nn.ReLU()
            self.blocks.append(block)
        self.blocks.append(ListOfCellSubNet())

    def construct(self, x):
        x = self.conv(x)
        for block in self.blocks:
            x = block(x)
        return x

def test_list_of_cells():
    """
    Feature: for parser for list of cells.
    Description: parse for statement when the iterator is list of cells, then insert, erase and replace nodes in it.
    Expectation: Success.
    """
    net = ListOfCellNet()
    stree = SymbolTree.create(net)
    # check for nodes
    for_node = stree.get_node("for_node")
    assert for_node.get_node_type() == NodeType.ControlFlow
    assert len(for_node.get_handler().nodes()) == 1
    # get nodes in celllist
    blocks = stree.get_node("self.blocks")
    assert len(blocks.get_handler().nodes()) == 4
    relu_node = stree.get_node("ReLU")
    relu_node_1 = stree.get_node("ReLU_1")
    relu_node_2 = stree.get_node("ReLU_2")
    subnet_node = stree.get_node("ListOfCellSubNet")
    assert relu_node is not None
    assert relu_node_1 is not None
    assert relu_node_2 is not None
    assert subnet_node is not None
    # insert node into celllist
    new_node = Node.create_call_cell(nn.Tanhshrink(), [relu_node.get_targets()[0]], name="new_node0")
    stree.insert(stree.after(relu_node), new_node)
    assert len(blocks.get_handler().nodes()) == 5
    assert stree.get_node("new_node0") is not None
    # erase node from celllist
    stree.erase(relu_node_1)
    assert len(blocks.get_handler().nodes()) == 4
    assert stree.get_node("ReLU_1") is None
    # replace node in celllist
    new_node_1 = Node.create_call_cell(nn.Tanhshrink(), [relu_node.get_targets()[0]], name="new_node1")
    stree.replace(relu_node_2, [new_node_1])
    assert len(blocks.get_handler().nodes()) == 4
    assert stree.get_node("ReLU_2") is None
    assert stree.get_node("new_node1") is not None
    # check codes
    codes = stree.get_code()
    assert codes.count("self.blocks[3] = ListOfCellSubNetOpt(self.blocks[3])") == 1
    assert codes.count("self.blocks.insert(1, self.new_node0)") == 1
    assert codes.count("del self.blocks[2]") == 2 # del RELU_1 & replace RELU_2
    assert codes.count("self.blocks.insert(3, self.new_node1)") == 1
    assert codes.count("self_blocks = self.blocks") == 1
    assert codes.count("for block in self_blocks:") == 1
    assert codes.count("x = block(x)") == 1
