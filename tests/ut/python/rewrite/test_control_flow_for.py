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
from mindspore.rewrite import SymbolTree, Node, NodeType
from mindspore import nn

class SubNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(x)

class MyNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 16, 3)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        for _ in range(3):
            x = self.relu(x)
            x = self.relu(x)
        return x

def test_for_control_flow():
    """
    Feature: Test rewrite for control flow node.
    Description: Test rewrite parse `for` statement to control flow node.
    Expectation: Success.
    """
    net = MyNet()
    stree = SymbolTree.create(net)
    # get nodes in for node
    for_node = stree.get_node("for_node")
    assert for_node is not None
    assert for_node.get_node_type() == NodeType.ControlFlow
    assert len(for_node.get_handler().nodes()) == 2
    relu_node = stree.get_node("relu")
    assert relu_node is not None
    assert relu_node.get_handler().get_node_manager() == for_node.get_handler()
    relu_node_1 = stree.get_node("relu_1")
    assert relu_node_1 is not None
    assert relu_node_1.get_handler().get_node_manager() == for_node.get_handler()
    codes = stree.get_code()
    assert codes.count("range_var = range(3)") == 1
    assert codes.count("for _ in range_var:") == 1
    assert codes.count("x = self.relu(x)") == 2
    # insert node into for node
    new_node = Node.create_call_cell(nn.Tanhshrink(), targets=[relu_node.get_targets()[0]],
                                     args=[relu_node.get_targets()[0]], name="new_node0")
    stree.insert(stree.after(relu_node), new_node)
    assert len(for_node.get_handler().nodes()) == 3
    assert stree.get_node("new_node0") is not None
    codes = stree.get_code()
    assert codes.count("self.new_node0 = obj.new_node0") == 1
    assert codes.count("x = self.new_node0(x)") == 1
    # erase node from for node
    stree.erase(relu_node)
    assert len(for_node.get_handler().nodes()) == 2
    assert stree.get_node("relu") is None
    codes = stree.get_code()
    assert codes.count("x = self.relu(x)") == 1
    # replace node in for node
    new_node_1 = Node.create_call_cell(nn.Tanhshrink(), targets=[relu_node_1.get_targets()[0]],
                                       args=[relu_node_1.get_targets()[0]], name="new_node1")
    stree.replace(relu_node_1, [new_node_1])
    assert len(for_node.get_handler().nodes()) == 2
    assert stree.get_node("relu_1") is None
    assert stree.get_node("new_node1") is not None
    codes = stree.get_code()
    assert codes.count("x = self.relu(x)") == 0
    assert codes.count("self.new_node1 = obj.new_node1") == 1
    assert codes.count("x = self.new_node1(x)") == 1
