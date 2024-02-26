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
from mindspore.rewrite import SymbolTree as SymbolTreeApi
from mindspore.rewrite import NodeType, Node
from mindspore import nn


class MyNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def construct(self, x):
        count = 10
        while count > 0:
            x = self.relu(x)
            count -= 1
        else: # pylint: disable=useless-else-on-loop
            count = 0
        return x


def test_while_control_flow():
    """
    Feature: Test rewrite while control flow node.
    Description: Test rewrite parse `while` statement to control flow node.
    Expectation: Success.
    """
    net = MyNet()
    stree = SymbolTreeApi.create(net)
    codes = stree.get_code()
    # test while test statement flatten results
    assert codes.count("while (count > 0):") == 1
    assert codes.count("x = self.relu(x)") == 1
    assert codes.count("count -= 1") == 1
    assert codes.count("else:") == 1
    assert codes.count("count = 0") == 1
    # test nodes in while
    while_node = stree.get_node("while_node")
    assert while_node.get_node_type() == NodeType.ControlFlow
    assert len(while_node.get_handler().nodes()) == 2
    while_else_node = stree.get_node("while_else_node")
    assert while_else_node.get_node_type() == NodeType.ControlFlow
    assert len(while_else_node.get_handler().nodes()) == 1
    # insert node to while node
    node_relu = stree.get_node("relu")
    new_relu = Node.create_call_cell(nn.ReLU(), targets=node_relu.get_targets(), args=node_relu.get_targets(),
                                     name="new_relu")
    stree.insert(stree.after(node_relu), new_relu)
    codes = stree.get_code()
    assert codes.count("x = self.new_relu(x)") == 1
    # delete node from if node
    node_relu = stree.get_node("relu")
    stree.erase(node_relu)
    codes = stree.get_code()
    assert codes.count("x = self.relu(x)") == 0
    # replace node in if node
    new_relu_2 = Node.create_call_cell(nn.ReLU(), targets=new_relu.get_targets(), args=new_relu.get_targets(),
                                       name="new_relu_2")
    stree.replace(new_relu, [new_relu_2])
    codes = stree.get_code()
    assert codes.count("x = self.new_relu(x)") == 0
    assert codes.count("x = self.new_relu_2(x)") == 1
    # erase while node
    codes = stree.get_code()
    stree.erase(while_node)
    codes = stree.get_code()
    assert codes.count("while (count > 0):") == 1
    assert codes.count("x = self.new_relu_2(x)") == 0
    assert codes.count("count -= 1") == 0
    assert codes.count("pass") == 1
    assert codes.count("else:") == 1
    assert codes.count("count = 0") == 1
    stree.erase(while_else_node)
    codes = stree.get_code()
    assert codes.count("while (count > 0):") == 0
    assert codes.count("x = self.relu(x)") == 0
    assert codes.count("count -= 1") == 0
    assert codes.count("pass") == 0
    assert codes.count("else:") == 0
    assert codes.count("count = 0") == 0
