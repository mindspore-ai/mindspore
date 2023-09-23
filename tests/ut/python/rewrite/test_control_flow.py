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
from mindspore import nn, ops


def external_func(x):
    x = ops.abs(x)
    return x


class SubSubNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.add = ops.Add()

    def construct(self, x):
        x = self.add(x, x)
        return x


class SubNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.subsubnet = SubSubNet()
        self.abs = ops.Abs()

    def construct(self, x):
        if self.abs(x):
            x = external_func(x)
            x = self.relu(x)
        else:
            return self.subsubnet(x)
        return x


class MyNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.abs = ops.Abs()
        self.sub_net = SubNet()

    def construct(self, x):
        if self.abs(x): # pylint: disable=no-else-return
            return self.internal_func(x)
        else:
            x = self.abs(x)
            x = self.sub_net(self.abs(x))
        return x

    def internal_func(self, x):
        if self.abs(x):
            x = self.external_func(self.abs(x))
        return x


def test_rewrite_control_flow():
    """
    Feature: Test rewrite control flow node.
    Description: Test rewrite parse `if` statement to control flow node.
    Expectation: Success.
    """
    net = MyNet()
    stree = SymbolTreeApi.create(net)
    codes = stree.get_code()
    # test if statement flatten results
    assert codes.count("internal_func_var = self.internal_func(x)") == 1
    assert codes.count("return internal_func_var") == 1
    assert codes.count("subsubnet_var = self.subsubnet(x)") == 1
    assert codes.count("return subsubnet_var") == 1
    # test control flow nodes
    if_node_1 = stree.get_node("if_node_1")
    assert if_node_1.get_node_type() == NodeType.ControlFlow
    assert len(if_node_1.get_handler().nodes()) == 2
    else_node = stree.get_node("else_node")
    assert else_node.get_node_type() == NodeType.ControlFlow
    assert len(else_node.get_handler().nodes()) == 3
    if_node = stree.get_node("if_node")
    assert if_node.get_node_type() == NodeType.ControlFlow
    assert len(if_node.get_handler().nodes()) == 2
    sub_stree = SymbolTreeApi(stree.get_node("sub_net").get_handler().symbol_tree)
    sub_if_node = sub_stree.get_node("if_node")
    assert sub_if_node.get_node_type() == NodeType.ControlFlow
    assert len(sub_if_node.get_handler().nodes()) == 2
    sub_else_node = sub_stree.get_node("else_node")
    assert sub_else_node.get_node_type() == NodeType.ControlFlow
    assert len(sub_else_node.get_handler().nodes()) == 2
    # insert node to control flow
    node_abs_2 = stree.get_node("abs_2")
    new_relu = Node.create_call_cell(nn.ReLU(), targets=node_abs_2.get_targets(), args=node_abs_2.get_targets(),
                                     name="new_relu")
    stree.insert(stree.after(node_abs_2), new_relu)
    codes = stree.get_code()
    assert codes.count("abs_var_3 = self.new_relu(abs_var_3)") == 1
    # delete node from control flow
    node_abs_3 = stree.get_node("abs_3")
    stree.erase(node_abs_3)
    codes = stree.get_code()
    assert codes.count("x = self.abs(x)") == 0
    # replace node in control flow
    sub_relu = sub_stree.get_node("relu")
    new_relu_2 = Node.create_call_cell(nn.ReLU(), targets=node_abs_3.get_targets(), args=node_abs_3.get_targets(),
                                       name="new_relu_2")
    sub_stree.replace(sub_relu, [new_relu_2])
    codes = stree.get_code()
    assert codes.count("x = self.relu(x)") == 0
    assert codes.count("x = self.new_relu_2(x)") == 1
