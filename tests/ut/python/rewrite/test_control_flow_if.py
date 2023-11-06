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
from mindspore import nn, ops, context


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
            x = self.relu(x)
            x = self.sub_net(x)
        return x

    def internal_func(self, x):
        x = self.external_func(x)
        return x


def test_if_control_flow():
    """
    Feature: Test rewrite if control flow node.
    Description: Test rewrite parse `if` statement to control flow node.
    Expectation: Success.
    """
    net = MyNet()
    stree = SymbolTreeApi.create(net)
    codes = stree.get_code()
    # test if test statement flatten results
    assert codes.count("internal_func_var = self.internal_func(x)") == 1
    assert codes.count("return internal_func_var") == 1
    assert codes.count("subsubnet_var = self.subsubnet(x)") == 1
    assert codes.count("return subsubnet_var") == 1
    # test nodes in if
    if_node = stree.get_node("if_node")
    assert if_node.get_node_type() == NodeType.ControlFlow
    assert len(if_node.get_handler().nodes()) == 2
    else_node = stree.get_node("else_node")
    assert else_node.get_node_type() == NodeType.ControlFlow
    assert len(else_node.get_handler().nodes()) == 2
    sub_stree = SymbolTreeApi(stree.get_node("sub_net").get_handler().symbol_tree)
    sub_if_node = sub_stree.get_node("if_node")
    assert sub_if_node.get_node_type() == NodeType.ControlFlow
    assert len(sub_if_node.get_handler().nodes()) == 2
    sub_else_node = sub_stree.get_node("else_node")
    assert sub_else_node.get_node_type() == NodeType.ControlFlow
    assert len(sub_else_node.get_handler().nodes()) == 2
    # insert node to if node
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
    assert codes.count("x = self.relu(x)") == 1
    # replace node in if node
    sub_relu = sub_stree.get_node("relu")
    new_relu_2 = Node.create_call_cell(nn.ReLU(), targets=sub_relu.get_targets(), args=sub_relu.get_targets(),
                                       name="new_relu_2")
    sub_stree.replace(sub_relu, [new_relu_2])
    codes = stree.get_code()
    assert codes.count("x = self.relu(x)") == 0
    assert codes.count("x = self.new_relu_2(x)") == 1
    # erase if node
    codes = stree.get_code()
    stree.erase(if_node)
    codes = stree.get_code()
    assert codes.count("pass") == 1
    assert codes.count("internal_func_var = self.internal_func(x)") == 0
    assert codes.count("return internal_func_var") == 0
    stree.erase(else_node)
    codes = stree.get_code()
    assert codes.count("pass") == 0
    assert codes.count("if abs_var:") == 0
    assert codes.count("x = self.new_relu(x)") == 0
    assert codes.count("x = self.sub_net(x)") == 0

class MyNet2(nn.Cell):
    DEVICE_TARGET = "CPU"

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.abs = ops.Abs()
        self.relu1 = nn.ReLU()
        self.abs1 = ops.Abs()
        self.relu2 = nn.ReLU()
        self.abs2 = ops.Abs()
        self.relu3 = nn.ReLU()
        self.abs3 = ops.Abs()
        self.relu4 = nn.ReLU()
        self.abs4 = ops.Abs()
        self.abs5 = ops.Abs()
        self.device_target = "CPU"

    def construct(self, x):
        if self.abs(x):
            x = self.abs(x)
        else:
            x = self.relu(x)

        if context.get_context("device_target") == "CPU":
            x = self.abs1(x)
        else:
            x = self.relu1(x)

        if MyNet2.DEVICE_TARGET == "CPU":
            x = self.abs2(x)
        else:
            x = self.relu2(x)

        if self.device_target == "CPU":
            x = self.abs3(x)
        else:
            x = self.relu3(x)

        if self.device_target == "Ascend":
            x = self.abs4(x)
        else:
            x = self.relu4(x)

        if self.device_target == "Ascend":
            x = self.abs5(x)

        return x

def test_flatten_if_control_flow():
    """
    Feature: Test flatten rewrite if control flow node.
    Description: Test flatten static if control flow node.
    Expectation: Success.
    """
    net = MyNet2()
    stree = SymbolTreeApi.create(net)
    stree.flatten_static_if_control_flow()
    codes = stree.get_code()
    assert codes.count("x = self.abs(x)") == 1
    assert codes.count("x = self.relu(x)") == 1
    assert codes.count("x = self.abs1(x)") == 1
    assert codes.count("x = self.relu1(x)") == 0
    assert codes.count("x = self.abs2(x)") == 1
    assert codes.count("x = self.relu2(x)") == 0
    assert codes.count("x = self.abs3(x)") == 1
    assert codes.count("x = self.relu3(x)") == 0
    assert codes.count("x = self.abs4(x)") == 0
    assert codes.count("x = self.relu4(x)") == 1
    assert codes.count("x = self.abs5(x)") == 0
