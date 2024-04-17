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
from mindspore.rewrite import SymbolTree, ScopedValue, ValueType, Node
from mindspore.common.initializer import Normal
from mindspore.common.api import _cell_graph_executor


class SimpleNet(nn.Cell):
    def __init__(self, num_class=10, num_channel=1):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))
        self.var = 10

    def construct(self, x):
        x = self.conv1(x)
        x = x
        y = self.var
        y = y * 5
        y = y and True
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MyCell(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv = nn.Dense(5, 16)

    def construct(self, x, y):
        x = self.conv(x)
        x = mindspore.ops.Add()(x, y)
        return x


def add_conv_before_flatten(stree: SymbolTree):
    new_conv_node = None
    for node in stree.nodes():
        if node.get_instance_type() == mindspore.nn.Flatten:
            position = stree.before(node)
            new_conv = nn.Conv2d(16, 16, 3)
            new_conv_node = Node.create_call_cell(new_conv, targets=['x_1'], name='new_conv',
                                                  args=[ScopedValue.create_naming_value('self_max_po')])
            stree.insert(position, new_conv_node)
            break
    if new_conv_node is not None:
        for node in stree.nodes():
            if node.get_instance_type() == mindspore.nn.Flatten:
                inputs = node.get_inputs()
                assert len(inputs) == 1
                new_conv_node.set_arg_by_node(0, inputs[0])


def add_my_cell_after_x_12(stree: SymbolTree):
    for node in stree.nodes():
        targets = node.get_targets()
        if not targets:
            continue
        assert targets[0].type == ValueType.NamingValue
        target = str(targets[0])
        if target == "x_12":
            position = stree.after(node)
            custom_cell = MyCell()
            bias = Tensor(1, mindspore.int32)
            new_custom_node = Node.create_call_cell(custom_cell, targets=['nx2'],
                                                    args=[ScopedValue.create_naming_value('nx3'),
                                                          ScopedValue.create_variable_value(bias)], name='my_cell')
            stree.insert(position, new_custom_node)
            new_custom_node.set_arg(0, "x_12")
            break


def erase_node_x_11(stree: SymbolTree):
    return_node = None
    for node in stree.nodes():
        if not node.get_targets():
            return_node = node
            break
    assert return_node is not None
    for node in stree.nodes():
        targets = node.get_targets()
        if not targets:
            continue
        assert targets[0].type == ValueType.NamingValue
        target = str(targets[0])
        if target == "x_11":
            stree.set_output(0, "x_10")
            stree.erase_node(node)
            break


def transform(stree: SymbolTree):
    add_conv_before_flatten(stree)
    erase_node_x_11(stree)


def test_simple_net():
    """
    Feature: Module rewrite.
    Description: Resolve a simple network by rewrite and do some transform on it.
    Expectation: Result of rewrite can be compiled.
    """
    net = SimpleNet(10)
    stree = SymbolTree.create(net)
    transform(stree)
    net_opt = stree.get_network()
    data_in = Tensor(np.ones([1, 1, 32, 32]), mindspore.float32)
    _cell_graph_executor.compile(net_opt, data_in)
