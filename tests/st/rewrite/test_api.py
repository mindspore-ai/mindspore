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
from mindspore.rewrite import SymbolTree, Node, ScopedValue
from mindspore import Tensor
from mindspore.common.api import _cell_graph_executor
import numpy as np
import pytest
from tests.mark_utils import arg_mark


class LeNet5(nn.Cell):
    """
    Args:
        num_class (int): Number of classes. Default: 10.
        num_channel (int): Number of channels. Default: 1.
    """
    def __init__(self, num_class=10, num_channel=1, include_top=True):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.include_top = include_top
        if self.include_top:
            self.flatten = nn.Flatten()
            self.fc1 = nn.Dense(16 * 5 * 5, 120)
            self.fc2 = nn.Dense(120, 84)
            self.fc3 = nn.Dense(84, num_class)

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        if not self.include_top:
            return x
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_rewrite_apis():
    """
    Feature: Test rewrite apis.
    Description: Test rewrite SymbolTree and Node apis.
    Expectation: success.
    """
    net = LeNet5()
    stree = SymbolTree.create(net)
    assert isinstance(stree, SymbolTree) is True
    assert len(list(stree.nodes())) == 16
    conv1_node = stree.get_node('conv1')
    assert isinstance(conv1_node, Node) is True
    node_name = conv1_node.get_name()
    assert node_name == 'conv1'
    position = stree.after(conv1_node)
    new_node = Node.create_call_cell(cell=nn.ReLU(), targets=['x_1'],
                                     args=[ScopedValue.create_naming_value('x')], name='new_relu')
    for user in conv1_node.get_users():
        user.set_arg(0, new_node.get_targets()[0])
    stree.insert(position, new_node)
    assert conv1_node.get_users()[0] == new_node
    assert new_node.get_inputs()[0] == conv1_node
    assert len(list(stree.nodes())) == 17
    conv2_node = stree.get_node('conv2')
    position = stree.before(conv2_node)
    new_node2 = Node.create_call_cell(cell=nn.ReLU(), targets=['x_2'],
                                      args=[ScopedValue.create_naming_value('x')], name='new_relu2')
    conv2_node.set_arg_by_node(0, new_node2, 0)
    stree.insert(position, new_node2)
    assert new_node2.get_users()[0] == conv2_node
    assert conv2_node.get_inputs()[0] == new_node2
    relu_node = stree.get_node("relu")
    assert len(list(stree.nodes())) == 18
    assert "relu" in [node.get_name() for node in stree.nodes()]
    stree.erase(relu_node)
    assert len(list(stree.nodes())) == 17
    assert "relu" not in [node.get_name() for node in stree.nodes()]
    new_node3 = Node.create_call_cell(cell=nn.Flatten(), targets=[stree.unique_name('x')],
                                      args=[ScopedValue.create_naming_value('x')], name='new_flatten')
    assert new_node3.get_targets()[0] == ScopedValue.create_naming_value('x_3')
    flatten_node = None
    for node in stree.nodes():
        if node.get_instance_type() == nn.Flatten:
            flatten_node = node
            break
    assert flatten_node is not None
    for user in flatten_node.get_users():
        user.set_arg_by_node(0, new_node3, 0)
    assert "flatten" in [node.get_name() for node in stree.nodes()]
    stree.replace(flatten_node, [new_node3])
    assert "flatten" not in [node.get_name() for node in stree.nodes()]
    assert "new_flatten" in [node.get_name() for node in stree.nodes()]
    codes = stree.get_code()
    assert codes.find("self.new_relu")
    assert codes.find("self.new_relu2")
    assert codes.find("self.new_relu3")
    net = stree.get_network()
    data_in = Tensor(np.ones([1, 1, 32, 32]).astype(np.float32) * 0.01)
    _cell_graph_executor.compile(net, data_in)
