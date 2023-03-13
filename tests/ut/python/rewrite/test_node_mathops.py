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
import numpy as np

import mindspore
from mindspore import Tensor, nn
from mindspore.rewrite import SymbolTree, ScopedValue
from mindspore.common.api import _cell_graph_executor


class SimpleNet(nn.Cell):
    def __init__(self, num_channel=1):
        super(SimpleNet, self).__init__()
        self.conv = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = 0.5 * x
        x = x + x
        x = self.relu(x)
        return x


def erase_node_mathops_add(stree: SymbolTree):
    """ erase addition operation in the symbol tree """
    for node in stree.nodes():
        if node.get_name() == "binop_add":
            for user in node.get_users():
                user.set_arg(0, node.get_args()[0])
            stree.erase_node(node)
            break


def modify_node_multiply_value(stree: SymbolTree):
    """ modify value in multiplication operation in the symbol tree """
    for node in stree.nodes():
        if node.get_name() == "binop_mult":
            node.set_arg(0, ScopedValue.create_variable_value(0.8))
            break


def transform(stree: SymbolTree):
    """ Do some transform on math operations, such as erase add operation, add sub operation and modify values """
    erase_node_mathops_add(stree)
    modify_node_multiply_value(stree)


def test_simple_net():
    """
    Feature: Module rewrite.
    Description: Resolve a simple network by rewrite and do some transform on math operations.
    Expectation: Result of rewrite can be compiled.
    """
    net = SimpleNet()
    stree = SymbolTree.create(net)
    transform(stree)
    net_opt = stree.get_network()
    data_in = Tensor(np.ones([1, 1, 32, 32]), mindspore.float32)
    _cell_graph_executor.compile(net_opt, data_in)
