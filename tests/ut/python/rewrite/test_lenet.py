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
"""LeNet."""
from collections import OrderedDict

import mindspore.nn as nn
from mindspore.common.initializer import Normal
from mindspore.rewrite import SymbolTree, PatternEngine, Replacement, PatternNode, Node


class LeNet5(nn.Cell):
    def __init__(self, num_class=10, num_channel=1, include_top=True):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.include_top = include_top
        if self.include_top:
            self.flatten = nn.Flatten()
            self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
            self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
            self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))

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


class ConvActReplace(Replacement):
    def build(self, pattern: PatternNode, is_chain_pattern: bool, matched: OrderedDict) -> [Node]:
        conv_p = pattern.get_inputs()[0]
        conv_node: Node = matched.get(conv_p.name())
        conv: nn.Conv2d = conv_node.get_instance()
        newconv = nn.Conv2dBnAct(conv.in_channels,
                                 conv.out_channels,
                                 conv.kernel_size,
                                 conv.stride,
                                 conv.pad_mode,
                                 conv.padding,
                                 conv.dilation,
                                 conv.group,
                                 conv.has_bias,
                                 conv.weight_init,
                                 conv.bias_init,
                                 False,
                                 activation="relu")
        newconv_node = Node.create_call_cell(newconv, conv_node.get_targets(), conv_node.get_args(),
                                             conv_node.get_kwargs(), "Conv2dBnAct")
        return [newconv_node]


class ConvReLUPattern(PatternEngine):
    def __init__(self):
        super().__init__([nn.Conv2d, nn.ReLU], ConvActReplace())


def test_lenet():
    """
    Feature: Test PatternEngine.
    Description: Test PatternEngine on Lenet5.
    Expectation: Success.
    """
    net = LeNet5(10)
    stree = SymbolTree.create(net)
    original_nodes_size = len(stree.get_handler()._nodes)
    ConvReLUPattern().apply(stree)
    assert len(stree.get_handler()._nodes) == original_nodes_size - 2
