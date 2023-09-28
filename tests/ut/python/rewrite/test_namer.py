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
from mindspore.nn import Cell, Conv2d, ReLU
from mindspore.rewrite import SymbolTree


class Network(Cell):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = Conv2d(1, 6, 5, pad_mode='valid')
        self.conv2 = Conv2d(6, 16, 5, pad_mode='valid')
        self.relu = ReLU()
        self.relu_ = ReLU()

    def construct(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x_1)
        x_1 = self.relu(x_2)
        x = self.relu(x_1)
        x = self.relu(x)
        x = self.relu_(x)
        return x


def test_target_namer():
    """
    Feature: Python api `get_real_arg` of `TargetNamer` of Rewrite.
    Description: Construct a network and check topological relation.
    Expectation: Success.
    """
    stree = SymbolTree.create(Network())
    relu1 = stree.get_node('relu_1')
    assert relu1
    inputs = relu1.get_inputs()
    assert len(inputs) == 1
    input0 = inputs[0]
    assert input0.get_name() == 'relu'
    relu_ = stree.get_node('relu_')
    assert relu_.get_name() == 'relu_'
    stree.get_network()
