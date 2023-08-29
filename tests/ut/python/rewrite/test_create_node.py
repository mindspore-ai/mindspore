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
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.rewrite import SymbolTree, ScopedValue, Node
import mindspore.nn as nn
import pytest


class SimpleNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.mul = P.Mul()
        self.dense = nn.Dense(in_channels=32, out_channels=32, weight_init="ones")
        self.mean = P.ReduceMean(keep_dims=False)
        self.split = P.Split(axis=1, output_num=3)

    def construct(self, x, y):
        x = self.dense(x)
        y, _, _ = self.split(y)
        y = self.mean(y, (2, 3))
        x = self.mul(x, 1)
        return x, y


def test_create_call_function_ok():
    """
    Feature: Create a call function node.
    Description: Call create_call_function to create a call function node.
    Expectation: Success.
    """
    net = SimpleNet()
    stree = SymbolTree.create(net)

    new_node = Node.create_call_function(F.abs, ["x"], [ScopedValue.create_variable_value("abc")])
    for node in stree.nodes():
        if node.get_instance_type() == P.ReduceMean:
            pos = stree.after(node)
            stree.insert(pos, new_node)
            new_node_1 = Node.create_call_function(F.abs, ["x"], node.get_targets())
            stree.insert(pos, new_node_1)
            new_node_2 = Node.create_call_function(F.scalar_to_tensor, ["x"],
                                                   [ScopedValue.create_variable_value("abc")],
                                                   {"dtype": ScopedValue.create_naming_value("float16", "mindspore")})
            stree.insert(pos, new_node_2)
            new_node_3 = Node.create_call_function(F.scalar_to_tensor, ["x"],
                                                   [ScopedValue.create_variable_value(2)],
                                                   {"dtype": ScopedValue.create_naming_value("float16", "mindspore")})
            stree.insert(pos, new_node_3)


def test_create_call_function_fail():
    """
    Feature: Create a call function node.
    Description: Call create_call_function to create a call function node.
    Expectation: raise TypeError.
    """
    with pytest.raises(TypeError):
        _ = Node.create_call_function(F.cast(), ["x"], [ScopedValue.create_variable_value("abc")])
    with pytest.raises(TypeError):
        _ = Node.create_call_function(F.cast, [2], [ScopedValue.create_variable_value("abc")])
    with pytest.raises(TypeError):
        _ = Node.create_call_function(F.cast, ["x"], [2])
    with pytest.raises(TypeError):
        _ = Node.create_call_function(F.cast, ["x"], [ScopedValue.create_variable_value("abc")],
                                      {2: ScopedValue.create_naming_value("float16", "mindspore")})
    with pytest.raises(TypeError):
        _ = Node.create_call_function(F.cast, ["x"], [ScopedValue.create_variable_value("abc")],
                                      {"dtype": 2})
