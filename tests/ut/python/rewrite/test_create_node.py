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

import mindspore
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.rewrite import SymbolTree
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

    new_node = stree.create_call_function(F.abs, ["x"], "abc")
    for node in stree.nodes():
        if node.get_instance_type() == P.ReduceMean:
            pos = stree.after(node)
            stree.insert(pos, new_node)
            new_node_1 = stree.create_call_function(F.abs, ["x"], node)
            stree.insert(pos, new_node_1)
            new_node_2 = stree.create_call_function(F.scalar_to_tensor, ["x"], 2, dtype=mindspore.float16)
            stree.insert(pos, new_node_2)
            new_node_3 = stree.create_call_function(F.scalar_to_tensor, ["x"], 2, mindspore.float16)
            stree.insert(pos, new_node_3)


def test_create_call_function_fail():
    """
    Feature: Create a call function node.
    Description: Call create_call_function to create a call function node.
    Expectation: raise TypeError.
    """
    net = SimpleNet()
    stree = SymbolTree.create(net)

    for node in stree.nodes():
        if node.get_instance_type() == P.ReduceMean:
            with pytest.raises(TypeError):
                _ = stree.create_call_function(F.cast, ["x"], node, mindspore.float16)


def test_create_call_function_fail_0():
    """
    Feature: Create a call function node.
    Description: Call create_call_function to create a call function node.
    Expectation: raise TypeError.
    """
    net = SimpleNet()
    stree = SymbolTree.create(net)

    for node in stree.nodes():
        if node.get_instance_type() == P.ReduceMean:
            with pytest.raises(TypeError):
                _ = stree.create_call_function(F.scalar_to_tensor, ["x"], 2, dtype=mindspore.int32)


def test_create_call_function_fail_1():
    """
    Feature: Create a call function node.
    Description: Call create_call_function to create a call function node.
    Expectation: raise TypeError.
    """
    net = SimpleNet()
    stree = SymbolTree.create(net)

    with pytest.raises(TypeError):
        _ = stree.create_call_function(F.abs, ["x"], "abc", [1, 2])
    with pytest.raises(TypeError):
        _ = stree.create_call_function(F.abs, "x")
    with pytest.raises(TypeError):
        _ = stree.create_call_function(F.scalar_to_tensor, ["x"], "2", dtype=mindspore.int32)
    with pytest.raises(TypeError):
        t = mindspore.Tensor(1, mindspore.int32)
        _ = stree.create_call_function(F.scalar_to_tensor, ["x"], t, mindspore.float16)
