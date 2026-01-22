# Copyright 2025 Huawei Technologies Co., Ltd
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
import re
import pytest
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore import Symbol
from mindspore.graph.api import _cell_graph_executor
from mindspore.ops.auto_generate.gen_ops_prim import Max, Mul

def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")

class MaxNet0(nn.Cell):
    def __init__(self, strategy=None):
        super().__init__()
        if strategy:
            self.max_op = Max().shard(strategy)
        else:
            self.max_op = Max()

    def construct(self, input_data):
        out = self.max_op(input_data)
        return out


class MaxNet1(nn.Cell):
    def __init__(self, strategy):
        super().__init__()
        self.mul = Mul().shard(strategy)
        self.max_op = Max()

    def construct(self, input_data):
        out = self.mul(input_data, input_data)
        out = self.max_op(out)
        return out

def compile_graph(net, device_num, parallel_mode, input_data, search_mode="sharding_propagation"):
    context.set_auto_parallel_context(device_num=device_num, global_rank=0, parallel_mode=parallel_mode)
    net.set_train()
    phase, _ = _cell_graph_executor.compile(net, input_data)
    return phase


def test_max_shard_basic_0():
    """
    Feature: distribute operator max in semi auto parallel.
    Description: basic
    Expectation: compile done without error.
    """
    context.set_context(save_graphs=True)
    strategy = ((2, 4),)
    net = MaxNet0(strategy)
    input_data = Tensor(np.ones([128, 128]), dtype=ms.int32)
    compile_graph(net, 8, "semi_auto_parallel", input_data)


def test_max_shard_basic_1():
    """
    Feature: distribute operator max in semi auto parallel.
    Description: repeated calc
    Expectation: compile done without error.
    """
    context.set_context(save_graphs=True)
    strategy = ((2, 4),)
    net = MaxNet0(strategy)
    input_data = Tensor(np.ones([128, 128]), dtype=ms.int32)
    compile_graph(net, 16, "semi_auto_parallel", input_data)


def test_max_shard_dynamic_0():
    """
    Feature: distribute operator max in semi auto parallel.
    Description:max(1): dynamic shape (can be divided)
    Expectation: compile done without error.
    """
    context.set_context(save_graphs=True)
    strategy = ((2, 4),)
    s1 = Symbol(divisor=2)
    s2 = Symbol(divisor=4)
    input_data = Tensor(shape=[s1, s2], dtype=ms.int32)
    net = MaxNet0(strategy)
    compile_graph(net, 8, "semi_auto_parallel", input_data)


def test_max_shard_dynamic_1():
    """
    Feature: distribute operator max in semi auto parallel.
    Description:
    Expectation: compile done without error.
    """
    context.set_context(save_graphs=True)
    strategy = ((8, 1),)
    s1 = Symbol(divisor=8)
    s2 = Symbol(divisor=1)
    input_data = Tensor(shape=[s1, s2], dtype=ms.int32)
    net = MaxNet0(strategy)
    compile_graph(net, 8, "semi_auto_parallel", input_data)

def test_max_auto_parallel_sharding_propagation0():
    """
    Feature: distribute operator max in auto parallel.
    Description: sharding propagation for max ops.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                      search_mode="sharding_propagation")
    strategy = ((2, 4), (2, 4))
    input_data = Tensor(np.ones([32, 16]), dtype=ms.float32)
    net = MaxNet1(strategy)
    _cell_graph_executor.compile(net, input_data, phase="train")
    strategies = _cell_graph_executor._get_shard_strategy(net)
    for (k, v) in strategies.items():
        if re.search("Max", k) is not None:
            assert v == [[2, 4],]

def test_max_auto_parallel_sharding_propagation1():
    """
    Feature: distribute operator max in auto parallel.
    Description: sharding propagation for max ops.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                      search_mode="sharding_propagation")
    strategy = ((2, 2, 2), (2, 2, 2))
    input_data = Tensor(np.ones([8, 8, 8]), dtype=ms.float32)
    net = MaxNet1(strategy)
    _cell_graph_executor.compile(net, input_data, phase="train")
    strategies = _cell_graph_executor._get_shard_strategy(net)
    for (k, v) in strategies.items():
        if re.search("Max", k) is not None:
            assert v == [[2, 2, 2],]

def test_max_shard_error():
    """
    Feature: test parallel error strategy
    Description: error strategy
    Expectation: raise RuntimeError
    """
    context.set_context(save_graphs=True)
    strategy = ((2, 4), (1,))
    net = MaxNet0(strategy)
    input_data = Tensor(np.ones([128, 128]), dtype=ms.int32)
    with pytest.raises(RuntimeError):
        compile_graph(net, 16, "semi_auto_parallel", input_data)
