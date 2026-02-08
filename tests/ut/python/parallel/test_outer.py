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
from mindspore.parallel.shard import Layout
from mindspore.graph.api import _cell_graph_executor
from mindspore.ops.auto_generate.gen_ops_prim import Outer, Mul

def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")

class OuterNet0(nn.Cell):
    def __init__(self, strategy=None):
        super().__init__()
        if strategy:
            self.outer_op = Outer().shard(strategy)
        else:
            self.outer_op = Outer()

    def construct(self, input_data, vec2):
        out = self.outer_op(input_data, vec2)
        return out


class OuterNet1(nn.Cell):
    def __init__(self, strategy):
        super().__init__()
        self.mul = Mul().shard(strategy)
        self.outer_op = Outer()

    def construct(self, input_data, vec2):
        out = self.mul(input_data, input_data)
        out = self.outer_op(out, vec2)
        return out[0]


def compile_graph(net, device_num, parallel_mode, input_data, vec2, search_mode="sharding_propagation"):
    context.set_auto_parallel_context(device_num=device_num, global_rank=0, parallel_mode=parallel_mode)
    net.set_train()
    phase, _ = _cell_graph_executor.compile(net, input_data, vec2)
    return phase


def test_outer_shard():
    """
    Feature: distribute operator outer in semi auto parallel.
    Description: basic
    Expectation: compile done without error.
    """
    context.set_context(save_graphs=True)
    strategy = ((2,), (2,))
    net = OuterNet0(strategy)
    input_data = Tensor(np.ones([128]), dtype=ms.int32)
    vec2 = Tensor(np.ones([128]), dtype=ms.int32)
    compile_graph(net, 8, "semi_auto_parallel", input_data, vec2)


def test_outer_layout():
    """
    Feature: distribute operator outer in semi auto parallel.
    Description: basic
    Expectation: compile done without error.
    """
    context.set_context(save_graphs=True)
    layout = Layout((2, 2, 2), ("dp", "mp", "tp"))
    strategy = (layout("dp"), layout("mp"))
    net = OuterNet0(strategy)
    input_data = Tensor(np.ones([128]), dtype=ms.int32)
    vec2 = Tensor(np.ones([128]), dtype=ms.int32)
    compile_graph(net, 8, "semi_auto_parallel", input_data, vec2)

def test_outer_dynamic_shard():
    """
    Feature: distribute operator outer in semi auto parallel.
    Description:outer(1): dynamic shape (can be divided)
    Expectation: compile done without error.
    """
    context.set_context(save_graphs=True)
    strategy = ((2,), (2,))
    s1 = Symbol(divisor=2)
    s2 = Symbol(divisor=2)
    input_data = Tensor(shape=[s1], dtype=ms.int32)
    vec2 = Tensor(shape=[s2], dtype=ms.int32)
    net = OuterNet0(strategy)
    compile_graph(net, 8, "semi_auto_parallel", input_data, vec2)

def test_outer_dynamic_layout():
    """
    Feature: distribute operator outer in semi auto parallel.
    Description:outer(1): dynamic shape (can be divided)
    Expectation: compile done without error.
    """
    context.set_context(save_graphs=True)
    layout = Layout((2, 2, 2), ("dp", "mp", "tp"))
    strategy = (layout("dp"), layout("mp"))
    s1 = Symbol(divisor=2)
    s2 = Symbol(divisor=2)
    input_data = Tensor(shape=[s1], dtype=ms.int32)
    vec2 = Tensor(shape=[s2], dtype=ms.int32)
    net = OuterNet0(strategy)
    compile_graph(net, 8, "semi_auto_parallel", input_data, vec2)

def test_outer_auto_parallel_sharding_propagation0():
    """
    Feature: distribute operator outer in auto parallel.
    Description: sharding_propagation for outer
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                      search_mode="sharding_propagation")
    strategy = ((2,), (2,))
    input_data = Tensor(np.ones([4]), dtype=ms.float32)
    vec2 = Tensor(np.ones([8]), dtype=ms.float32)
    net = OuterNet1(strategy=strategy)
    _cell_graph_executor.compile(net, input_data, vec2, phase="train")
    strategies = _cell_graph_executor._get_shard_strategy(net)
    for (k, v) in strategies.items():
        if re.search("Outer", k) is not None:
            assert v == [[2], [2]]

def test_outer_repeat_tensor_map():
    """
    Feature: test parallel error strategy.
    Description: repeat tensor_map.
    Expectation: raise RuntimeError.
    """
    context.set_context(save_graphs=True)
    layout = Layout((2, 2, 2), ("dp", "mp", "tp"))
    strategy = (layout("dp"), layout("dp"))
    net = OuterNet0(strategy)
    input_data = Tensor(np.ones([128]), dtype=ms.int32)
    vec2 = Tensor(np.ones([128]), dtype=ms.int32)
    with pytest.raises(RuntimeError):
        compile_graph(net, 8, "semi_auto_parallel", input_data, vec2)

def test_outer_interleaved_parallel_1():
    """
    Feature: test parallel error input_data strategy.
    Description: interleaved parallel.
    Expectation: raise RuntimeError.
    """
    context.set_context(save_graphs=True)
    layout = Layout((2, 2, 2, 2), ("dp", "mp", "tp", "interleaved_parallel"))
    strategy = (layout("interleaved_parallel"), layout("dp"))
    net = OuterNet0(strategy)
    input_data = Tensor(np.ones([128]), dtype=ms.int32)
    vec2 = Tensor(np.ones([128]), dtype=ms.int32)
    with pytest.raises(RuntimeError):
        compile_graph(net, 8, "semi_auto_parallel", input_data, vec2)

def test_outer_interleaved_parallel_2():
    """
    Feature: test parallel error vec2 strategy.
    Description: interleaved parallel.
    Expectation: raise RuntimeError.
    """
    context.set_context(save_graphs=True)
    layout = Layout((2, 2, 2, 2), ("dp", "mp", "tp", "interleaved_parallel"))
    strategy = (layout("dp"), layout("interleaved_parallel"))
    net = OuterNet0(strategy)
    input_data = Tensor(np.ones([128]), dtype=ms.int32)
    vec2 = Tensor(np.ones([128]), dtype=ms.int32)
    with pytest.raises(RuntimeError):
        compile_graph(net, 8, "semi_auto_parallel", input_data, vec2)
