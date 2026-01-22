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
from mindspore.ops.auto_generate.gen_ops_prim import CumsumExt, Mul

def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")

class CumsumExtNet0(nn.Cell):
    def __init__(self, strategy=None):
        super().__init__()
        if strategy:
            self.cumsum_ext_op = CumsumExt().shard(strategy)
        else:
            self.cumsum_ext_op = CumsumExt()

    def construct(self, input_data, dim):
        out = self.cumsum_ext_op(input_data, dim)
        return out


class CumsumExtNet1(nn.Cell):
    def __init__(self, strategy):
        super().__init__()
        self.mul = Mul().shard(strategy)
        self.cumsum_ext_op = CumsumExt()

    def construct(self, input_data, dim):
        out = self.mul(input_data, input_data)
        out = self.cumsum_ext_op(out, dim)
        return out[0]


def compile_graph(net, device_num, parallel_mode, input_data, dim, search_mode="sharding_propagation"):
    context.set_auto_parallel_context(device_num=device_num, global_rank=0, parallel_mode=parallel_mode)
    net.set_train()
    phase, _ = _cell_graph_executor.compile(net, input_data, dim)
    return phase


def test_cumsum_ext_not_shard_dim_01():
    """
    Feature: distribute operator cumsum_ext in semi auto parallel.
    Description: basic
    Expectation: compile done without error.
    """
    context.set_context(save_graphs=True)
    strategy = ((1, 4),)
    net = CumsumExtNet0(strategy)
    input_data = Tensor(np.ones([128, 128]), dtype=ms.int32)
    dim = 0
    compile_graph(net, 8, "semi_auto_parallel", input_data, dim)

def test_cumsum_ext_not_shard_dim_02():
    """
    Feature: distribute operator cumsum_ext in semi auto parallel.
    Description: basic
    Expectation: compile done without error.
    """
    context.set_context(save_graphs=True)
    strategy = ((2, 1),)
    net = CumsumExtNet0(strategy)
    input_data = Tensor(np.ones([128, 128]), dtype=ms.int32)
    dim = -1
    compile_graph(net, 8, "semi_auto_parallel", input_data, dim)

def test_cumsum_ext_not_shard_dim_layout_01():
    """
    Feature: distribute operator cumsum_ext in semi auto parallel.
    Description: basic
    Expectation: compile done without error.
    """
    context.set_context(save_graphs=True)
    layout = Layout((2, 1, 4), ("ap", "bp", "cp"))
    strategy = (layout("bp", "ap"),)
    net = CumsumExtNet0(strategy)
    input_data = Tensor(np.ones([128, 128]), dtype=ms.int32)
    dim = 0
    compile_graph(net, 8, "semi_auto_parallel", input_data, dim)

def test_cumsum_ext_not_shard_dim_layout_02():
    """
    Feature: distribute operator cumsum_ext in semi auto parallel.
    Description: basic
    Expectation: compile done without error.
    """
    context.set_context(save_graphs=True)
    layout = Layout((2, 1, 4), ("ap", "bp", "cp"))
    strategy = (layout("cp", "bp"),)
    net = CumsumExtNet0(strategy)
    input_data = Tensor(np.ones([128, 128]), dtype=ms.int32)
    dim = -1
    compile_graph(net, 8, "semi_auto_parallel", input_data, dim)

def test_cumsum_ext_dynamic_not_shard_dim_01():
    """
    Feature: distribute operator cumsum_ext in semi auto parallel.
    Description:cumsum_ext(1): dynamic shape (can be divided)
    Expectation: compile done without error.
    """
    context.set_context(save_graphs=True)
    strategy = ((2, 1),)
    s1 = Symbol(divisor=2)
    s2 = Symbol(divisor=1)
    input_data = Tensor(shape=[s1, s2], dtype=ms.int32)
    dim = 1
    net = CumsumExtNet0(strategy)
    compile_graph(net, 8, "semi_auto_parallel", input_data, dim)

def test_cumsum_ext_dynamic_not_shard_dim_02():
    """
    Feature: distribute operator cumsum_ext in semi auto parallel.
    Description:cumsum_ext(1): dynamic shape (can be divided)
    Expectation: compile done without error.
    """
    context.set_context(save_graphs=True)
    strategy = ((1, 4),)
    s1 = Symbol(divisor=1)
    s2 = Symbol(divisor=4)
    input_data = Tensor(shape=[s1, s2], dtype=ms.int32)
    dim = -2
    net = CumsumExtNet0(strategy)
    compile_graph(net, 8, "semi_auto_parallel", input_data, dim)

def test_cumsum_ext_auto_parallel_sharding_propagation0():
    """
    Feature: distribute operator cumsum_ext in auto parallel.
    Description: sharding propagation for cumsum_ext ops, dim is 0, so the input's dimension 0 can not be sharded.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                      search_mode="sharding_propagation")
    strategy = ((1, 2, 4), (1, 2, 4))
    input_data = Tensor(np.ones([8, 8, 8]), dtype=ms.float32)
    dim = 0
    net = CumsumExtNet1(strategy=strategy)
    _cell_graph_executor.compile(net, input_data, dim, phase="train")
    strategies = _cell_graph_executor._get_shard_strategy(net)
    for (k, v) in strategies.items():
        if re.search("CumsumExt", k) is not None:
            assert v == [[1, 2, 4],]

def test_cumsum_ext_auto_parallel_sharding_propagation1():
    """
    Feature: distribute operator cumsum_ext in auto parallel.
    Description: sharding propagation for cumsum_ext ops, dim is 1, so the input's dimension 1 can not be sharded.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                      search_mode="sharding_propagation")
    strategy = ((2, 1, 4), (2, 1, 4))
    input_data = Tensor(np.ones([8, 8, 8]), dtype=ms.float32)
    dim = 1
    net = CumsumExtNet1(strategy)
    _cell_graph_executor.compile(net, input_data, dim, phase="train")
    strategies = _cell_graph_executor._get_shard_strategy(net)
    for (k, v) in strategies.items():
        if re.search("CumsumExt", k) is not None:
            assert v == [[2, 1, 4],]

def test_cumsum_ext_shard_dim_error():
    """
    Feature: test parallel error dim.
    Description: not support shard the input's dimension 'dim'.
    Expectation: raise RuntimeError.
    """
    context.set_context(save_graphs=True)
    strategy = ((2, 4),)
    net = CumsumExtNet0(strategy)
    input_data = Tensor(np.ones([128, 128]), dtype=ms.int32)
    dim = 0
    with pytest.raises(RuntimeError):
        compile_graph(net, 8, "semi_auto_parallel", input_data, dim)

def test_cumsum_ext_shard_dim_out_of_range_error():
    """
    Feature: test parallel error dim.
    Description: the 'dim' is out of range.
    Expectation: raise ValueError.
    """
    context.set_context(save_graphs=True)
    strategy = ((2, 4),)
    net = CumsumExtNet0(strategy)
    input_data = Tensor(np.ones([128, 128]), dtype=ms.int32)
    dim = 3
    with pytest.raises(ValueError):
        compile_graph(net, 8, "semi_auto_parallel", input_data, dim)

def test_cumsum_ext_shard_strategy_error():
    """
    Feature: test parallel error strategy.
    Description: error strategy, not support shard dim or keepdim.
    Expectation: raise RuntimeError.
    """
    context.set_context(save_graphs=True)
    strategy = ((1, 4), (1,))
    net = CumsumExtNet0(strategy)
    input_data = Tensor(np.ones([128, 128]), dtype=ms.int32)
    dim = 0
    with pytest.raises(RuntimeError):
        compile_graph(net, 16, "semi_auto_parallel", input_data, dim)
