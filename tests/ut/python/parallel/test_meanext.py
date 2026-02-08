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
from mindspore import Tensor, context, Symbol, Layout
from mindspore.graph.api import _cell_graph_executor
from mindspore.ops.auto_generate.gen_ops_prim import MeanExt, Mul

def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")

class MeanNet0(nn.Cell):
    def __init__(self, strategy=None):
        super().__init__()
        if strategy:
            self.mean_op = MeanExt().shard(strategy)
        else:
            self.mean_op = MeanExt()

    def construct(self, input_data, dim, keepdim):
        out = self.mean_op(input_data, dim, keepdim)
        return out

class MeanNet1(nn.Cell):
    def __init__(self, strategy):
        super().__init__()
        self.mul = Mul().shard(strategy)
        self.mean_op = MeanExt()

    def construct(self, input_data, dim, keepdim):
        out = self.mul(input_data, input_data)
        out = self.mean_op(out, dim, keepdim)
        return out[0]


def compile_graph(net, device_num, parallel_mode, input_data, dim, keepdim, search_mode="sharding_propagation"):
    context.set_auto_parallel_context(device_num=device_num, global_rank=0, parallel_mode=parallel_mode)
    net.set_train()
    phase, _ = _cell_graph_executor.compile(net, input_data, dim, keepdim)
    return phase

def test_mean_ext_shard_keepdims_false():
    """
    Feature: distribute operator mean_ext in semi auto parallel.
    Description: basic
    Expectation: compile done without error.
    """
    context.set_context(save_graphs=True)
    strategy = ((2, 4),)
    net = MeanNet0(strategy)
    input_data = Tensor(np.ones([64, 128]), dtype=ms.float32)
    dim = 0
    keepdim = False
    compile_graph(net, 8, "semi_auto_parallel", input_data, dim, keepdim)

def test_mean_ext_shard_keepdims_true():
    """
    Feature: distribute operator mean_ext in semi auto parallel.
    Description: basic
    Expectation: compile done without error.
    """
    context.set_context(save_graphs=True)
    strategy = ((2, 4),)
    net = MeanNet0(strategy)
    input_data = Tensor(np.ones([64, 128]), dtype=ms.float32)
    dim = 0
    keepdim = True
    compile_graph(net, 8, "semi_auto_parallel", input_data, dim, keepdim)

def test_mean_ext_shard_dim_list():
    """
    Feature: distribute operator mean_ext in semi auto parallel.
    Description: basic
    Expectation: compile done without error.
    """
    context.set_context(save_graphs=True)
    strategy = ((2, 4),)
    net = MeanNet0(strategy)
    input_data = Tensor(np.ones([64, 128]), dtype=ms.float32)
    dim = (0, 1)
    keepdim = True
    compile_graph(net, 8, "semi_auto_parallel", input_data, dim, keepdim)


def test_mean_ext_shard_dim_none():
    """
    Feature: distribute operator mean_ext in semi auto parallel.
    Description: basic
    Expectation: compile done without error.
    """
    context.set_context(save_graphs=True)
    strategy = ((2, 4),)
    net = MeanNet0(strategy)
    input_data = Tensor(np.ones([64, 128]), dtype=ms.float32)
    dim = None
    keepdim = True
    compile_graph(net, 8, "semi_auto_parallel", input_data, dim, keepdim)


def test_mean_ext_shard_dynamic_keepdims_true():
    """
    Feature: distribute operator mean_ext in semi auto parallel.
    Description: dynamic
    Expectation: compile done without error.
    """
    context.set_context(save_graphs=True)
    strategy = ((2, 4),)
    net = MeanNet0(strategy)
    s1 = Symbol(divisor=2)
    s2 = Symbol(divisor=4)
    input_data = Tensor(shape=[s1, s2], dtype=ms.float32)
    dim = 0
    keepdim = True
    compile_graph(net, 8, "semi_auto_parallel", input_data, dim, keepdim)


def test_mean_ext_shard_dynamic_keepdims_false():
    """
    Feature: distribute operator mean_ext in semi auto parallel.
    Description: dynamic
    Expectation: compile done without error.
    """
    context.set_context(save_graphs=True)
    strategy = ((2, 4),)
    net = MeanNet0(strategy)
    s1 = Symbol(divisor=2)
    s2 = Symbol(divisor=4)
    input_data = Tensor(shape=[s1, s2], dtype=ms.float32)
    dim = 0
    keepdim = False
    compile_graph(net, 8, "semi_auto_parallel", input_data, dim, keepdim)


def test_mean_ext_shard_dynamic_dim_list():
    """
    Feature: distribute operator mean_ext in semi auto parallel.
    Description: dynamic
    Expectation: compile done without error.
    """
    context.set_context(save_graphs=True)
    strategy = ((2, 4),)
    net = MeanNet0(strategy)
    s1 = Symbol(divisor=2)
    s2 = Symbol(divisor=4)
    input_data = Tensor(shape=[s1, s2], dtype=ms.float32)
    dim = (0, 1)
    keepdim = True
    compile_graph(net, 8, "semi_auto_parallel", input_data, dim, keepdim)


def test_mean_ext_shard_dynamic_dim_None():
    """
    Feature: distribute operator mean_ext in semi auto parallel.
    Description: dynamic
    Expectation: compile done without error.
    """
    context.set_context(save_graphs=True)
    strategy = ((2, 4),)
    net = MeanNet0(strategy)
    s1 = Symbol(divisor=2)
    s2 = Symbol(divisor=4)
    input_data = Tensor(shape=[s1, s2], dtype=ms.float32)
    dim = None
    keepdim = True
    compile_graph(net, 8, "semi_auto_parallel", input_data, dim, keepdim)


def test_mean_ext_auto_parallel_sharding_propagation_keepdims_true():
    """
    Feature: distribute operator mean_ext in auto parallel.
    Description: sharding propagation for mean_ext ops.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                      search_mode="sharding_propagation")
    strategy = ((1, 2, 4), (1, 2, 4))
    input_data = Tensor(np.ones([8, 8, 8]), dtype=ms.float32)
    dim = 1
    keepdim = True
    net = MeanNet1(strategy=strategy)
    _cell_graph_executor.compile(net, input_data, dim, keepdim, phase="train")
    strategies = _cell_graph_executor._get_shard_strategy(net)
    for (k, v) in strategies.items():
        if re.search("MeanExt", k) is not None:
            assert v == [[1, 2, 4],]


def test_mean_ext_auto_parallel_sharding_propagation_keepdims_false():
    """
    Feature: distribute operator mean_ext in auto parallel.
    Description: sharding propagation for mean_dim ops.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                      search_mode="sharding_propagation")
    strategy = ((2, 1, 4), (2, 1, 4))
    input_data = Tensor(np.ones([8, 8, 8]), dtype=ms.float32)
    dim = 0
    keepdim = False
    net = MeanNet1(strategy)
    _cell_graph_executor.compile(net, input_data, dim, keepdim, phase="train")
    strategies = _cell_graph_executor._get_shard_strategy(net)
    for (k, v) in strategies.items():
        if re.search("MeanExt", k) is not None:
            assert v == [[2, 1, 4],]


def test_mean_ext_auto_parallel_sharding_propagation_dim_list():
    """
    Feature: distribute operator mean_ext in auto parallel.
    Description: sharding propagation for mean_ext ops, dim is list.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                      search_mode="sharding_propagation")
    strategy = ((1, 2, 4), (1, 2, 4))
    input_data = Tensor(np.ones([8, 8, 8]), dtype=ms.float32)
    dim = (0, 1)
    keepdim = True
    net = MeanNet1(strategy=strategy)
    _cell_graph_executor.compile(net, input_data, dim, keepdim, phase="train")
    strategies = _cell_graph_executor._get_shard_strategy(net)
    for (k, v) in strategies.items():
        if re.search("MeanExt", k) is not None:
            assert v == [[1, 2, 4],]


def test_mean_ext_auto_parallel_sharding_propagation_dim_none():
    """
    Feature: distribute operator mean_ext in auto parallel.
    Description: sharding propagation for mean_ext ops, dim is None.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                      search_mode="sharding_propagation")
    strategy = ((1, 2, 4), (1, 2, 4))
    input_data = Tensor(np.ones([8, 8, 8]), dtype=ms.float32)
    dim = None
    keepdim = True
    net = MeanNet1(strategy=strategy)
    _cell_graph_executor.compile(net, input_data, dim, keepdim, phase="train")
    strategies = _cell_graph_executor._get_shard_strategy(net)
    for (k, v) in strategies.items():
        if re.search("MeanExt", k) is not None:
            assert v == [[1, 2, 4],]


def test_mean_shard_dim_out_of_range_error():
    """
    Feature: test parallel error dim.
    Description: the 'dim' is out of range.
    Expectation: raise ValueError.
    """
    context.set_context(save_graphs=True)
    strategy = ((2, 4),)
    net = MeanNet0(strategy)
    input_data = Tensor(np.ones([128, 128]), dtype=ms.float32)
    dim = 3
    keepdim = True
    with pytest.raises(ValueError):
        compile_graph(net, 8, "semi_auto_parallel", input_data, dim, keepdim)


def test_mean_shard_strategy_error():
    """
    Feature: test parallel error strategy.
    Description: error strategy, not support shard dim or keepdim.
    Expectation: raise RuntimeError.
    """
    context.set_context(save_graphs=True)
    strategy = ((1, 4), (1,))
    net = MeanNet0(strategy)
    input_data = Tensor(np.ones([128, 128]), dtype=ms.float32)
    dim = 0
    keepdim = True
    with pytest.raises(RuntimeError):
        compile_graph(net, 16, "semi_auto_parallel", input_data, dim, keepdim)


def test_mean_ext_shard_keepdims_true_layout():
    """
    Feature: distribute operator mean_ext in semi auto parallel.
    Description: basic
    Expectation: compile done without error.
    """
    context.set_context(save_graphs=True)
    layout = Layout((2, 4), ('a', 'b'))
    strategy = (layout('a', 'b'),)
    net = MeanNet0(strategy)
    input_data = Tensor(np.ones([64, 128]), dtype=ms.float32)
    dim = 0
    keepdim = True
    compile_graph(net, 8, "semi_auto_parallel", input_data, dim, keepdim)


def test_mean_ext_shard_keepdims_false_layout():
    """
    Feature: distribute operator mean_ext in semi auto parallel.
    Description: basic
    Expectation: compile done without error.
    """
    context.set_context(save_graphs=True)
    layout = Layout((2, 4), ('a', 'b'))
    strategy = (layout('a', 'b'),)
    net = MeanNet0(strategy)
    input_data = Tensor(np.ones([64, 128]), dtype=ms.float32)
    dim = 0
    keepdim = False
    compile_graph(net, 8, "semi_auto_parallel", input_data, dim, keepdim)


def test_mean_ext_shard_dim_list_layout():
    """
    Feature: distribute operator mean_ext in semi auto parallel.
    Description: basic
    Expectation: compile done without error.
    """
    context.set_context(save_graphs=True)
    layout = Layout((2, 4), ('a', 'b'))
    strategy = (layout('a', 'b'),)
    net = MeanNet0(strategy)
    input_data = Tensor(np.ones([64, 128]), dtype=ms.float32)
    dim = (0, 1)
    keepdim = True
    compile_graph(net, 8, "semi_auto_parallel", input_data, dim, keepdim)


def test_mean_ext_shard_dim_none_layout():
    """
    Feature: distribute operator mean_ext in semi auto parallel.
    Description: basic
    Expectation: compile done without error.
    """
    context.set_context(save_graphs=True)
    layout = Layout((2, 4), ('a', 'b'))
    strategy = (layout('a', 'b'),)
    net = MeanNet0(strategy)
    input_data = Tensor(np.ones([64, 128]), dtype=ms.float32)
    dim = None
    keepdim = True
    compile_graph(net, 8, "semi_auto_parallel", input_data, dim, keepdim)

def test_mean_ext_shard_false_layout():
    """
    Feature: distribute operator mean_ext in semi auto parallel.
    Description: basic
    Expectation: compile done without error.
    """
    context.set_context(save_graphs=True)
    layout = Layout((1, 4, 1, 2), ('a', 'b', 'c', 'd'))
    strategy = (layout(('a', 'b', 'c', 'd'), "None"),)
    net = MeanNet0(strategy)
    input_data = Tensor(np.ones([16, 128]), dtype=ms.float32)
    dim = None
    keepdim = False
    compile_graph(net, 8, "semi_auto_parallel", input_data, dim, keepdim)
