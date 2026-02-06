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
import numpy as np
import pytest
import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore import Parameter
from mindspore import Symbol
from mindspore import Tensor
from mindspore.ops.auto_generate.gen_ops_prim import IndexAddExt, Mul
from mindspore.parallel.auto_parallel import AutoParallel
from mindspore.parallel.shard import Layout
from mindspore.graph.api import _cell_graph_executor
from mindspore.nn.utils import no_init_parameters
from hccl_test.manage.api import Hccl


inputs = Tensor(np.random.randn(4, 8, 8).astype(np.float32), ms.float32)
source = Tensor(np.random.randn(4, 8, 4).astype(np.float32), ms.float32)
index = Tensor(np.array([0, 3, 1, 2]), ms.int32)
dim = 2
alpha = 1
layout = Layout((2, 4, 1), ("dp", "cp", "mp"))

# init hccl


def init_hccl(global_rank, device_num):
    hccl = Hccl()
    hccl.rank_id = global_rank
    hccl.rank_size = device_num


def setup_function():
    parallel_ir_path = "index_add_ir"
    context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend",
                        save_graphs=True, save_graphs_path=parallel_ir_path)


def compile_net(parallel_mode, strategy):
    init_hccl(0, 8)
    with no_init_parameters():
        net = IndexNet(inputs, dim, source, alpha, strategy)
    parallel_net = AutoParallel(net, parallel_mode=parallel_mode)
    _ = _cell_graph_executor.compile(parallel_net, index)


class IndexNet(nn.Cell):
    def __init__(self, in_inputs, in_dim, in_source, in_alpha, strategy=None):
        super().__init__()
        if strategy:
            self.index_add = IndexAddExt().shard(strategy)
        else:
            self.index_add = IndexAddExt()
        self.input = Parameter(in_inputs, "input")
        self.dim = in_dim
        self.source = Parameter(in_source, "source")
        self.alpha = in_alpha

    def construct(self, in_index):
        out = self.index_add(self.input, self.dim, in_index,
                             self.source, self.alpha)
        return out


class ShardingPropagationNet(nn.Cell):
    def __init__(self, in_inputs, in_dim, in_index, in_source, in_alpha, strategy=None):
        super().__init__()
        self.mul = Mul().shard(strategy)
        self.index_add = IndexAddExt()
        self.input = Parameter(in_inputs, "input")
        self.dim = in_dim
        self.index = in_index
        self.source = Parameter(in_source, "source")
        self.alpha = in_alpha

    def construct(self):
        self.input = self.mul(self.input, self.input)
        out = self.index_add(self.input, self.dim,
                             self.index, self.source, self.alpha)
        return out


def test_index_add_fully_shard():
    """
    Feature: distribute operator index add in "semi_auto" mode
    Description: test IndexAddExt sharding tensor across all cards
    Expectation: compile success
    """
    strategy = ((2, 4, 1), (1,), (2, 4, 1))
    compile_net("semi_auto", strategy)


def test_index_add_layout():
    """
    Feature: distribute operator index add in "semi_auto" mode
    Description: test IndexAddExt using layout
    Expectation: compile success
    """
    strategy = (layout("dp", "cp", "mp"), layout(
        "mp"), layout("dp", "cp", "mp"))
    compile_net("semi_auto", strategy)


def test_index_add_dynamic_shape():
    """
    Feature: distribute operator index add in "semi_auto" mode
    Description: test IndexAddExt using dynamic shape
    Expectation: compile success
    """
    init_hccl(0, 8)
    strategy = ((2, 4, 1), (1,), (2, 4, 1))
    s2 = Symbol(devisor=2)
    dynamic_index = Tensor(shape=[s2], dtype=ms.int32)
    with no_init_parameters():
        net = IndexNet(inputs, dim, source, alpha, strategy)
        net.set_inputs(dynamic_index)
    parallel_net = AutoParallel(net, parallel_mode="semi_auto")
    _ = _cell_graph_executor.compile(parallel_net, dynamic_index)


def test_index_add_sharding_propagation():
    """
    Feature: distribute operator index add in "sharding_propagation" mode
    Description: sharding propagation for index_add ops
    Expectation: strategy is in line with expectations
    """
    init_hccl(0, 8)
    strategy = ((2, 2, 1), (2, 2, 1))
    with no_init_parameters():
        net = ShardingPropagationNet(
            inputs, dim, index, source, alpha, strategy)
    parallel_net = AutoParallel(net, parallel_mode="sharding_propagation")
    _ = _cell_graph_executor.compile(parallel_net, phase="train")
    strategies = _cell_graph_executor._get_shard_strategy(parallel_net)
    for (k, v) in strategies.items():
        if re.search("IndexAddExt", k) is not None:
            assert v == [[2, 2, 1], [1], [2, 2, 1]]


def test_index_add_split_dim_axis():
    """
    Feature: test IndexAddExt using invalid strategy
    Description: distribute operator index doesn't support spilt the 'dim' axis
    Expectation: compile with error
    """
    strategy = ((2, 2, 1), (2,), (2, 2, 1))
    with pytest.raises(RuntimeError):
        compile_net("semi_auto", strategy)


def test_index_add_split_dim_axis_layout():
    """
    Feature: test IndexAddExt using invalid layout strategy
    Description: distribute operator index doesn't support spilt the 'dim' axis
    Expectation: compile with error
    """
    new_layout = Layout((2, 2, 2), ("dp", "cp", "mp"))
    strategy = (new_layout("dp", "cp", "mp"), new_layout(
        "mp"), new_layout("dp", "cp", "mp"))
    with pytest.raises(RuntimeError):
        compile_net("semi_auto", strategy)


def test_index_add_strategy_inconsistency():
    """
    Feature: test IndexAddExt using invalid strategy
    Description: The input strategy must match the source strategy; mismatches will trigger a runtime error
    Expectation: compile with error
    """
    strategy = ((2, 2, 2), (2,), (2, 2, 1))
    with pytest.raises(RuntimeError):
        compile_net("semi_auto", strategy)


def test_index_add_strategy_inconsistency_layout():
    """
    Feature: test IndexAddExt using layout strategy
    Description: The input strategy must match the source strategy; mismatches will trigger a runtime error
    Expectation: compile with error
    """
    strategy = (layout("dp", "cp", "mp"), layout("cp"), layout("cp", "dp", "mp"))
    with pytest.raises(RuntimeError):
        compile_net("semi_auto", strategy)


def test_index_add_strategy_wrong_size():
    """
    Feature: test IndexAddExt using invalid strategy
    Description: The input strategy size must be 3;
    Expectation: compile with error
    """
    strategy = ((2, 2, 2), (2,), (2, 2, 1), (1,))
    with pytest.raises(RuntimeError):
        compile_net("semi_auto", strategy)


def test_index_add_strategy_wrong_size_layout():
    """
    Feature: test IndexAddExt using invalid layout strategy
    Description: The input strategy size must be 3;
    Expectation: compile with error
    """
    strategy = (layout("dp", "cp", "mp"), layout(
        "mp"), layout("dp", "cp", "mp"), layout("mp"))
    with pytest.raises(RuntimeError):
        compile_net("semi_auto", strategy)


def test_index_add_no_fully_shard():
    """
    Feature: distribute operator index add in "semi_auto" mode
    Description: test IndexAddExt not sharding tensor across all cards
    Expectation: compile success
    """
    strategy = ((2, 2, 1), (1,), (2, 2, 1))
    compile_net("semi_auto", strategy)
