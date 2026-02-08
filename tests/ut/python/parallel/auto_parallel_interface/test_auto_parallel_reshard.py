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

import numpy as np
import pytest

import mindspore as ms
from mindspore import ops, nn, context, Tensor
from mindspore.graph.api import _cell_graph_executor
from mindspore.parallel.shard import Layout
from mindspore.nn import Cell
from mindspore.nn.utils import no_init_parameters
from parallel.auto_parallel_interface._utils import init_hccl, set_parallel_mode, find_ir_file_path, remove_files
from parallel.utils.utils import check_layout_config


def setup_function():
    keyword = 'reshard'
    base_dir = './test_auto_parallel'
    remove_files(keyword, base_dir)


def teardown_function():
    keyword = 'reshard'
    base_dir = './test_auto_parallel'
    remove_files(keyword, base_dir)


class BasicBlock(nn.Cell):
    def __init__(self):
        super(BasicBlock, self).__init__()
        self.layer1_dense1 = nn.Dense(128, 128)
        self.layer1_gelu = nn.GELU()

        def test_function(x, y):
            x = ops.abs(x)
            return x + y

        self.test_fn = ms.shard(test_function, in_strategy=((2, 2), (1, 4)), out_strategy=((4, 1),))

    def construct(self, x, u):
        x1 = self.layer1_gelu(x)
        y = self.layer1_gelu(u)
        y = x1 * y
        x = self.layer1_dense1(x)
        x = self.layer1_gelu(x)
        x = self.test_fn(x, y)
        return x


class NetForward(nn.Cell):
    def __init__(self):
        super(NetForward, self).__init__()
        self.layer2_block0 = BasicBlock()
        self.layer2_block1 = BasicBlock()
        self.layer2_block2 = BasicBlock()
        self.layer2_block2_graph = ms.shard(self.layer2_block2, in_strategy=((4, 1), (1, 4)), out_strategy=((4, 1),),
                                            parameter_plan={"self.layer2_block2.layer1_dense1.weight": (4, 1)})
        self.layer2_block3 = BasicBlock()

    def construct(self, x):
        x = self.layer2_block0(x, x)
        x = self.layer2_block2_graph(x, x)
        return x


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.layer3_net = NetForward()
        self.layer3_net_graph = ms.shard(self.layer3_net, in_strategy=((4, 1),),
                                         parameter_plan={"self.layer3_net.layer2_block0.layer1_dense1.weight": (4, 1)})
        self.layer3_net1 = NetForward()
        self.layer3_net1_graph = ms.shard(self.layer3_net1, in_strategy=((2, 2),))

        self.layer3_flatten = nn.Flatten()
        self.layer3_layer1 = nn.Dense(28 * 28, 128)
        self.layer3_layer2 = nn.Dense(128, 10)
        self.layer3_add = ops.Add()
        self.matmul = ops.MatMul()

    def construct(self, x, layout1_, layout2_):
        x1 = self.layer3_flatten(x)
        x2 = self.layer3_layer1(x1)
        x3 = self.layer3_net_graph(x2)
        x4 = self.layer3_net1_graph(x3)
        x4_reshard = ms.reshard(x4, layout1_)
        y = Tensor(np.ones(shape=(128, 128)), dtype=ms.float32)
        y_reshard = ms.reshard(y, layout2_)
        out = self.matmul(x4_reshard, y_reshard)
        return out


layout = Layout((2, 2), ("dp", "mp"))
layout1 = layout("dp", "None")
layout2 = layout("None", "mp")
layout_ = Layout((4, 2, 1), ("dp", "mp", "sp"))
layout3 = layout_("dp", "mp")
layout4 = layout_("mp", "sp")


def compile_net(net: Cell, parallel_config, *inputs):
    net.set_train()
    net = set_parallel_mode(net, parallel_config)
    phase, _ = _cell_graph_executor.compile(net, *inputs)
    return phase


def before_test(case_name):
    init_hccl(global_rank=0, device_num=4)

    # save graph
    context.set_context(mode=ms.GRAPH_MODE)
    ir_graph_path = f"./test_auto_parallel/{case_name}"
    context.set_context(save_graphs=True, save_graphs_path=ir_graph_path)

    with no_init_parameters():
        net = Net()

    x = Tensor(np.ones(shape=(32, 1, 28, 28)), dtype=ms.float32)
    return net, x, ir_graph_path


def test_shard_with_in_strategy_4x1_sharding_propagation():
    """
    Feature: Test reshard in sharding_propagation mode.
    Description: Test shard given (4, 1) tuple as in_strategy.
    Expectation: In strategy of the identity node is ((4, 1)).
    """
    case_name = "test_shard_with_in_strategy_4x1_sharding_propagation"
    net, x, ir_graph_path = before_test(case_name)
    parallel_config = {"parallel_mode": "sharding_propagation"}
    compile_net(net, parallel_config, x, layout1, layout2)

    file = find_ir_file_path(ir_graph_path, "step_parallel_end")

    para1 = "PrimFunc_AShardIdentity(%36)"
    in_strategy1 = "in_strategy: ((4, 1))"

    para2 = "PrimFunc_AShardIdentity(%40)"
    in_strategy2 = "in_strategy: ((4, 1))"

    check_layout_config(para1, file, in_strategy1)
    check_layout_config(para2, file, in_strategy2)


def test_shard_with_in_strategy_4x1_semi_auto():
    """
    Feature: Test reshard in semi_auto_parallel mode.
    Description: Test shard given (4, 1) tuple as in_strategy.
    Expectation: In strategy of the identity node is ((4, 1)).
    """
    case_name = "test_shard_with_in_strategy_4x1_semi_auto"
    net, x, ir_graph_path = before_test(case_name)
    parallel_config = {"parallel_mode": "semi_auto"}
    compile_net(net, parallel_config, x, layout1, layout2)

    file = find_ir_file_path(ir_graph_path, "step_parallel_end")

    para1 = "PrimFunc_AShardIdentity(%36)"
    in_strategy1 = "in_strategy: ((4, 1))"

    para2 = "PrimFunc_AShardIdentity(%40)"
    in_strategy2 = "in_strategy: ((4, 1))"

    check_layout_config(para1, file, in_strategy1)
    check_layout_config(para2, file, in_strategy2)


def test_shard_with_in_strategy_4x1_recursive_programming():
    """
    Feature: Test reshard in recursive_programming mode.
    Description: 'search_mode' must be 'sharding_propagation' for 'Shard' when the 'parallel_mode' is 'auto_parallel'.
    Expectation: raise RuntimeError.
    """
    case_name = "test_shard_with_in_strategy_4x1_recursive_programming"
    net, x, _ = before_test(case_name)
    parallel_config = {"parallel_mode": "recursive_programming"}
    with pytest.raises(RuntimeError) as e:
        compile_net(net, parallel_config, x, layout1, layout2)
    assert "'search_mode' must be 'sharding_propagation' for 'Shard'" in str(e.value)


def test_shard_with_data_parallel():
    """
    Feature: shard in data_parallel mode
    Description: test usage of shard nested shard
    Expectation: compile success.
    """
    from mindspore.parallel.strategy import get_strategy_metadata, get_current_strategy_metadata, \
        enable_save_strategy_online
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
    case_name = "test_shard_with_data_parallel"
    net, x, _ = before_test(case_name)
    parallel_config = None
    enable_save_strategy_online()
    compile_net(net, parallel_config, x, layout1, layout2)

    # data parallel does not support save strategies
    local_info = get_current_strategy_metadata(network=net)
    assert local_info is None

    global_layout = get_strategy_metadata(network=net)
    assert global_layout is None
