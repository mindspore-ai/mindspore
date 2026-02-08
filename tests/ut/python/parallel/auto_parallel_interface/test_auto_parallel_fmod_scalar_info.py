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
# limitations under the License

import mindspore as ms
from mindspore import context, Tensor, nn
from mindspore.graph.api import _cell_graph_executor
from mindspore.ops import operations as P
from mindspore.parallel.shard import Layout
from parallel.auto_parallel_interface._utils import init_hccl, set_parallel_mode, remove_files

context.set_context(mode=ms.GRAPH_MODE)


def setup_function():
    keyword = 'fmod_scalar'
    base_dir = './test_auto_parallel'
    remove_files(keyword, base_dir)


def teardown_function():
    keyword = 'fmod_scalar'
    base_dir = './test_auto_parallel'
    remove_files(keyword, base_dir)


class Net(nn.Cell):
    def __init__(self, fmod_scalar=1, in_strategy=None, out_strategy=None):
        super(Net, self).__init__()
        self.matmul = P.MatMul()
        self.scalar = fmod_scalar
        self.fmodscalar = ms.ops.auto_generate.gen_ops_prim.FmodScalar().shard(in_strategy, out_strategy)

    def construct(self, x, y):
        out = self.matmul(x, y)
        out = self.fmodscalar(out, self.scalar)
        return out


def before_test(case_name):
    init_hccl(global_rank=0, device_num=8)

    # save graph
    case_name = "test_fmod_scalar_cell_shard"
    ir_graph_path = f"./test_auto_parallel/{case_name}"
    context.set_context(save_graphs=True, save_graphs_path=ir_graph_path)


def compile_net(net: nn.Cell, parallel_config, *inputs):
    net.set_train()
    net = set_parallel_mode(net, parallel_config)
    phase, _ = _cell_graph_executor.compile(net, *inputs)
    return phase


# dataset
a = Tensor([[2.0, 1.0, 0.1],
            [2.0, 1.0, 0.1]], ms.float32)
b = Tensor([[0.7, 0.2, 0.1, 0.5],
            [0.1, 0.8, 0.1, 0.3],
            [0.7, 0.2, 0.1, 0.4]], ms.float32)
scalar1 = 1.5


def test_fmod_scalar_semi_auto_with_strategy():
    """
    Feature: distribute operator FmodScalar in semi_auto_parallel mode.
    Description: primitive shard with strategy.
    Expectation: compile done without error.
    """
    case_name = "test_fmod_scalar_cell_shard"
    before_test(case_name)

    strategy1 = ((2, 1),)
    net = Net(fmod_scalar=scalar1, in_strategy=strategy1)
    parallel_config = {"parallel_mode": "semi_auto", "dataset_strategy": "full_batch"}
    compile_net(net, parallel_config, a, b)


def test_fmod_scalar_semi_auto_with_layout():
    """
    Feature: distribute operator FmodScalar in semi_auto_parallel mode.
    Description: primitive shard with in_layout.
    Expectation: compile done without error.
    """
    case_name = "test_fmod_scalar_semi_auto_with_layout"
    before_test(case_name)

    layout = Layout((4, 2, 1), ("dp", "cp", "mp"))
    int_layout = (layout("cp", "dp"),)
    net = Net(fmod_scalar=scalar1, in_strategy=int_layout)
    parallel_config = {"parallel_mode": "semi_auto", "dataset_strategy": "full_batch"}
    compile_net(net, parallel_config, a, b)


def test_fmod_scalar_semi_auto_without_strategy():
    """
    Feature: distribute operator FmodScalar in semi_auto_parallel mode.
    Description: primitive shard without strategy.
    Expectation: compile done without error.
    """
    case_name = "test_fmod_scalar_semi_auto_without_strategy"
    before_test(case_name)

    strategy1 = None
    net = Net(fmod_scalar=scalar1, in_strategy=strategy1)
    parallel_config = {"parallel_mode": "semi_auto", "dataset_strategy": "full_batch"}
    compile_net(net, parallel_config, a, b)


def test_fmod_scalar_sharding_propagation():
    """
    Feature: distribute operator FmodScalar in auto_parallel mode.
    Description: primitive shard with sharding propagation.
    Expectation: compile done without error.
    """
    case_name = "test_fmod_scalar_sharding_propagation"
    before_test(case_name)

    strategy1 = None
    net = Net(fmod_scalar=scalar1, in_strategy=strategy1)
    parallel_config = {"parallel_mode": "sharding_propagation", "dataset_strategy": "full_batch"}
    compile_net(net, parallel_config, a, b)


def test_fmod_scalar_dyanamic_shape():
    """
    Feature: test dynamic shape for fmod scalar in semi_auto_parallel mode
    Description: shard with dynamic shape input
    Expectation: compile success
    """
    case_name = "test_fmod_scalar_dyanamic_shape"
    before_test(case_name)

    class Net1(nn.Cell):
        def __init__(self, fmod_scalar=1, in_strategy=None, out_strategy=None):
            super(Net1, self).__init__()
            self.scalar = fmod_scalar
            self.fmodscalar = ms.ops.auto_generate.gen_ops_prim.FmodScalar().shard(in_strategy, out_strategy)

        def construct(self, x):
            out = self.fmodscalar(x, self.scalar)
            return out

    strategy1 = ((1, 4),)
    net = Net1(fmod_scalar=scalar1, in_strategy=strategy1)
    input_dyn = Tensor(shape=[None, 8], dtype=ms.float32)
    net.set_inputs(input_dyn)

    parallel_config = {"parallel_mode": "semi_auto", "dataset_strategy": "full_batch"}
    compile_net(net, parallel_config, input_dyn)
