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

import pytest

import mindspore as ms
from mindspore import context, Tensor, Parameter
from mindspore.ops import operations as P
from mindspore.parallel.shard import Layout
from mindspore.graph.api import _cell_graph_executor
from mindspore.nn import Cell, TrainOneStepCell, Momentum
from mindspore.nn.utils import no_init_parameters
from mindspore.common.initializer import initializer
from parallel.utils.utils import ParallelValidator
from parallel.auto_parallel_interface._utils import init_hccl, set_parallel_mode, remove_files, find_ir_file_path,\
    check_node_attrs_pair


def setup_function():
    keyword = 'dump_local_norm'
    base_dir = './test_auto_parallel'
    remove_files(keyword, base_dir)


def teardown_function():
    keyword = 'dump_local_norm'
    base_dir = './test_auto_parallel'
    remove_files(keyword, base_dir)


class DynamicMulNet(Cell):
    def __init__(self, strategy1):
        super().__init__()
        self.mul = P.Mul().shard(strategy1)
        layout = Layout((8, 1, 1), ("dp", "mp", "xp"))
        layout1 = (layout("dp", "mp", "xp"),)

        self.gelu = P.GeLU().shard(layout1)
        self.w = Parameter(initializer("ones", [1], dtype=ms.float32), "w2")

    def construct(self, x, y):
        out = self.mul(x, self.w)
        out = self.gelu(out)
        return out


def test_layout_dump_local_norm_true():
    """
    Feature: config layout for dynamic shape, dump_local_norm is true
    Description: no redistribution, 'Print("dump local norm: ", "w2", %13)' is in validate.ir
    Expectation: compile success, check_node_attrs_pair pass
    """
    graph_path = "./test_auto_parallel/test_dump_local_norm_true_graphs"
    context.set_context(save_graphs=True, save_graphs_path=graph_path)

    strategy1 = ((8, 1, 1), (1,))
    with no_init_parameters():
        net = DynamicMulNet(strategy1)
        optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)

    x = Tensor(shape=[16, None, 1], dtype=ms.float32)
    y = Tensor(shape=[None, None, None], dtype=ms.float32)
    net.set_inputs(x, y)

    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    train_net.set_inputs(x, y)

    # set auto_parallel
    init_hccl(global_rank=0, device_num=8)
    parallel_config = {"parallel_mode": "semi_auto", "gradients_mean": False, "dataset_strategy": "full_batch",
                       "dump_local_norm": True}
    train_net = set_parallel_mode(train_net, parallel_config)
    phase, _ = _cell_graph_executor.compile(train_net, x, y)

    # validation
    validator = ParallelValidator(train_net, phase)
    assert validator.check_node_inputs_has('GeLU-0', ['Mul-0'])
    validate_ir = find_ir_file_path(graph_path, "validate")
    graph_dict = {'"dump local norm: ", "w2"': 1}
    check_pairs = {"Print": graph_dict}
    check_node_attrs_pair(validate_ir, check_pairs)


def test_layout_dump_local_norm_false():
    """
    Feature: config layout for dynamic shape, dump_local_norm is false
    Description: no redistribution, 'Print("dump local norm: ", "w2", %13)' is not in validate.ir
    Expectation: compile success, raise Exception in check_node_attrs_pair
    """
    graph_path = "./test_auto_parallel/test_dump_local_norm_false_graphs"
    context.set_context(save_graphs=True, save_graphs_path=graph_path)

    strategy1 = ((8, 1, 1), (1,))
    with no_init_parameters():
        net = DynamicMulNet(strategy1)
        optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)

    x = Tensor(shape=[16, None, 1], dtype=ms.float32)
    y = Tensor(shape=[None, None, None], dtype=ms.float32)
    net.set_inputs(x, y)

    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    train_net.set_inputs(x, y)

    # set auto_parallel
    init_hccl(global_rank=0, device_num=8)
    parallel_config = {"parallel_mode": "semi_auto", "gradients_mean": False, "dataset_strategy": "full_batch",
                       "dump_local_norm": False}
    train_net = set_parallel_mode(train_net, parallel_config)
    phase, _ = _cell_graph_executor.compile(train_net, x, y)

    # validation
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('GeLU-0', ['Mul-0'])
    validate_ir = find_ir_file_path(graph_path, "validate")
    graph_dict = {'"dump local norm: ", "network.w"': 1}
    check_pairs = {"Print": graph_dict}
    with pytest.raises(ValueError):
        check_node_attrs_pair(validate_ir, check_pairs)
