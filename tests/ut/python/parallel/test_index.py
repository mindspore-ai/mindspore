# Copyright 2024 Huawei Technologies Co., Ltd
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
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore import Symbol
from mindspore.graph.api import _cell_graph_executor
from mindspore.ops.auto_generate.gen_ops_prim import Index
from parallel.utils.utils import ParallelValidator

def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")

class IndexNet(nn.Cell):
    def __init__(self, strategy=None):
        super().__init__()
        if strategy:
            self.index_op = Index().shard(strategy)
        else:
            self.index_op = Index()

    def construct(self, input_data, indices):
        out = self.index_op(input_data, indices)
        return out


def compile_graph(net, device_num, parallel_mode, input_data, indices, search_mode="sharding_propagation"):
    context.set_auto_parallel_context(device_num=device_num, global_rank=0, parallel_mode=parallel_mode)
    net.set_train()
    phase, _ = _cell_graph_executor.compile(net, input_data, indices)
    return phase


def test_index_shard_basic_0():
    """
    Feature: distribute operator index in semi auto parallel.
    Description: basic
    Expectation: compile done without error.
    """
    context.set_context(save_graphs=True)
    strategy = ((2, 4), ((1,), (1,)))
    index_1 = Tensor(np.ones([64]), dtype=ms.int32)
    index_2 = Tensor(np.zeros([64]), dtype=ms.int32)
    indices = (index_1, index_2,)
    net = IndexNet(strategy)
    input_data = Tensor(np.ones([128, 128]), dtype=ms.int32)
    phase = compile_graph(net, 8, "semi_auto_parallel", input_data, indices)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('ReLU-0', ['Sub-0'])
    assert validator.check_node_inputs_has('LogicalAnd-0', ['Equal-0', 'Equal-1'])
    assert validator.check_node_attrs("Sub-0", {"keep_alive": True})


def test_index_shard_basic_1():
    """
    Feature: distribute operator index in semi auto parallel.
    Description: repeated calc
    Expectation: compile done without error.
    """
    context.set_context(save_graphs=True)
    strategy = ((2, 4), ((1,), (1,)))
    index_1 = Tensor(np.ones([32]), dtype=ms.int32)
    index_2 = Tensor(np.zeros([32]), dtype=ms.int32)
    indices = (index_1, index_2,)
    net = IndexNet(strategy)
    input_data = Tensor(np.ones([128, 128]), dtype=ms.int32)
    phase = compile_graph(net, 16, "semi_auto_parallel", input_data, indices)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('ReLU-0', ['Sub-0'])
    assert validator.check_node_inputs_has('LogicalAnd-0', ['Equal-0', 'Equal-1'])
    assert validator.check_node_attrs("Sub-0", {"keep_alive": True})


def test_index_shard_dynamic_0():
    """
    Feature: distribute operator index in semi auto parallel.
    Description:index(1): dynamic shape (can be divided)
    Expectation: compile done without error.
    """
    context.set_context(save_graphs=True)
    strategy = ((2, 4), ((1,), (1,)))
    s1 = Symbol(divisor=2)
    s2 = Symbol(divisor=4)
    input_data = Tensor(shape=[s1, s2], dtype=ms.int32)
    index_1 = Tensor(np.ones([32]), dtype=ms.int32)
    index_2 = Tensor(np.zeros([32]), dtype=ms.int32)
    indices = (index_1, index_2)
    net = IndexNet(strategy)
    phase = compile_graph(net, 8, "semi_auto_parallel", input_data, indices)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('ReLU-0', ['Sub-0'])
    assert validator.check_node_inputs_has('LogicalAnd-0', ['Equal-0', 'Equal-1'])
    assert validator.check_node_attrs("Sub-0", {"keep_alive": True})


def test_index_shard_dynamic_1():
    """
    Feature: distribute operator index in semi auto parallel.
    Description:
    Expectation: compile done without error.
    """
    context.set_context(save_graphs=True)
    strategy = ((2, 4), ((1,), (1,)))
    s1 = Symbol(divisor=2)
    s2 = Symbol(divisor=4)
    input_data = Tensor(shape=[s1, s2], dtype=ms.int32)
    index_1 = Tensor(np.ones([64]), dtype=ms.int32)
    index_2 = Tensor(np.zeros([64]), dtype=ms.int32)
    indices = (index_1, index_2,)
    net = IndexNet(strategy)
    phase = compile_graph(net, 8, "semi_auto_parallel", input_data, indices)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('ReLU-0', ['Sub-0'])
    assert validator.check_node_inputs_has('LogicalAnd-0', ['Equal-0', 'Equal-1'])
    assert validator.check_node_attrs("Sub-0", {"keep_alive": True})
