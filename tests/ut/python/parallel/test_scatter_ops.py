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
""" test scatter update """
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P
from mindspore import context
from mindspore.common.api import _cell_graph_executor
from parallel.utils.utils import ParallelValidator

scatter_ops_dict = {"Add": P.ScatterAdd(), "Mul": P.ScatterMul(), "Div": P.ScatterDiv(),
                    "Min": P.ScatterMin(), "Max": P.ScatterMax(), "Sub": P.ScatterSub()}


class Net(nn.Cell):
    """Net definition"""
    def __init__(self, input_shape, indices_shape, updates_shape, strategy1=None, strategy2=None, ops="Add"):
        super(Net, self).__init__()
        self.inputs = Parameter(Tensor(np.ones(input_shape).astype(np.float32)), "input")
        self.indices = Tensor(np.ones(indices_shape).astype(np.int32))
        self.updates = Tensor(np.ones(updates_shape).astype(np.float32))
        self.scatter_ops = scatter_ops_dict.get(ops)
        self.scatter_ops.shard(strategy1)
        self.add = P.TensorAdd().shard(strategy2)
        self.relu = P.ReLU()

    def construct(self, x):
        out = self.scatter_ops(self.inputs, self.indices, self.updates)
        out = self.add(x, out)
        out = self.relu(out)
        return out


def compile_net(net, *inputs):
    net.set_auto_parallel()
    net.set_train()
    phase, _ = _cell_graph_executor.compile(net, *inputs)
    context.reset_auto_parallel_context()
    return phase


def test_scatter_add_column_split():
    """
    Feature: test scatter ops auto parallel
    Description: test scatter add column split
    Expectation: compile success
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, full_batch=True)
    inputs = Tensor(np.ones([32, 64, 128]).astype(np.float32))
    input_shape = [32, 64, 128]
    indices_shape = [4, 8]
    updates_shape = [4, 8, 64, 128]
    strategy1 = ((1, 2, 4), (1, 1), (1, 1, 2, 4))
    strategy2 = ((1, 2, 4), (1, 2, 4))
    net = Net(input_shape, indices_shape, updates_shape, strategy1, strategy2)
    compile_net(net, inputs)


def test_scatter_min_column_split():
    """
    Feature: test scatter ops auto parallel
    Description: test scatter add column split
    Expectation: compile success
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, full_batch=True)
    inputs = Tensor(np.ones([32, 64, 128]).astype(np.float32))
    input_shape = [32, 64, 128]
    indices_shape = [4, 8]
    updates_shape = [4, 8, 64, 128]
    strategy1 = ((1, 2, 4), (1, 1), (1, 1, 2, 4))
    strategy2 = ((1, 2, 4), (1, 2, 4))
    net = Net(input_shape, indices_shape, updates_shape, strategy1, strategy2, "Min")
    compile_net(net, inputs)


def test_scatter_max_column_split():
    """
    Feature: test scatter ops auto parallel
    Description: test scatter add column split
    Expectation: compile success
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, full_batch=True)
    inputs = Tensor(np.ones([32, 64, 128]).astype(np.float32))
    input_shape = [32, 64, 128]
    indices_shape = [4, 8]
    updates_shape = [4, 8, 64, 128]
    strategy1 = ((1, 2, 4), (1, 1), (1, 1, 2, 4))
    strategy2 = ((1, 2, 4), (1, 2, 4))
    net = Net(input_shape, indices_shape, updates_shape, strategy1, strategy2, "Max")
    compile_net(net, inputs)


def test_scatter_add_row_split():
    """
    Feature: test scatter ops auto parallel
    Description: test scatter add row split
    Expectation: compile success
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, full_batch=True)
    inputs = Tensor(np.ones([32, 64, 128]).astype(np.float32))
    input_shape = [32, 64, 128]
    indices_shape = [4, 8]
    updates_shape = [4, 8, 64, 128]
    strategy1 = ((8, 1, 1), (1, 1), (1, 1, 1, 1))
    strategy2 = ((1, 1, 1), (1, 1, 1))
    net = Net(input_shape, indices_shape, updates_shape, strategy1, strategy2)
    phase = compile_net(net, inputs)
    validator = ParallelValidator(net, phase)
    # check sub_graph
    sub_graph = {
        'ScatterAdd-0': ['input', 'Minimum-0', 'Mul-0'],
        'Mul-0': ['_GetTensorSlice-1', 'Reshape-0'],
    }
    assert validator.check_graph_structure(sub_graph)


def test_scatter_add_mix_split():
    """
    Feature: test scatter ops auto parallel
    Description: test scatter add mix split
    Expectation: compile success
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=24, full_batch=True)
    inputs = Tensor(np.ones([32 * 3, 64, 128]).astype(np.float32))
    input_shape = [32 * 3, 64, 128]
    indices_shape = [4, 8]
    updates_shape = [4, 8, 64, 128]
    strategy1 = ((3, 2, 4), (1, 1), (1, 1, 2, 4))
    strategy2 = ((1, 1, 1), (1, 1, 1))
    net = Net(input_shape, indices_shape, updates_shape, strategy1, strategy2)
    compile_net(net, inputs)


def test_scatter_mul_mix_split():
    """
    Feature: test scatter ops auto parallel
    Description: test scatter mul mix split
    Expectation: compile success
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=64, full_batch=True)
    inputs = Tensor(np.ones([32 * 4, 64]).astype(np.float32))
    input_shape = [32 * 4, 64]
    indices_shape = [4]
    updates_shape = [4, 64]
    strategy1 = ((8, 8), (1,), (1, 8))
    strategy2 = ((1, 1), (1, 1))
    net = Net(input_shape, indices_shape, updates_shape, strategy1, strategy2, "Mul")
    phase = compile_net(net, inputs)
    validator = ParallelValidator(net, phase)
    # check sub_graph
    sub_graph = {
        'ScatterMul-0': ['input', 'Minimum-0', 'Add-0'],
        'Add-0': ['Mul-0', 'Sub-1'],
    }
    assert validator.check_graph_structure(sub_graph)


def test_scatter_div_mix_split():
    """
    Feature: test scatter ops auto parallel
    Description: test scatter div mix split
    Expectation: compile success
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=64, full_batch=True)
    inputs = Tensor(np.ones([32 * 4, 64]).astype(np.float32))
    input_shape = [32 * 4, 64]
    indices_shape = [4, 4]
    updates_shape = [4, 4, 64]
    strategy1 = ((8, 8), (1, 1), (1, 1, 8))
    strategy2 = ((1, 1), (1, 1))
    net = Net(input_shape, indices_shape, updates_shape, strategy1, strategy2, "Div")
    compile_net(net, inputs)


def test_scatter_add_mix_split_auto_parallel_sharding_prop():
    """
    Feature: test scatter ops auto parallel
    Description: test scatter add mix split
    Expectation: compile success
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=24, full_batch=True,
                                      search_mode="sharding_propagation")
    inputs = Tensor(np.ones([32 * 3, 64, 128]).astype(np.float32))
    input_shape = [32 * 3, 64, 128]
    indices_shape = [4, 8]
    updates_shape = [4, 8, 64, 128]
    strategy1 = None
    strategy2 = ((3, 2, 4), (3, 2, 4))
    net = Net(input_shape, indices_shape, updates_shape, strategy1, strategy2)
    compile_net(net, inputs)


def test_scatter_mul_mix_split_auto_parallel():
    """
    Feature: test scatter ops auto parallel
    Description: test scatter mul mix split
    Expectation: compile success
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=64, full_batch=True)
    inputs = Tensor(np.ones([32 * 4, 64]).astype(np.float32))
    input_shape = [32 * 4, 64]
    indices_shape = [4]
    updates_shape = [4, 64]
    strategy1 = None
    strategy2 = None
    net = Net(input_shape, indices_shape, updates_shape, strategy1, strategy2, "Mul")
    compile_net(net, inputs)


def test_scatter_div_mix_split_auto_parallel_rec():
    """
    Feature: test scatter ops auto parallel
    Description: test scatter div mix split
    Expectation: compile success
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=64, full_batch=True,
                                      search_mode="recursive_programming")
    inputs = Tensor(np.ones([32 * 4, 64]).astype(np.float32))
    input_shape = [32 * 4, 64]
    indices_shape = [4, 4]
    updates_shape = [4, 4, 64]
    strategy1 = None
    strategy2 = None
    net = Net(input_shape, indices_shape, updates_shape, strategy1, strategy2, "Div")
    compile_net(net, inputs)


def test_scatter_sub_mix_split():
    """
    Feature: test scatter ops auto parallel
    Description: test scatter sub mix split
    Expectation: compile success
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=64, full_batch=True)
    inputs = Tensor(np.ones([32 * 4, 64]).astype(np.float32))
    input_shape = [32 * 4, 64]
    indices_shape = [4, 4]
    updates_shape = [4, 4, 64]
    strategy1 = ((8, 8), (1, 1), (1, 1, 8))
    strategy2 = ((1, 1), (1, 1))
    net = Net(input_shape, indices_shape, updates_shape, strategy1, strategy2, "Sub")
    compile_net(net, inputs)
