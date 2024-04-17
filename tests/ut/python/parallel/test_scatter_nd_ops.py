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
import pytest
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, Model, Parameter
from mindspore.ops import operations as P
from mindspore import context
from mindspore.common.api import _cell_graph_executor
from parallel.utils.utils import ParallelValidator

scatter_nd_ops_map = {"Add": P.ScatterNdAdd(), "Update": P.ScatterNdUpdate(), "Sub": P.ScatterNdSub()}
tensor_scatter_ops_map = {"Add": P.TensorScatterAdd(), "Update": P.TensorScatterUpdate(),
                          "Sub": P.TensorScatterSub(), "Mul": P.TensorScatterMul(), "Div": P.TensorScatterDiv()}


# The shape of input:   [A, B, C, D], the strategy of input: (a, b, c, d)
# The shape of indices: [Q, W, 2], the strategy of indices: (1, 1, 1)
# here the 2 respect to the size of [A, B]
# The shape of updates: [Q, W, C, D], the strategy of updates: (1, 1, c, d)
# The shape of output:  [A, B, C, D], the strategy of output: (a, b, c, d)
class Net(nn.Cell):
    """Net definition"""

    def __init__(self, strategy1=None, strategy2=None, ops_type="Add"):
        super(Net, self).__init__()
        self.inputs = Parameter(Tensor(np.ones([32, 64, 128]).astype(np.float32)), "input")
        self.indices = Tensor(np.ones([4, 2]).astype(np.int32))
        self.updates = Tensor(np.ones([4, 128]).astype(np.float32))
        self.scatter_ops = scatter_nd_ops_map.get(ops_type).shard(strategy1)
        self.add = P.TensorAdd().shard(strategy2)
        self.relu = P.ReLU()

    def construct(self, x):
        out = self.scatter_ops(self.inputs, self.indices, self.updates)
        out = self.add(x, out)
        out = self.relu(out)
        return out


class Net1(nn.Cell):
    """Net definition"""

    def __init__(self, strategy1=None, strategy2=None, ops_type="Add"):
        super(Net1, self).__init__()
        self.inputs = Parameter(Tensor(np.ones([32, 64, 128]).astype(np.float32)), "input")
        self.indices = Tensor(np.ones([4, 3]).astype(np.int32))
        self.updates = Tensor(np.ones([4]).astype(np.float32))
        self.scatter_ops = scatter_nd_ops_map.get(ops_type).shard(strategy1)
        self.add = P.TensorAdd().shard(strategy2)
        self.relu = P.ReLU()

    def construct(self, x):
        out = self.scatter_ops(self.inputs, self.indices, self.updates)
        out = self.add(x, out)
        out = self.relu(out)
        return out


class Net2(nn.Cell):
    """Net definition"""

    def __init__(self, strategy1=None, strategy2=None, ops_type="Add"):
        super(Net2, self).__init__()
        self.indices = Tensor(np.ones([4, 3]).astype(np.int32))
        self.updates = Tensor(np.ones([4]).astype(np.float32))
        self.scatter_ops = tensor_scatter_ops_map.get(ops_type).shard(strategy1)
        self.add = P.TensorAdd().shard(strategy2)
        self.relu = P.ReLU()

    def construct(self, inputs, x):
        inputs = self.relu(inputs)
        out = self.scatter_ops(inputs, self.indices, self.updates)
        out = self.add(x, out)
        out = self.relu(out)
        return out


class Net3(nn.Cell):
    """Net definition"""

    def __init__(self, strategy1=None, strategy2=None, ops_type="Add"):
        super(Net3, self).__init__()
        self.inputs = Parameter(Tensor(np.ones([8, 8, 64]).astype(np.float32)), "input")
        self.inputs1 = Parameter(Tensor(np.ones([8, 8, 64]).astype(np.float32)), "input1")
        self.indices = Tensor(np.ones([8, 8, 2]).astype(np.int32))
        self.updates = Tensor(np.ones([8, 8, 64]).astype(np.float32))
        self.scatter_ops = scatter_nd_ops_map.get(ops_type).shard(strategy1)
        self.add = P.TensorAdd().shard(strategy2)
        self.relu = P.ReLU()

    def construct(self, x):
        out = self.scatter_ops(self.inputs, self.indices, self.updates)
        out = self.scatter_ops(self.inputs1, self.indices, out)
        out = self.add(x, out)
        out = self.relu(out)
        return out


class Net4(nn.Cell):
    """Net definition"""

    def __init__(self, strategy1=None, ops_type="Add"):
        super(Net4, self).__init__()
        self.indices = Tensor(np.ones([4, 3]).astype(np.int32))
        self.updates = Tensor(np.ones([4]).astype(np.float32))
        self.scatter_ops = tensor_scatter_ops_map.get(ops_type).shard(strategy1)
        self.relu = P.ReLU()

    def construct(self, inputs, x):
        inputs = self.relu(inputs)
        out = self.scatter_ops(inputs, self.indices, self.updates)
        return out


def compile_net(net, *inputs):
    net.set_auto_parallel()
    net.set_train(False)
    phase, _ = _cell_graph_executor.compile(net, *inputs)
    context.reset_auto_parallel_context()
    return phase


def test_scatter_nd_add():
    """
    Feature: distribute operator scatter_nd_add in auto parallel.
    Description: scatter_nd_add net with sharding updating strategy in semi auto parallel.
    Expectation: assert ok.
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, full_batch=True)
    inputs = Tensor(np.ones([32, 64, 128]).astype(np.float32))
    strategy1 = ((1, 2, 4), (1, 1), (1, 4))
    strategy2 = ((1, 2, 4), (1, 2, 4))
    net = Net(strategy1, strategy2)
    phase = compile_net(net, inputs)
    validator = ParallelValidator(net, phase)
    # check layout
    inputs_expect_layout = ([2, 4], [-1, 1, 0], [32, 32, 32], 0, True, '')
    assert validator.check_parameter_layout('input', inputs_expect_layout)


def test_two_scatter_nd_add():
    """
    Feature: distribute operator scatter_nd_add in auto parallel.
    Description: scatter_nd_add net with sharding updating strategy in semi auto parallel.
    Expectation: assert ok.
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, full_batch=True)
    inputs = Tensor(np.ones([8, 8, 64]).astype(np.float32))
    strategy1 = ((1, 8, 1), (1, 1, 1), (1, 1, 1))
    strategy2 = ((1, 8, 1), (1, 8, 1))
    net = Net3(strategy1, strategy2)
    phase = compile_net(net, inputs)
    validator = ParallelValidator(net, phase)
    # check sub_graph
    sub_graph = {
        'ScatterNdAdd-0': ['input', 'Sub-1', 'Mul-3'],
        'ScatterNdAdd-1': ['input1', 'Sub-0', 'Mul-5'],
    }
    assert validator.check_graph_structure(sub_graph)


def test_scatter_nd_wrong_strategy():
    """
    Feature: distribute operator scatter_nd_add in auto parallel.
    Description: scatter_nd_add net with wrong strategy in semi auto parallel.
    Expectation: raise runtime error.
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, full_batch=True)
    inputs = Tensor(np.ones([32, 64, 128]).astype(np.float32))
    strategy1 = ((1, 2, 4), (1, 1), (1, 2))
    strategy2 = ((1, 2, 4), (1, 2, 4))
    net = Net(strategy1, strategy2)
    model = Model(net)
    with pytest.raises(RuntimeError):
        model.predict(inputs)
    context.reset_auto_parallel_context()


def test_scatter_nd_sub():
    """
    Feature: distribute operator scatter_nd_sub in auto parallel.
    Description: scatter_nd_sub net with sharding input gather dims in semi auto parallel.
    Expectation: assert ok.
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, full_batch=True)
    inputs = Tensor(np.ones([32, 64, 128]).astype(np.float32))
    strategy1 = ((2, 2, 2), (1, 1), (1,))
    strategy2 = ((2, 2, 2), (2, 2, 2))
    net = Net1(strategy1, strategy2, ops_type="Sub")
    phase = compile_net(net, inputs)
    validator = ParallelValidator(net, phase)
    # check layout
    inputs_expect_layout = ([2, 2, 2], [2, 1, 0], [16, 32, 64], 0, True, '')
    assert validator.check_parameter_layout('input', inputs_expect_layout)
    # check sub_graph
    sub_graph = {
        'ScatterNdSub-0': ['input', 'Sub-0', 'Mul-2'],
        'FloorDiv-0': ['_GetTensorSlice-0', 'Reshape-0'],
    }
    assert validator.check_graph_structure(sub_graph)


def test_scatter_nd_update():
    """
    Feature: distribute operator scatter_nd_update in auto parallel.
    Description: scatter_nd_update net with sharding input gather dims and tables in semi auto parallel.
    Expectation: assert ok.
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, full_batch=True)
    inputs = Tensor(np.ones([32, 64, 128]).astype(np.float32))
    strategy1 = ((2, 2, 2), (1, 1), (1, 2))
    strategy2 = ((2, 2, 2), (2, 2, 2))
    net = Net(strategy1, strategy2, ops_type="Update")
    model = Model(net)
    with pytest.raises(RuntimeError):
        model.predict(inputs)
    context.reset_auto_parallel_context()


def test_tensor_scatter_add():
    """
    Feature: distribute operator tensor_scatter_add in auto parallel.
    Description: tensor_scatter_update net with sharding input gather dims and tables in semi auto parallel.
    Expectation: assert ok.
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, full_batch=True)
    input1 = Tensor(np.ones([32, 64, 128]).astype(np.float32))
    input2 = Tensor(np.ones([32, 64, 128]).astype(np.float32))
    strategy1 = ((2, 2, 2), (1, 1), (1,))
    strategy2 = ((1, 2, 2), (1, 2, 2))
    net = Net2(strategy1, strategy2, ops_type="Add")
    phase = compile_net(net, input1, input2)
    validator = ParallelValidator(net, phase)
    # check sub_graph
    sub_graph = {
        'TensorScatterAdd-0': ['Reshape-1', 'Sub-0', 'Mul-2'],
        'Equal-0': ['Sub-1', 'Minimum-0'],
        'AllGather-2': ['TensorScatterAdd-0']
    }
    assert validator.check_graph_structure(sub_graph)


def test_tensor_scatter_mul():
    """
    Feature: distribute operator tensor_scatter_mul in auto parallel.
    Description: tensor_scatter_update net with sharding input gather dims and tables in semi auto parallel.
    Expectation: assert ok.
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, full_batch=True)
    input1 = Tensor(np.ones([32, 64, 128]).astype(np.float32))
    input2 = Tensor(np.ones([32, 64, 128]).astype(np.float32))
    strategy1 = ((2, 2, 2), (1, 1), (1,))
    strategy2 = ((1, 2, 2), (1, 2, 2))
    net = Net2(strategy1, strategy2, ops_type="Mul")
    phase = compile_net(net, input1, input2)
    validator = ParallelValidator(net, phase)
    # check sub_graph
    sub_graph = {
        'TensorScatterMul-0': ['Reshape-1', 'Sub-0', 'Add-0'],
        'Equal-0': ['Sub-1', 'Minimum-0'],
        'AllGather-2': ['TensorScatterMul-0']
    }
    assert validator.check_graph_structure(sub_graph)


def test_tensor_scatter_mul_auto_parallel():
    """
    Feature: distribute operator tensor_scatter_mul in auto parallel.
    Description: tensor_scatter_update net with sharding input gather dims and tables in auto parallel.
    Expectation: assert ok.
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="dynamic_programming", device_num=8,
                                      full_batch=True)
    input1 = Tensor(np.ones([32, 64, 128]).astype(np.float32))
    input2 = Tensor(np.ones([32, 64, 128]).astype(np.float32))
    strategy1 = ((2, 2, 2), (1, 1), (1,))
    strategy2 = None
    net = Net2(strategy1, strategy2, ops_type="Mul")
    phase = compile_net(net, input1, input2)
    validator = ParallelValidator(net, phase)
    # check sub_graph
    sub_graph = {
        'TensorScatterMul-0': ['ReLU-0', 'Sub-0', 'Add-0'],
        'Equal-0': ['Sub-1', 'Minimum-0']
    }
    assert validator.check_graph_structure(sub_graph)


def test_scatter_nd_add_dynamic_constraint():
    """
    Feature: distribute operator scatter_nd_add dynamic shape
    Description: need replace graph
    Expectation: compile failed
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, full_batch=False)
    input1 = Tensor(shape=[None, 64, 128], dtype=ms.float32)
    input2 = Tensor(np.ones([32, 64, 128]).astype(np.float32))
    strategy1 = ((2, 2, 2), (1, 1), (1,))
    net = Net4(strategy1, ops_type="Add")
    with pytest.raises(RuntimeError):
        compile_net(net, input1, input2)
