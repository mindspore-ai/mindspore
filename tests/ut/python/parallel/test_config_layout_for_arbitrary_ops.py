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

import pytest
import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore import context
from mindspore.common.api import _cell_graph_executor
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.parallel.shard import Layout
from tests.ut.python.ops.test_math_ops import VirtualLoss
from parallel.utils.utils import ParallelValidator


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


grad_all = C.GradOperation(get_all=True)


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, y):
        predict = self.network(y)
        return self.loss(predict)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, y):
        return grad_all(self.network)(y)


def compile_net(net, x, y, z):
    net.set_auto_parallel()
    net.set_train()
    phase, _ = _cell_graph_executor.compile(net, x, y, z)
    return phase


class Net(nn.Cell):
    def __init__(self, in_layout, out_layout, self_define_shard=True):
        super().__init__()
        self.tensor_scatter_update = P.TensorScatterUpdate()
        self.tensor_scatter_update.shard(in_strategy=in_layout, out_strategy=out_layout)
        self.tensor_scatter_update.add_prim_attr("self_define_shard", self_define_shard)
        self.relu = P.ReLU()
        self.mul = P.Mul()

    def construct(self, input_x, indices, update):
        out = self.relu(input_x)
        out = self.tensor_scatter_update(out, indices, update)
        out = self.mul(out, 2)
        return out


class Net2(nn.Cell):
    def __init__(self, in_layout, out_layout, self_define_shard=True):
        super().__init__()
        self.add = P.Add()
        self.add.shard(in_strategy=in_layout, out_strategy=out_layout)
        self.add.add_prim_attr("self_define_shard", self_define_shard)
        self.relu = P.ReLU()
        self.mul = P.Mul()

    def construct(self, x, y, z):
        out = self.relu(x)
        out = self.add(out, y)
        out = self.mul(out, z)
        return out


class Net3(nn.Cell):
    def __init__(self, in_layout, out_layout, self_define_shard=True):
        super().__init__()
        self.layernorm = P.LayerNorm(begin_norm_axis=2)
        self.layernorm.shard(in_strategy=in_layout, out_strategy=out_layout)
        self.layernorm.add_prim_attr("self_define_shard", self_define_shard)
        self.relu = P.ReLU()
        self.mul = P.Mul()
        self.gamma = Parameter(Tensor(np.ones([16, 32]), dtype=ms.float32))
        self.beta = Parameter(Tensor(np.ones([16, 32]), dtype=ms.float32))

    def construct(self, x, y, z):
        out = self.relu(x)
        out = self.layernorm(out, self.beta, self.gamma)
        out = self.mul(out[0], z)
        return out


def test_config_layout_for_ops_success():
    """
    Feature: test layout extend
    Description: dev_num is 2.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=2, global_rank=0)
    layout = Layout((2, 1), ("dp", "mp"))
    layout1 = (layout("dp", "mp", "None"), layout("dp", "mp", "None"), layout("dp", "mp", "None"))
    layout2 = (layout("dp", "mp", "None"),)
    net = Net(layout1, layout2)
    input_x = Tensor(np.zeros((2, 2, 3)).astype(np.float32))
    indices = Tensor(np.array([[[0, 0], [1, 1]], [[0, 0], [1, 1]]]).astype(np.int32))
    update = Tensor(np.ones((2, 2, 3)).astype(np.float32))
    compile_net(net, input_x, indices, update)


def test_config_layout_for_parallel_supported_ops_success():
    """
    Feature: test layout extend
    Description: dev_num is 2.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=2, global_rank=0)
    layout = Layout((2, 1), ("dp", "mp"))
    layout1 = (layout("dp", "mp"), layout("dp", "mp"))
    layout2 = (layout("dp", "mp"),)
    net = Net2(layout1, layout2)
    x = Tensor(np.ones((2, 2)).astype(np.float32))
    y = Tensor(np.ones((2, 2)).astype(np.float32))
    z = Tensor(np.ones((2, 2)).astype(np.float32))
    compile_net(net, x, y, z)


def test_config_layout_for_parallel_supported_ops_failed():
    """
    Feature: test layout extend
    Description: dev_num is 2.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=2, global_rank=0)
    layout1 = ((2, 1,),)
    net = Net2(layout1, None)
    x = Tensor(np.ones((2, 2)).astype(np.float32))
    y = Tensor(np.ones((2, 2)).astype(np.float32))
    z = Tensor(np.ones((2, 2)).astype(np.float32))
    with pytest.raises(RuntimeError):
        compile_net(net, x, y, z)


def test_config_layout_for_ops_no_attr():
    """
    Feature: test layout extend
    Description: dev_num is 2.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=2, global_rank=0)
    layout = Layout((2, 1), ("dp", "mp"))
    layout1 = (layout("dp", "mp", "None"), layout("dp", "mp", "None"), layout("dp", "mp", "None"))
    layout2 = (layout("dp", "mp", "None"),)
    net = Net(layout1, layout2, False)
    input_x = Tensor(np.zeros((2, 2, 3)).astype(np.float32))
    indices = Tensor(np.array([[[0, 0], [1, 1]], [[0, 0], [1, 1]]]).astype(np.int32))
    update = Tensor(np.ones((2, 2, 3)).astype(np.float32))
    with pytest.raises(RuntimeError):
        compile_net(net, input_x, indices, update)


def test_config_layout_for_ops_not_dividable():
    """
    Feature: test layout extend
    Description: dev_num is 2.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)
    layout = Layout((4, 1), ("dp", "mp"))
    layout1 = (layout("dp", "mp", "None"), layout("dp", "mp", "None"), layout("dp", "mp", "None"))
    layout2 = (layout("dp", "mp", "None"),)
    net = Net(layout1, layout2, False)
    input_x = Tensor(np.zeros((2, 2, 3)).astype(np.float32))
    indices = Tensor(np.array([[[0, 0], [1, 1]], [[0, 0], [1, 1]]]).astype(np.int32))
    update = Tensor(np.ones((2, 2, 3)).astype(np.float32))
    with pytest.raises(RuntimeError):
        compile_net(net, input_x, indices, update)


def test_config_layout_for_ops_incorrect_dtype():
    """
    Feature: test layout extend
    Description: dev_num is 2.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)
    layout = Layout((4, 1), ("dp", "mp"))
    layout1 = (layout("dp", "mp", "None"), layout("dp", "mp", "None"), layout("dp", "mp", "None"))
    layout2 = (layout("dp", "mp", "None"),)
    net = Net(layout1, layout2, 1)
    input_x = Tensor(np.zeros((2, 2, 3)).astype(np.float32))
    indices = Tensor(np.array([[[0, 0], [1, 1]], [[0, 0], [1, 1]]]).astype(np.int32))
    update = Tensor(np.ones((2, 2, 3)).astype(np.float32))
    with pytest.raises(RuntimeError):
        compile_net(net, input_x, indices, update)


def test_config_layout_for_multi_out_ops():
    """1
    Feature: test layout extend
    Description: dev_num is 2.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 2, 2), ("dp", "sp", "mp"))
    layout1 = (layout("dp", "sp", "None"), layout("sp", "None"), layout("sp", "None"))
    layout2 = (layout("dp", "mp", "None"), layout("dp", "mp", "None"), layout("dp", "mp", "None"))
    net = Net3(layout1, layout2, True)
    x = Tensor(np.ones([128, 16, 32]), dtype=ms.float32)
    y = None
    z = Tensor(np.ones((1,)).astype(np.float32))
    phase = compile_net(net, x, y, z)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('LayerNorm-0', ['Reshape-1'])
