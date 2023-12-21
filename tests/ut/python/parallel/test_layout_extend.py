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


def compile_net(net, input_x):
    net.set_auto_parallel()
    net.set_train()
    phase, _ = _cell_graph_executor.compile(net, input_x)
    return phase


class Net(nn.Cell):
    def __init__(self, weight, in_layout, out_layout=None):
        super().__init__()
        self.matmul1 = P.MatMul().shard(in_strategy=in_layout, out_strategy=out_layout)
        self.relu = P.ReLU()
        self.w = Parameter(weight, "w1")

    def construct(self, y):
        out1 = self.matmul1(y, self.w)
        out2 = self.relu(out1)
        out = out1 + out2
        return out

class Net1(nn.Cell):
    def __init__(self, weight, in_layout, out_layout=None):
        super().__init__()
        self.matmul1 = P.MatMul().shard(in_strategy=in_layout, out_strategy=out_layout)
        self.relu = P.ReLU()
        self.w = Parameter(weight, "w1")

    def construct(self, y):
        y = self.relu(y)
        out1 = self.matmul1(y, self.w)
        return out1

x = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
w = Tensor(np.ones([1024, 1024]), dtype=ms.float32)


def test_layout_extend_base():
    """
    Feature: test layout extend
    Description: dev_num is 8.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout(2, 2, 2)
    layout1 = (layout(2, 1), layout(1, 0))
    net = Net(w, layout1)
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_parameter_shape('w1', [512, 512])


def test_layout_extend_base_reduce_scatter():
    """
    Feature: test layout extend
    Description: dev_num is 8.
    Expectation: compile success, forward reduce_scatter
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout(2, 2, 2)
    layout1 = (layout(2, 1), layout(1, 0))
    out_layout = (layout((2, 1), 0),)
    net = Net(w, layout1, out_layout)
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_parameter_shape('w1', [512, 512])
    assert validator.check_node_inputs_has('ReduceScatter-0', ['MatMul'])

def test_layout_extend_batch_multi_shard():
    """
    Feature: test layout extend
    Description: dev_num is 16, batch dim multi shard.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    layout = Layout(2, 2, 2, 2)
    layout1 = (layout((3, 0), 2), layout(2, 1))
    net = Net(w, layout1)
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_parameter_shape('w1', [512, 512])

def test_layout_extend_batch_multi_shard_reduce_scatter():
    """
    Feature: test layout extend
    Description: dev_num is 16, batch dim multi shard.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    layout = Layout(2, 2, 2, 2)
    layout1 = (layout((3, 0), 2), layout(2, 1))
    out_layout = (layout((3, 0, 2), 1),)
    net = Net(w, layout1, out_layout)
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_parameter_shape('w1', [512, 512])
    assert validator.check_node_inputs_has('ReduceScatter-0', ['MatMul'])

def test_layout_extend_batch_multi_shard_reduce_scatter_net1():
    """
    Feature: test layout extend
    Description: dev_num is 16, batch dim multi shard.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    layout = Layout(2, 2, 2, 2)
    layout1 = (layout((3, 0), 2), layout(2, 1))
    out_layout = (layout((3, 0, 2), 1),)
    net = Net1(w, layout1, out_layout)
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_parameter_shape('w1', [512, 512])
    assert validator.check_node_inputs_has('ReduceScatter-0', ['MatMul'])

def test_layout_extend_reduce_axis_multi_shard():
    """
    Feature: test layout extend
    Description: dev_num is 16, reduce dim multi shard.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    layout = Layout(2, 2, 2, 2)
    layout1 = (layout(3, (2, 0)), layout((2, 0), 1))
    net = Net(w, layout1)
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_parameter_shape('w1', [256, 512])

def test_layout_extend_reduce_axis_multi_shard_reduce_scatter():
    """
    Feature: test layout extend
    Description: dev_num is 16, reduce dim multi shard.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    layout = Layout(2, 2, 2, 2)
    layout1 = (layout(3, (2, 0)), layout((2, 0), 1))
    out_layout = (layout((3, 2, 0), 1),)
    net = GradWrap(NetWithLoss(Net(w, layout1, out_layout)))
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_parameter_shape('network.network.w', [256, 512])
    assert validator.check_node_inputs_has('ReduceScatter-0', ['MatMul'])

def test_layout_extend_reduce_axis_multi_shard_reduce_scatter_opt_shard():
    """
    Feature: test layout extend
    Description: dev_num is 16, reduce dim multi shard, enable optimizer parallel.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0,
                                      enable_parallel_optimizer=True)
    layout = Layout(2, 2, 2, 2)
    layout1 = (layout(3, (2, 0)), layout((2, 0), 1))
    out_layout = (layout((3, 2, 0), 1),)
    net = GradWrap(NetWithLoss(Net(w, layout1, out_layout)))
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    context.reset_auto_parallel_context()
    assert validator.check_parameter_layout('network.network.w',
                                            ([2, 2, 2, 2], [2, 0, 1], [256, 512], 0, True, '2-16557109384257890687'))
    assert validator.check_node_inputs_has('ReduceScatter-0', ['MatMul'])

def test_layout_extend_reduce_axis_multi_shard_reduce_scatter_opt_shard_not_full():
    """
    Feature: test layout extend
    Description: dev_num is 32, reduce dim multi shard, enable optimizer parallel, opt_shard=2.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=32, global_rank=0,
                                      enable_parallel_optimizer=True,
                                      parallel_optimizer_config={"optimizer_weight_shard_size": 2})
    layout = Layout(4, 2, 2, 2)
    layout1 = (layout(3, (2, 0)), layout((2, 0), 1))
    out_layout = (layout((3, 2, 0), 1),)
    net = GradWrap(NetWithLoss(Net(w, layout1, out_layout)))
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    context.reset_auto_parallel_context()
    assert validator.check_parameter_layout('network.network.w',
                                            ([4, 2, 2, 2], [2, 0, 1], [256, 512], 0, True, '2-16557109384257890687'))
    assert validator.check_node_inputs_has('ReduceScatter-0', ['MatMul'])
