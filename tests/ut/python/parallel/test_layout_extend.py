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
import pytest

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

class Net2(nn.Cell):
    def __init__(self, weight, in_layout, out_layout=None):
        super().__init__()
        self.add = P.Add()
        self.matmul1 = P.MatMul().shard(in_strategy=in_layout, out_strategy=out_layout)
        self.relu = P.ReLU()
        self.w = Parameter(weight, "w1")

    def construct(self, y):
        y = self.add(y, y)
        out1 = self.matmul1(y, self.w)
        out1 = self.relu(out1)
        return out1

class Net3(nn.Cell):
    def __init__(self, weight, in_layout1, in_layout2):
        super().__init__()
        self.matmul1 = P.BatchMatMul().shard(in_strategy=in_layout2)
        self.gelu = P.GeLU().shard(in_strategy=in_layout1)
        self.w = Parameter(weight, "w1")

    def construct(self, y):
        out1 = self.gelu(y)
        out1 = self.matmul1(out1, self.w)
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
    layout = Layout((2, 2, 2), ("dp", "sp", "mp"))
    layout1 = (layout("dp", "sp"), layout("sp", "mp"))
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
    layout = Layout((2, 2, 2), ("dp", "sp", "mp"))
    layout1 = (layout("dp", "sp"), layout("sp", "mp"))
    out_layout = (layout(("dp", "sp"), "mp"),)
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
    layout = Layout((2, 2, 2, 2), ("dp", "sp", "vp", "mp"))
    layout1 = (layout(("dp", "mp"), "sp"), layout("sp", "vp"))
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
    layout = Layout((2, 2, 2, 2), ("dp", "sp", "vp", "mp"))
    layout1 = (layout(("dp", "mp"), "sp"), layout("sp", "vp"))
    out_layout = (layout(("dp", "mp", "sp"), "vp"),)
    net = Net(w, layout1, out_layout)
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
    layout = Layout((2, 2, 2, 2), ("dp", "sp", "vp", "mp"))
    layout1 = (layout("dp", ("sp", "mp")), layout(("sp", "mp"), "vp"))
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
    layout = Layout((2, 2, 2, 2), ("dp", "sp", "vp", "mp"))
    layout1 = (layout("dp", ("sp", "mp")), layout(("sp", "mp"), "vp"))
    out_layout = (layout(("dp", "sp", "mp"), "vp"),)
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
    layout = Layout((2, 2, 2, 2), ("dp", "sp", "vp", "mp"))
    layout1 = (layout("dp", ("sp", "mp")), layout(("sp", "mp"), "vp"))
    out_layout = (layout(("dp", "sp", "mp"), "vp"),)
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
    layout = Layout((4, 2, 2, 2), ("dp", "sp", "vp", "mp"))
    layout1 = (layout("dp", ("sp", "mp")), layout(("sp", "mp"), "vp"))
    out_layout = (layout(("dp", "sp", "mp"), "vp"),)
    net = GradWrap(NetWithLoss(Net(w, layout1, out_layout)))
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    context.reset_auto_parallel_context()
    assert validator.check_parameter_layout('network.network.w',
                                            ([4, 2, 2, 2], [2, 0, 1], [256, 512], 0, True, '2-16557109384257890687'))
    assert validator.check_node_inputs_has('ReduceScatter-0', ['MatMul'])

def test_layout_extend_reduce_axis_multi_shard_reduce_scatter_including_dev1():
    """
    Feature: test layout extend
    Description: dev_num is 8, reduce dim multi shard.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 2, 2, 1), ("dp", "sp", "vp", "mp"))
    layout1 = (layout("dp", ("sp", "mp")), layout(("sp", "mp"), "vp"))
    out_layout = (layout(("dp", "sp", "mp"), "vp"),)
    net = GradWrap(NetWithLoss(Net(w, layout1, out_layout)))
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_parameter_shape('network.network.w', [512, 512])
    assert validator.check_node_inputs_has('ReduceScatter-0', ['MatMul'])

def test_layout_extend_reduce_axis_multi_shard_reduce_scatter_including_axis_none():
    """
    Feature: test layout extend
    Description: dev_num is 8, reduce dim multi shard.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 2, 2), ("dp", "sp", "mp"))
    layout1 = (layout("dp", ("sp", "mp")), layout(("sp", "mp"), "None"))
    out_layout = (layout(("dp", "sp", "mp"), "None"),)
    net = GradWrap(NetWithLoss(Net(w, layout1, out_layout)))
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_parameter_shape('network.network.w', [256, 1024])
    assert validator.check_node_inputs_has('ReduceScatter-0', ['MatMul'])

def test_layout_extend_reduce_axis_multi_shard_reduce_scatter_including_reduce_axis_none():
    """
    Feature: test layout extend
    Description: dev_num is 8, reduce dim multi shard with None.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 2, 2), ("dp", "sp", "mp"))
    layout1 = (layout("dp", ("sp", "None")), layout(("sp", "None"), "mp"))
    out_layout = (layout(("dp", "sp", "None"), "mp"),)
    net = GradWrap(NetWithLoss(Net(w, layout1, out_layout)))
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_parameter_shape('network.network.w', [512, 512])
    assert validator.check_node_inputs_has('ReduceScatter-0', ['MatMul'])

def test_layout_extend_reduce_axis_multi_shard_reduce_scatter_including_reduce_axis_none_and_not_full():
    """
    Feature: test layout extend
    Description: dev_num is 8, reduce dim multi shard with None, and not shard full.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 2, 2), ("dp", "sp", "mp"))
    layout1 = (layout("dp", ("sp", "None")), layout(("sp", "None"), "None"))
    out_layout = (layout(("dp", "sp", "None"), "None"),)
    net = GradWrap(NetWithLoss(Net(w, layout1, out_layout)))
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_parameter_shape('network.network.w', [512, 1024])
    assert validator.check_node_inputs_has('ReduceScatter-0', ['MatMul'])

def test_layout_extend_error_case1():
    """
    Feature: test layout extend
    Description: error case, the strategy is incorrect.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 2, 2), ("dp", "sp", "mp"))
    layout1 = (layout("dp", ("sp", "mp")), layout(("sp", "mp"), "dp"))
    net = GradWrap(NetWithLoss(Net(w, layout1)))
    with pytest.raises(RuntimeError):
        compile_net(net, x)

def test_layout_extend_error_case2():
    """
    Feature: test layout extend
    Description: error case, the strategy is incorrect.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 2, 2), ("dp", "sp", "mp"))
    with pytest.raises(ValueError):
        layout1 = (layout("dp", ("sp", "mp")), layout(("sp", "mp"), "mp"))
        net = GradWrap(NetWithLoss(Net(w, layout1)))
        compile_net(net, x)

def test_layout_extend_only_reshape_redis():
    """
    Feature: test layout extend
    Description: dev_num is 8, only reshape redistribution.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((4, 1, 2), ("dp", "sp", "mp"))
    layout1 = (layout(("dp", "mp"), "sp"), layout("sp", "None"))
    net = GradWrap(NetWithLoss(Net2(w, layout1)))
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_parameter_shape('network.network.w', [1024, 1024])

def test_layout_extend_moe():
    """
    Feature: test layout extend
    Description: dev_num is 16, modify MOE.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16,
                                      global_rank=0, enable_alltoall=True)
    layout = Layout((2, 2, 4), ("vp", "dp", "mp"))
    layout1 = (layout("None", ("dp", "vp"), "None"),)
    layout2 = (layout(("dp", "vp"), "None", "None"), layout(("dp", "vp"), "None", "mp"))
    x1 = Tensor(np.ones([8, 1024, 1024]), dtype=ms.float32)
    w1 = Tensor(np.ones([8, 1024, 1024]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(Net3(w1, layout1, layout2)))
    phase = compile_net(net, x1)
    validator = ParallelValidator(net, phase)
    assert validator.check_parameter_shape('network.network.w', [2, 1024, 256])
