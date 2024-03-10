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
        self.add = P.Add().shard(in_strategy=in_layout, out_strategy=out_layout)
        self.relu = P.ReLU()
        self.w = Parameter(weight, "w1")

    def construct(self, y):
        out1 = self.add(y, self.w)
        out2 = self.relu(out1)
        out = out1 + out2
        return out

x = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
w = Tensor(np.ones([1024, 1024]), dtype=ms.float32)

input_1024 = Tensor(np.ones([1024]), dtype=ms.float32)
input_1_1024 = Tensor(np.ones([1, 1024]), dtype=ms.float32)
input_1024_1024 = Tensor(np.ones([1024, 1024]), dtype=ms.float32)


def test_layout_extend_add_same_shape_same_shard():
    """
    Feature: test layout extend
    Description: dev_num is 8.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 2, 2), ("dp", "sp", "mp"))
    layout1 = (layout(("dp", "sp"), "mp"), layout(("dp", "sp"), "mp"))
    first, second = input_1024_1024, x
    net = Net(second, layout1)
    phase = compile_net(net, first)
    validator = ParallelValidator(net, phase)
    assert validator.check_parameter_shape('w1', [256, 512])

def test_layout_extend_add_same_shape_wrong_shard():
    """
    Feature: test layout extend
    Description: dev_num is 8.
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 2, 2), ("dp", "sp", "mp"))
    layout1 = (layout(("dp", "sp"), "mp"), layout(("dp", "mp"), "sp"))
    first, second = input_1024_1024, x
    net = Net(second, layout1)
    with pytest.raises(RuntimeError):
        compile_net(net, first)

def test_layout_extend_add_same_dim_broadcast():
    """
    Feature: test layout extend
    Description: dev_num is 8.
    Expectation: compile success, second input broadcast
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 2, 2), ("dp", "sp", "mp"))
    layout1 = (layout(("dp", "sp"), "mp"), layout("None", "mp"))
    first, second = input_1024_1024, input_1_1024
    net = Net(second, layout1)
    phase = compile_net(net, first)
    validator = ParallelValidator(net, phase)
    assert validator.check_parameter_shape('w1', [1, 512])

def test_layout_extend_add_different_dim_broadcast():
    """
    Feature: test layout extend
    Description: dev_num is 8.
    Expectation: compile success, second input broadcast
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 2, 2), ("dp", "sp", "mp"))
    layout1 = (layout(("dp", "sp"), "mp"), layout("mp",))
    first, second = input_1024_1024, input_1024
    net = Net(second, layout1)
    phase = compile_net(net, first)
    validator = ParallelValidator(net, phase)
    assert validator.check_parameter_shape('w1', [512])

def test_layout_extend_add_different_dim_broadcast_failed():
    """
    Feature: test layout extend
    Description: dev_num is 8.
    Expectation: compile success, second input broadcast
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 2, 2), ("dp", "sp", "mp"))
    layout1 = (layout(("dp", "sp"), "mp"), layout("None",))
    first, second = input_1024_1024, input_1024
    net = Net(second, layout1)
    with pytest.raises(RuntimeError):
        compile_net(net, first)

def test_layout_extend_add_same_shape_same_shard_outputlayout_not_allowed():
    """
    Feature: test layout extend
    Description: dev_num is 8.
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 2, 2), ("dp", "sp", "mp"))
    layout1 = (layout(("dp", "sp"), "mp"), layout(("dp", "sp"), "mp"))
    out_layout = (layout(("dp", "sp"), "mp"),)
    first, second = input_1024_1024, x
    net = Net(second, layout1, out_layout)
    with pytest.raises(RuntimeError):
        compile_net(net, first)
