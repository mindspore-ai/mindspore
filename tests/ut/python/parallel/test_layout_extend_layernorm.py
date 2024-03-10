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


def compile_net(net, input_x):
    net.set_auto_parallel()
    net.set_train()
    phase, _ = _cell_graph_executor.compile(net, input_x)
    return phase


class Net(nn.Cell):
    def __init__(self, x_gamma, x_beta, in_layout, out_layout=None, begin_norm_axis=1):
        super().__init__()
        self.relu = P.ReLU()
        self.layernorm = P.LayerNorm(begin_norm_axis).shard(in_strategy=in_layout,
                                                            out_strategy=out_layout)
        self.gamma = Parameter(x_gamma, "gamma")
        self.beta = Parameter(x_beta, "beta")

    def construct(self, y):
        out1, _, _ = self.layernorm(y, self.gamma, self.beta)
        out2 = self.relu(out1)
        out = out1 + out2
        return out

x = Tensor(np.ones([128, 16, 32]), dtype=ms.float32)
gamma = Tensor(np.ones([16, 32]), dtype=ms.float32)
beta = Tensor(np.ones([16, 32]), dtype=ms.float32)

def test_layout_layernorm_base():
    """
    Feature: test layout extend
    Description: dev_num is 8.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 2, 2), ("dp", "sp", "mp"))
    layout1 = (layout("dp", "sp", "None"), layout("sp", "None"), layout("sp", "None"))
    net = Net(gamma, beta, layout1, begin_norm_axis=2)
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_parameter_shape('gamma', [8, 32])


def test_layout_layernorm_multi_shard():
    """
    Feature: test layout extend for multi shard
    Description: dev_num is 16.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    layout = Layout((2, 2, 2, 2), ("dp", "sp", "vp", "mp"))
    layout1 = (layout(("dp", "mp"), "sp", "None"), layout("sp", "None"), layout("sp", "None"))
    net = Net(gamma, beta, layout1, begin_norm_axis=2)
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_parameter_shape('gamma', [8, 32])


def test_layout_layernorm_multi_shard1():
    """
    Feature: test layout extend for multi shard
    Description: dev_num is 16.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    layout = Layout((2, 2, 2, 2), ("dp", "sp", "vp", "mp"))
    layout1 = (layout("dp", ("sp", "mp"), "None"), layout(("sp", "mp"), "None"), layout(("sp", "mp"), "None"))
    net = Net(gamma, beta, layout1, begin_norm_axis=2)
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_parameter_shape('gamma', [4, 32])


def test_layout_layernorm_out_check():
    """
    Feature: test layout extend for output layout check
    Description: dev_num is 16.
    Expectation: compile failed, throw exception
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    layout = Layout((2, 2, 2, 2), ("dp", "sp", "vp", "mp"))
    layout1 = (layout(("dp", "mp"), "sp", "None"), layout("sp", "None"), layout("sp", "None"))
    out_layout = (layout(("dp", "mp"), "sp", "None"), layout(("dp", "mp"), "sp", "None"),
                  layout(("dp", "mp"), "sp", "None"))
    net = Net(gamma, beta, layout1, out_layout=out_layout, begin_norm_axis=2)
    with pytest.raises(RuntimeError):
        compile_net(net, x)


def test_layout_layernorm_multi_shard_with_grad():
    """
    Feature: test layout extend with grad
    Description: dev_num is 16.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    layout = Layout((2, 2, 2, 2), ("dp", "sp", "vp", "mp"))
    layout1 = (layout(("dp", "mp"), "sp", "None"), layout("sp", "None"), layout("sp", "None"))
    net = GradWrap(NetWithLoss(Net(gamma, beta, layout1, begin_norm_axis=2)))
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_parameter_shape('network.network.gamma', [8, 32])
