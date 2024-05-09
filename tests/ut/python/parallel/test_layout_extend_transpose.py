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
from mindspore import Tensor
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
    def __init__(self, in_layout, out_layout=None):
        super().__init__()
        self.transpose = P.Transpose().shard(in_strategy=in_layout, out_strategy=out_layout)
        self.relu = P.ReLU()

    def construct(self, y):
        out1 = self.transpose(y, (1, 0))
        out2 = self.relu(out1)
        out = out1 + out2
        return out


input_512_1024 = Tensor(np.ones([512, 1024]), dtype=ms.float32)


def test_layout_extend_transpose_without_output_layout1():
    """
    Feature: test layout extend
    Description: dev_num is 8.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 2, 2), ("dp", "sp", "mp"))
    layout1 = (layout(("dp", "sp"), "mp"),)
    net = Net(layout1)
    phase = compile_net(net, input_512_1024)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs("Transpose-0", ["Reshape-1", "(1, 0)"])


def test_layout_extend_transpose_without_output_layout2():
    """
    Feature: test layout extend
    Description: dev_num is 8.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 4), ("dp", "sp"))
    layout1 = (layout("dp", "sp"),)
    net = Net(layout1)
    phase = compile_net(net, input_512_1024)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs("Transpose-0", ["StridedSlice-1", "(1, 0)"])


def test_layout_extend_transpose_with_output_layout1():
    """
    Feature: test layout extend
    Description: dev_num is 8.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 2, 2), ("dp", "sp", "mp"))
    layout1 = (layout(("dp", "sp"), "mp"),)
    layout2 = (layout("mp", ("dp", "sp")),)
    net = Net(layout1, layout2)
    phase = compile_net(net, input_512_1024)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs("Transpose-0", ["Reshape-1", "(1, 0)"])


def test_layout_extend_transpose_with_output_layout2():
    """
    Feature: test layout extend
    Description: dev_num is 8.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout_in = Layout((2, 4), ("dp", "sp"))
    layout1 = (layout_in("dp", "sp"),)
    layout_out = Layout((4, 2), ("dp", "sp"))
    layout2 = (layout_out("dp", "sp"),)
    net = Net(layout1, layout2)
    phase = compile_net(net, input_512_1024)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs("Transpose-0", ["StridedSlice-1", "(1, 0)"])


def test_layout_extend_transpose_with_output_layout_fail1():
    """
    Feature: test layout extend
    Description: dev_num is 8.
    Expectation: compile fail
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout_in = Layout((2, 4), ("dp", "sp"))
    layout1 = (layout_in("dp", "sp"),)
    layout_out = Layout((2, 4), ("dp", "sp"))
    layout2 = (layout_out("dp", "sp"),)
    net = Net(layout1, layout2)
    with pytest.raises(RuntimeError):
        compile_net(net, input_512_1024)
