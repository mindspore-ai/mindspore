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
        self.matmul1.add_prim_attr("recompute_comm_op", True)
        self.relu = P.ReLU()
        self.w = Parameter(weight, "w1")

    def construct(self, y):
        out1 = self.matmul1(y, self.w)
        out2 = self.relu(out1)
        out = out1 + out2
        return out

class NetWithReshape(nn.Cell):
    def __init__(self, weight1, weight2, in_layout1, in_layout2, out_layout1=None, out_layout2=None):
        super().__init__()
        self.matmul1 = P.MatMul().shard(in_strategy=in_layout1, out_strategy=out_layout1)
        self.matmul2 = P.MatMul().shard(in_strategy=in_layout2, out_strategy=out_layout2)
        self.transpose = P.Transpose().shard(out_layout2)
        self.matmul2.add_prim_attr("recompute_comm_op", True)
        self.reshape = P.Reshape().add_prim_attr("recompute_comm_op", True)
        self.relu = P.ReLU()
        self.cast = P.Cast()
        self.gelu = P.GeLU()
        self.depend = P.Depend()
        self.w1 = Parameter(weight1, "w1")
        self.w2 = Parameter(weight2, "w2")

    def construct(self, y):
        y_new = self.gelu(y)
        y_new = self.cast(y_new, ms.float32)
        y_new = self.reshape(y_new, (1024, 1024))
        out1 = self.matmul1(y_new, self.w1)
        out1 = self.cast(out1, ms.float16)
        out1 = self.transpose(out1, (1, 0))
        out1 = self.reshape(out1, (512, 2048))
        out2 = self.matmul2(out1, self.w2)
        out2 = self.reshape(out2, (1024, 1024))
        return self.relu(out2)

class NetTwoMatMul(nn.Cell):
    def __init__(self, weight1, weight2, in_layout1, in_layout2, out_layout1=None, out_layout2=None):
        super().__init__()
        self.matmul1 = P.MatMul().shard(in_strategy=in_layout1, out_strategy=out_layout1)
        self.matmul2 = P.MatMul().shard(in_strategy=in_layout2, out_strategy=out_layout2)
        self.matmul2.add_prim_attr("recompute_comm_op", True)
        self.relu = P.ReLU()
        self.cast = P.Cast()
        self.gelu = P.GeLU()
        self.depend = P.Depend()
        self.w1 = Parameter(weight1, "w1")
        self.w2 = Parameter(weight2, "w2")

    def construct(self, y):
        y = self.relu(y)
        y_new = self.gelu(y)
        y_new = self.cast(y_new, ms.float16)
        out1 = self.matmul1(y, self.w1)
        out1 = self.cast(out1, ms.float16)
        out1 = self.depend(out1, y_new)
        out2 = self.matmul2(out1, self.w2)
        return self.relu(out2) + y_new

def test_interleaved_base():
    """
    Feature: test micro interleaved
    Description: dev_num is 8.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 4, 2), ("dp", "mp", "interleaved_parallel"))
    layout1 = (layout(("dp", "interleaved_parallel"), "mp"), layout("mp", "None"))
    x = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    w = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(Net(w, layout1)))
    _ = compile_net(net, x)


def test_interleaved_two_matmul():
    """
    Feature: test micro interleaved using two matmul
    Description: dev_num is 16.
    Expectation: compile success, forward reduce_scatter
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    layout = Layout((2, 4, 2, 2), ("dp", "mp", "sp", "interleaved_parallel"))
    layout1 = (layout(("dp", "interleaved_parallel"), "mp"), layout("mp", "sp"))
    out_layout1 = (layout(("dp", "interleaved_parallel", "mp"), "sp"),)
    layout2 = (layout(("dp", "interleaved_parallel", "mp"), "sp"), layout("sp", "None"))
    out_layout2 = (layout(("dp", "interleaved_parallel", "mp", "sp"), "None"),)
    x = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    w1 = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    w2 = Tensor(np.ones([1024, 1024]), dtype=ms.float16)
    net = GradWrap(NetWithLoss(NetTwoMatMul(w1, w2, layout1, layout2, out_layout1, out_layout2)))
    phase = compile_net(net, x)
    _ = ParallelValidator(net, phase)

def test_interleaved_with_reshape():
    """
    Feature: test micro interleaved using two matmul
    Description: dev_num is 16.
    Expectation: compile success, forward reduce_scatter
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    layout = Layout((2, 4, 2, 2), ("dp", "mp", "sp", "interleaved_parallel"))
    layout1 = (layout(("dp", "interleaved_parallel"), "mp"), layout("mp", "sp"))
    out_layout1 = (layout(("dp", "interleaved_parallel", "mp"), "sp"),)
    layout2 = (layout(("dp", "interleaved_parallel", "mp"), "sp"), layout("sp", "None"))
    out_layout2 = (layout(("dp", "interleaved_parallel", "mp", "sp"), "None"),)
    x = Tensor(np.ones([1024, 1024]), dtype=ms.float16)
    w1 = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    w2 = Tensor(np.ones([2048, 2048]), dtype=ms.float16)
    net = GradWrap(NetWithLoss(NetWithReshape(w1, w2, layout1, layout2, out_layout1, out_layout2)))
    phase = compile_net(net, x)
    _ = ParallelValidator(net, phase)
