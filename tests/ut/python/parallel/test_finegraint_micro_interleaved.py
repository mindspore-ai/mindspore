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
from mindspore import Tensor, Parameter, context
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


def compile_net_unsupported(net, input_x, indices, update):
    net.set_auto_parallel()
    net.set_train()
    phase, _ = _cell_graph_executor.compile(net, input_x, indices, update)
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


class NetWithAddActivation(nn.Cell):
    def __init__(self, weight, bias, matmul_in_layout, add_in_layout, relu_in_layout=None, softmax_in_layout=None,
                 out_layout=None):
        super().__init__()
        self.matmul1 = P.MatMul().shard(in_strategy=matmul_in_layout, out_strategy=out_layout)
        self.matmul1.add_prim_attr("recompute_comm_op", True)
        self.add = P.Add().shard(in_strategy=add_in_layout)
        self.gelu = P.GeLU()
        if relu_in_layout is not None:
            self.activation = P.ReLU().shard(in_strategy=relu_in_layout)
        elif softmax_in_layout is not None:
            self.activation = P.Softmax().shard(in_strategy=softmax_in_layout)
        else:
            self.activation = P.Softmax()
        self.w = Parameter(weight, "w1")
        self.bias = Parameter(bias, "bias")

    def construct(self, y):
        y_new = self.gelu(y)
        out1 = self.matmul1(y, self.w)
        out1 = self.add(out1, self.bias)
        out1 = self.activation(out1)
        out = out1 + y_new
        return out


class NetWithAdd(nn.Cell):
    def __init__(self, weight, bias, matmul_in_layout, add_1_in_layout, add_2_in_layout, add_3_in_layout):
        super().__init__()
        self.matmul1 = P.MatMul().shard(in_strategy=matmul_in_layout)
        self.matmul1.add_prim_attr("recompute_comm_op", True)
        self.add_1 = P.Add().shard(in_strategy=add_1_in_layout)
        self.add_2 = P.Add().shard(in_strategy=add_2_in_layout)
        self.add_3 = P.Add().shard(in_strategy=add_3_in_layout)
        self.gelu = P.GeLU()
        self.w = Parameter(weight, "w1")
        self.bias = Parameter(bias, "bias")

    def construct(self, y):
        y_new = self.gelu(y)
        out1 = self.matmul1(y, self.w)
        out1 = self.add_1(out1, self.bias)
        out2 = self.add_2(self.bias, out1)
        out3 = self.add_3(out1, out2)
        out = out3 + y_new
        return out


class NetWithAdd2(nn.Cell):
    def __init__(self, bias, add_in_layout):
        super().__init__()
        self.add = P.Add().shard(in_strategy=add_in_layout)
        self.gelu = P.GeLU()
        self.bias = Parameter(bias, "bias")

    def construct(self, y):
        y_new = self.gelu(y)
        out1 = self.add(y, self.bias)
        out = out1 + y_new
        return out


class NetWithAdd3(nn.Cell):
    def __init__(self, bias, add_in_layout):
        super().__init__()
        self.add = P.Add().shard(in_strategy=add_in_layout)
        self.gelu = P.GeLU()
        self.bias = Parameter(bias, "bias")

    def construct(self, y):
        out1 = self.add(y, self.bias)
        out = self.gelu(out1)
        return out


class NetWithRmsNorm(nn.Cell):
    def __init__(self, weight, bias, gamma, matmul_in_layout, add_in_layout, relu_in_layout, rmsnorm_in_layout,
                 out_layout=None):
        super().__init__()
        self.matmul1 = P.MatMul().shard(in_strategy=matmul_in_layout, out_strategy=out_layout)
        self.matmul1.add_prim_attr("recompute_comm_op", True)
        self.add = P.Add().shard(in_strategy=add_in_layout)
        self.gelu = P.GeLU()
        self.activation = P.ReLU().shard(in_strategy=relu_in_layout)
        self.rmsnorm = P.RmsNorm().shard(in_strategy=rmsnorm_in_layout)
        self.w = Parameter(weight, "w1")
        self.bias = Parameter(bias, "bias")
        self.gamma = Parameter(gamma, "gamma")

    def construct(self, y):
        y_new = self.gelu(y)
        out1 = self.matmul1(y, self.w)
        out1 = self.add(out1, self.bias)
        out1 = self.activation(out1)
        out1 = self.rmsnorm(out1, self.gamma)
        out = out1[0] + y_new
        return out


class NetWithLayernorm(nn.Cell):
    def __init__(self, weight, bias, gamma, beta, matmul_in_layout, add_in_layout, relu_in_layout, layernorm_in_layout,
                 out_layout=None):
        super().__init__()
        self.matmul1 = P.MatMul().shard(in_strategy=matmul_in_layout, out_strategy=out_layout)
        self.matmul1.add_prim_attr("recompute_comm_op", True)
        self.add = P.Add().shard(in_strategy=add_in_layout)
        self.gelu = P.GeLU()
        self.activation = P.ReLU().shard(in_strategy=relu_in_layout)
        self.layernorm = P.LayerNorm(begin_norm_axis=1, begin_params_axis=1, epsilon=1e-7).shard(
            in_strategy=layernorm_in_layout)
        self.w = Parameter(weight, "w1")
        self.bias = Parameter(bias, "bias")
        self.gamma = Parameter(gamma, "gamma")
        self.beta = Parameter(beta, "beta")

    def construct(self, y):
        y_new = self.gelu(y)
        out1 = self.matmul1(y, self.w)
        out1 = self.add(out1, self.bias)
        out1 = self.activation(out1)
        out1 = self.layernorm(out1, self.gamma, self.beta)
        out = out1[0] + y_new
        return out


class NetWithTranspose(nn.Cell):
    def __init__(self, mul_size, perm=(1, 0), in_strategy=None, out_strategy=None):
        super().__init__()
        mul_np = np.full(mul_size, 0.5, dtype=np.float32)
        self.mul_weight = Parameter(Tensor(mul_np), name="mul_weight")
        self.mul = P.Mul()
        self.transpose = P.Transpose()
        self.perm = perm
        self.transpose.shard(in_strategy=in_strategy, out_strategy=out_strategy)
        self.add = P.Add()

    def construct(self, inputs):
        x = self.mul(inputs, self.mul_weight)
        x = self.transpose(x, self.perm)
        x = self.add(x, x)
        return x


class NetWithUnsupportedOps(nn.Cell):
    def __init__(self, in_layout, out_layout):
        super().__init__()
        self.tensor_scatter_update = P.TensorScatterUpdate()
        self.tensor_scatter_update.shard(in_strategy=in_layout, out_strategy=out_layout)
        self.relu = P.ReLU()
        self.mul = P.Mul()

    def construct(self, input_x, indices, update):
        out = self.relu(input_x)
        out = self.tensor_scatter_update(out, indices, update)
        out = self.mul(out, 2)
        return out


class NetWithSplit(nn.Cell):
    def __init__(self, mul_size, in_strategy=None, out_strategy=None):
        super().__init__()
        mul_np = np.full(mul_size, 0.5, dtype=np.float32)
        self.mul_weight = Parameter(Tensor(mul_np), name="mul_weight")
        self.mul = P.Mul()
        self.split = P.Split(-1, 2).add_prim_attr("skip_redistribution", True)
        self.split.shard(in_strategy=in_strategy, out_strategy=out_strategy)
        self.add = P.Add()

    def construct(self, inputs):
        x = self.mul(inputs, self.mul_weight)
        x1, x2 = self.split(x)
        out = self.add(x1, x2)
        return out


class NetWithSplitAxis0(nn.Cell):
    def __init__(self, mul_size, in_strategy=None, out_strategy=None):
        super().__init__()
        mul_np = np.full(mul_size, 0.5, dtype=np.float32)
        self.mul_weight = Parameter(Tensor(mul_np), name="mul_weight")
        self.mul = P.Mul()
        self.split = P.Split(0, 2)
        self.split.shard(in_strategy=in_strategy, out_strategy=out_strategy)
        self.add = P.Add()

    def construct(self, inputs):
        x = self.mul(inputs, self.mul_weight)
        x1, x2 = self.split(x)
        out = self.add(x1, x2)
        return out


class AddNet(nn.Cell):
    def __init__(self, add_shape, in_layout=None):
        super().__init__()
        self.add = P.Add()
        self.add.shard(in_layout)
        self.relu = P.ReLU()
        self.weight = Parameter(Tensor(np.ones(add_shape).astype(np.float32)), name="add_weight")

    def construct(self, y):
        out1 = self.add(y, self.weight)
        out = self.relu(out1)
        return out


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


def test_interleaved_with_relu_add():
    """
    Feature: test micro interleaved
    Description: dev_num is 8.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 4, 2), ("dp", "mp", "interleaved_parallel"))
    matmul_layout = (layout(("dp", "interleaved_parallel"), "mp"), layout("mp", "None"))
    add_layout = (layout(("dp", "interleaved_parallel"), "None"), layout("None"))
    activation_layout = (layout(("dp", "interleaved_parallel"), "None"),)
    x = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    w = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    bias = Tensor(np.ones([1024,]), dtype=ms.float32)
    net = GradWrap(
        NetWithLoss(NetWithAddActivation(w, bias, matmul_layout, add_layout, relu_in_layout=activation_layout)))
    _ = compile_net(net, x)


def test_interleaved_with_add():
    """
    Feature: test micro interleaved
    Description: dev_num is 8.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 4, 2), ("dp", "mp", "interleaved_parallel"))
    matmul_layout = (layout(("dp", "interleaved_parallel"), "mp"), layout("mp", "None"))
    add_1_layout = (layout(("dp", "interleaved_parallel"), "None"), layout("None"))
    add_2_layout = (layout("None"), layout(("dp", "interleaved_parallel"), "None"))
    add_3_layout = (layout(("dp", "interleaved_parallel"), "None"), layout(("dp", "interleaved_parallel"), "None"))
    x = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    w = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    bias = Tensor(np.ones([1024,]), dtype=ms.float32)
    net = GradWrap(
        NetWithLoss(NetWithAdd(w, bias, matmul_layout, add_1_layout, add_2_layout, add_3_layout)))
    _ = compile_net(net, x)


def test_interleaved_with_add_failed():
    """
    Feature: test micro interleaved in parameter
    Description: dev_num is 8.
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 4, 2), ("dp", "mp", "interleaved_parallel"))
    add_layout = (layout("None"), layout(("dp", "interleaved_parallel"), "None"))
    x = Tensor(np.ones([1024,]), dtype=ms.float32)
    bias = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    net = GradWrap(
        NetWithLoss(NetWithAdd2(bias, add_layout)))
    with pytest.raises(RuntimeError):
        _ = compile_net(net, x)


def test_interleaved_with_softmax_add():
    """
    Feature: test micro interleaved
    Description: dev_num is 8.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 4, 2), ("dp", "mp", "interleaved_parallel"))
    matmul_layout = (layout(("dp", "interleaved_parallel"), "mp"), layout("mp", "None"))
    add_layout = (layout(("dp", "interleaved_parallel"), "None"), layout("None"))
    activation_layout = (layout(("dp", "interleaved_parallel"), "None"),)
    x = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    w = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    bias = Tensor(np.ones([1024,]), dtype=ms.float32)
    net = GradWrap(
        NetWithLoss(NetWithAddActivation(w, bias, matmul_layout, add_layout, softmax_in_layout=activation_layout)))
    _ = compile_net(net, x)


def test_interleaved_with_rmsnorm():
    """
    Feature: test micro interleaved
    Description: dev_num is 8.
    Expectation: compile success
    """

    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 4, 2), ("dp", "mp", "interleaved_parallel"))
    matmul_layout = (layout(("dp", "interleaved_parallel"), "mp"), layout("mp", "None"))
    add_layout = (layout(("dp", "interleaved_parallel"), "None"), layout("None"))
    activation_layout = (layout(("dp", "interleaved_parallel"), "None"),)
    rmsnorm_layout = (layout(("dp", "interleaved_parallel"), "None"), layout("None"))
    x = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    w = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    bias = Tensor(np.ones([1024,]), dtype=ms.float32)
    net = GradWrap(
        NetWithLoss(NetWithRmsNorm(w, bias, bias, matmul_layout, add_layout, activation_layout, rmsnorm_layout)))
    _ = compile_net(net, x)


def test_interleaved_with_layernorm():
    """
    Feature: test micro interleaved
    Description: dev_num is 8.
    Expectation: compile success
    """

    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 4, 2), ("dp", "mp", "interleaved_parallel"))
    matmul_layout = (layout(("dp", "interleaved_parallel"), "mp"), layout("mp", "None"))
    add_layout = (layout(("dp", "interleaved_parallel"), "None"), layout("None"))
    activation_layout = (layout(("dp", "interleaved_parallel"), "None"),)
    layernorm_layout = (layout(("dp", "interleaved_parallel"), "None"), layout("None"), layout("None"))
    x = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    w = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    bias = Tensor(np.ones([1024,]), dtype=ms.float32)
    net = GradWrap(
        NetWithLoss(
            NetWithLayernorm(w, bias, bias, bias, matmul_layout, add_layout, activation_layout, layernorm_layout)))
    _ = compile_net(net, x)


def test_interleaved_with_transpose():
    """
    Feature: test micro interleaved
    Description: dev_num is 4.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)
    layout = Layout((2, 2, 1, 1, 2), ("dp", "sp", "mp", "cp", "interleaved_parallel"))
    in_strategy = (layout("dp", ("sp", "cp"), "mp", "interleaved_parallel"),)
    out_strategy = (layout(("sp", "cp"), "interleaved_parallel", "dp", "mp"),)
    x = Tensor(np.ones([64, 96, 16, 16]), dtype=ms.float32)
    net = GradWrap(
        NetWithLoss(
            NetWithTranspose((64, 96, 16, 16), (1, 3, 0, 2), in_strategy, out_strategy)))
    _ = compile_net(net, x)


def test_interleaved_with_unsupported_ops():
    """
    Feature: test layout extend
    Description: dev_num is 2.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=2, global_rank=0)
    layout = Layout((2, 1), ("dp", "mp"))
    layout1 = (layout("dp", "mp", "None"), layout("dp", "mp", "None"), layout("dp", "mp", "None"))
    layout2 = (layout("dp", "mp", "None"),)
    net = NetWithUnsupportedOps(layout1, layout2)
    input_x = Tensor(np.zeros((2, 2, 3)).astype(np.float32))
    indices = Tensor(np.array([[[0, 0], [1, 1]], [[0, 0], [1, 1]]]).astype(np.int32))
    update = Tensor(np.ones((2, 2, 3)).astype(np.float32))
    with pytest.raises(RuntimeError):
        compile_net_unsupported(net, input_x, indices, update)


def test_interleaved_size4():
    """
    Feature: test micro interleaved in parameter
    Description: dev_num is 16.
    Expectation: compile ok
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)
    layout = Layout((2, 1, 2, 4), ("dp", "sp", "mp", "interleaved_parallel"))
    add_layout = (layout(("interleaved_parallel", "dp"), "mp"), layout("None", "mp"))
    x = Tensor(np.ones([1024, 1024]), dtype=ms.float32)
    bias = Tensor(np.ones([1, 1024]), dtype=ms.float32)
    net = GradWrap(
        NetWithLoss(NetWithAdd3(bias, add_layout)))
    _ = compile_net(net, x)


def test_interleaved_with_split():
    """
    Feature: test micro interleaved
    Description: dev_num is 8.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 4, 2), ("dp", "mp", "interleaved_parallel"))
    in_strategy = (layout(("dp", "interleaved_parallel"), "mp"),)
    x = Tensor(np.ones([64, 128]), dtype=ms.float32)
    net = GradWrap(
        NetWithLoss(
            NetWithSplit((64, 128), in_strategy)))
    _ = compile_net(net, x)


def test_interleaved_with_split_with_error():
    """
    Feature: test micro interleaved
    Description: dev_num is 8.
    Expectation: raise error
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 4, 2), ("dp", "mp", "interleaved_parallel"))
    in_strategy = (layout(("interleaved_parallel"), ("dp", "mp")),)
    x = Tensor(np.ones([64, 128]), dtype=ms.float32)
    net = GradWrap(
        NetWithLoss(
            NetWithSplitAxis0((64, 128), in_strategy)))
    with pytest.raises(RuntimeError):
        _ = compile_net(net, x)


def test_interleaved_with_add_interleave3():
    """
    Feature: test micro interleaved, interleave size 3.
    Description: dev_num is 8.
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 2, 2, 3), ("dp", "sp", "mp", "interleaved_parallel"))
    add_layout = (layout(("interleaved_parallel", "dp"), "mp"), layout("None", "mp"))
    x = Tensor(np.ones([3 * 1024, 1024]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(AddNet((1, 1024), add_layout)))
    _ = compile_net(net, x)
