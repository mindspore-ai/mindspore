# Copyright 2019 Huawei Technologies Co., Ltd
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
from mindspore import Tensor, context
from mindspore.common import dtype as mstype
from mindspore.common import Parameter
from mindspore.common.api import _cell_graph_executor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from tests.ut.python.ops.test_math_ops import VirtualLoss


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")

grad_all = C.GradOperation(get_all=True)


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x):
        predict = self.network(x)
        return self.loss(predict)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x):
        return grad_all(self.network)(x)

class NetWithLossTwoInput(nn.Cell):
    def __init__(self, network):
        super(NetWithLossTwoInput, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x, y):
        predict = self.network(x, y)
        return self.loss(predict)

class GradWrapTwoInput(nn.Cell):
    def __init__(self, network):
        super(GradWrapTwoInput, self).__init__()
        self.network = network

    def construct(self, x, y):
        return grad_all(self.network)(x, y)


def compile_graph(net, device_num, x):
    context.set_auto_parallel_context(device_num=device_num, global_rank=0, parallel_mode="auto_parallel",
                                      search_mode="sharding_propagation")
    net.set_train()
    _cell_graph_executor.compile(net, x)


def compile_graph_two_input(net, device_num, x, y):
    context.set_auto_parallel_context(device_num=device_num, global_rank=0, parallel_mode="auto_parallel",
                                      search_mode="sharding_propagation")
    net.set_train()
    _cell_graph_executor.compile(net, x, y)


def test_reshape_reshape():
    """
    Feature: Sharding propagation for Reshape.
    Description: ReLU->Reshape
    Expectation: compile done without error.
    """
    device_num = 8
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.reshape = P.Reshape()
            self.relu = P.ReLU().shard(((1, 1, 1, 1),))

        def construct(self, x):
            x = self.relu(x)
            out = self.reshape(x, (64, 28))
            out = self.reshape(out, (64, 28, 1))
            return out

    x = Tensor(np.ones([device_num * 8, 28, 1, 1]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(Net()))
    compile_graph(net, device_num, x)


def test_reshape_auto_1():
    """
    Feature: Sharding propagation for Reshape.
    Description: ReLU->Reshape->MatMul
    Expectation: compile done without error.
    """
    device_num = 8
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU().shard(((1, 1, 1, 1),))
            self.reshape = P.Reshape()
            self.matmul = P.MatMul().shard(((2, 1), (1, 4)))
            self.matmul_weight = Parameter(Tensor(np.ones([28, 64]), dtype=ms.float32), name="weight")

        def construct(self, x):
            x = self.relu(x)
            out = self.reshape(x, (64, 28))
            out = self.matmul(out, self.matmul_weight)
            return out

    x = Tensor(np.ones([device_num * 8, 28, 1, 1]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(Net()))
    compile_graph(net, device_num, x)


def test_reshape_auto_2():
    """
    Feature: Sharding propagation for Reshape.
    Description: ReLU->Reshape->MatMul->Reshape->Add
    Expectation: compile done without error.
    """
    device_num = 8
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.relu2 = P.ReLU()
            self.reshape = P.Reshape()
            self.matmul = P.MatMul().shard(((2, 1), (1, 4)))
            self.matmul_weight = Parameter(Tensor(np.ones([28, 64]), dtype=ms.float32), name="weight")
            self.add = P.Add().shard(((2, 4), (2, 4)))
            self.add_weight = Parameter(Tensor(np.ones([128, 32]), dtype=ms.float32), name="weight1")

        def construct(self, x):
            out = self.relu(x)
            out = self.relu2(out)
            out = self.reshape(out, (64, 28))
            out = self.matmul(out, self.matmul_weight)
            out = self.reshape(out, (128, 32))
            out = self.add(out, self.add_weight)
            return out

    x = Tensor(np.ones([device_num * 8, 28, 1, 1]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(Net()))
    compile_graph(net, device_num, x)


def test_reshape_auto_3():
    """
    Feature: Sharding propagation for Reshape.
    Description: Mul->Add->Cast->Reshape->Cast->ReduceMean
    Expectation: compile done without error.
    """
    device_num = 8
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.gamma = Parameter(Tensor(np.ones([1024]), dtype=ms.float32), name="gamma")
            self.beta = Parameter(Tensor(np.ones([1024]), dtype=ms.float32), name="beta")
            self.add = P.TensorAdd().shard(((8, 1, 1), (1,)))
            self.mul = P.Mul().shard(((8, 1, 1), (1,)))
            self.mean = P.ReduceMean(keep_dims=True).shard(((8, 1),))
            self.reshape = P.Reshape()
            self.dtype1 = mstype.float16
            self.dtype2 = mstype.float32

        def construct(self, x):
            out = self.add(self.mul(x, self.gamma), self.beta)
            out = F.cast(out, self.dtype1)
            out = self.reshape(out, (-1, 1024))
            out = F.cast(out, self.dtype2)
            out = self.mean(out, -1)
            return out

    x = Tensor(np.ones([2048, 30, 1024]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(Net()))
    compile_graph(net, device_num, x)


def test_reshape_auto_4():
    """
    Feature: Sharding propagation for Reshape.
    Description: Mul->Add->Cast->Reshape->Cast->ReduceMean
    Expectation: compile done without error.
    """
    device_num = 8
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.gamma = Parameter(Tensor(np.ones([1024]), dtype=ms.float32), name="gamma")
            self.beta = Parameter(Tensor(np.ones([1024]), dtype=ms.float32), name="beta")
            self.add = P.TensorAdd().shard(((8, 1, 1), (1,)))
            self.mul = P.Mul().shard(((8, 1, 1), (1,)))
            self.mean = P.ReduceMean(keep_dims=True)
            self.reshape = P.Reshape()
            self.dtype1 = mstype.float16
            self.dtype2 = mstype.float32

        def construct(self, x):
            out = self.add(self.mul(x, self.gamma), self.beta)
            out = F.cast(out, self.dtype1)
            out = self.reshape(out, (-1, 1024))
            out = F.cast(out, self.dtype2)
            out = self.mean(out, -1)
            return out

    x = Tensor(np.ones([2048, 30, 1024]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(Net()))
    compile_graph(net, device_num, x)


def test_reshape_auto_5():
    """
    Feature: Sharding propagation for Reshape.
    Description: Mul->Add->Cast->Reshape->Cast->ReduceMean
    Expectation: compile done without error.
    """
    device_num = 8
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.gamma = Parameter(Tensor(np.ones([1024]), dtype=ms.float32), name="gamma")
            self.beta = Parameter(Tensor(np.ones([1024]), dtype=ms.float32), name="beta")
            self.add = P.TensorAdd().shard(((8, 1, 1), (1,)))
            self.mul = P.Mul()
            self.mean = P.ReduceMean(keep_dims=True).shard(((2, 4),))
            self.reshape = P.Reshape()
            self.dtype1 = mstype.float16
            self.dtype2 = mstype.float32

        def construct(self, x):
            out = self.add(self.mul(x, self.gamma), self.beta)
            out = self.reshape(out, (-1, 1024))
            out = self.mean(out, -1)
            return out

    x = Tensor(np.ones([2048, 30, 1024]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(Net()))
    compile_graph(net, device_num, x)


def test_reshape_auto_6():
    """
    Feature: Sharding propagation for Reshape.
    Description: Reshape->ReLU->Mul->Reshape->Add->Mul->Reshape->Add
    Expectation: compile done without error.
    """
    device_num = 8
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.mul = P.Mul().shard(((8, 1, 1), (8, 1, 1)))
            self.reshape = P.Reshape()
            self.reduce_sum = P.ReduceSum()
            self.wide_w = Parameter(Tensor(np.ones([8, 1024*8, 64]), dtype=ms.float32), name="weight")

        def construct(self, x, y):
            mask = self.reshape(y, (8, 1024*8, 1))
            w_id = self.relu(x)
            wx = self.mul(w_id, mask)
            wide_out = self.reshape(self.reduce_sum(wx, 1), (-1, 1))
            deep_id = x + self.wide_w
            vx = self.mul(deep_id, mask)
            deep_in = self.reshape(vx, (-1, 1024*8*64))
            out = wide_out + deep_in
            return out

    x = Tensor(np.ones([8, 1024*device_num, 1]), dtype=ms.float32)
    y = Tensor(np.ones([8, 1024*device_num]), dtype=ms.float32)
    net = GradWrapTwoInput(NetWithLossTwoInput(Net()))
    compile_graph_two_input(net, device_num, x, y)


def test_reshape_depend_reshape():
    """
    Feature: Sharding propagation for Reshape.
    Description: Mul->ReLU->Reshape->Reshape->Add
    Expectation: compile with error.
    """
    device_num = 8
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.reshape1 = P.Reshape()
            self.reshape2 = P.Reshape()
            self.relu = P.ReLU()
            self.depend = P.Depend()
            self.mul = P.Mul().shard(((2, 4), (2, 4)))
            self.mul_weight = Parameter(Tensor(np.ones([128, 96]), dtype=ms.float32), name="weight")
            self.add = P.Add().shard(((4, 2), (4, 2)))

        def construct(self, x, y):
            out1 = self.mul(x, self.mul_weight)
            y = self.relu(y)
            out2 = self.reshape1(y, (96, 32, 4))
            out3 = self.depend(out2, out1)
            out3 = self.reshape2(out3, (128, 96))
            out = out1 + out3
            return out

    class NetWithLoss1(nn.Cell):
        def __init__(self, network):
            super(NetWithLoss1, self).__init__()
            self.mean = P.ReduceMean(keep_dims=False)
            self.network = network

        def construct(self, x, y):
            predict = self.network(x, y)
            return self.mean(predict, ())

    x = Tensor(np.ones([128, 96]), dtype=ms.float32)
    y = Tensor(np.ones([256, 48]), dtype=ms.float32)
    net = GradWrapTwoInput(NetWithLoss1(Net()))
    with pytest.raises(RuntimeError):
        compile_graph_two_input(net, device_num, x, y)

def test_reshape_auto_8():
    """
    Feature: Sharding propagation for common parameter being used by multiple ops.
    Description: relu->add->mul->mean
    Expectation: compile done without error.
    """
    device_num = 8
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.gamma = Parameter(Tensor(np.ones([2048, 2048]), dtype=ms.float32), name="gamma")
            self.add = P.TensorAdd()
            self.relu = P.ReLU().shard(((1, 1),))
            self.mul2 = P.MatMul().shard(((1, 1), (1, 8)))
            self.mean = P.ReduceMean(keep_dims=True)

        def construct(self, x):
            out = self.add(x, self.relu(self.gamma))
            out = self.mul2(out, self.gamma)
            out = self.mean(out, -1)
            return out

    x = Tensor(np.ones([2048, 2048]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(Net()))
    with pytest.raises(RuntimeError):
        compile_graph(net, device_num, x)

def test_reshape_auto_9():
    """
    Feature: Sharding propagation for common parameter being used by multiple ops.
    Description: relu->add->mul->mean
    Expectation: compile done without error.
    """
    device_num = 8
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.gamma = Parameter(Tensor(np.ones([2048, 2048]), dtype=ms.float32), name="gamma")
            self.add = P.TensorAdd()
            self.relu = P.ReLU().shard(((1, 1),))
            self.mul2 = P.MatMul().shard(((8, 1), (1, 1)))
            self.mean = P.ReduceMean(keep_dims=True)

        def construct(self, x):
            out = self.add(x, self.relu(self.gamma))
            out = self.mul2(out, self.gamma)
            out = self.mean(out, -1)
            return out

    x = Tensor(np.ones([2048, 2048]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(Net()))
    compile_graph(net, device_num, x)
