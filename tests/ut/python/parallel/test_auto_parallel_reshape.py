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

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import _executor
from mindspore.common.parameter import Parameter
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from tests.ut.python.ops.test_math_ops import VirtualLoss


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
        return C.grad_all(self.network)(x)


# core dump, step_auto_parallel should SetInputs for transpose axis
def test_reshape_matmul():
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.reshape = P.Reshape()
            self.matmul = P.MatMul()
            self.matmul_weight = Parameter(Tensor(np.ones([28, 64]), dtype=ms.float32), name="weight")

        def construct(self, x):
            out = self.reshape(x, (64, 28))
            out = self.matmul(out, self.matmul_weight)
            return out

    size = 8
    context.set_auto_parallel_context(device_num=size, global_rank=0)
    x = Tensor(np.ones([8 * size, 28, 1, 1]), dtype=ms.float32)

    net = GradWrap(NetWithLoss(Net()))
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    net.set_auto_parallel()
    _executor.compile(net, x)


def test_reshape_auto_1():
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.reshape = P.Reshape()
            self.matmul = P.MatMul()
            self.matmul_weight = Parameter(Tensor(np.ones([28, 64]), dtype=ms.float32), name="weight")

        def construct(self, x):
            out = self.relu(x)
            out = self.reshape(out, (64, 28))
            out = self.matmul(out, self.matmul_weight)
            return out

    size = 8
    context.set_auto_parallel_context(device_num=size, global_rank=0)
    x = Tensor(np.ones([8 * size, 28, 1, 1]), dtype=ms.float32)

    net = GradWrap(NetWithLoss(Net()))
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    net.set_auto_parallel()
    _executor.compile(net, x)


def test_reshape_auto_2():
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.reshape = P.Reshape()
            self.matmul = P.MatMul()
            self.add_weight = Parameter(Tensor(np.ones([128, 32]), dtype=ms.float32), name="weight1")
            self.matmul_weight = Parameter(Tensor(np.ones([28, 64]), dtype=ms.float32), name="weight")

        def construct(self, x):
            out = self.relu(x)
            out = self.reshape(out, (64, 28))
            out = self.matmul(out, self.matmul_weight)
            out = self.reshape(out, (128, 32))
            out = out + self.add_weight
            return out

    size = 8
    context.set_auto_parallel_context(device_num=size, global_rank=0)
    x = Tensor(np.ones([8 * size, 28, 1, 1]), dtype=ms.float32)

    net = GradWrap(NetWithLoss(Net()))
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    net.set_auto_parallel()
    _executor.compile(net, x)


def test_reshape_auto_3():
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.reshape = P.Reshape()
            self.matmul = P.MatMul()
            self.matmul_weight = Parameter(Tensor(np.ones([28, 64]), dtype=ms.float32), name="weight")

        def construct(self, x):
            out = self.relu(x)
            out = self.matmul(out, self.matmul_weight)
            out = self.reshape(out, (8, 8, 8, 8))
            return out

    size = 8
    context.set_auto_parallel_context(device_num=size, global_rank=0)
    x = Tensor(np.ones([8 * size, 28]), dtype=ms.float32)

    net = GradWrap(NetWithLoss(Net()))
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    net.set_auto_parallel()
    _executor.compile(net, x)


def test_reshape_auto_4():
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.reshape = P.Reshape()
            self.matmul = P.MatMul()
            self.matmul_weight = Parameter(Tensor(np.ones([28 * 64]), dtype=ms.float32), name="weight")

        def construct(self, x):
            out = self.relu(x)
            out = self.reshape(out, (64, 28))
            w = self.reshape(self.matmul_weight, (28, 64))
            out = self.matmul(out, w)
            return out

    size = 8
    context.set_auto_parallel_context(device_num=size, global_rank=0)
    x = Tensor(np.ones([8 * size, 28, 1, 1]), dtype=ms.float32)

    net = GradWrap(NetWithLoss(Net()))
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    net.set_auto_parallel()
    _executor.compile(net, x)


def test_reshape_auto_5():
    class NetWithLoss5(nn.Cell):
        def __init__(self, network):
            super(NetWithLoss5, self).__init__()
            self.loss = VirtualLoss()
            self.network = network

        def construct(self, x, y):
            predict = self.network(x, y)
            return self.loss(predict)

    class GradWrap5(nn.Cell):
        def __init__(self, network):
            super(GradWrap5, self).__init__()
            self.network = network

        def construct(self, x, y):
            return C.grad_all(self.network)(x, y)

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.mul = P.Mul()
            self.reshape = P.Reshape()
            self.reduce_sum = P.ReduceSum()
            self.wide_w = Parameter(Tensor(np.ones([4, 1024 * 8, 64]), dtype=ms.float32), name="weight")

        def construct(self, x, y):
            mask = self.reshape(y, (4, 1024 * 8, 1))
            w_id = self.relu(x)
            wx = self.mul(w_id, mask)
            wide_out = self.reshape(self.reduce_sum(wx, 1), (-1, 1))
            deep_id = x + self.wide_w
            vx = self.mul(deep_id, mask)
            deep_in = self.reshape(vx, (-1, 1024 * 8 * 64))
            out = wide_out + deep_in
            return out

    size = 8
    context.set_auto_parallel_context(device_num=size, global_rank=0)
    x = Tensor(np.ones([4, 1024 * size, 1]), dtype=ms.float32)
    y = Tensor(np.ones([4, 1024 * size,]), dtype=ms.float32)

    net = GradWrap5(NetWithLoss5(Net()))
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    net.set_auto_parallel()
    _executor.compile(net, x, y)


def test_reshape_auto_6():
    class NetWithLoss6(nn.Cell):
        def __init__(self, network):
            super(NetWithLoss6, self).__init__()
            self.loss = VirtualLoss()
            self.network = network

        def construct(self, x, y):
            predict = self.network(x, y)
            return self.loss(predict)

    class GradWrap6(nn.Cell):
        def __init__(self, network):
            super(GradWrap6, self).__init__()
            self.network = network

        def construct(self, x, y):
            return C.grad_all(self.network)(x, y)

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()
            self.mul = P.Mul()
            self.reshape = P.Reshape()
            self.reduce_mean = P.ReduceMean()
            self.wide_w = Parameter(Tensor(np.ones([4, 1024, 1]), dtype=ms.float32), name="weight")

        def construct(self, x, y):
            out1 = x + self.wide_w
            w = self.reshape(self.wide_w, (4, 1024))
            out1 = self.reduce_mean(out1, 1)
            out1 = out1 - w
            out2 = self.mul(y, w)
            out = out1 + out2
            return out

    size = 8
    context.set_auto_parallel_context(device_num=size, global_rank=0)
    x = Tensor(np.ones([4, 1024, 1]), dtype=ms.float32)
    y = Tensor(np.ones([4, 1024,]), dtype=ms.float32)

    net = GradWrap6(NetWithLoss6(Net()))
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    net.set_auto_parallel()
    _executor.compile(net, x, y)
