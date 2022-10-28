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
from mindspore.common.api import _cell_graph_executor
from mindspore.common.parameter import Parameter
from mindspore.ops import composite as C
from mindspore.ops import operations as P
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

class NetWithReduceLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithReduceLoss, self).__init__()
        self.mean = P.ReduceMean(keep_dims=False)
        self.network = network

    def construct(self, x, y):
        predict = self.network(x, y)
        return self.mean(predict, ())

class GradWrapTwoInput(nn.Cell):
    def __init__(self, network):
        super(GradWrapTwoInput, self).__init__()
        self.network = network

    def construct(self, x, y):
        return grad_all(self.network)(x, y)


def compile_graph(net, parallel_mode, device_num, x):
    context.set_auto_parallel_context(device_num=device_num, global_rank=0, parallel_mode=parallel_mode)
    net.set_train()
    _cell_graph_executor.compile(net, x)

def compile_graph_two_input(net, parallel_mode, device_num, x, y):
    context.set_auto_parallel_context(device_num=device_num, global_rank=0, parallel_mode=parallel_mode)
    net.set_train()
    _cell_graph_executor.compile(net, x, y)


def test_reshape_matmul():
    """
    Feature: distribute operator reshape in auto parallel.
    Description: reshape - matmul net in auto parallel.
    Expectation: compile done without error.
    """
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
    x = Tensor(np.ones([8 * size, 28, 1, 1]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(Net()))
    compile_graph(net, "auto_parallel", size, x)

def test_reshape_reshape():
    """
    Feature: distribute operator reshape in auto parallel.
    Description: reshape - reshape net in auto parallel.
    Expectation: compile done without error.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.reshape = P.Reshape()
            self.relu = P.ReLU()

        def construct(self, x):
            x = self.relu(x)
            out = self.reshape(x, (64, 28))
            out = self.reshape(out, (64, 28, 1))
            return out

    size = 8
    x = Tensor(np.ones([8 * size, 28, 1, 1]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(Net()))
    compile_graph(net, "auto_parallel", size, x)


def test_reshape_auto_1():
    """
    Feature: distribute operator reshape in auto parallel.
    Description: relu - reshape - matmul net in auto parallel.
    Expectation: compile done without error.
    """
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
    x = Tensor(np.ones([8 * size, 28, 1, 1]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(Net()))
    compile_graph(net, "auto_parallel", size, x)


def test_reshape_auto_2():
    """
    Feature: distribute operator reshape in auto parallel.
    Description: reshape - matmul -reshape net in auto parallel.
    Expectation: compile done without error.
    """
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
    x = Tensor(np.ones([8 * size, 28, 1, 1]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(Net()))
    compile_graph(net, "auto_parallel", size, x)


def test_reshape_auto_3():
    """
    Feature: distribute operator reshape in auto parallel.
    Description: reshape as last node net in auto parallel.
    Expectation: compile done without error.
    """
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
    x = Tensor(np.ones([8 * size, 28]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(Net()))
    compile_graph(net, "auto_parallel", size, x)


def test_reshape_auto_4():
    """
    Feature: distribute operator reshape in auto parallel.
    Description: reshape - reshape net in auto parallel.
    Expectation: compile done without error.
    """
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
    x = Tensor(np.ones([8 * size, 28, 1, 1]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(Net()))
    compile_graph(net, "auto_parallel", size, x)


def test_reshape_auto_5():
    """
    Feature: distribute operator reshape in auto parallel.
    Description: modify wide&deep small net in auto parallel.
    Expectation: compile done without error.
    """
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
    context.set_auto_parallel_context(dataset_strategy="full_batch")
    x = Tensor(np.ones([4, 1024 * size, 1]), dtype=ms.float32)
    y = Tensor(np.ones([4, 1024 * size,]), dtype=ms.float32)
    net = GradWrapTwoInput(NetWithLossTwoInput(Net()))
    compile_graph_two_input(net, "auto_parallel", size, x, y)

def test_reshape_auto_6():
    """
    Feature: distribute operator reshape in auto parallel.
    Description: modify wide&deep small net in auto parallel.
    Expectation: compile done without error.
    """
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
    context.set_auto_parallel_context(dataset_strategy="full_batch")
    x = Tensor(np.ones([4, 1024, 1]), dtype=ms.float32)
    y = Tensor(np.ones([4, 1024,]), dtype=ms.float32)
    net = GradWrapTwoInput(NetWithLossTwoInput(Net()))
    compile_graph_two_input(net, "auto_parallel", size, x, y)

def test_reshape_auto_7():
    """
    Feature: distribute operator reshape in auto parallel.
    Description: reshape weight net in semi auto parallel.
    Expectation: compile done without error.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.reshape = P.Reshape()
            self.mul = P.Mul().shard(((1, 2, 4), (2, 4)))
            self.mul_weight = Parameter(Tensor(np.ones([128, 96]), dtype=ms.float32), name="weight")

        def construct(self, x):
            weight = self.reshape(self.mul_weight, (1, 128, 96))
            out = self.mul(weight, self.mul_weight)
            return out

    size = 8
    x = Tensor(np.ones([128, 28]), dtype=ms.float32)
    net = GradWrap(NetWithLoss(Net()))
    compile_graph(net, "semi_auto_parallel", size, x)

def test_reshape_depend_reshape():
    """
    Feature: distribute operator reshape in auto parallel.
    Description: reshape - depend -reshape net in semi auto parallel.
    Expectation: compile done without error.
    """
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

    size = 8
    x = Tensor(np.ones([128, 96]), dtype=ms.float32)
    y = Tensor(np.ones([256, 48]), dtype=ms.float32)
    net = GradWrapTwoInput(NetWithReduceLoss(Net()))
    compile_graph_two_input(net, "semi_auto_parallel", size, x, y)
    net_auto = GradWrapTwoInput(NetWithReduceLoss(Net()))
    compile_graph_two_input(net_auto, "auto_parallel", size, x, y)

def test_appeq_reshape():
    """
    Feature: distribute operator reshape in auto parallel.
    Description: app_eq - reshape - cast - relu net in semi auto parallel / auto parallel.
    Expectation: compile done without error.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.app_eq = P.ApproximateEqual(2.)
            self.reshape = P.Reshape()
            self.cast = P.Cast()
            self.relu = P.ReLU().shard(((1, 8),))

        def construct(self, x, y):
            out1 = self.app_eq(x, y)
            out2 = self.reshape(out1, (64, 192))
            out3 = self.cast(out2, ms.int32)
            out = self.relu(out3)
            return out

    size = 8
    x = Tensor(np.ones([128, 96]), dtype=ms.float32)
    y = Tensor(np.ones([128, 96]), dtype=ms.float32)
    net = GradWrapTwoInput(NetWithReduceLoss(Net()))
    compile_graph_two_input(net, "semi_auto_parallel", size, x, y)
    net_auto = GradWrapTwoInput(NetWithReduceLoss(Net()))
    context.set_auto_parallel_context(search_mode="recursive_programming")
    compile_graph_two_input(net_auto, "auto_parallel", size, x, y)
