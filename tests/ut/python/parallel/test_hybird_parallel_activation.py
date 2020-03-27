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
from mindspore import context
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor
from tests.ut.python.ops.test_math_ops import VirtualLoss
from mindspore.common.api import _executor
from mindspore.ops import composite as C


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x, y, b):
        predict = self.network(x, y, b)
        return self.loss(predict)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x, y, b):
        return C.grad_all(self.network)(x, y, b)


def test_matmul_tanh():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2, strategy3):
            super().__init__()
            self.matmul1 = P.MatMul().set_strategy(strategy1)
            self.matmul2 = P.MatMul().set_strategy(strategy2)
            self.tanh = P.Tanh().set_strategy(strategy3)

        def construct(self, x, y, b):
            out = self.tanh(self.matmul1(x, y))
            out = self.matmul2(out, b)
            return out

    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((1, 1), (1, 16))
    strategy3 = ((4, 4), )
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2, strategy3)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_auto_parallel_context(device_num=16, global_rank=0)

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    _executor.compile(net, x, y, b)


def test_matmul_activation():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2, strategy3):
            super().__init__()
            self.matmul1 = P.MatMul().set_strategy(strategy1)
            self.matmul2 = P.MatMul().set_strategy(strategy2)
            self.activation = P.ReLU().set_strategy(strategy3)

        def construct(self, x, y, b):
            out = self.activation(self.matmul1(x, y))
            out = self.matmul2(out, b)
            return out

    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((1, 1), (1, 16))
    strategy3 = ((4, 4), )
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2, strategy3)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_auto_parallel_context(device_num=16, global_rank=0)

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    _executor.compile(net, x, y, b)


def test_matmul_softmax():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2, strategy3):
            super().__init__()
            self.matmul1 = P.MatMul().set_strategy(strategy1)
            self.matmul2 = P.MatMul().set_strategy(strategy2)
            self.softmax = P.Softmax().set_strategy(strategy3)

        def construct(self, x, y, b):
            out = self.softmax(self.matmul1(x, y))
            out = self.matmul2(out, b)
            return out

    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((1, 1), (1, 16))
    strategy3 = ((16, 1), )
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2, strategy3)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_auto_parallel_context(device_num=16, global_rank=0)

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    _executor.compile(net, x, y, b)


def test_matmul_logsoftmax():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2, strategy3):
            super().__init__()
            self.matmul1 = P.MatMul().set_strategy(strategy1)
            self.matmul2 = P.MatMul().set_strategy(strategy2)
            self.logsoftmax = P.LogSoftmax().set_strategy(strategy3)

        def construct(self, x, y, b):
            out = self.logsoftmax(self.matmul1(x, y))
            out = self.matmul2(out, b)
            return out

    strategy1 = ((4, 2), (2, 2))
    strategy2 = ((2, 4), (4, 2))
    strategy3 = ((16, 1), )
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2, strategy3)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_auto_parallel_context(device_num=16, global_rank=0)

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    _executor.compile(net, x, y, b)


def test_activations():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2, strategy3):
            super().__init__()
            self.matmul1 = P.MatMul().set_strategy(strategy1)
            self.matmul2 = P.MatMul().set_strategy(strategy2)
            self.gelu = P.Gelu().set_strategy(strategy3)
            self.tanh = P.Tanh().set_strategy(strategy3)
            self.softmax = P.Softmax().set_strategy(strategy3)
            self.logsoftmax = P.LogSoftmax().set_strategy(strategy3)

        def construct(self, x, y, b):
            out = self.gelu(self.tanh(self.matmul1(x, y)))
            out = self.logsoftmax(self.softmax(self.matmul2(out, b)))
            return out

    strategy1 = ((1, 2), (2, 2))
    strategy2 = ((2, 2), (2, 1))
    strategy3 = ((4, 1), )
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2, strategy3)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_auto_parallel_context(device_num=4, global_rank=0)

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    _executor.compile(net, x, y, b)

def test_activations_repeated_calculation():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2, strategy3, strategy4, strategy5, strategy6):
            super().__init__()
            self.matmul1 = P.MatMul().set_strategy(strategy1)
            self.matmul2 = P.MatMul().set_strategy(strategy2)
            self.gelu = P.Gelu().set_strategy(strategy3)
            self.tanh = P.Tanh().set_strategy(strategy4)
            self.softmax = P.Softmax().set_strategy(strategy5)
            self.logsoftmax = P.LogSoftmax().set_strategy(strategy6)

        def construct(self, x, y, b):
            out = self.gelu(self.tanh(self.matmul1(x, y)))
            out = self.logsoftmax(self.softmax(self.matmul2(out, b)))
            return out

    strategy1 = ((2, 4), (4, 8))
    strategy2 = ((2, 2), (2, 1))
    strategy3 = ((2, 1), )
    strategy4 = ((2, 2), )
    strategy5 = ((4, 1), )
    strategy6 = ((8, 1), )
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2, strategy3, strategy4, strategy5, strategy6)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_auto_parallel_context(device_num=64, global_rank=0)

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    _executor.compile(net, x, y, b)


def test_activations_axis_tuple():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2, strategy3, strategy4, strategy5, strategy6):
            super().__init__()
            self.matmul1 = P.MatMul().set_strategy(strategy1)
            self.matmul2 = P.MatMul().set_strategy(strategy2)
            self.gelu = P.Gelu().set_strategy(strategy3)
            self.tanh = P.Tanh().set_strategy(strategy4)
            self.softmax = P.Softmax(axis=(0, 1)).set_strategy(strategy5)
            self.logsoftmax = P.LogSoftmax().set_strategy(strategy6)

        def construct(self, x, y, b):
            out = self.gelu(self.tanh(self.matmul1(x, y)))
            out = self.logsoftmax(self.softmax(self.matmul2(out, b)))
            return out

    strategy1 = ((2, 4), (4, 8))
    strategy2 = ((2, 2), (2, 1))
    strategy3 = ((2, 1), )
    strategy4 = ((2, 2), )
    strategy5 = ((1, 1), )
    strategy6 = ((8, 1), )
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2, strategy3, strategy4, strategy5, strategy6)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_auto_parallel_context(device_num=64, global_rank=0)

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    _executor.compile(net, x, y, b)
