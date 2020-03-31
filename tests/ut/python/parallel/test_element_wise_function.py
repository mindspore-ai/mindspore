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


def test_matmul_pow():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().set_strategy(strategy1)
            self.pow = P.Pow().set_strategy(strategy2)
            self.matmul2 = P.MatMul().set_strategy(strategy1)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.pow(out, 2.0)
            out = self.matmul2(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), )
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    _executor.compile(net, x, y, b)


def test_matmul_exp():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().set_strategy(strategy1)
            self.exp = P.Exp().set_strategy(strategy2)
            self.matmul2 = P.MatMul().set_strategy(strategy1)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.exp(out)
            out = self.matmul2(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), )
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    _executor.compile(net, x, y, b)


def test_matmul_log():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().set_strategy(strategy1)
            self.log = P.Log().set_strategy(strategy2)
            self.matmul2 = P.MatMul().set_strategy(strategy1)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.log(out)
            out = self.matmul2(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), )
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    _executor.compile(net, x, y, b)


def test_matmul_logical_not():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2, strategy3):
            super().__init__()
            self.matmul = P.MatMul().set_strategy(strategy1)
            self.logicalnot = P.LogicalNot().set_strategy(strategy2)
            self.equal = P.Equal().set_strategy(strategy3)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.equal(out, b)
            out = self.logicalnot(out)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), )
    strategy3 = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2, strategy3)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([128, 64]), dtype=ms.float32)
    _executor.compile(net, x, y, b)

def test_matmul_cast():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2, strategy3):
            super().__init__()
            self.matmul = P.MatMul().set_strategy(strategy1)
            self.cast = P.Cast().set_strategy(strategy2)
            self.matmul2 = P.MatMul().set_strategy(strategy3)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            b = self.cast(b, ms.float32)
            out = self.matmul2(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), )
    strategy3 = ((1, 4), (4, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2, strategy3)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.int32)
    _executor.compile(net, x, y, b)


def test_cast_before_mirror():
    class Net(nn.Cell):
        def __init__(self, strategy1):
            super().__init__()
            self.matmul = P.MatMul().set_strategy(strategy1)
            self.cast = P.Cast()

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            b = self.cast(b, ms.float32)
            out = self.matmul(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0, cast_before_mirror=True)
    strategy1 = ((2, 2), (2, 2))
    net = GradWrap(NetWithLoss(Net(strategy1)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    
    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float16)
    _executor.compile(net, x, y, b)


def test_cast_before_mirror1():
    class Net(nn.Cell):
        def __init__(self, strategy1):
            super().__init__()
            self.matmul = P.MatMul().set_strategy(strategy1)
            self.cast = P.Cast()

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            b = self.cast(b, ms.float16)
            out = self.matmul(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0, cast_before_mirror=True)
    strategy1 = ((2, 2), (2, 2))
    net = GradWrap(NetWithLoss(Net(strategy1)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    
    x = Tensor(np.ones([128, 32]), dtype=ms.float16)
    y = Tensor(np.ones([32, 64]), dtype=ms.float16)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    _executor.compile(net, x, y, b)


def test_cast_before_mirror2():
    class Net(nn.Cell):
        def __init__(self, strategy1):
            super().__init__()
            self.matmul = P.MatMul().set_strategy(strategy1)
            self.cast = P.Cast()

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            b = self.cast(b, ms.float16)
            out = self.matmul(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0, cast_before_mirror=False)
    strategy1 = ((2, 2), (2, 2))
    net = GradWrap(NetWithLoss(Net(strategy1)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    
    x = Tensor(np.ones([128, 32]), dtype=ms.float16)
    y = Tensor(np.ones([32, 64]), dtype=ms.float16)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    _executor.compile(net, x, y, b)


def test_cast_before_mirror3():
    class Net(nn.Cell):
        def __init__(self, strategy1):
            super().__init__()
            self.matmul = P.MatMul().set_strategy(strategy1)
            self.cast = P.Cast()

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            b = self.cast(b, ms.float16)
            out = self.matmul(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((2, 2), (2, 2))
    net = GradWrap(NetWithLoss(Net(strategy1)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    
    x = Tensor(np.ones([128, 32]), dtype=ms.float16)
    y = Tensor(np.ones([32, 64]), dtype=ms.float16)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    _executor.compile(net, x, y, b)
