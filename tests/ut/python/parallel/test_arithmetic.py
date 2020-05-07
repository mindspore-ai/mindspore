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
from mindspore import Parameter, Tensor, context
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.common.api import _executor
from tests.ut.python.ops.test_math_ops import VirtualLoss


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


def compile(net, x, y, b):
    net.set_auto_parallel()
    _executor.compile(net, x, y, b)


def test_matmul_sub():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().set_strategy(strategy1)
            self.sub = P.Sub().set_strategy(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.sub(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    compile(net, x, y, b)


def test_matmul_add():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().set_strategy(strategy1)
            self.add = P.TensorAdd().set_strategy(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.add(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    compile(net, x, y, b)


def test_matmul_mul():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().set_strategy(strategy1)
            self.mul = P.Mul().set_strategy(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.mul(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    compile(net, x, y, b)


def test_matmul_div():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().set_strategy(strategy1)
            self.div = P.Div().set_strategy(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.div(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    compile(net, x, y, b)

def test_matmul_greater():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().set_strategy(strategy1)
            self.greater = P.Greater().set_strategy(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.greater(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    compile(net, x, y, b)

def test_matmul_add_broadcast():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().set_strategy(strategy1)
            self.add = P.TensorAdd().set_strategy(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.add(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (2, ))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64]), dtype=ms.float32)
    compile(net, x, y, b)


def test_matmul_add_broadcast2():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().set_strategy(strategy1)
            self.add = P.TensorAdd().set_strategy(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.add(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((2, 4), (4, 1))
    strategy2 = ((4, 1), (1, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 1]), dtype=ms.float32)
    b = Tensor(np.ones([1, 64]), dtype=ms.float32)
    compile(net, x, y, b)


def test_matmul_sub_broadcast():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().set_strategy(strategy1)
            self.sub = P.Sub().set_strategy(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.sub(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (2, ))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64]), dtype=ms.float32)
    compile(net, x, y, b)


def test_matmul_sub_broadcast2():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().set_strategy(strategy1)
            self.sub = P.Sub().set_strategy(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.sub(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((2, 4), (4, 1))
    strategy2 = ((4, 1), (1, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 1]), dtype=ms.float32)
    b = Tensor(np.ones([1, 64]), dtype=ms.float32)
    compile(net, x, y, b)


def test_matmul_mul_broadcast():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().set_strategy(strategy1)
            self.mul = P.Mul().set_strategy(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.mul(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (2, ))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64]), dtype=ms.float32)
    compile(net, x, y, b)


def test_matmul_mul_broadcast2():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().set_strategy(strategy1)
            self.mul = P.Mul().set_strategy(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.mul(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((2, 4), (4, 1))
    strategy2 = ((4, 1), (1, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 1]), dtype=ms.float32)
    b = Tensor(np.ones([1, 64]), dtype=ms.float32)
    compile(net, x, y, b)


def test_matmul_div_broadcast():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().set_strategy(strategy1)
            self.div = P.Div().set_strategy(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.div(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (2, ))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64]), dtype=ms.float32)
    compile(net, x, y, b)


def test_matmul_div_broadcast2():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().set_strategy(strategy1)
            self.div = P.Div().set_strategy(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.div(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((2, 4), (4, 1))
    strategy2 = ((4, 1), (1, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 1]), dtype=ms.float32)
    b = Tensor(np.ones([1, 64]), dtype=ms.float32)
    compile(net, x, y, b)

def test_matmul_greater_broadcast():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().set_strategy(strategy1)
            self.greater = P.Greater().set_strategy(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.greater(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (2, ))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64]), dtype=ms.float32)
    compile(net, x, y, b)


def test_matmul_greater_broadcast2():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().set_strategy(strategy1)
            self.greater = P.Greater().set_strategy(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.greater(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((2, 4), (4, 1))
    strategy2 = ((4, 1), (1, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 1]), dtype=ms.float32)
    b = Tensor(np.ones([1, 64]), dtype=ms.float32)
    compile(net, x, y, b)

def test_matmul_floordiv():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().set_strategy(strategy1)
            self.floordiv = P.FloorDiv().set_strategy(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.floordiv(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    compile(net, x, y, b)


def test_matmul_floordiv_broadcast():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().set_strategy(strategy1)
            self.floordiv = P.FloorDiv().set_strategy(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.floordiv(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (2, ))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64]), dtype=ms.float32)
    compile(net, x, y, b)


def test_matmul_floordiv_broadcast2():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().set_strategy(strategy1)
            self.floordiv = P.FloorDiv().set_strategy(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.floordiv(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    strategy1 = ((2, 4), (4, 1))
    strategy2 = ((4, 1), (1, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 1]), dtype=ms.float32)
    b = Tensor(np.ones([1, 64]), dtype=ms.float32)
    compile(net, x, y, b)


def test_assign_sub():
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assign_sub = P.AssignSub()
            self.mul = P.Mul()
            self.mul_weight = Parameter(Tensor(np.full([128, 32],
                                        0.5, dtype=np.float32)),
                                        name="mul_weight")
            self.assignsub_weight = Parameter(Tensor(np.full([128, 32],
                                              1.1, dtype=np.float32)),
                                              name="assignsub_weight")

        def construct(self, x, y, z):
            out = self.mul(x, self.mul_weight)
            out = self.assign_sub(self.assignsub_weight, out)
            return out

    context.set_auto_parallel_context(device_num=64, global_rank=15)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    net = GradWrap(NetWithLoss(Net()))
    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([128, 32]), dtype=ms.float32)
    z = Tensor(np.ones([128, 32]), dtype=ms.float32)
    compile(net, x, y, z)
