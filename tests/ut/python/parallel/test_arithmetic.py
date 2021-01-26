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
from mindspore import Parameter, Tensor, context
from mindspore.common.api import _executor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from tests.ut.python.ops.test_math_ops import VirtualLoss


grad_all = C.GradOperation(get_all=True)


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
        return grad_all(self.network)(x, y, b)


def compile_net(net, x, y, b):
    net.set_auto_parallel()
    net.set_train()
    _executor.compile(net, x, y, b)


def test_matmul_sub():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.sub = P.Sub().shard(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.sub(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_add():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.add = P.Add().shard(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.add(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_mul():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.mul = P.Mul().shard(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.mul(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)

def test_matmul_mod():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.mod = P.Mod().shard(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.mod(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)

def test_matmul_floormod():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.floormod = P.FloorMod().shard(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.floormod(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_atan2():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.atan2 = P.Atan2().shard(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.atan2(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_divNoNan():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.divNoNan = P.DivNoNan().shard(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.divNoNan(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_logicaland():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.equal = P.Equal().shard(strategy2)
            self.notequal = P.NotEqual().shard(strategy2)
            self.logical = P.LogicalAnd().shard(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out1 = self.equal(out, b)
            out = self.matmul(x, y)
            out2 = self.notequal(out, b)
            out = self.logical(out1, out2)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_logicalor():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.equal = P.Equal().shard(strategy2)
            self.notequal = P.NotEqual().shard(strategy2)
            self.logical = P.LogicalOr().shard(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out1 = self.equal(out, b)
            out = self.matmul(x, y)
            out2 = self.notequal(out, b)
            out = self.logical(out1, out2)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_div():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.div = P.Div().shard(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.div(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_add_broadcast():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.add = P.Add().shard(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.add(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (2,))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_add_broadcast2():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.add = P.Add().shard(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.add(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 4), (4, 1))
    strategy2 = ((4, 1), (1, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 1]), dtype=ms.float32)
    b = Tensor(np.ones([1, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_sub_broadcast():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.sub = P.Sub().shard(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.sub(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (2,))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_sub_broadcast2():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.sub = P.Sub().shard(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.sub(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 4), (4, 1))
    strategy2 = ((4, 1), (1, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 1]), dtype=ms.float32)
    b = Tensor(np.ones([1, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_mul_broadcast():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.mul = P.Mul().shard(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.mul(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (2,))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_mul_broadcast2():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.mul = P.Mul().shard(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.mul(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 4), (4, 1))
    strategy2 = ((4, 1), (1, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 1]), dtype=ms.float32)
    b = Tensor(np.ones([1, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_div_broadcast():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.div = P.Div().shard(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.div(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (2,))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_div_broadcast2():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.div = P.Div().shard(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.div(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 4), (4, 1))
    strategy2 = ((4, 1), (1, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 1]), dtype=ms.float32)
    b = Tensor(np.ones([1, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_greater_broadcast():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.greater = P.Greater().shard(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.greater(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (2,))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_greater_broadcast2():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.greater = P.Greater().shard(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.greater(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 4), (4, 1))
    strategy2 = ((4, 1), (1, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 1]), dtype=ms.float32)
    b = Tensor(np.ones([1, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_floordiv():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.floordiv = P.FloorDiv().shard(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.floordiv(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_floordiv_broadcast():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.floordiv = P.FloorDiv().shard(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.floordiv(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((4, 2), (2,))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_floordiv_broadcast2():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.floordiv = P.FloorDiv().shard(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.floordiv(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 4), (4, 1))
    strategy2 = ((4, 1), (1, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 1]), dtype=ms.float32)
    b = Tensor(np.ones([1, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


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

        def construct(self, x):
            out = self.mul(x, self.mul_weight)
            out = self.assign_sub(self.assignsub_weight, out)
            return out

    class SubNetWithLoss(nn.Cell):
        def __init__(self, network):
            super(SubNetWithLoss, self).__init__()
            self.loss = VirtualLoss()
            self.network = network

        def construct(self, x):
            predict = self.network(x,)
            return self.loss(predict)

    class SubGradWrap(nn.Cell):
        def __init__(self, network):
            super(SubGradWrap, self).__init__()
            self.network = network

        def construct(self, x):
            return grad_all(self.network)(x)

    def compile_sub_net(net, x):
        net.set_auto_parallel()
        net.set_train()
        _executor.compile(net, x)

    context.set_auto_parallel_context(device_num=64, global_rank=15)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    net = SubGradWrap(SubNetWithLoss(Net()))
    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    compile_sub_net(net, x)


def test_assign_add():
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assign_sub = P.AssignAdd()
            self.mul = P.Mul()
            self.mul_weight = Parameter(Tensor(np.full([128, 32],
                                                       0.5, dtype=np.float32)),
                                        name="mul_weight")
            self.assignsub_weight = Parameter(Tensor(np.full([128, 32],
                                                             1.1, dtype=np.float32)),
                                              name="assignsub_weight")

        def construct(self, x):
            out = self.mul(x, self.mul_weight)
            out = self.assign_sub(self.assignsub_weight, out)
            return out

    class SubNetWithLoss(nn.Cell):
        def __init__(self, network):
            super(SubNetWithLoss, self).__init__()
            self.loss = VirtualLoss()
            self.network = network

        def construct(self, x):
            predict = self.network(x,)
            return self.loss(predict)

    class SubGradWrap(nn.Cell):
        def __init__(self, network):
            super(SubGradWrap, self).__init__()
            self.network = network

        def construct(self, x):
            return grad_all(self.network)(x)

    def compile_sub_net(net, x):
        net.set_auto_parallel()
        net.set_train()
        _executor.compile(net, x)

    context.set_auto_parallel_context(device_num=64, global_rank=15)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    net = SubGradWrap(SubNetWithLoss(Net()))
    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    compile_sub_net(net, x)


def test_assign():
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assign_sub = P.Assign()
            self.mul = P.Mul()
            self.mul_weight = Parameter(Tensor(np.full([128, 32],
                                                       0.5, dtype=np.float32)),
                                        name="mul_weight")
            self.assignsub_weight = Parameter(Tensor(np.full([128, 32],
                                                             1.1, dtype=np.float32)),
                                              name="assignsub_weight")

        def construct(self, x):
            out = self.mul(x, self.mul_weight)
            out = self.assign_sub(self.assignsub_weight, out)
            return out

    class SubNetWithLoss(nn.Cell):
        def __init__(self, network):
            super(SubNetWithLoss, self).__init__()
            self.loss = VirtualLoss()
            self.network = network

        def construct(self, x):
            predict = self.network(x,)
            return self.loss(predict)

    class SubGradWrap(nn.Cell):
        def __init__(self, network):
            super(SubGradWrap, self).__init__()
            self.network = network

        def construct(self, x):
            return grad_all(self.network)(x)

    def compile_sub_net(net, x):
        net.set_auto_parallel()
        net.set_train()
        _executor.compile(net, x)

    context.set_auto_parallel_context(device_num=64, global_rank=15)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    net = SubGradWrap(SubNetWithLoss(Net()))
    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    compile_sub_net(net, x)
