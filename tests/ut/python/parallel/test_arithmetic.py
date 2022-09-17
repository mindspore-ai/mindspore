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
from mindspore.common.api import _cell_graph_executor
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
    _cell_graph_executor.compile(net, x, y, b)


def test_matmul_sub():
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-sub net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """

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
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-add net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """

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
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-mul net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """

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
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-mod net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """

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
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-floormod net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """

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
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-atan2 net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """

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
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-divNoNan net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """

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
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-logical_and net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """

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
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-logical_or net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """

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
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-div net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """

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
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-add broadcast net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """

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
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-add broadcast net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """

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
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-sub broadcast net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """

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
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-sub broadcast net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """

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
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-mul broadcast net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """

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
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-mul broadcast net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """

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
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-div broadcast net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """

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
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-div broadcast net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """

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
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-greater broadcast net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """

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
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-greater broadcast net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """

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
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-floordiv net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """

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
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-floordiv broadcast net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """

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
    """
    Feature: distribute operator sub in auto parallel.
    Description: matmul-floordiv broadcast net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """

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
    """
    Feature: distribute operator sub in auto parallel.
    Description: mul-assign_sub net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """

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
            self.assign_sub(self.assignsub_weight, out)
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
        _cell_graph_executor.compile(net, x)

    context.set_auto_parallel_context(device_num=64, global_rank=15)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    net = SubGradWrap(SubNetWithLoss(Net()))
    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    compile_sub_net(net, x)


def test_assign_add():
    """
    Feature: distribute operator add in auto parallel.
    Description: mul-assign_add net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assign_add = P.AssignAdd()
            self.mul = P.Mul()
            self.mul_weight = Parameter(Tensor(np.full([128, 32],
                                                       0.5, dtype=np.float32)),
                                        name="mul_weight")
            self.assignadd_weight = Parameter(Tensor(np.full([128, 32],
                                                             1.1, dtype=np.float32)),
                                              name="assignadd_weight")

        def construct(self, x):
            out = self.mul(x, self.mul_weight)
            self.assign_add(self.assignadd_weight, out)
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
        _cell_graph_executor.compile(net, x)

    context.set_auto_parallel_context(device_num=64, global_rank=15)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    net = SubGradWrap(SubNetWithLoss(Net()))
    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    compile_sub_net(net, x)


def test_assign():
    """
    Feature: distribute operator assign in auto parallel.
    Description: mul-assign_sub net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.assign = P.Assign()
            self.mul = P.Mul()
            self.mul_weight = Parameter(Tensor(np.full([128, 32],
                                                       0.5, dtype=np.float32)),
                                        name="mul_weight")
            self.assign_weight = Parameter(Tensor(np.full([128, 32],
                                                          1.1, dtype=np.float32)),
                                           name="assign_weight")

        def construct(self, x):
            out = self.mul(x, self.mul_weight)
            self.assign(self.assign_weight, out)
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
        _cell_graph_executor.compile(net, x)

    context.set_auto_parallel_context(device_num=64, global_rank=15)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    net = SubGradWrap(SubNetWithLoss(Net()))
    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    compile_sub_net(net, x)


def test_matmul_bitwise_and_broadcast():
    """
    Feature: distribute operator BitwiseAnd in auto parallel.
    Description: mul-BitwiseAnd net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """

    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.bitwise_and = P.BitwiseAnd().shard(strategy1)
            self.matmul = P.MatMul().shard(strategy2)

        def construct(self, x, y, z):
            out = self.bitwise_and(x, y)
            out = self.matmul(out, z)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 1), (1, 4))
    strategy2 = ((1, 4), (4, 2))
    net = Net(strategy1, strategy2)

    x = Tensor(np.ones([64, 1]), dtype=ms.int32)
    y = Tensor(np.ones([1, 64]), dtype=ms.int32)
    z = Tensor(np.ones([64, 32]), dtype=ms.int32)
    compile_net(net, x, y, z)


def test_matmul_bitwise_or_broadcast():
    """
    Feature: distribute operator BitwiseOr in auto parallel.
    Description: mul-BitwiseOr net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """

    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.bitwise_or = P.BitwiseOr().shard(strategy1)
            self.matmul = P.MatMul().shard(strategy2)

        def construct(self, x, y, z):
            out = self.bitwise_or(x, y)
            out = self.matmul(out, z)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 1), (1, 4))
    strategy2 = ((1, 4), (4, 2))
    net = Net(strategy1, strategy2)

    x = Tensor(np.ones([64, 1]), dtype=ms.int32)
    y = Tensor(np.ones([1, 64]), dtype=ms.int32)
    z = Tensor(np.ones([64, 32]), dtype=ms.int32)
    compile_net(net, x, y, z)


def test_matmul_bitwise_xor_broadcast():
    """
    Feature: distribute operator BitwiseXor in auto parallel.
    Description: mul-BitwiseXor net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """

    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.bitwise_xor = P.BitwiseXor().shard(strategy1)
            self.matmul = P.MatMul().shard(strategy2)

        def construct(self, x, y, z):
            out = self.bitwise_xor(x, y)
            out = self.matmul(out, z)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 1), (1, 4))
    strategy2 = ((1, 4), (4, 2))
    net = Net(strategy1, strategy2)

    x = Tensor(np.ones([64, 1]), dtype=ms.int32)
    y = Tensor(np.ones([1, 64]), dtype=ms.int32)
    z = Tensor(np.ones([64, 32]), dtype=ms.int32)
    compile_net(net, x, y, z)


def test_matmul_mul_no_nan_broadcast():
    """
    Feature: distribute operator MulNoNan in auto parallel.
    Description: mul-MulNoNan net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """

    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.mul_no_nan = P.MulNoNan().shard(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.mul_no_nan(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 4), (4, 1))
    strategy2 = ((4, 1), (1, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 1]), dtype=ms.float32)
    b = Tensor(np.ones([1, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_truncate_div_broadcast():
    """
    Feature: distribute operator TruncateDiv in auto parallel.
    Description: mul-TruncateDiv net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """

    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.truncate_div = P.TruncateDiv().shard(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.truncate_div(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 4), (4, 1))
    strategy2 = ((4, 1), (1, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 1]), dtype=ms.float32)
    b = Tensor(np.ones([1, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_truncate_mod_broadcast():
    """
    Feature: distribute operator TruncateMod in auto parallel.
    Description: mul-TruncateMod net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """

    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.truncate_mod = P.TruncateMod().shard(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.truncate_mod(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 4), (4, 1))
    strategy2 = ((4, 1), (1, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 1]), dtype=ms.float32)
    b = Tensor(np.ones([1, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_xdivy_broadcast():
    """
    Feature: distribute operator Xdivy in auto parallel.
    Description: mul-Xdivy net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """

    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.xdivy = P.Xdivy().shard(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.xdivy(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 4), (4, 1))
    strategy2 = ((4, 1), (1, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 1]), dtype=ms.float32)
    b = Tensor(np.ones([1, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_xlogy_broadcast():
    """
    Feature: distribute operator Xlogy in auto parallel.
    Description: mul-Xlogy net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """

    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.xlogy = P.Xlogy().shard(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.xlogy(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 4), (4, 1))
    strategy2 = ((4, 1), (1, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 1]), dtype=ms.float32)
    b = Tensor(np.ones([1, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_squared_difference_broadcast():
    """
    Feature: distribute operator SquaredDifference in auto parallel.
    Description: mul-SquaredDifference net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """

    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.squared_difference = P.SquaredDifference().shard(strategy2)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.squared_difference(out, b)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 4), (4, 1))
    strategy2 = ((4, 1), (1, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 1]), dtype=ms.float32)
    b = Tensor(np.ones([1, 64]), dtype=ms.float32)
    compile_net(net, x, y, b)


def test_matmul_masked_fill_broadcast_with_value_float():
    """
    Feature: distribute operator MaskedFill in auto parallel.
    Description: mul-MaskedFill net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """

    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.masked_fill = P.MaskedFill().shard(strategy2)
            self.value = 1.0

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.masked_fill(out, b, self.value)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 4), (4, 1))
    strategy2 = ((4, 1), (1, 2))
    net = Net(strategy1, strategy2)

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 1]), dtype=ms.float32)
    b = Tensor(np.ones([1, 64]), dtype=ms.bool_)
    compile_net(net, x, y, b)


def test_matmul_masked_fill_broadcast_with_value_tensor():
    """
    Feature: distribute operator MaskedFill in auto parallel.
    Description: mul-MaskedFill net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """

    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.masked_fill = P.MaskedFill().shard(strategy2)
            self.value = Tensor(1.0, ms.float32)

        def construct(self, x, y, b):
            out = self.matmul(x, y)
            out = self.masked_fill(out, b, self.value)
            return out

    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 4), (4, 1))
    strategy2 = ((4, 1), (1, 2), ())
    net = Net(strategy1, strategy2)

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 1]), dtype=ms.float32)
    b = Tensor(np.ones([1, 64]), dtype=ms.bool_)
    compile_net(net, x, y, b)
