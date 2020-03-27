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
from mindspore import context
from mindspore.context import set_auto_parallel_context
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor
from tests.ut.python.ops.test_math_ops import VirtualLoss
import mindspore as ms
from mindspore.common.api import _executor
from mindspore.ops import composite as C


# model_parallel test
# export PARALLEL_CHECKPOINT_ON=on
# export PARALLEL_TRAIN_TIMES=4
def test_six_matmul():
    class NetWithLoss(nn.Cell):
        def __init__(self, network):
            super(NetWithLoss, self).__init__()
            self.loss = VirtualLoss()
            self.network = network

        def construct(self, x1, x2, x3, x4, x5, x6, x7):
            predict = self.network(x1, x2, x3, x4, x5, x6, x7)
            return self.loss(predict)


    class GradWrap(nn.Cell):
        def __init__(self, network):
            super(GradWrap, self).__init__()
            self.network = network

        def construct(self, x1, x2, x3, x4, x5, x6, x7):
            return C.grad_all(self.network)(x1, x2, x3, x4, x5, x6, x7)

    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2, strategy3, strategy4, strategy5, strategy6):
            super().__init__()
            self.matmul1 = P.MatMul().set_strategy(strategy1)
            self.matmul2 = P.MatMul().set_strategy(strategy2)
            self.matmul3 = P.MatMul().set_strategy(strategy3)
            self.matmul4 = P.MatMul().set_strategy(strategy4)
            self.matmul5 = P.MatMul().set_strategy(strategy5)
            self.matmul6 = P.MatMul().set_strategy(strategy6)

        def construct(self, x1, x2, x3, x4, x5, x6, x7):
            out = self.matmul1(x1, x2)
            out = self.matmul2(out, x3)
            out = self.matmul3(out, x4)
            out = self.matmul4(out, x5)
            out = self.matmul5(out, x6)
            out = self.matmul6(out, x7)
            return out

    set_auto_parallel_context(device_num=512, global_rank=0)
    strategy1 = ((8, 1), (1, 1))
    strategy2 = ((1, 8), (8, 1))
    strategy3 = ((2, 2), (2, 2))
    strategy4 = ((4, 2), (2, 4))
    strategy5 = ((2, 4), (4, 2))
    strategy6 = ((4, 4), (4, 4))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2, strategy3, strategy4, strategy5, strategy6)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x1 = Tensor(np.ones([128, 32]), dtype=ms.float32)
    x2 = Tensor(np.ones([32, 64]), dtype=ms.float32)
    x3 = Tensor(np.ones([64, 64]), dtype=ms.float32)
    x4 = Tensor(np.ones([64, 128]), dtype=ms.float32)
    x5 = Tensor(np.ones([128, 64]), dtype=ms.float32)
    x6 = Tensor(np.ones([64, 32]), dtype=ms.float32)
    x7 = Tensor(np.ones([32, 32]), dtype=ms.float32)
    _executor.compile(net, x1, x2, x3, x4, x5, x6, x7)

# remove matmul2
def test_six_matmul_repeated1():
    class NetWithLoss(nn.Cell):
        def __init__(self, network):
            super(NetWithLoss, self).__init__()
            self.loss = VirtualLoss()
            self.network = network

        def construct(self, x1, x2, x4, x5, x6, x7):
            predict = self.network(x1, x2, x4, x5, x6, x7)
            return self.loss(predict)


    class GradWrap(nn.Cell):
        def __init__(self, network):
            super(GradWrap, self).__init__()
            self.network = network

        def construct(self, x1, x2, x4, x5, x6, x7):
            return C.grad_all(self.network)(x1, x2, x4, x5, x6, x7)

    class Net(nn.Cell):
        def __init__(self, strategy1, strategy3, strategy4, strategy5, strategy6):
            super().__init__()
            self.matmul1 = P.MatMul().set_strategy(strategy1)
            self.matmul3 = P.MatMul().set_strategy(strategy3)
            self.matmul4 = P.MatMul().set_strategy(strategy4)
            self.matmul5 = P.MatMul().set_strategy(strategy5)
            self.matmul6 = P.MatMul().set_strategy(strategy6)

        def construct(self, x1, x2, x4, x5, x6, x7):
            out = self.matmul1(x1, x2)
            out = self.matmul3(out, x4)
            out = self.matmul4(out, x5)
            out = self.matmul5(out, x6)
            out = self.matmul6(out, x7)
            return out

    set_auto_parallel_context(device_num=512, global_rank=0)
    strategy1 = ((8, 1), (1, 1))
    strategy3 = ((8, 1), (1, 1))
    strategy4 = ((8, 1), (1, 1))
    strategy5 = ((8, 1), (1, 1))
    strategy6 = ((8, 1), (1, 1))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy3, strategy4, strategy5, strategy6)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x1 = Tensor(np.ones([128, 32]), dtype=ms.float32)
    x2 = Tensor(np.ones([32, 64]), dtype=ms.float32)
    x4 = Tensor(np.ones([64, 128]), dtype=ms.float32)
    x5 = Tensor(np.ones([128, 64]), dtype=ms.float32)
    x6 = Tensor(np.ones([64, 32]), dtype=ms.float32)
    x7 = Tensor(np.ones([32, 32]), dtype=ms.float32)
    _executor.compile(net, x1, x2, x4, x5, x6, x7)

# add matmul7
def test_six_matmul_repeated2():
    class NetWithLoss(nn.Cell):
        def __init__(self, network):
            super(NetWithLoss, self).__init__()
            self.loss = VirtualLoss()
            self.network = network

        def construct(self, x1, x2, x4, x5, x6, x7, x8):
            predict = self.network(x1, x2, x4, x5, x6, x7, x8)
            return self.loss(predict)


    class GradWrap(nn.Cell):
        def __init__(self, network):
            super(GradWrap, self).__init__()
            self.network = network

        def construct(self, x1, x2, x4, x5, x6, x7, x8):
            return C.grad_all(self.network)(x1, x2, x4, x5, x6, x7, x8)

    class Net(nn.Cell):
        def __init__(self, strategy1, strategy3, strategy4, strategy5, strategy6, strategy7):
            super().__init__()
            self.matmul1 = P.MatMul().set_strategy(strategy1)
            self.matmul3 = P.MatMul().set_strategy(strategy3)
            self.matmul4 = P.MatMul().set_strategy(strategy4)
            self.matmul5 = P.MatMul().set_strategy(strategy5)
            self.matmul6 = P.MatMul().set_strategy(strategy6)
            self.matmul7 = P.MatMul().set_strategy(strategy7)

        def construct(self, x1, x2, x4, x5, x6, x7, x8):
            out = self.matmul1(x1, x2)
            out = self.matmul3(out, x4)
            out = self.matmul4(out, x5)
            out = self.matmul5(out, x6)
            out = self.matmul6(out, x7)
            out = self.matmul7(out, x8)
            return out

    set_auto_parallel_context(device_num=512, global_rank=0)
    strategy1 = ((8, 1), (1, 1))
    strategy3 = ((8, 1), (1, 1))
    strategy4 = ((8, 1), (1, 1))
    strategy5 = ((8, 1), (1, 1))
    strategy6 = ((8, 1), (1, 1))
    strategy7 = ((8, 1), (1, 1))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy3, strategy4, strategy5, strategy6, strategy7)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x1 = Tensor(np.ones([128, 32]), dtype=ms.float32)
    x2 = Tensor(np.ones([32, 64]), dtype=ms.float32)
    x4 = Tensor(np.ones([64, 128]), dtype=ms.float32)
    x5 = Tensor(np.ones([128, 64]), dtype=ms.float32)
    x6 = Tensor(np.ones([64, 32]), dtype=ms.float32)
    x7 = Tensor(np.ones([32, 32]), dtype=ms.float32)
    x8 = Tensor(np.ones([32, 128]), dtype=ms.float32)
    _executor.compile(net, x1, x2, x4, x5, x6, x7, x8)


# add scope2
def test_six_matmul_repeated3():
    class NetWithLoss(nn.Cell):
        def __init__(self, network1, network2):
            super(NetWithLoss, self).__init__()
            self.loss = VirtualLoss()
            self.network = network1
            self.network2 = network2

        def construct(self, x1, x2, x4, x5, x6, x7, x8, x9, x10):
            predict = self.network(x1, x2, x4, x5, x6, x7, x8)
            predict = self.network2(predict, x9, x10)
            return self.loss(predict)


    class GradWrap(nn.Cell):
        def __init__(self, network):
            super(GradWrap, self).__init__()
            self.network = network

        def construct(self, x1, x2, x4, x5, x6, x7, x8, x9, x10):
            return C.grad_all(self.network)(x1, x2, x4, x5, x6, x7, x8, x9, x10)

    class Net(nn.Cell):
        def __init__(self, strategy1, strategy3, strategy4, strategy5, strategy6, strategy7):
            super().__init__()
            self.matmul1 = P.MatMul().set_strategy(strategy1)
            self.matmul3 = P.MatMul().set_strategy(strategy3)
            self.matmul4 = P.MatMul().set_strategy(strategy4)
            self.matmul5 = P.MatMul().set_strategy(strategy5)
            self.matmul6 = P.MatMul().set_strategy(strategy6)
            self.matmul7 = P.MatMul().set_strategy(strategy7)

        def construct(self, x1, x2, x4, x5, x6, x7, x8):
            out = self.matmul1(x1, x2)
            out = self.matmul3(out, x4)
            out = self.matmul4(out, x5)
            out = self.matmul5(out, x6)
            out = self.matmul6(out, x7)
            out = self.matmul7(out, x8)
            return out

    class Net1(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul1 = P.MatMul().set_strategy(strategy1)
            self.matmul2 = P.MatMul().set_strategy(strategy2)

        def construct(self, x1, x2, x3):
            out = self.matmul1(x1, x2)
            out = self.matmul2(out, x3)
            return out


    set_auto_parallel_context(device_num=512, global_rank=0)
    strategy1 = ((8, 1), (1, 1))
    strategy3 = ((8, 1), (1, 1))
    strategy4 = ((8, 1), (1, 1))
    strategy5 = ((8, 1), (1, 1))
    strategy6 = ((8, 1), (1, 1))
    strategy7 = ((8, 1), (1, 1))
    strategy8 = ((8, 1), (1, 1))
    strategy9 = ((8, 1), (1, 1))
    net1 = Net(strategy1, strategy3, strategy4, strategy5, strategy6, strategy7)
    net2 = Net1(strategy8, strategy9)
    net = GradWrap(NetWithLoss(net1, net2))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    x1 = Tensor(np.ones([128, 32]), dtype=ms.float32)
    x2 = Tensor(np.ones([32, 64]), dtype=ms.float32)
    x4 = Tensor(np.ones([64, 128]), dtype=ms.float32)
    x5 = Tensor(np.ones([128, 64]), dtype=ms.float32)
    x6 = Tensor(np.ones([64, 32]), dtype=ms.float32)
    x7 = Tensor(np.ones([32, 32]), dtype=ms.float32)
    x8 = Tensor(np.ones([32, 128]), dtype=ms.float32)
    x9 = Tensor(np.ones([128, 64]), dtype=ms.float32)
    x10 = Tensor(np.ones([64, 64]), dtype=ms.float32)
    _executor.compile(net, x1, x2, x4, x5, x6, x7, x8, x9, x10)

