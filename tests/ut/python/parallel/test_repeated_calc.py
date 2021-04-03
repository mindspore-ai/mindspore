# Copyright 2020 Huawei Technologies Co., Ltd
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


# it has not redistribution
def test_tensoradd_reshape_matmul():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.add = P.Add().shard(strategy1)
            self.reshape = P.Reshape()
            self.matmul = P.MatMul().shard(strategy2)

        def construct(self, x, y, b):
            out = self.add(x, y)
            out = self.reshape(out, (256, 16))
            out = self.matmul(out, b)
            return out

    context.set_auto_parallel_context(device_num=64, global_rank=0, gradients_mean=True)
    strategy1 = ((8, 1, 1), (8, 1, 1))
    strategy2 = ((8, 1), (1, 8))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_context(save_graphs=False)

    x = Tensor(np.ones([32, 8, 16]), dtype=ms.float32)
    y = Tensor(np.ones([32, 8, 16]), dtype=ms.float32)
    b = Tensor(np.ones([16, 16]), dtype=ms.float32)

    compile_net(net, x, y, b)


def test_two_matmul():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul1 = P.MatMul().shard(strategy1)
            self.matmul2 = P.MatMul().shard(strategy2)

        def construct(self, x, y, b):
            out = self.matmul1(x, y)
            out = self.matmul2(out, b)
            return out

    context.set_auto_parallel_context(device_num=64, global_rank=0, gradients_mean=True)
    strategy1 = ((8, 8), (8, 1))
    strategy2 = ((8, 1), (1, 1))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_context(save_graphs=False)

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)

    compile_net(net, x, y, b)
