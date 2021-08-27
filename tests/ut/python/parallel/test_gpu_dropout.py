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
from mindspore.common.api import _cell_graph_executor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from tests.ut.python.ops.test_math_ops import VirtualLoss


grad_all = C.GradOperation(get_all=True)


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x, y):
        predict = self.network(x, y)
        return self.loss(predict)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x, y):
        return grad_all(self.network)(x, y)


class Net(nn.Cell):
    def __init__(self, strategy1=None, strategy2=None):
        super().__init__()
        self.dropout = P.Dropout(keep_prob=0.6).shard(strategy1)
        self.matmul = P.MatMul().shard(strategy2)

    def construct(self, x, y):
        out = self.matmul(x, y)
        out, _ = self.dropout(out)
        return out


def test_dropout_semi_auto():
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel")
    net = GradWrap(NetWithLoss(Net()))
    net.set_auto_parallel()

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 128]), dtype=ms.float32)
    net.set_train()
    _cell_graph_executor.compile(net, x, y)


def test_dropout_semi_auto2():
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel")
    strategy1 = ((8, 1),)
    strategy2 = ((4, 2), (2, 1))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    net.set_auto_parallel()

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 128]), dtype=ms.float32)
    net.set_train()
    _cell_graph_executor.compile(net, x, y)


def test_dropout_semi_auto3():
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 4),)
    strategy2 = ((4, 2), (2, 1))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    net.set_auto_parallel()

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 128]), dtype=ms.float32)
    net.set_train()
    _cell_graph_executor.compile(net, x, y)


def test_dropout_auto():
    context.set_auto_parallel_context(device_num=8, global_rank=0, parallel_mode="auto_parallel")
    net = GradWrap(NetWithLoss(Net()))
    net.set_auto_parallel()

    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 128]), dtype=ms.float32)
    net.set_train()
    _cell_graph_executor.compile(net, x, y)
