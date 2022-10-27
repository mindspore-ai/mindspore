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
from mindspore.context import set_auto_parallel_context
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
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


def compile_net(net, x):
    net.set_train()
    _cell_graph_executor.compile(net, x)


class Net(nn.Cell):
    def __init__(self, strategy1, strategy2, strategy3, strategy4, strategy5):
        super().__init__()
        self.query_w = Parameter(initializer(
            "normal", [8, 16], ms.float32), name='query')
        self.query = P.MatMul().shard(strategy1)

        self.key_w = Parameter(initializer(
            "normal", [8, 16], ms.float32), name='key')
        self.key = P.MatMul().shard(strategy2)

        self.value_w = Parameter(initializer(
            "normal", [8, 16], ms.float32), name='value')
        self.value = P.MatMul().shard(strategy3)

        self.score = P.MatMul().shard(strategy4)
        self.context = P.MatMul().shard(strategy5)
        self.transpose1 = P.Transpose()
        self.transpose2 = P.Transpose()
        self.relu = P.ReLU()

    def construct(self, x):
        q = self.query(x, self.query_w)
        k = self.key(x, self.key_w)
        v = self.value(x, self.value_w)

        k = self.transpose1(k, (1, 0))
        s = self.score(q, k)

        v = self.transpose2(v, (1, 0))
        c = self.context(v, s)
        out = self.relu(c)

        return out


def test_self_attention_standalone():
    context.reset_auto_parallel_context()
    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="stand_alone")
    net = GradWrap(NetWithLoss(
        Net(None, None, None, None, None)))

    x = Tensor(np.ones([32, 8]), dtype=ms.float32)

    compile_net(net, x)


def test_self_attention_semi():
    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((2, 2), (2, 2))
    strategy3 = ((2, 2), (2, 2))
    strategy4 = ((2, 4), (4, 1))
    strategy5 = ((2, 1), (1, 4))

    net = GradWrap(NetWithLoss(
        Net(strategy1, strategy2, strategy3, strategy4, strategy5)))

    x = Tensor(np.ones([32, 8]), dtype=ms.float32)

    compile_net(net, x)


def test_self_attention_dp():
    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

    strategy1 = ((8, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    strategy3 = ((8, 1), (1, 1))
    strategy4 = ((8, 1), (1, 1))
    strategy5 = ((8, 1), (1, 1))

    net = GradWrap(NetWithLoss(
        Net(strategy1, strategy2, strategy3, strategy4, strategy5)))

    x = Tensor(np.ones([32, 8]), dtype=ms.float32)

    compile_net(net, x)


def test_self_attention_auto():
    set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    net = GradWrap(NetWithLoss(
        Net(None, None, None, None, None)))

    x = Tensor(np.ones([32, 8]), dtype=ms.float32)

    compile_net(net, x)
