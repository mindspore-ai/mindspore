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


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


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

def compile_graph(net, device_num, parallel_mode, x, y):
    context.set_auto_parallel_context(device_num=device_num, global_rank=0, parallel_mode=parallel_mode)
    net.set_train()
    _cell_graph_executor.compile(net, x, y)

def test_dropout_semi_auto():
    """
    Feature: distribute operator dropout in auto parallel with gpu backend.
    Description: dropout net without strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    net = GradWrap(NetWithLoss(Net()))
    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 128]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_dropout_semi_auto2():
    """
    Feature: distribute operator dropout in auto parallel with gpu backend.
    Description: dropout net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    strategy1 = ((8, 1),)
    strategy2 = ((4, 2), (2, 1))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 128]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_dropout_semi_auto3():
    """
    Feature: distribute operator dropout in auto parallel with gpu backend.
    Description: dropout net with strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    strategy1 = ((2, 4),)
    strategy2 = ((4, 2), (2, 1))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 128]), dtype=ms.float32)
    compile_graph(net, 8, "semi_auto_parallel", x, y)


def test_dropout_auto():
    """
    Feature: distribute operator dropout in auto parallel with gpu backend.
    Description: dropout net in auto parallel.
    Expectation: compile done without error.
    """
    net = GradWrap(NetWithLoss(Net()))
    x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 128]), dtype=ms.float32)
    compile_graph(net, 8, "auto_parallel", x, y)
