# Copyright 2021 Huawei Technologies Co., Ltd
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
from mindspore import Tensor, context
from mindspore.ops import operations as P
from mindspore.common.api import _cell_graph_executor
from mindspore.nn.wrap.cell_wrapper import MicroBatchInterleaved


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(nn.Cell):
    def __init__(self, strategy1, strategy2):
        super().__init__()
        self.matmul1 = P.MatMul().shard(strategy1)
        self.matmul2 = P.MatMul().shard(strategy2)

    def construct(self, x, y, b):
        out = self.matmul1(x, y)
        out = self.matmul2(out, b)
        return out


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = P.ReLU()
        self.network = network

    def construct(self, x, y, b):
        predict = self.network(x, y, b)
        return self.loss(predict)


def compile_net(net, x, y, b):
    net.set_train()
    _cell_graph_executor.compile(net, x, y, b)


def test_micro_batch_interleaved():
    """
    Feature: test MicroBatchInterleaved in auto parallel.
    Description: net with MicroBatchInterleaved in semi auto parallel.
    Expectation: compile done without error.
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_auto_parallel_context(device_num=8, global_rank=0, gradients_mean=True)
    strategy1 = ((4, 2), (2, 1))
    strategy2 = ((2, 4), (4, 1))
    micro_batch_interleaved = 2
    net = MicroBatchInterleaved(NetWithLoss(Net(strategy1, strategy2)), micro_batch_interleaved)

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32 * micro_batch_interleaved, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64 * micro_batch_interleaved, 64]), dtype=ms.float32)

    compile_net(net, x, y, b)
