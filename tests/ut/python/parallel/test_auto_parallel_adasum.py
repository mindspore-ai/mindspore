# Copyright 2022 Huawei Technologies Co., Ltd
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
from mindspore import context, Tensor, Parameter
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import Cell, TrainOneStepCell, Momentum, AdaSumByDeltaWeightWrapCell, AdaSumByGradWrapCell
from mindspore.ops import operations as P


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(Cell):
    def __init__(self, strategy1=None, strategy2=None, strategy3=None):
        super().__init__()
        self.mul = P.Mul().shard(strategy1)
        self.matmul = P.MatMul().shard(strategy2)
        self.gather = P.Gather().shard(strategy3)
        self.reduce_sum = P.ReduceSum()
        self.mul_weight = Parameter(Tensor(np.ones([64, 32]), dtype=ms.float32), "w1")
        self.matmul_weight = Parameter(Tensor(np.ones([32, 32]), dtype=ms.float32), "w2")
        self.embedding_table = Parameter(Tensor(np.ones([64, 32]), dtype=ms.float32), "embedding_table")

    def construct(self, x, b):
        out = self.gather(self.embedding_table, x, 0)
        out = self.matmul(out, self.matmul_weight)
        out = self.mul(out, self.mul_weight)
        out = out + b
        return self.reduce_sum(out)


_x = Tensor(np.ones([64]), dtype=ms.int32)
_b = Tensor(np.ones([64, 32]), dtype=ms.float32)


def compile_net(net, by_grad=True):
    if by_grad:
        optimizer = AdaSumByGradWrapCell(Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9))
    else:
        optimizer = AdaSumByDeltaWeightWrapCell(Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9))
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, _x, _b)
    context.reset_auto_parallel_context()


def test_auto_parallel_adasum1():
    """
    Feature: adasum in auto parallel.
    Description: verify adasum by mul/matmul/gather, rank0, dp, mp, not_full_dp
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=32, global_rank=0)
    mul_strategy1 = ((8, 4), (8, 4))
    matmul_strategy2 = ((8, 1), (1, 1))
    gather_strategy3 = ((1, 1), (32,))
    net = Net(mul_strategy1, matmul_strategy2, gather_strategy3)
    compile_net(net)

def test_auto_parallel_adasum2():
    """
    Feature: adasum in auto parallel.
    Description: verify adasum by mul/matmul/gather, rank0, dp, mp, not_full_dp
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=32, global_rank=0)
    mul_strategy1 = ((8, 4), (8, 4))
    matmul_strategy2 = ((8, 1), (1, 1))
    gather_strategy3 = ((1, 1), (32,))
    net = Net(mul_strategy1, matmul_strategy2, gather_strategy3)
    compile_net(net, by_grad=False)

def test_auto_parallel_adasum3():
    """
    Feature: adasum in auto parallel.
    Description: verify adasum by mul/matmul/gather, rank0, mix_dp_mp, mp
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=32, global_rank=0)
    mul_strategy1 = ((8, 4), (8, 4))
    matmul_strategy2 = ((8, 4), (4, 1))
    gather_strategy3 = ((32, 1), (1,))
    net = Net(mul_strategy1, matmul_strategy2, gather_strategy3)
    compile_net(net)

def test_auto_parallel_adasum4():
    """
    Feature: adasum in auto parallel.
    Description: verify adasum by mul/matmul/gather, rank0, mix_dp_mp, mp
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=32, global_rank=0)
    mul_strategy1 = ((8, 4), (8, 4))
    matmul_strategy2 = ((8, 4), (4, 1))
    gather_strategy3 = ((32, 1), (1,))
    net = Net(mul_strategy1, matmul_strategy2, gather_strategy3)
    compile_net(net, by_grad=False)

def test_auto_parallel_adasum5():
    """
    Feature: adasum in auto parallel.
    Description: verify adasum by mul/matmul/gather, rank16, dp, mp, not_full_dp
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=32, global_rank=16)
    mul_strategy1 = ((8, 4), (8, 4))
    matmul_strategy2 = ((8, 1), (1, 1))
    gather_strategy3 = ((1, 1), (32,))
    net = Net(mul_strategy1, matmul_strategy2, gather_strategy3)
    compile_net(net)

def test_auto_parallel_adasum6():
    """
    Feature: adasum in auto parallel.
    Description: verify adasum by mul/matmul/gather, rank16, dp, mp, not_full_dp
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=32, global_rank=16)
    mul_strategy1 = ((8, 4), (8, 4))
    matmul_strategy2 = ((8, 1), (1, 1))
    gather_strategy3 = ((1, 1), (32,))
    net = Net(mul_strategy1, matmul_strategy2, gather_strategy3)
    compile_net(net, by_grad=False)
