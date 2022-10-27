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
from mindspore import context, Tensor, Parameter
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import Cell, TrainOneStepCell, Momentum
from mindspore.ops import operations as P


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(Cell):
    def __init__(self, mul_weight, batch_matmul_weight, transpose_b=False, strategy1=None, strategy2=None):
        super().__init__()
        self.mul = P.Mul().shard(strategy1)
        self.batch_matmul = P.BatchMatMul(transpose_b=transpose_b).shard(strategy2)
        self.mul_weight = Parameter(mul_weight, "w1")
        self.batch_matmul_weight = Parameter(batch_matmul_weight, "w2")

    def construct(self, x, b):
        out = self.mul(x, self.mul_weight)
        out = self.batch_matmul(out, self.batch_matmul_weight)
        return out


_x = Tensor(np.ones([128, 64, 32]), dtype=ms.float32)
_w1 = Tensor(np.ones([128, 64, 32]), dtype=ms.float32)
_w2 = Tensor(np.ones([128, 32, 32]), dtype=ms.float32)
_b = Tensor(np.ones([128, 64, 16]), dtype=ms.float32)


def compile_net(net):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, _x, _b)
    context.reset_auto_parallel_context()


def test_batch_matmul_data_parallel():
    """
    Feature: distribute operator batch_matmul in auto parallel.
    Description: mul-batch_matmul net with data parallel strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((16, 1, 1), (16, 1, 1))
    strategy2 = ((16, 1, 1), (16, 1, 1))
    net = Net(_w1, _w2, False, strategy1, strategy2)
    compile_net(net)


def test_batch_matmul_model_parallel():
    """
    Feature: distribute operator batch_matmul in auto parallel.
    Description: mul-batch_matmul net with model parallel strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((1, 1, 1), (1, 1, 1))
    strategy2 = ((1, 1, 1), (1, 1, 16))
    net = Net(_w1, _w2, False, strategy1, strategy2)
    compile_net(net)


def test_batch_matmul_hybrid_parallel():
    """
    Feature: distribute operator batch_matmul in auto parallel.
    Description: mul-batch_matmul net with mixed strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((2, 2, 2), (2, 2, 2))
    strategy2 = ((2, 2, 2), (2, 2, 2))
    net = Net(_w1, _w2, False, strategy1, strategy2)
    compile_net(net)


def test_batch_matmul_auto_parallel():
    """
    Feature: distribute operator batch_matmul in auto parallel.
    Description: mul-batch_matmul net in auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=16, global_rank=0)
    net = Net(_w1, _w2, False)
    compile_net(net)


def test_batch_matmul_repeat_calc():
    """
    Feature: distribute operator batch_matmul in auto parallel.
    Description: mul-batch_matmul net with repeated strategy in semi auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((2, 2, 4), (2, 2, 4))
    strategy2 = ((1, 2, 2), (1, 2, 2))
    net = Net(_w1, _w2, False, strategy1, strategy2)
    compile_net(net)


def test_batch_matmul_transpose_b():
    """
    Feature: distribute operator batch_matmul in auto parallel.
    Description: mul-batch_matmul net with strategy in semi auto parallel, transpose_b.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((2, 2, 4), (2, 2, 4))
    strategy2 = ((1, 2, 2), (1, 2, 2))
    net = Net(_w1, _w2, True, strategy1, strategy2)
    compile_net(net)
