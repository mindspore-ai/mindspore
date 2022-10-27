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

import pytest
import numpy as np
import mindspore as ms
from mindspore import context, Tensor, Parameter
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import Cell, TrainOneStepCell, Momentum
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
        self.mul_weight = Parameter(Tensor(np.ones([64 * 3, 32 * 3]), dtype=ms.float32), "w1")
        self.matmul_weight = Parameter(Tensor(np.ones([32 * 3, 32 * 3]), dtype=ms.float32), "w2")
        self.embedding_table = Parameter(Tensor(np.ones([64 * 3, 32 * 3]), dtype=ms.float32), "embedding_table")

    def construct(self, x, b):
        out = self.gather(self.embedding_table, x, 0)
        out = self.matmul(out, self.matmul_weight)
        out = self.mul(out, self.mul_weight)
        out = out + b
        return self.reduce_sum(out)


class Net1(Cell):
    def __init__(self):
        super().__init__()
        self.mul = P.Mul()
        self.matmul1 = P.MatMul(transpose_a=False, transpose_b=False)
        self.matmul2 = P.MatMul(transpose_a=False, transpose_b=True)
        self.matmul3 = P.MatMul(transpose_a=False, transpose_b=True)
        self.matmul4 = P.MatMul(transpose_a=False, transpose_b=False)
        self.reduce_sum = P.ReduceSum()
        self.matmul_weight1 = Parameter(Tensor(np.ones([32 * 3, 32 * 3]), dtype=ms.float32), "mat_w1")
        self.matmul_weight2 = Parameter(Tensor(np.ones([32 * 3, 32 * 3]), dtype=ms.float32), "mat_w2")
        self.matmul_weight3 = Parameter(Tensor(np.ones([32 * 3, 32 * 3]), dtype=ms.float32), "mat_w3")
        self.matmul_weight4 = Parameter(Tensor(np.ones([32 * 3, 32 * 3]), dtype=ms.float32), "mat_w4")

    def construct(self, x, b):
        out = self.matmul1(x, self.matmul_weight1)
        out = self.matmul2(out, self.matmul_weight2)
        out = self.matmul3(out, self.matmul_weight3)
        out = self.matmul4(out, self.matmul_weight4)
        out = self.mul(out, b)
        return self.reduce_sum(out)

_x = Tensor(np.ones([3 * 64]), dtype=ms.int32)
_b = Tensor(np.ones([64 * 3, 32 * 3]), dtype=ms.float32)

_x1 = Tensor(np.ones([3 * 32, 3 * 32]), dtype=ms.float32)
_b1 = Tensor(np.ones([32 * 3]), dtype=ms.float32)

def compile_net(net, change_input=False):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    if change_input:
        _cell_graph_executor.compile(train_net, _x1, _b1)
    else:
        _cell_graph_executor.compile(train_net, _x, _b)
    context.reset_auto_parallel_context()


def test_auto_parallel_device_num_24():
    """
    Feature: device num 24 in auto parallel.
    Description: verify device_num 24 in auto parallel  by mul/matmul/gather
    Expectation: compile done without error.
    """
    context.set_context(device_target="Ascend")
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=24, global_rank=0)
    mul_strategy1 = ((3 * 8, 1), (3 * 8, 1))
    matmul_strategy2 = ((24, 1), (1, 1))
    gather_strategy3 = ((1, 1), (24,))
    net = Net(mul_strategy1, matmul_strategy2, gather_strategy3)
    compile_net(net)

def test_auto_parallel_device_num_24_1():
    """
    Feature: device num 24 in auto parallel.
    Description: verify device_num 24 in auto parallel  by mul/matmul/gather
    Expectation: compile done without error.
    """
    context.set_context(device_target="Ascend")
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=24, global_rank=0)
    mul_strategy1 = ((3, 8), (3, 8))
    matmul_strategy2 = ((3, 8), (8, 1))
    gather_strategy3 = ((24, 1), (1,))
    net = Net(mul_strategy1, matmul_strategy2, gather_strategy3)
    compile_net(net)

def test_auto_parallel_device_num_24_2():
    """
    Feature: device num 24 in auto parallel.
    Description: verify device_num 24 in auto parallel  by mul/matmul/gather
    Expectation: compile done without error.
    """
    context.set_context(device_target="Ascend")
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=24, global_rank=0)
    mul_strategy1 = ((8, 3), (8, 3))
    matmul_strategy2 = ((8, 3), (3, 1))
    gather_strategy3 = ((3, 1), (1,))
    net = Net(mul_strategy1, matmul_strategy2, gather_strategy3)
    with pytest.raises(RuntimeError):
        compile_net(net)

def test_auto_parallel_device_num_24_3():
    """
    Feature: device num 24 in auto parallel.
    Description: verify device_num 24 in auto parallel  by mul/matmul/gather
    Expectation: compile done without error.
    """
    context.set_context(device_target="Ascend")
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=24, global_rank=0)
    mul_strategy1 = ((8, 3), (8, 3))
    matmul_strategy2 = ((8, 1), (1, 3))
    gather_strategy3 = ((24, 1), (1,))
    net = Net(mul_strategy1, matmul_strategy2, gather_strategy3)
    with pytest.raises(RuntimeError):
        compile_net(net)

def test_auto_parallel_device_num_24_dyn_search():
    """
    Feature: device num 24 in auto parallel.
    Description: verify device_num 24 in auto parallel  by mul/matmul/gather
    Expectation: compile done without error.
    """
    context.set_context(device_target="Ascend")
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=24, global_rank=0,
                                      search_mode="dynamic_programming")
    mul_strategy1 = None
    matmul_strategy2 = None
    gather_strategy3 = None
    net = Net(mul_strategy1, matmul_strategy2, gather_strategy3)
    compile_net(net)

def test_auto_parallel_device_num_24_prop_search():
    """
    Feature: device num 24 in auto parallel.
    Description: verify device_num 24 in auto parallel  by mul/matmul/gather
    Expectation: compile done without error.
    """
    context.set_context(device_target="Ascend")
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=24, global_rank=0,
                                      search_mode="sharding_propagation")
    mul_strategy1 = None
    matmul_strategy2 = ((3, 8), (8, 1))
    gather_strategy3 = None
    net = Net(mul_strategy1, matmul_strategy2, gather_strategy3)
    compile_net(net)

def test_auto_parallel_device_num_24_dyn_search_1():
    """
    Feature: device num 24 in auto parallel.
    Description: verify device_num 24 in auto parallel, by 4 matmul
    Expectation: compile done without error.
    """
    context.set_context(device_target="Ascend")
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=24, global_rank=0,
                                      search_mode="dynamic_programming")
    net = Net1()
    compile_net(net, True)
