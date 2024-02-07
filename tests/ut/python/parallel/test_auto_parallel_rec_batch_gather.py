# Copyright 2023 Huawei Technologies Co., Ltd
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
from mindspore import context, Tensor
from mindspore.common.api import _cell_graph_executor
from mindspore.common.parameter import Parameter
from mindspore.nn import Cell, TrainOneStepCell, Momentum
from mindspore.ops import operations as P


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class ParallelMulGatherMulNet(Cell):
    def __init__(self, param_size, mul_size1, mul_size2, batch_dim=0, axis=-1, strategy=None):
        super(ParallelMulGatherMulNet, self).__init__()
        param_np = np.full(param_size, 0.5, dtype=np.float32)
        mul_np1 = np.full(mul_size1, 1, dtype=np.int32)
        mul_np2 = np.full(mul_size2, 0.2, dtype=np.float32)
        self.param_weight = Parameter(Tensor(param_np), name="param_weight")
        self.mul_weight1 = Tensor(mul_np1)
        self.mul_weight2 = Tensor(mul_np2)
        self.axis = axis
        self.mul1 = P.Mul()
        self.mul2 = P.Mul()
        self.gather = P.Gather(batch_dim)

    def construct(self, inputs, label):
        x = self.mul1(inputs, self.mul_weight1)
        x = self.gather(self.param_weight, x, self.axis)
        x = self.mul2(x, self.mul_weight2)
        return x


class BatchGatherNet(Cell):
    def __init__(self, param_size, mul_size1, mul_size2, batch_dim=0, axis=-1):
        super(BatchGatherNet, self).__init__()
        param_np = np.full(param_size, 0.5, dtype=np.float32)
        mul_np1 = np.full(mul_size1, 1, dtype=np.int32)
        mul_np2 = np.full(mul_size2, 0.2, dtype=np.float32)
        self.param_weight = Parameter(Tensor(param_np), name="param_weight")
        self.mul_weight1 = Tensor(mul_np1)
        self.mul_weight2 = Tensor(mul_np2)
        self.axis = axis
        self.mul1 = P.Mul()
        self.mul2 = P.Mul()
        self.gather = P.Gather(batch_dim)

    def construct(self, inputs, label):
        x = self.mul1(inputs, self.mul_weight1)
        x = self.gather(self.param_weight, x, self.axis)
        x = self.mul2(x, self.mul_weight2)


inputs_np = Tensor(np.ones([64, 32, 32]), dtype=ms.int32)
label_ = Tensor(np.random.randn(1, 1), dtype=ms.int32)


def compile_net(net):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, inputs_np, label_)
    context.reset_auto_parallel_context()


def test_auto_parallel_rec_batch_gather():
    """
    Feature: test batch_gather of auto parallel
    Description: using recursive algorithm
    Expectation: compile success
    """
    context.set_auto_parallel_context(
        parallel_mode="auto_parallel",
        device_num=8,
        global_rank=0,
        search_mode="recursive_programming",
    )
    net = BatchGatherNet(
        param_size=(64, 32, 128),
        mul_size1=(64, 32, 32),
        mul_size2=(64, 64, 32, 32),
        axis=2,
        batch_dim=2,
    )
    compile_net(net)


def test_auto_parallel_mul_gather_mul():
    """
    Feature: test mul_gather_mul of auto parallel
    Description: using recursive algorithm
    Expectation: compile success
    """
    context.set_auto_parallel_context(
        parallel_mode="auto_parallel",
        device_num=8,
        global_rank=0,
        search_mode="recursive_programming",
    )
    net = ParallelMulGatherMulNet(
        param_size=(64, 32, 128),
        mul_size1=(64, 32, 32),
        mul_size2=(64, 64, 32, 32),
        axis=2,
        batch_dim=2,
    )
    compile_net(net)
