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


class Net(Cell):
    def __init__(self, seed=0, seed2=0):
        super().__init__()
        mul_np = np.full((1, 1), 0.1, dtype=np.float32)
        self.mul_weight = Parameter(Tensor(mul_np), name="mul_weight")
        self.seed = seed
        self.seed2 = seed2
        self.cast = P.Cast()
        self.mul = P.Mul()
        self.random_choice_with_mask = P.RandomChoiceWithMask(count=256, seed=self.seed,
                                                              seed2=self.seed2)

    def construct(self, input_a, label):
        x, _ = self.random_choice_with_mask(input_a)
        x = self.cast(x, ms.float32)
        x = self.mul(x, self.mul_weight)
        return x


inputs_np = Tensor(np.ones([32, 4, 12]).astype(np.bool_))
label_ = Tensor(np.random.randn(1, 1).astype(np.float32))


def compile_net(net):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, inputs_np, label_)
    context.reset_auto_parallel_context()


def test_auto_parallel_unknownop_rec():
    """
    Feature: test unknownop net of auto parallel
    Description: using recursive algorithm
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                      search_mode="recursive_programming")
    net = Net(1, 1)
    compile_net(net)
