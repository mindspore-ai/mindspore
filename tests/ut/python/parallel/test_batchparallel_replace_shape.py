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
import pytest

import mindspore as ms
from mindspore import context, Tensor, Parameter
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import Cell, TrainOneStepCell, Momentum
from mindspore.ops import operations as P


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(Cell):
    def __init__(self, mul_weight, strategy1=None, strategy2=None):
        super().__init__()
        self.mul = P.Mul().shard(strategy1)
        self.neg = P.Neg().shard(strategy2)
        self.mul_weight = Parameter(mul_weight, "w1")
        self.uniform_real = P.UniformReal()
        self.shape = P.Shape()

    def construct(self, x, b):
        out = self.mul(x, self.mul_weight)
        out = self.neg(out)
        shape = self.shape(out)[0]
        z = self.uniform_real((shape, 64, 32))
        out = out + z
        return out


_x = Tensor(np.ones([128, 64, 32]), dtype=ms.float32)
_w1 = Tensor(np.ones([64, 32]), dtype=ms.float32)
_b = Tensor(np.ones([128, 64, 32]), dtype=ms.float32)


def compile_net(net, x=_x, b=_b):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, x, b)
    context.reset_auto_parallel_context()


def test_batch_parallel_replace_shape():
    """
    Feature: test dynamic shape
    Description:
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((16, 1, 1), (1, 1))
    strategy2 = ((16, 1, 1),)
    net = Net(_w1, strategy1, strategy2)
    compile_net(net)


def test_batch_parallel_dynamic_shape_constraint():
    """
    Feature: test dynamic shape
    Description:
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0,
                                      full_batch=False)
    strategy1 = ((16, 1, 1), (1, 1))
    strategy2 = ((16, 1, 1),)
    net = Net(_w1, strategy1, strategy2)
    dynamic_x = Tensor(shape=[None, 64, 32], dtype=ms.float32)
    dynamic_b = Tensor(shape=[None, 64, 32], dtype=ms.float32)
    with pytest.raises(RuntimeError):
        compile_net(net, dynamic_x, dynamic_b)
