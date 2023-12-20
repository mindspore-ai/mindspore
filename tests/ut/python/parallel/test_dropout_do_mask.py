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
        self.dropout_do_mask = P.DropoutDoMask().shard(strategy2)
        self.dropout_gen_mask = P.DropoutGenMask()
        self.get_shape = P.Shape()
        self.cast = P.Cast()
        self.mul_weight = Parameter(mul_weight, "w1")
        self.keep_prob = Tensor(0.9)

    def construct(self, x, b):
        out = self.mul(x, self.mul_weight)
        shape = self.get_shape(out)
        dtype = P.DType()(out)
        keep_prob = self.cast(self.keep_prob, dtype)
        mask = self.dropout_gen_mask(shape, keep_prob)
        out = self.dropout_do_mask(out, mask, keep_prob)
        return out


_x = Tensor(np.ones([128, 64]), dtype=ms.float32)
_w1 = Tensor(np.ones([64]), dtype=ms.float32)
_b = Tensor(np.ones([128, 64]), dtype=ms.float32)


def compile_net(net, x=_x, b=_b):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, x, b)
    context.reset_auto_parallel_context()


def test_dropout_do_mask_data_parallel():
    """
    Feature: test data parallel
    Description: data parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((16, 1), (1,))
    strategy2 = ((16, 1),)
    net = Net(_w1, strategy1, strategy2)
    compile_net(net)


def test_dropout_do_mask_model_parallel():
    """
    Feature: test model parallel
    Description: model parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((1, 16), (16,))
    strategy2 = ((1, 16),)
    net = Net(_w1, strategy1, strategy2)
    compile_net(net)


def test_dropout_do_mask_hybrid_parallel():
    """
    Feature: test hybrid parallel
    Description: hybrid parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((4, 4), (4,))
    strategy2 = ((4, 4),)
    net = Net(_w1, strategy1, strategy2)
    compile_net(net)


def test_dropout_do_mask_auto_parallel():
    """
    Feature: test auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="dynamic_programming", device_num=16,
                                      global_rank=0)
    net = Net(_w1)
    compile_net(net)


def test_dropout_do_mask_repeat_calc():
    """
    Feature: test repeat calc
    Description: repeat calc
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((4, 4), (4,))
    strategy2 = ((2, 4),)
    net = Net(_w1, strategy1, strategy2)
    compile_net(net)


def test_dropout_do_mask_dynamic_shape_constraint():
    """
    Feature: test dynamic shape
    Description:
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((4, 4), (4,))
    strategy2 = ((2, 4),)
    net = Net(_w1, strategy1, strategy2)
    dynamic_x = Tensor(shape=[None, 64], dtype=ms.float32)
    with pytest.raises(RuntimeError):
        compile_net(net, x=dynamic_x)
