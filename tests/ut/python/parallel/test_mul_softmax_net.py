
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
# ============================================================================
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
    def __init__(self, mul_weight, strategy1=None, strategy2=None, strategy3=None):
        super().__init__()
        self.begin_norm_axis = 2
        self.begin_params_axis = 1
        self.mul = P.Mul().shard(strategy1)
        self.softmax = P.Softmax().shard(strategy2)
        self.mul2 = P.Mul().shard(strategy3)
        self.mul_weight = Parameter(mul_weight, "w1")
        self.normalized_shape = [64, 32, 16]

    def construct(self, x, b):
        out = self.mul(x, self.mul_weight)
        out = self.softmax(out)
        out = self.mul2(out, b)
        return out


_x = Tensor(np.ones([16, 64, 32, 16]), dtype=ms.float32)
_w = Tensor(np.ones([16, 64, 32, 16]), dtype=ms.float32)
_b = Tensor(np.ones([16, 64, 32, 16]), dtype=ms.float32)


def compile_net(net):
    optimizer = Momentum(net.trainable_params(),
                         learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, _x, _b)
    context.reset_auto_parallel_context()


def test_softmax_data_parallel():
    """
    Feature: distribute operator softmax in auto parallel.
    Description: data parallel softmax net in auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((16, 1, 1, 1), (16, 1, 1, 1))
    strategy2 = ((16, 1, 1, 1),)
    strategy3 = ((16, 1, 1, 1), (16, 1, 1, 1))
    net = Net(_w, strategy1, strategy2, strategy3)
    compile_net(net)


def test_softmax_data_parallel_with_half_repeat():
    """
    Feature: distribute operator softmax in auto parallel.
    Description: data parallel and half repeat softmax net in auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((16, 1, 1, 1), (16, 1, 1, 1))
    strategy2 = ((8, 1, 1, 1),)
    strategy3 = ((16, 1, 1, 1), (16, 1, 1, 1))
    net = Net(_w, strategy1, strategy2, strategy3)
    compile_net(net)


def test_softmax_model_parallel():
    """
    Feature: distribute operator softmax in auto parallel.
    Description: model parallel softmax net in auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((1, 16, 1, 1), (1, 16, 1, 1))
    strategy2 = ((16, 1, 1, 1),)
    strategy3 = ((1, 16, 1, 1), (1, 16, 1, 1))
    net = Net(_w, strategy1, strategy2, strategy3)
    compile_net(net)


def test_softmax_model_parallel_with_repeat():
    """
    Feature: distribute operator softmax in auto parallel.
    Description: model parallel with repeate softmax net in auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((1, 16, 1, 1), (1, 16, 1, 1))
    strategy2 = ((1, 1, 1, 1),)
    strategy3 = ((1, 16, 1, 1), (1, 16, 1, 1))
    net = Net(_w, strategy1, strategy2, strategy3)
    compile_net(net)


def test_softmax_model_parallel_with_half_repeat():
    """
    Feature: distribute operator softmax in auto parallel.
    Description: model parallel with half repeate softmax net in auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((1, 16, 1, 1), (1, 16, 1, 1))
    strategy2 = ((8, 1, 1, 1),)
    strategy3 = ((1, 16, 1, 1), (1, 16, 1, 1))
    net = Net(_w, strategy1, strategy2, strategy3)
    compile_net(net)


def test_softmax_hybrid_parallel():
    """
    Feature: distribute operator softmax in auto parallel.
    Description: hybrid parallel softmax net in auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((2, 8, 1, 1), (2, 8, 1, 1))
    strategy2 = ((16, 1, 1, 1),)
    strategy3 = ((2, 8, 1, 1), (2, 8, 1, 1))
    net = Net(_w, strategy1, strategy2, strategy3)
    compile_net(net)


def test_softmax_hybrid_with_repeat_parallel():
    """
    Feature: distribute operator softmax in auto parallel.
    Description: hybrid parallel with repat softmax net in auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((2, 8, 1, 1), (2, 8, 1, 1))
    strategy2 = ((1, 1, 1, 1),)
    strategy3 = ((2, 8, 1, 1), (2, 8, 1, 1))
    net = Net(_w, strategy1, strategy2, strategy3)
    compile_net(net)


def test_softmax_hybrid_with_half_repeat_parallel():
    """
    Feature: distribute operator softmax in auto parallel.
    Description: hybrid parallel with half repeate softmax net in auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((2, 8, 1, 1), (2, 8, 1, 1))
    strategy2 = ((8, 1, 1, 1),)
    strategy3 = ((2, 8, 1, 1), (2, 8, 1, 1))
    net = Net(_w, strategy1, strategy2, strategy3)
    compile_net(net)


def test_softmax_hybrid_parallel_with_quater_repeate():
    """
    Feature: distribute operator softmax in auto parallel.
    Description: hybrid parallel softmax net in auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((2, 8, 1, 1), (2, 8, 1, 1))
    strategy2 = ((4, 1, 1, 1),)
    strategy3 = ((2, 8, 1, 1), (2, 8, 1, 1))
    net = Net(_w, strategy1, strategy2, strategy3)
    compile_net(net)


def test_softmax_auto_parallel():
    """
    Feature: distribute operator softmax in auto parallel.
    Description: softmax net in auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(
        parallel_mode="auto_parallel", device_num=16, global_rank=0)
    net = Net(_w)
    compile_net(net)


def test_softmax_repeat_calc():
    """
    Feature: distribute operator softmax in auto parallel.
    Description: repeated calculation softmax net in auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((2, 2, 4, 1), (2, 2, 4, 1))
    strategy2 = ((1, 1, 1, 1),)
    strategy3 = ((2, 2, 4, 1), (2, 2, 4, 1))
    net = Net(_w, strategy1, strategy2, strategy3)
    compile_net(net)


def test_softmax_wrong_strategy():
    """
    Feature: distribute operator softmax in auto parallel.
    Description: wrong strategy net in auto parallel.
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((2, 2, 4, 1), (2, 2, 4, 1))
    strategy2 = ((1, 2, 1, 2),)
    strategy3 = ((2, 2, 4, 1), (2, 2, 4, 1))
    net = Net(_w, strategy1, strategy2, strategy3)
    with pytest.raises(RuntimeError):
        compile_net(net)
