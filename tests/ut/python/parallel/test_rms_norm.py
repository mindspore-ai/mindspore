# Copyright 2024 Huawei Technologies Co., Ltd
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
from mindspore.common.initializer import initializer
from mindspore.nn import Cell, TrainOneStepCell, Momentum
from mindspore.ops import operations as P


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(Cell):
    def __init__(self, mul_weight, strategy1=None, strategy2=None, strategy3=None):
        super().__init__()
        self.mul = P.Mul().shard(strategy1)
        self.rms_norm = P.RmsNorm().shard(strategy2)
        self.mul2 = P.Mul().shard(strategy3)
        self.mul_weight = Parameter(mul_weight, "w1")
        self.normalized_shape = mul_weight.shape[1:]
        self.gamma = Parameter(initializer('ones', self.normalized_shape), name="gamma")

    def construct(self, x, b):
        out = self.mul(x, self.mul_weight)
        out, _ = self.rms_norm(out, self.gamma)
        out = self.mul2(out, b)
        return out


_x = Tensor(np.ones([16, 64, 32]), dtype=ms.float32)
_w = Tensor(np.ones([16, 64, 32]), dtype=ms.float32)
_b = Tensor(np.ones([16, 64, 32]), dtype=ms.float32)


def compile_net(net, x, b):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, x, b)
    context.reset_auto_parallel_context()


def test_rms_norm_data_parallel():
    """
    Feature: test RmsNorm data parallel
    Description: test RmsNorm data parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((16, 1, 1), (16, 1, 1))
    strategy2 = ((16, 1, 1), (1, 1))
    strategy3 = ((16, 1, 1), (16, 1, 1))
    net = Net(_w, strategy1, strategy2, strategy3)
    compile_net(net, _x, _b)


def test_rms_norm_model_parallel():
    """
    Feature: test RmsNorm model parallel
    Description: test RmsNorm model parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((1, 16, 1), (1, 16, 1))
    strategy2 = ((16, 1, 1), (1, 1))
    strategy3 = ((1, 16, 1), (1, 16, 1))
    net = Net(_w, strategy1, strategy2, strategy3)
    compile_net(net, _x, _b)


def test_rms_norm_hybrid_parallel():
    """
    Feature: test RmsNorm hybrid parallel
    Description: test RmsNorm hybrid parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((2, 8, 1), (2, 8, 1))
    strategy2 = ((2, 1, 1), (1, 1))
    strategy3 = ((2, 8, 1), (2, 8, 1))
    net = Net(_w, strategy1, strategy2, strategy3)
    compile_net(net, _x, _b)


def test_rms_norm_auto_parallel():
    """
    Feature: test auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="dynamic_programming", device_num=16,
                                      global_rank=0)
    net = Net(_w)
    compile_net(net, _x, _b)


def test_rms_norm_repeat_calc():
    """
    Feature: test RmsNorm repeat calc
    Description: test RmsNorm repeat calc
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((2, 2, 4), (2, 2, 4))
    strategy2 = ((2, 1, 1), (1, 1))
    strategy3 = ((2, 2, 4), (2, 2, 4))
    net = Net(_w, strategy1, strategy2, strategy3)
    compile_net(net, _x, _b)


def test_rms_norm_wrong_strategy():
    """
    Feature: test RmsNorm different input length strategy
    Description: test RmsNorm different input length strategy
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((2, 2, 4, 1), (2, 2, 4, 1))
    strategy2 = ((1, 2, 1, 2), (2, 1, 2))
    strategy3 = ((2, 2, 4, 1), (2, 2, 4, 1))
    net = Net(_w, strategy1, strategy2, strategy3)
    with pytest.raises(RuntimeError):
        compile_net(net, _x, _b)


_x_2d = Tensor(np.ones([16, 64]), dtype=ms.float32)
_w_2d = Tensor(np.ones([16, 64]), dtype=ms.float32)
_b_2d = Tensor(np.ones([16, 64]), dtype=ms.float32)


def test_rms_norm_data_parallel_2d():
    """
    Feature: test RmsNorm data parallel 2d input
    Description: test RmsNorm data parallel 2d input
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((16, 1), (16, 1))
    strategy2 = ((1, 1), (1,))
    strategy3 = ((16, 1), (16, 1))
    net = Net(_w_2d, strategy1, strategy2, strategy3)
    compile_net(net, _x_2d, _b_2d)


def test_rms_norm_auto_parallel_2d():
    """
    Feature: test auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="dynamic_programming", device_num=16,
                                      global_rank=0)
    net = Net(_w_2d)
    compile_net(net, _x_2d, _b_2d)


def test_rms_norm_wrong_strategy_2d():
    """
    Feature: test RmsNorm different input length strategy with invalid axis
    Description: test RmsNorm different input length strategy with invalid axis
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((2, 8), (2, 8))
    strategy2 = ((2, 8), (8,))
    strategy3 = ((2, 8), (2, 8))
    net = Net(_w_2d, strategy1, strategy2, strategy3)
    with pytest.raises(RuntimeError):
        compile_net(net, _x_2d, _b_2d)
