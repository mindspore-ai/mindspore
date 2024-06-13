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

import numpy as np
import pytest

import mindspore as ms
from mindspore import context, Tensor, Parameter
from mindspore.nn import Cell
from mindspore.ops import operations as P
from parallel.utils.utils import ParallelValidator, compile_net

def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class TopKRouterNet(Cell):
    def __init__(self, strategy):
        super().__init__()
        self.topkrouter = P._inner_ops.TopKRouter().shard(strategy)

    def construct(self, x, capacity, expert_num):
        out = self.topkrouter(x, capacity, expert_num)
        return out


def test_topkrouter():
    """
    Feature: test auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=2, global_rank=0)
    strategy = ((2, 1, 1),)
    net = TopKRouterNet(strategy)
    x = Parameter(Tensor(np.ones([2, 16, 2]), dtype=ms.int32), "x")
    capacity = 3
    expert_num = 4
    net.set_inputs(x, capacity, expert_num)
    phase = compile_net(net, x, capacity, expert_num)
    validator = ParallelValidator(net, phase)
    assert validator.check_parameter_shape("x", [1, 16, 2])


def test_topkrouter_strategy_error():
    """
    Feature: test auto parallel
    Description: test tokrouter shard strategy with (1, 2, 1).
    Expectation: compile error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=2, global_rank=0)
    strategy = ((1, 2, 1),)
    net = TopKRouterNet(strategy)
    x = Parameter(Tensor(np.ones([2, 16, 2]), dtype=ms.int32), "x")
    capacity = 3
    expert_num = 4
    net.set_inputs(x, capacity, expert_num)
    with pytest.raises(RuntimeError):
        compile_net(net, x, capacity, expert_num)
    