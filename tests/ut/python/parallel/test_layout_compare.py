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
import pytest

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore import context
from mindspore.common.api import _cell_graph_executor
from mindspore.ops import operations as P


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


def compile_net(net, input_x):
    net.set_auto_parallel()
    net.set_train()
    _cell_graph_executor.compile(net, input_x)


class Net(nn.Cell):
    def __init__(self, weight, strategy1, strategy2):
        super().__init__()
        self.matmul1 = P.MatMul().shard(strategy1)
        self.relu = P.ReLU().shard(strategy2)
        self.w = Parameter(weight, "w1")

    def construct(self, y):
        out1 = self.matmul1(y, self.w)
        out2 = self.relu(self.w)
        out = out1 + out2
        return out


x = Tensor(np.ones([8, 8]), dtype=ms.float32)
w = Tensor(np.ones([8, 8]), dtype=ms.float32)


def test_layout_equal_01():
    """
    Feature: test layout equal
    Description: dev_num is 8, strategy is (2, 4)
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 2), (2, 4))
    strategy2 = ((2, 4),)
    net = Net(w, strategy1, strategy2)
    compile_net(net, x)


def test_layout_equal_02():
    """
    Feature: test layout equal
    Description: dev_num is 8, strategy is (1, 1), the dev_matrix is different
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 1), (1, 1))
    strategy2 = ((1, 1),)
    net = Net(w, strategy1, strategy2)
    compile_net(net, x)


def test_layout_equal_03():
    """
    Feature: test layout equal
    Description: dev_num is 8, strategy is (2, 1), the dev_matrix is different
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 2), (2, 1))
    strategy2 = ((2, 1),)
    net = Net(w, strategy1, strategy2)
    compile_net(net, x)


def test_layout_no_equal_01():
    """
    Feature: test layout equal
    Description: dev_num is 8, strategies are different
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 2), (2, 4))
    strategy2 = ((4, 2),)
    net = Net(w, strategy1, strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net, x)


def test_layout_no_equal_02():
    """
    Feature: test layout equal
    Description: dev_num is 8, strategy is (2, 2), the tensor map is different
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2), (2, 2))
    strategy2 = ((2, 2),)
    net = Net(w, strategy1, strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net, x)
