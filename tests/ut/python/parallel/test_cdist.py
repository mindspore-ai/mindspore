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

from mindspore import Tensor, context
from mindspore.nn import Cell
import mindspore.ops as ops

from parallel.utils.utils import compile_net


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")

B = 8
P = 8
R = 8
M = 2

input_x_2d_ = Tensor(np.random.normal(size=[P, M]).astype(np.float32))
input_y_2d_ = Tensor(np.random.normal(size=[R, M]).astype(np.float32))
input_x_3d_ = Tensor(np.random.normal(size=[B, P, M]).astype(np.float32))
input_y_3d_ = Tensor(np.random.normal(size=[B, R, M]).astype(np.float32))


class Net(Cell):
    def __init__(self, strategy=None):
        super(Net, self).__init__()
        self.cdist = ops.Cdist().shard(strategy)

    def construct(self, *inputs):
        output = self.cdist(*inputs)
        return output


def test_cdist_2d_auto_parallel():
    """
    Feature: test Cdist-2d in parallel
    Description: auto parallel with 2d inputs
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0)
    net = Net()
    compile_net(net, input_x_2d_, input_y_2d_)


def test_cdist_2d_data_parallel():
    """
    Feature: test Cdist-2d in parallel
    Description: data parallel with 2d inputs
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((4, 1), (2, 1))
    net = Net(strategy)
    compile_net(net, input_x_2d_, input_y_2d_)


def test_cdist_2d_data_parallel_with_repeated_cal():
    """
    Feature: test Cdist-2d in parallel with repeated calculation
    Description: data parallel with 2d inputs
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((2, 1), (2, 1))
    net = Net(strategy)
    compile_net(net, input_x_2d_, input_y_2d_)


def test_cdist_2d_strategy_error():
    """
    Feature: test invalid strategy for Cdist 2d
    Description: illegal strategy with 2d inputs
    Expectation: raise RuntimeError
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((2, 2), (2, 1))
    net = Net(strategy)
    with pytest.raises(RuntimeError):
        compile_net(net, input_x_2d_, input_y_2d_)


def test_cdist_3d_auto_parallel():
    """
    Feature: test Cdist-3d in parallel
    Description: auto parallel with 3d inputs
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0)
    net = Net()
    compile_net(net, input_x_3d_, input_y_3d_)


def test_cdist_3d_data_parallel():
    """
    Feature: test Cdist-3d in parallel
    Description: data parallel with 3d inputs
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((2, 2, 1), (2, 2, 1))
    net = Net(strategy)
    compile_net(net, input_x_3d_, input_y_3d_)


def test_cdist_3d_data_parallel_with_repeated_cal():
    """
    Feature: test Cdist-3d in parallel with repeated calculation
    Description: data parallel with 3d inputs
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((2, 2, 1), (2, 1, 1))
    net = Net(strategy)
    compile_net(net, input_x_3d_, input_y_3d_)


def test_cdist_3d_strategy_error():
    """
    Feature: test invalid strategy for Cdist 3d
    Description: illegal strategy with 3d inputs
    Expectation: raise RuntimeError
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((2, 2, 1), (1, 2, 1))
    net = Net(strategy)
    with pytest.raises(RuntimeError):
        compile_net(net, input_x_3d_, input_y_3d_)
