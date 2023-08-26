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
import mindspore.context as context
from mindspore import Tensor, Parameter
import mindspore.nn as nn
from mindspore.ops import operations as P

from parallel.utils.utils import compile_net


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


input_x_tensor_ = Tensor(np.ones([8, 8, 8]), ms.float32)
input_x_parameter_ = Parameter(Tensor(np.ones([8, 8, 8]), ms.float32), "input_x")
SIZE_SPLIT = [3, 3, 2]
NUM_SPLIT = 3


class Net(nn.Cell):
    def __init__(self, size_split, split_dim, num_split, strategy1=None, strategy2=None):
        super(Net, self).__init__()
        self.splitv = P.SplitV(size_split, split_dim, num_split).shard(strategy1)
        self.mul = P.Mul().shard(strategy2)
        self.weight = Parameter(np.array([1.0]), name="mul_weight")

    def construct(self, x):
        out = self.splitv(x)
        out = self.mul(out[0], self.weight)
        return out


class NetWithParameter(nn.Cell):
    def __init__(self, size_split, split_dim, num_split, strategy1=None, strategy2=None):
        super(NetWithParameter, self).__init__()
        self.splitv = P.SplitV(size_split, split_dim, num_split).shard(strategy1)
        self.mul = P.Mul().shard(strategy2)
        self.weight = input_x_parameter_

    def construct(self, x):
        out = self.splitv(self.weight)
        out = self.mul(x, out[0])
        return out


def test_splitv_auto_parallel():
    """
    Feature: test SplitV auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="dynamic_programming", device_num=8,
                                      global_rank=0)
    net = Net(SIZE_SPLIT, 0, NUM_SPLIT)
    compile_net(net, input_x_tensor_)


def test_splitv_auto_parallel_with_parameter():
    """
    Feature: test SplitV auto parallel with parameter input
    Description: auto parallel with parameter input
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="dynamic_programming", device_num=8,
                                      global_rank=0)
    net = NetWithParameter(SIZE_SPLIT, 2, NUM_SPLIT)
    x = Tensor(np.ones([8, 8, 3]), ms.float32)
    compile_net(net, x)


def test_splitv_data_parallel():
    """
    Feature: test SplitV data parallel
    Description: data parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    net = Net(SIZE_SPLIT, 1, NUM_SPLIT)
    compile_net(net, input_x_tensor_)


def test_splitv_model_parallel():
    """
    Feature: test SplitV model parallel
    Description: model parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 2, 2),)
    strategy2 = ((1, 2, 2), (1,))
    net = Net(SIZE_SPLIT, 0, NUM_SPLIT, strategy1, strategy2)
    compile_net(net, input_x_tensor_)


def test_splitv_strategy_error():
    """
    Feature: test SplitV parallel with invalid strategy
    Description: config invalid strategy
    Expectation: raise RuntimeError
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 2),)
    strategy2 = ((8, 1, 1),)
    net = Net(SIZE_SPLIT, 0, NUM_SPLIT, strategy1, strategy2)
    with pytest.raises(RuntimeError):
        compile_net(net, input_x_tensor_)
    context.reset_auto_parallel_context()
