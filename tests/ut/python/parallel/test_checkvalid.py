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
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.train import Model


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(Cell):
    def __init__(self, weight, strategy):
        super().__init__()
        self.check_valid = P.CheckValid().shard(strategy)
        self.mul = P.Mul()
        cast_strategy = None
        if strategy:
            cast_strategy = (strategy[0],)
        self.cast = P.Cast().shard(cast_strategy)
        self.relu = P.ReLU()
        self.weight = Parameter(weight, "w1")

    def construct(self, x, b):
        out = self.mul(x, self.weight)
        out = self.check_valid(out, b)
        out = self.cast(out, ms.float32)
        out = self.relu(out)
        return out


_x = Tensor(np.ones([16, 4]), dtype=ms.float32)
_w = Tensor(np.ones([16, 4]), dtype=ms.float32)
_b = Tensor(np.ones([3]), dtype=ms.float32)


def compile_net(net):
    model = Model(net)
    model.predict(_x, _b)
    context.reset_auto_parallel_context()


def test_check_valid_data_parallel():
    """
    Feature: test check valid data parallel
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, full_batch=True)
    strategy = ((8, 1), (1,))
    net = Net(_w, strategy)
    compile_net(net)


def test_check_valid_repeated_calc():
    """
    Feature: test check valid repeated calculation
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, full_batch=True)
    strategy = ((2, 1), (1,))
    net = Net(_w, strategy)
    compile_net(net)


def test_check_valid_no_shard():
    """
    Feature: test check valid no shard
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, full_batch=True)
    strategy = ((1, 1), (1,))
    net = Net(_w, strategy)
    compile_net(net)


def test_check_valid_strategy_none():
    """
    Feature: test check valid strategy none
    Description: generator batch parallel strategy
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, full_batch=True)
    strategy = None
    net = Net(_w, strategy)
    compile_net(net)


def test_check_valid_auto_parallel():
    """
    Feature: test check valid auto parallel
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="dynamic_programming", device_num=8,
                                      full_batch=True)
    strategy = None
    net = Net(_w, strategy)
    compile_net(net)


def test_check_valid_shard_img():
    """
    Feature: test check valid shard img
    Description:
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, full_batch=True)
    strategy = ((2, 1), (4,))
    net = Net(_w, strategy)
    with pytest.raises(RuntimeError):
        compile_net(net)


def test_check_valid_shard_bbox_second_dimension():
    """
    Feature: test check valid shard bbox second dimension
    Description:
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, full_batch=True)
    strategy = ((2, 2), (1,))
    net = Net(_w, strategy)
    with pytest.raises(RuntimeError):
        compile_net(net)
