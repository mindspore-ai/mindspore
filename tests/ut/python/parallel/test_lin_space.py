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

import pytest

import mindspore.common.dtype as mstype
from mindspore import Tensor, context
from mindspore.nn import Cell
from mindspore.ops import operations as P

from parallel.utils.utils import ParallelValidator, compile_net


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(Cell):
    def __init__(self, strategy=None):
        super(Net, self).__init__()
        self.lin_space = P.LinSpace().shard(strategy)

    def construct(self, start, end, x):
        return self.lin_space(start, end, x)


def test_lin_space_loss_auto_parallel():
    """
    Feature: test LinSpace auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="dynamic_programming", device_num=8,
                                      global_rank=0)
    start = Tensor(1, mstype.float32)
    end = Tensor(10, mstype.float32)
    x = 8
    net = Net()
    compile_net(net, start, end, x)


def test_lin_space_parallel_with_repeated_cal():
    """
    Feature: test LinSpace parallel
    Description: parallel with repeated_cal
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    start = Tensor(1, mstype.float32)
    end = Tensor(10, mstype.float32)
    x = 8
    strategy = ((4,),)
    net = Net(strategy)
    phase = compile_net(net, start, end, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs('LinSpace-0', ['Add-0', 'Add-1', 2])


def test_lin_space_parallel_with_x_2():
    """
    Feature: test LinSpace parallel
    Description: parallel with input x is 2
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    start = Tensor(1, mstype.float32)
    end = Tensor(10, mstype.float32)
    x = 2
    strategy = ((2,),)
    net = Net(strategy)
    phase = compile_net(net, start, end, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs('LinSpace-0', ['Add-0', 'Add-1', 1])


def test_lin_space_data_parallel():
    """
    Feature: test LinSpace data parallel
    Description: data parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    start = Tensor(1, mstype.float32)
    end = Tensor(10, mstype.float32)
    x = 8
    net = Net()
    phase = compile_net(net, start, end, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs('LinSpace-0', ['Add-0', 'Add-1', 1])


def test_lin_space_parallel_strategy_wrong():
    """
    Feature: test LinSpace parallel with invalid strategy
    Description: invalid strategy
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    start = Tensor(1, mstype.float32)
    end = Tensor(10, mstype.float32)
    x = 6
    strategy = ((4,),)
    net = Net(strategy)
    with pytest.raises(RuntimeError):
        compile_net(net, start, end, x)


def test_lin_space_dynamic_shape_constraint():
    """
    Feature: test LinSpace parallel with dynamic shape
    Description: strategy > 1
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    start = Tensor(shape=[None], dtype=mstype.float32)
    end = Tensor(10, mstype.float32)
    x = 6
    strategy = ((4,),)
    net = Net(strategy)
    with pytest.raises(RuntimeError):
        compile_net(net, start, end, x)
