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
from mindspore import context
from mindspore.nn import Cell
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops.auto_generate.gen_ops_prim import LinSpaceExt

from parallel.utils.utils import ParallelValidator, compile_net



def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(Cell):
    def __init__(self, strategy=None):
        super(Net, self).__init__()
        if strategy is None:
            self.lin_space = _get_cache_prim(LinSpaceExt)()
        else:
            self.lin_space = _get_cache_prim(LinSpaceExt)().shard(strategy)

    def construct(self, start, end, x, dtype=None):
        return self.lin_space(start, end, x, dtype)


def test_lin_space_ext_data_parallel():
    """
    Feature: test LinSpaceExt data parallel
    Description: data parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    start = 1
    end = 10
    x = 8
    net = Net()
    phase = compile_net(net, start, end, x, mstype.float32)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs('LinSpaceExt-0', ['ScalarAdd-0', 'ScalarAdd-1', 1, 43])


def test_lin_space_ext_parallel_all_constant_input():
    """
    Feature: test LinSpaceExt parallel
    Description: parallel with repeated_cal
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    start = 1
    end = 10
    x = 8
    strategy = ((8,),)
    net = Net(strategy)
    phase = compile_net(net, start, end, x, mstype.float32)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs('LinSpaceExt-0', ['ScalarAdd-0', 'ScalarAdd-1', 1, 43])


def test_lin_space_ext_parallel_with_repeated_cal():
    """
    Feature: test LinSpaceExt parallel
    Description: parallel with repeated_cal
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    start = 1
    end = 10
    x = 8
    strategy = ((4,),)
    net = Net(strategy)
    phase = compile_net(net, start, end, x, mstype.float32)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs('LinSpaceExt-0', ['ScalarAdd-0', 'ScalarAdd-1', 2, 43])


def test_lin_space_ext_parallel_with_x_2():
    """
    Feature: test LinSpaceExt parallel
    Description: parallel with input x is 2
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    start = 1
    end = 10
    x = 2
    strategy = ((2,),)
    net = Net(strategy)
    phase = compile_net(net, start, end, x, mstype.float32)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs('LinSpaceExt-0', ['ScalarAdd-0', 'ScalarAdd-1', 1, 43])


def test_lin_space_ext_parallel_with_x_2_no_dtype_input():
    """
    Feature: test LinSpaceExt parallel
    Description: parallel with input x is 2
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    start = 1
    end = 10
    x = 2
    strategy = ((2,),)
    net = Net(strategy)
    phase = compile_net(net, start, end, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs('LinSpaceExt-0', ['ScalarAdd-0', 'ScalarAdd-1', 1, 'None'])


def test_lin_space_ext_parallel_strategy_wrong():
    """
    Feature: test LinSpaceExt parallel with invalid strategy
    Description: invalid strategy
    Expectation: compile failed
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    start = 1
    end = 10
    x = 6
    strategy = ((4,),)
    net = Net(strategy)
    with pytest.raises(RuntimeError):
        compile_net(net, start, end, x, mstype.float32)


def test_lin_space_ext_dynamic_shape():
    """
    Feature: test LinSpaceExt parallel with dynamic shape
    Description: strategy > 1
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    start = 8
    end = 10
    x = 8
    strategy = ((4,),)
    net = Net(strategy)
    phase = compile_net(net, start, end, x, mstype.float32)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs('LinSpaceExt-0', ['ScalarAdd-0', 'ScalarAdd-1', 2, 43])
