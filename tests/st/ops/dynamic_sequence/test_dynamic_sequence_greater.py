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
# ============================================================================
import pytest

import mindspore.nn as nn
from mindspore import context
from mindspore.common import mutable
from mindspore.ops.composite import GradOperation
from mindspore.ops.operations import _sequence_ops as seq
from sequence_help import TupleFactory, context_prepare

context.set_context(mode=context.GRAPH_MODE)
context_prepare()


class GreaterThanNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.func = seq.tuple_greater_than()

    def construct(self, x, y):
        if self.func(x, y):
            return 1
        return 0


class GreaterEqualNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.func = seq.tuple_greater_equal()

    def construct(self, x, y):
        if self.func(x, y):
            return 1
        return 0


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_seq_greater_than():
    """
    Feature: test tuple_greater_than op
    Description: first, second inputs are dynamic tuple
    Expectation: the result match with tuple result
    """
    def func(x, y):
        return x > y

    net_ms = GreaterThanNet()
    input_x = tuple([2, 3, 4, 5])
    input_y = tuple([1, 2, 3, 4])
    fact = TupleFactory(net_ms, func, (input_x, input_y))
    fact.forward_cmp()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_seq_greater_equal():
    """
    Feature: test tuple_greater_equal op
    Description: first input is dynamic tuple
    Expectation: the result match with tuple result
    """
    def func(x, y):
        return x >= y

    net_ms = GreaterEqualNet()
    input_x = tuple([2, 2, 4, 5])
    input_y = tuple([1, 2, 4, 5])
    fact = TupleFactory(net_ms, func, (input_x, input_y))
    fact.forward_cmp()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_seq_greater_than_grad():
    """
    Feature: test tuple_greater_than grad op
    Description: two inputs are dynamic tuple
    Expectation: the result match with tuple result
    """
    x = mutable((2, 3, 4), True)
    y = mutable((1, 2, 3), True)
    dout = mutable(1)
    net = GreaterThanNet()
    grad_func = GradOperation(get_all=True, sens_param=True)(net)
    expect = ((0, 0, 0), (0, 0, 0))
    assert grad_func(x, y, dout) == expect


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_seq_greater_equal_grad():
    """
    Feature: test tuple_greater_equal grad op
    Description: two inputs are dynamic tuple
    Expectation: the result match with tuple result
    """
    x = mutable((2, 2, 3), True)
    y = mutable((1, 2, 3), True)
    dout = mutable(0)
    net = GreaterEqualNet()
    grad_func = GradOperation(get_all=True, sens_param=True)(net)
    expect = ((0, 0, 0), (0, 0, 0))
    assert grad_func(x, y, dout) == expect
