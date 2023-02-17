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
from mindspore.ops.operations import _sequence_ops as seq
from mindspore import context
from mindspore.common import mutable
from mindspore.ops.composite import GradOperation
from sequence_help import context_prepare

context.set_context(mode=context.GRAPH_MODE)
context_prepare()


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.seq_count = seq.SequenceCount()

    def construct(self, x, y):
        return self.seq_count(x, y)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_seq_count_tuple_dy():
    """
    Feature: test sequence_count op
    Description: first input is dynamic sequence
    Expectation: the result match with tuple result
    """
    x = mutable((1, 2, 3), True)
    y = 3
    expect = 1
    net = Net()
    res = net(x, y)
    assert res == expect


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_seq_count_scalar_dy():
    """
    Feature: test sequence_count op
    Description: second input is dynamic scalar
    Expectation: the result match with tuple result
    """
    x = (0, 1, 1, 2)
    y = mutable(1)
    expect = 2
    net = Net()
    res = net(x, y)
    assert res == expect


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_seq_count_all_dy():
    """
    Feature: test sequence_count op
    Description: two inputs are dynamic sequence
    Expectation: the result match with tuple result
    """
    x = mutable((1, 2, 3, 3, 2, 3), True)
    y = mutable(3)
    expect = 3
    net = Net()
    res = net(x, y)
    assert res == expect


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_seq_count_grad():
    """
    Feature: test sequence_count grad op
    Description: two inputs are dynamic sequence
    Expectation: the result match with tuple result
    """
    x = mutable((1, 2, 3), True)
    y = mutable(2)
    dout = mutable(2)
    net = Net()
    grad_func = GradOperation(get_all=True, sens_param=True)(net)
    grad_func(x, y, dout)
