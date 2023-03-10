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
from mindspore import context, Tensor
from mindspore.common import mutable
from mindspore.ops.composite import GradOperation
from sequence_help import context_prepare

context.set_context(mode=context.GRAPH_MODE)
context_prepare()


class Net(nn.Cell):
    def construct(self, x, y):
        return x in y


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_seq_1():
    """
    Feature: test sequence_add op
    Description: two inputs are dynamic sequence
    Expectation: the result match with tuple result
    """
    x = mutable(3)
    y = mutable((4, 5, 6), True)
    expect = False
    net = Net()
    res = net(x, y)
    assert res == expect


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_seq_2():
    """
    Feature: test sequence_add op
    Description: two inputs are dynamic sequence
    Expectation: the result match with tuple result
    """
    x = 4
    y = mutable((4, 5, 6), True)
    expect = True
    net = Net()
    res = net(x, y)
    assert res == expect


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_seq_3():
    """
    Feature: test sequence_add op
    Description: two inputs are dynamic sequence
    Expectation: the result match with tuple result
    """
    x = Tensor(4)
    y = mutable((Tensor(1), Tensor(4), Tensor(5)), True)
    expect = True
    net = Net()
    res = net(x, y)
    assert res == expect


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_seq_4():
    """
    Feature: test sequence_add op
    Description: two inputs are dynamic sequence
    Expectation: the result match with tuple result
    """
    x = Tensor([[1, 2], [3, 4]])
    y = mutable((Tensor([[0, 2], [3, 4]]), Tensor([[1, 2], [3, 4]]), Tensor([[1, 2], [3, 0]])), True)
    expect = True
    net = Net()
    res = net(x, y)
    assert res == expect


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_in_seq_grad_other():
    """
    Feature: test sequence add grad op
    Description: inputs are dynamic sequence.
    Expectation: the result match with tuple result
    """
    net_ms = Net()
    input_x = 3
    input_y = mutable((3, 4, 5, 6), True)
    dout = True
    grad_func = GradOperation(get_all=True, sens_param=True)(net_ms)
    print("grad out1 = ", grad_func(input_x, input_y, dout))
