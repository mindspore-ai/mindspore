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
import mindspore.common.dtype as mstype
from mindspore import context, Tensor
from mindspore.common import mutable
from mindspore.ops import functional as F
from mindspore.nn import Cell
from mindspore.ops.composite import GradOperation
from sequence_help import context_prepare

context.set_context(mode=context.GRAPH_MODE)
context_prepare()


class Net(nn.Cell):
    def construct(self, x, y):
        return F.scalar_cast(x, mstype.int32), F.scalar_cast(y, mstype.float32)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scalar_cast():
    """
    Feature: test sequence_add op
    Description: two inputs are dynamic sequence
    Expectation: the result match with tuple result
    """
    x = mutable(2.1)
    y = mutable(3)
    expectx = 2
    expecty = 3.0
    net = Net()
    res = net(x, y)
    assert res[0] == expectx
    assert res[1] == expecty


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_scalar_cast1():
    """
    Feature: test sequence_add op
    Description: two inputs are dynamic sequence
    Expectation: the result match with tuple result
    """
    x = mutable(Tensor(2.1))
    y = mutable(Tensor([3]))
    expectx = 2
    expecty = 3.0
    net = Net()
    res = net(x, y)
    assert res[0] == expectx
    assert res[1] == expecty


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_cast_grad():
    """
    Feature: test sequence add grad op
    Description: inputs are dynamic sequence.
    Expectation: the result match with tuple result
    """
    class Net0(Cell):
        def construct(self, x, dtype):
            return F.scalar_cast(x, dtype)

    net_ms = Net0()
    input_x = mutable(2.2)
    dout = mutable(1)
    grad_func = GradOperation(get_all=True, sens_param=True)(net_ms)
    print("grad out = ", grad_func(input_x, mstype.int64, dout))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_cast_grad1():
    """
    Feature: test sequence add grad op
    Description: inputs are dynamic sequence.
    Expectation: the result match with tuple result
    """
    class Net1(Cell):
        def construct(self, y, dtype):
            return F.scalar_cast(y, dtype)

    net_ms = Net1()
    input_x = mutable(Tensor(2))
    dout = mutable(1.0)
    grad_func = GradOperation(get_all=True, sens_param=True)(net_ms)
    print("grad out1 = ", grad_func(input_x, mstype.float32, dout))
