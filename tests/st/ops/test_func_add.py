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
import numpy as np
import pytest
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore import ops


class Net(nn.Cell):
    def construct(self, x, other):
        return x.add(other)


class NetAdd(nn.Cell):
    def construct(self, x, other):
        return ops.add(x, other)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_add(mode):
    """
    Feature: tensor.subtract()
    Description: Verify the result of tensor.subtract
    Expectation: success
    """
    context.set_context(mode=mode)
    net = Net()
    x = Tensor([1, 2, 3], dtype=mstype.float32)
    y = Tensor([4, 5, 6], dtype=mstype.float32)
    output = net(x, y)
    expected = np.array([5, 7, 9], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_add_alpha(mode):
    """
    Feature: tensor.subtract()
    Description: Verify the result of tensor.subtract
    Expectation: success
    """
    context.set_context(mode=mode)
    net = Net()
    x = Tensor([1, 2, 3], dtype=mstype.float32)
    y = Tensor([2, 2, 2], dtype=mstype.float32)
    output = net(x, y)
    expected = np.array([3, 4, 5], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_add_ops(mode):
    """
    Feature: tensor.subtract()
    Description: Verify the result of tensor.subtract
    Expectation: success
    """
    context.set_context(mode=mode)
    net = NetAdd()
    x = Tensor([3, 4, 5], dtype=mstype.float32)
    y = Tensor([1, 2, 3], dtype=mstype.float32)
    output = net(x, y)
    expected = np.array([4, 6, 8], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expected)
