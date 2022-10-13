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
"""test python built-in functions in graph mode"""
import pytest
import numpy as np
from mindspore import Tensor, context, nn
from mindspore import dtype as mstype

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_sum_tensor_n_default_1():
    """
    Feature: JIT Fallback
    Description: Description: Test sum(Tensor) in graph mode with tensor and input n is default.
    Expectation: No exception
    """
    class Net(nn.Cell):
        def construct(self, x):
            return sum(x)

    net = Net()
    x = Tensor([3, 4, 5], dtype=mstype.float32)
    out = net(x)
    assert out == 12


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_sum_tensor_n_default_2():
    """
    Feature: JIT Fallback
    Description: Description: Test sum(Tensor) in graph mode with tensor and input n is default.
    Expectation: No exception
    """
    class Net(nn.Cell):
        def construct(self, x):
            return sum(x)

    net = Net()
    x = Tensor([[1, 2], [3, 4]], dtype=mstype.float32)
    out = net(x)
    assert np.allclose(out.asnumpy(), np.array([4, 6]))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_sum_with_x_tensor_n_not_default_1():
    """
    Feature: JIT Fallback
    Description: Test sum() in graph mode with input x tensor and input n not default.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            return sum(x, y)

    net = Net()
    x, y = Tensor([3, 4, 5], dtype=mstype.float32), 4
    out = net(x, y)
    assert out == 16


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_sum_with_x_tensor_n_not_default_2():
    """
    Feature: JIT Fallback
    Description: Test sum() in graph mode with input x tensor and input n not default.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            return sum(x, y)

    net = Net()
    x, y = Tensor([[1, 2], [3, 4]], dtype=mstype.float32), [5, 6]
    out = net(x, y)
    assert np.allclose(out.asnumpy(), np.array([9, 12]))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_sum_with_x_list_of_tensor():
    """
    Feature: JIT Fallback
    Description: Test sum() in graph mode when input x is list of tensor.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            return sum(x, y)

    net = Net()
    x, y = [1, Tensor([[1, 2], [3, 4]]), Tensor([[1, 2], [3, 4]])], Tensor([[1, 1], [1, 1]])
    out = net(x, y)
    assert np.allclose(out.asnumpy(), np.array([[4, 6], [8, 10]]))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_fallback_sum_with_tensor_0d(mode):
    """
    Feature: JIT Fallback
    Description: Test sum() in graph mode when input x is 0d tensor.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self):
            return sum(Tensor(1))

    with pytest.raises(TypeError, match="Cannot iterate over a scalar tensor."):
        context.set_context(mode=mode)
        net = Net()
        net()
