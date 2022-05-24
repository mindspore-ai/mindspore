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
""" test graph fallback control flow."""
import pytest
import numpy as np
from mindspore import Tensor, ms_function, context

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_while_after_while_in_while_tensor():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @ms_function
    def control_flow_while_after_while_in_while():
        x = Tensor([-1])
        y = Tensor([-2])
        z = Tensor([0])
        while z < abs(x + y):
            z = abs(x + y + 2)
            y -= 1
            while x < z:
                x = x - y
        while y > x:
            y -= x
            z = z + y
        return x, y, z
    res = control_flow_while_after_while_in_while()
    assert res == (2, -3, 1)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_while_after_while_in_while_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @ms_function
    def control_flow_while_after_while_in_while():
        x = Tensor([-1])
        y = Tensor([-2])
        z = Tensor([0])
        while abs(z) < abs(x + y):
            z = x + y + 2
            while x * y > z:
                x = x - y
            x = y - 2 * z
        y = y + 1
        while y >= x:
            y -= x
        z = z + y
        return x, y, z
    res = control_flow_while_after_while_in_while()
    assert res == (2, -1, -3)


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_while_after_while_in_while_numpy():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @ms_function
    def control_flow_while_after_while_in_while():
        x = Tensor([-1])
        y = Tensor([-2])
        while abs(x) <= abs(y):
            z = np.array([3, 4, 5])
            index = 0
            z_sum = 0
            while index < len(z):
                z_sum += z[index]
                index += 1
            x = x + Tensor(z_sum)
        while y < x:
            y += x
        return x, y
    res = control_flow_while_after_while_in_while()
    assert res == (11, 20)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_while_after_while_in_while_numpy_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @ms_function
    def control_flow_while_after_while_in_if():
        x = np.array([-1])
        y = np.array([-2])
        z_sum = Tensor([0])
        output = Tensor([0])
        while abs(x) <= abs(y):
            z = [Tensor([3]), Tensor([4]), Tensor([5])]
            index = 0
            while index < 3:
                z_sum += z[index]
                index += 1
            output = Tensor(x) + z_sum
            x += 1
        while y < x:
            y += 1
        return output + Tensor(y)
    res = control_flow_while_after_while_in_if()
    assert res == 53
