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
from mindspore import Tensor, jit, context

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_while_after_while_in_for_numpy():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_while_after_while_in_for():
        x = np.array([-1])
        y = np.array([-2]) + x
        z = [2, 4, 6]
        output = 0
        for index in range(2):
            while x * y < z[index]:
                x = y - x
            output = Tensor(x + y)
        while y <= x:
            y -= x
        output += Tensor(y)
        return output
    res = control_flow_while_after_while_in_for()
    assert res == -6


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_while_after_while_in_for_numpy_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_while_after_while_in_for():
        x = np.array([-1])
        y = np.array([-2]) + x
        z = [2, 4, 6]
        output = Tensor([0])
        for index in range(2):
            while x * y < z[index]:
                x = y - x
                z[index] += 1
        sum_z = z[0] + z[1] + z[2]
        while y != x and sum_z > x + y:
            y -= x
            output += Tensor(y)
        return output
    res = control_flow_while_after_while_in_for()
    assert res == 63


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_while_after_while_in_for_tensor():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_while_after_while_in_for():
        x = Tensor([1])
        y = Tensor([-2])
        z = [Tensor([2]), Tensor([4]), Tensor([6])]
        for index in z:
            while (x * y) > (x + y - index):
                x = x + 1
        while y >= -x:
            y -= x
        return x, y
    res_x, res_y = control_flow_while_after_while_in_for()
    assert res_x == 3
    assert res_y == -5


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_while_after_while_in_for_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_while_after_while_in_for():
        x = Tensor([1])
        y = Tensor([2])
        z = [Tensor([2]), Tensor([4]), Tensor([6])]
        for index in range(2):
            while (x * y) > z[index]:
                x = y * x
        while y >= x:
            y -= x
        return x, y
    res = control_flow_while_after_while_in_for()
    assert res == (1, 0)
