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
def test_if_after_while_in_for_tensor():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_after_while_in_for():
        x = Tensor(1)
        y = Tensor(2)
        z = Tensor(7)
        for _ in range(5):
            while y > x and x < z:
                y -= x
            z = z + y
        if x + y >= z:
            y = y * x - z
        return y + z
    res = control_flow_if_after_while_in_for()
    assert res == 13


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_if_after_while_in_for_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_after_while_in_for():
        x = Tensor(1)
        y = Tensor(2)
        z = Tensor(7)
        for i in range(-1, 2):
            while y > x and x < z:
                y -= i
                z = z - y
        y = y * x - z
        if x + y >= z:
            return y + z
        return y
    res = control_flow_if_after_while_in_for()
    assert res == 4


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_if_after_while_in_for_numpy():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_after_while_in_for():
        x = np.array([1, 2, 3, 4])
        y = Tensor(5)
        for _ in range(4):
            while y > Tensor(x[0]):
                y -= Tensor(x[1])
        if y != 0:
            y = y + Tensor(sum(x))
        return y
    res = control_flow_if_after_while_in_for()
    assert res == 11
