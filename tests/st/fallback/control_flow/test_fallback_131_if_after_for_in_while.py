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
def test_if_after_for_in_while_tensor():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_after_for_in_while():
        x = Tensor([1])
        y = Tensor([2])
        z = Tensor([7])
        while x < y and x < z:
            for _ in range(3):
                y -= x
            z = z + Tensor([7])
        if x + y >= z:
            y = y * x - Tensor([9])
        return y + z
    res = control_flow_if_after_for_in_while()
    assert res == 13


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_if_after_for_in_while_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_after_for_in_while():
        x = Tensor([1])
        y = Tensor([2])
        z = Tensor([7])
        while x < y:
            z = z * Tensor([7])
            for _ in range(3):
                y -= x
            x = y + z
        if x + y >= z and x != z:
            y = y * x - Tensor([9])
        return y + z
    res = control_flow_if_after_for_in_while()
    assert res == 48


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_if_after_for_in_while_numpy_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_after_for_in_while():
        x = np.array([5, 4, 3, 2, 1])
        y = (Tensor(1), Tensor(3), Tensor(5))
        while sum(x) >= 15:
            for _ in range(3):
                x -= 4
            x = x + 2
        if sum(y) == 9:
            return Tensor(x)
        return Tensor(x + 3)
    res = control_flow_if_after_for_in_while()
    assert (res.asnumpy() == [-5, -6, -7, -8, -9]).all()
