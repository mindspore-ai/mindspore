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
def test_if_after_if_in_while_tensor():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_after_if_in_while():
        x = Tensor(1)
        y = Tensor(10)
        z = x - y
        while x < Tensor(3):
            x += 1
            if y > x:
                y += x
            else:
                z = x * 2 - y
        if x + y >= z:
            y = y * x - z
        return y
    res = control_flow_if_after_if_in_while()
    assert res == 54


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_if_after_if_in_while_numpy():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_after_if_in_while():
        x = np.array([1, 2])
        y = np.array([3, 2])
        index = Tensor(1)
        while index < Tensor(3):
            index += 1
            if (y > x).all():
                y += x
        z = np.array([2, 4])
        if (x + y >= z).any():
            y = y * x - z
        return Tensor(y)
    res = control_flow_if_after_if_in_while()
    assert (res.asnumpy() == [1, 0]).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_if_after_if_in_while_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_after_if_in_while():
        x = Tensor(5)
        y = Tensor(2)
        z = x - y
        while x >= Tensor(3):
            if y > x:
                y += x
            z = x * 2 + y
            x -= 2
        if x + y >= z:
            return y * x - z
        return y
    res = control_flow_if_after_if_in_while()
    assert res == 2
