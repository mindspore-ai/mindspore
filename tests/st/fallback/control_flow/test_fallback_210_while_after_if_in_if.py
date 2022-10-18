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
def test_while_after_if_in_if_tensor():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_while_after_if_in_if():
        x = Tensor([1])
        y = Tensor([2])
        if x < y:
            y = y * x - Tensor([3])
            z = x + y
            if x * y < z:
                x = y + z
            else:
                x = y - z
        else:
            z = x - y
        while y > x and x < z:
            y -= x
            z = z + y
        return z
    res = control_flow_while_after_if_in_if()
    assert res == 0


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_while_after_if_in_if_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_while_after_if_in_if():
        x = Tensor([1])
        y = Tensor([2])
        z = x + y
        if x >= y:
            if x * y < z:
                x = y + z
            else:
                x = y - z
        print("x: ", x, " y: ", y, " z: ", z)
        while y >= x and x < z:
            y -= x
            z = z + y
        return z
    res = control_flow_while_after_if_in_if()
    assert res == 4


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_while_after_if_in_if_numpy():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_while_after_if_in_if():
        x = np.array([1])
        y = np.array([10])
        if x <= y:
            z = Tensor([2])
            if Tensor(x) > z:
                y = 0
            else:
                x = y - x
        while y > x:
            y[0] -= x[0]
        return Tensor(y)
    res = control_flow_while_after_if_in_if()
    assert res == 1


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_while_after_if_in_if_numpy_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_while_after_if_in_if():
        x = np.array([1])
        y = np.array([10])
        if x < y:
            z = Tensor([2])
            if Tensor(x) < z:
                x = y - x
            else:
                y = x + y
                return Tensor(x), Tensor(y)
        while y[0] > x[0]:
            y[0] -= x[0]
        return Tensor(x), Tensor(y)
    res_x, res_y = control_flow_while_after_if_in_if()
    assert res_x == 9
    assert res_y == 1
