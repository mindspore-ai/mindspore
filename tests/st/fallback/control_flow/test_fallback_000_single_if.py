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
from mindspore import Tensor, ms_function, context
from mindspore import dtype as mstype
import numpy as np

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single_if_1():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @ms_function
    def control_flow_if():
        x = Tensor(1)
        if x > Tensor(7):
            return x
        return x * 2
    res = control_flow_if()
    assert res == 2


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single_if_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @ms_function
    def control_flow_if():
        x = np.array([1, 2, 3, 4, 5])
        y = x % 2
        z = Tensor(y)
        if (x < y).any():
            z = Tensor(x)
        return z
    res = control_flow_if()
    assert np.all(res.asnumpy() == np.array([1, 0, 1, 0, 1]))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single_if_3():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @ms_function
    def control_flow_if():
        x = np.array([1])
        if x <= 1:
            x += 1
        return Tensor(x)
    res = control_flow_if()
    assert res == 2


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single_if_4():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @ms_function
    def control_flow_if():
        x = Tensor(7).astype("int32")
        y = Tensor(0).astype("int32")
        z = x + y
        if z > y:
            y = 5 * x + Tensor(7).astype("int32")
        return y
    res = control_flow_if()
    assert res == 42


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single_if_else_1():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @ms_function
    def control_flow_if_else():
        x = Tensor(1)
        if x > Tensor(7):
            return x
        x += Tensor(3)
        return x * 2
    res = control_flow_if_else()
    assert res == 8


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single_if_else_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @ms_function
    def control_flow_if_else():
        x = np.array([1, 2, 3, 4, 5])
        y = x % 2
        if (x < y).any():
            z = Tensor(x)
        else:
            z = Tensor(y)
        return z
    res = control_flow_if_else()
    assert np.all(res.asnumpy() == np.array([1, 0, 1, 0, 1]))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single_if_two_cond():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @ms_function
    def control_flow_if():
        x = Tensor(1)
        y = np.array(2)
        if x < Tensor(7) and x < Tensor(y):
            return x
        return x * 2
    res = control_flow_if()
    assert res == 1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single_if_builtin_function_abs():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @ms_function
    def control_flow_if():
        x = Tensor(-11, mstype.float32)
        if abs(x) > Tensor(np.array(2)):
            return x - Tensor(np.array(2))
        return x * 2
    res = control_flow_if()
    assert res == -13


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single_if_builtin_function_abs_min():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @ms_function
    def control_flow_if():
        x = Tensor(-11, mstype.float32)
        y = Tensor(12, mstype.float32)
        if abs(x) > Tensor(np.array(2)) and min(x, y) == x + y:
            return x - Tensor(np.array(2))
        return x * 2
    res = control_flow_if()
    assert res == -22


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single_if_builtin_function_sum():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @ms_function
    def control_flow_if():
        x = Tensor(-11, mstype.float32)
        y = Tensor(12, mstype.float32)
        if x + y > 0:
            return sum([x, y, 2 * x])
        return x * 2
    res = control_flow_if()
    assert res == -21
