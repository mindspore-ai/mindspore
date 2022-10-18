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
from mindspore import dtype as mstype

context.set_context(mode=context.GRAPH_MODE)


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
    @jit
    def control_flow_if():
        x = Tensor(7).astype("int32")
        y = Tensor(0).astype("int32")
        z = x + y
        if z > y:
            y = 5 * x + Tensor(7).astype("int32")
        return y
    res = control_flow_if()
    assert res == 42


@pytest.mark.level1
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
    @jit
    def control_flow_if():
        x = Tensor(1)
        y = np.array(2)
        if x < Tensor(7) and x < Tensor(y):
            return x
        return x * 2
    res = control_flow_if()
    assert res == 1


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single_if_builtin_function_abs():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if():
        x = Tensor(-11, mstype.float32)
        if abs(x) > Tensor(np.array(2)):
            return x - Tensor(np.array(2))
        return x * 2
    res = control_flow_if()
    assert res == -13


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single_if_builtin_function_abs_min():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if():
        x = Tensor(-11, mstype.float32)
        y = Tensor(12, mstype.float32)
        if abs(x) > Tensor(np.array(2)) and min(x, y) == x + y:
            return x - Tensor(np.array(2))
        return x * 2
    res = control_flow_if()
    assert res == -22
