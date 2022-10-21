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
""" test graph fallback control flow if in if scenario"""
import pytest
import numpy as np
from mindspore import Tensor, jit, context

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_if_in_if_5():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_in_if():
        x = list([1, 2, 3, 4])
        if max(x) >= 4:
            y = Tensor(sum(x) + max(x))
            if y < Tensor(10):
                return y
            return y - 10
        return x
    res = control_flow_if_in_if()
    assert res == 4


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_if_else_in_if_else_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_in_if():
        x = Tensor(10)
        y = Tensor(7)
        if x - y > Tensor(np.array([10])):
            x = x - Tensor(3)
            if x - y > Tensor(0):
                x = x - Tensor(4)
            else:
                x = x + Tensor(4)
            x = x * 2
        else:
            if x > Tensor(15):
                m = np.array([1, 2, 3, 4, 5])
            elif x < Tensor(-10):
                return Tensor(sum(np.array([5, 4, 3, 2, 1])))
            else:
                m = np.array([-1, -2, -3, -4, -5])
            x = Tensor(sum(m))
        return x - 1
    res = control_flow_if_in_if()
    assert res == -16


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_if_in_if_multi_conds_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_in_if():
        x = Tensor(10)
        y = Tensor(2)
        if x > y and x % y == Tensor(0):
            x -= Tensor(3)
            if x < y:
                m = Tensor(10)
            elif x > y or x % y == Tensor(0):
                m = Tensor(20)
            else:
                m = x + y
        else:
            m = Tensor(0)
        return m
    res = control_flow_if_in_if()
    assert res == 20


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_if_in_if_4():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_in_if():
        x = np.array([1, 2, 3, 4, 5])
        y = x % 2
        z = Tensor(y)
        if (x >= y).all():
            if sum(z) > Tensor(2):
                z = Tensor(x) + 1
        return z
    res = control_flow_if_in_if()
    assert np.all(res.asnumpy() == np.array([2, 3, 4, 5, 6]))
