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
""" test graph fallback control flow if in while scenario"""
import pytest
import numpy as np
from mindspore import Tensor, jit, context

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_if_in_while_1():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_in_while():
        x = Tensor(1)
        y = Tensor(0)
        while x < Tensor(5):
            if x % 2 == Tensor(0):
                y += Tensor(1)
            x += Tensor(1)
        return x + y
    res = control_flow_if_in_while()
    assert res == 7


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_if_in_while_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_in_while():
        x = Tensor(1)
        while x < Tensor(5):
            if x % 3 == Tensor(0):
                break
            x += Tensor(1)
        return x
    res = control_flow_if_in_while()
    assert res == 3


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_if_in_while_3():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_in_while():
        x = Tensor(1)
        y = Tensor(0)
        while x < Tensor(5):
            if x % 3 == Tensor(0):
                x += Tensor(1)
                y += Tensor(1)
                continue
            x += Tensor(1)
        return x + y
    res = control_flow_if_in_while()
    assert res == 6


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_if_in_while_4():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_in_while():
        x = Tensor(1)
        y = Tensor(0)
        while x < Tensor(10) and x + y < Tensor(20):
            if x % 3 == Tensor(0):
                x += Tensor(1)
                y += Tensor(1)
                continue
            elif y % 2 == Tensor(0):
                x += Tensor(1)
            elif (x+y) % 5 == Tensor(0):
                break
            else:
                x += Tensor(1)
        return x + y
    res = control_flow_if_in_while()
    assert res == 5


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_if_in_while_numpy():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_in_while():
        x = np.array([1, 2])
        y = np.array([3, 2])
        index = Tensor(1)
        while index < Tensor(3):
            index += 1
            if (y > x).all():
                y += x
        return Tensor(y)
    res = control_flow_if_in_while()
    assert (res.asnumpy() == [3, 2]).all()
