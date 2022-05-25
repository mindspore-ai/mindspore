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
""" test graph fallback control flow for after if in for scenario"""
import pytest
import numpy as np
from mindspore import Tensor, ms_function, context

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_for_after_if_in_for_tensor():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @ms_function
    def control_flow_for_after_if_in_for():
        x = Tensor([1])
        y = Tensor([0])
        for _ in range(5):
            x += 2
            y += 3
            if y - x > 2:
                y -= 4
        for _ in range(5):
            x += 1
            y -= 1
        return x - y
    res = control_flow_for_after_if_in_for()
    assert res == 10


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_for_after_if_in_for_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @ms_function
    def control_flow_for_after_if_in_for():
        x = Tensor([1])
        y = Tensor([0])
        for _ in range(5):
            x += 2
            y += 3
            if y > 8:
                break
            y += 1
        for _ in range(5):
            x += 1
        return x - y
    res = control_flow_for_after_if_in_for()
    assert res == 1


@pytest.mark.skip(reason='Not support to get attribute for InterpretObject.')
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_for_after_if_in_for_numpy():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @ms_function
    def control_flow_for_after_if_in_for():
        x = np.array([1, 2, 3, 4])
        y = np.array([0, 1, 1])
        for _ in range(3):
            x += 2
            y += 1
            if sum(x) > 15:
                break
        for _ in y:
            x += y
        return Tensor(max(x))
    res = control_flow_for_after_if_in_for()
    assert res == 11


@pytest.mark.skip(reason='Not support to get attribute for InterpretObject.')
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_for_after_if_in_for_numpy_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @ms_function
    def control_flow_for_after_if_in_for():
        x = np.array([1, 2, 3, 4])
        y = np.array([0, 1, 2])
        a = 0
        for i in x:
            y += 1
            if max(y) % 2 == 0:
                a += i
            if min(y) > 4:
                break
            y += 2
        for i in range(3):
            a += y[i]
        return Tensor(max(x))
    res = control_flow_for_after_if_in_for()
    assert res == 17
