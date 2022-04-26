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
""" test graph fallback control flow if after if scenario"""
import pytest
import numpy as np
from mindspore import Tensor, ms_function, context

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_if_in_if_tensor():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @ms_function
    def control_flow_if_in_while():
        x = Tensor(1)
        y = Tensor(0)
        if x < Tensor(5):
            y += Tensor(4)
        if y > Tensor(3):
            x += Tensor(3)
        return x + y
    res = control_flow_if_in_while()
    assert res == 8


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_if_in_if_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @ms_function
    def control_flow_if_in_while():
        x = Tensor(1)
        y = Tensor(0)
        if x > Tensor(5):
            y -= Tensor(4)
        elif x > Tensor(2):
            y -= Tensor(1)
        else:
            y += Tensor(4)
        if y < Tensor(3):
            x += Tensor(3)
        else:
            x += y
        return x + y
    res = control_flow_if_in_while()
    assert res == 9


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_if_in_if_tensor_3():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @ms_function
    def control_flow_if_in_while(a):
        if a > 15:
            y = Tensor(1)
        else:
            y = Tensor(2)
        if a == Tensor(10):
            a = Tensor(11)
        return a
    res = control_flow_if_in_while(Tensor(10))
    assert res == 11


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_if_in_if_numpy():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @ms_function
    def control_flow_if_in_while():
        x = np.array([1, 2, 3, 4])
        a = sum(x)
        if a > 15:
            y = np.array([1, 2, 3, 4])
        else:
            y = np.array([4, 5, 6])
        if np.all(y == np.array([1, 2, 3, 4])):
            ret = Tensor(1)
        else:
            ret = Tensor(2)
        return ret
    res = control_flow_if_in_while()
    assert res == 2
