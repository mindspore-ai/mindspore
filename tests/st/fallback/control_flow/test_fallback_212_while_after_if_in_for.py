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
def test_while_after_if_in_for_tensor():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_while_after_if_in_for():
        x = Tensor([1])
        y = Tensor([2])
        z = Tensor([0])
        tensor_list = [Tensor([3]), Tensor([5])]
        for index in tensor_list:
            y = y * x + Tensor([3])
            z = index + y
            if x * y < z:
                x = y + z
            else:
                x = y - z
        while y < x:
            y -= x
            z = z + y
        return z
    res = control_flow_while_after_if_in_for()
    assert res == 73


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_while_after_if_in_for_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_while_after_if_in_for():
        x = Tensor([1])
        y = Tensor([2])
        for _ in range(4):
            z = x + y
            if x * y < z:
                x = y + z
            elif x < y:
                x = y - 1
            else:
                x = y - z
        while y <= x:
            y += x
        return x, y
    res_x, res_y = control_flow_while_after_if_in_for()
    assert res_x == 3
    assert res_y == 5


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_while_after_if_in_for_numpy_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_while_after_if_in_for():
        x = np.array([1])
        y = np.array([10])
        tensor_list = [Tensor([3]), Tensor([5]), Tensor([7]), Tensor([9])]
        for index in tensor_list:
            z = Tensor([2]) + index
            if Tensor(x) < z:
                x = y - x
            elif Tensor(x) < Tensor(y) and Tensor(x[0] + y[0]) >= z:
                x += y
            else:
                y = x + y
        while y[0] > x[0]:
            y -= x[0]
        return Tensor(x), Tensor(y)
    res_x, res_y = control_flow_while_after_if_in_for()
    assert res_x == 48
    assert res_y == 29
