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
def test_while_after_for_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_while_after_for():
        x = Tensor([1])
        y = Tensor([2])
        for _ in range(3):
            y = y * x
        z = Tensor([4])
        while x + z < y:
            y -= x
        z = Tensor(7) - y
        return z
    res = control_flow_while_after_for()
    assert res == 5


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_while_after_for_numpy_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_while_after_for():
        x = np.array([3, 2])
        y = [1, 2, 3, 4]
        for i in x:
            y.append(i)
        z = Tensor([0])
        while Tensor(sum(x)) < Tensor(15):
            z += Tensor(15) - Tensor(sum(x))
            x = np.array([1, 2, 3, 4, 5, 6])
        return Tensor(x), Tensor(y), z
    res_x, res_y, res_z = control_flow_while_after_for()
    assert (res_x.asnumpy() == [1, 2, 3, 4, 5, 6]).all()
    assert (res_y.asnumpy() == [1, 2, 3, 4, 3, 2]).all()
    assert res_z == [10]


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_while_after_for_tensor():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_while_after_for():
        x = Tensor([1])
        y = Tensor([2])
        for _ in range(5):
            y = y * x - Tensor([3])
        z = y + x
        while y > x and x < z:
            y -= x
            z = z + y
        return z
    res = control_flow_while_after_for()
    assert res == -12


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_while_after_for_numpy():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_while_after_for():
        x = [1, 2, 3, 4, 5]
        y = Tensor([3])
        for i in range(4):
            x[i] += i
        while y >= 0:
            y -= Tensor(x[0])
        return Tensor(x), y
    res_x, res_y = control_flow_while_after_for()
    assert (res_x.asnumpy() == [1, 3, 5, 7, 5]).all()
    assert res_y == -1
