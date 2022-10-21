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
def test_for_after_if_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit
    def func():
        x = Tensor([2])
        y = Tensor([2])
        if x == y and x > Tensor([0]).astype("int32"):
            y = y + Tensor([3]).astype("int32")
        else:
            x = x + 1

        for i in range(3):
            z = Tensor([i]).astype("int32")
            x = x + z

        return x, y

    res_x, res_y = func()
    assert res_x == 5
    assert res_y == 5


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_for_after_if_3():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit
    def func():
        x = np.array([1, 2, 3])
        y = np.array([5, 6, 7])
        if x[2] > y[0]:
            y = y + x
        else:
            y = y - x

        z = Tensor(y)
        out = 0
        for i in z:
            out = out + i
        return out

    res = func()
    assert res == 12


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_for_after_if_4():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit
    def func():
        x = Tensor([1])
        y = Tensor([2])
        if max(x, y) == Tensor([2]):
            x = x + min(x, y)

        z = (Tensor(1), Tensor(2), Tensor(3))
        for i in z:
            x = x * i
        return x

    res = func()
    assert res == 12
