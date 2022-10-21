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
def test_while_after_for_in_for_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit
    def func2322():
        x = Tensor([0])
        for i in range(3):
            x = x + Tensor([i + 1])
            k = (Tensor(1), Tensor(1), Tensor(1))
            for j in zip(k):
                x = x + j

        y = Tensor([0])
        t = Tensor(np.array([1, 2, 3]))

        i = 0
        while i < 3:
            y = y + t[i]
            i += 1

        return x, y

    res_x, res_y = func2322()
    assert res_x == 15
    assert res_y == 6


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_while_after_for_in_for_3():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit
    def func2323():
        y = [5, 6, 7]
        for _ in (0, 1, 2):
            for j in range(3):
                y[j] = y[j] - 1

        y = np.array(y)
        z = Tensor(y)
        out = 0
        i = 0
        while i < len(z):
            out = out + z[i]
            i += 1
        return out

    res = func2323()
    assert res == 9
