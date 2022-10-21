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
import mindspore
from mindspore import Tensor, jit, context

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_for_after_while_in_for_1():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit
    def func3221():
        x = Tensor([0])
        y = np.array([1])
        for _ in range(2):
            while x < Tensor([3]):
                x = x + 1

        for i in range(3):
            y = y + i
        return x + Tensor(y, dtype=mindspore.int64)

    res = func3221()
    assert res == 7


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_for_after_while_in_for_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit
    def func3222():
        x = Tensor([0])
        y = Tensor([0])
        for i in range(3):
            x = x + Tensor([i + 1]).astype("int32")
            k = i
            while k < 2:
                k = k + 1
                y = y + 1

        t = Tensor(np.array([1, 2, 3]))
        for i in range(3):
            y = y + t[i]

        return x, y

    res_x, res_y = func3222()
    assert res_x == 6
    assert res_y == 9


def test_for_after_while_in_for_3():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit
    def func3224():
        x = Tensor([0])
        for i in range(3):
            x = x - Tensor([i])
            while x < Tensor([0]):
                x = x + abs(Tensor([-i]))

        z = (Tensor(0), Tensor(1), Tensor(2))
        for i in zip(z):
            x = x + i
        return x

    res = func3224()
    assert res == 3
