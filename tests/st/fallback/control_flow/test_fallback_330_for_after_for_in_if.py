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
from mindspore import Tensor, ms_function, context

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_for_after_for_in_if_1():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @ms_function
    def func3301():
        x = Tensor([0])
        y = Tensor([0])
        if x == y:
            x = x + 3
            for _ in range(2):
                x = x + 1
        for _ in range(3):
            y = y + 1
        return x + y

    res = func3301()
    assert res == 8


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_for_after_for_in_if_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @ms_function
    def func3302():
        x = np.array([1])
        y = np.array([1])
        if x + y == np.array([3]):
            x = np.array([0])
        else:
            for i in range(3):
                x = x + i

        for i in range(3):
            y = y + i

        return x, y

    res_x, res_y = func3302()
    assert res_x == 4
    assert res_y == 4


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_for_after_for_in_if_3():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @ms_function
    def func3303():
        x = np.array([1, 2, 3])
        y = np.array([5, 6, 7])
        k = []
        if x[2] < y[0]:
            y = y - x
            for i in y:
                k.append(i)

        z = Tensor(k)
        out = 0
        for i in z:
            out = out * i
        return out

    res = func3303()
    assert res == 64


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_for_after_for_in_if_4():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @ms_function
    def func3304():
        x = Tensor([1])
        y = Tensor([2])
        if max(x, y) == Tensor([1]) or min(x, y) == Tensor([2]):
            for _ in range(5):
                raise TypeError("Not expect to enter this branch")

        z = (Tensor(1), Tensor(2), Tensor(3))
        for i in zip(z):
            x = x * i
        return x

    res = func3304()
    assert res == 6
