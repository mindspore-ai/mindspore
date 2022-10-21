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
def test_while_after_for_in_while_1():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit
    def func2311():
        x = Tensor([2])
        y = Tensor([0])
        k = 1
        while y < Tensor([3]):
            for _ in range(3):
                y = y + Tensor([k])
        z = y + x
        while y > x and x < z:
            y -= x
            z = z + y
        return z

    res = func2311()
    assert res == 6


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_while_after_for_in_while_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit
    def func2312():
        x = Tensor([2])
        y = Tensor([2])
        while y < Tensor([5]) and x > Tensor([1]):
            x -= 1
            for _ in range(3):
                y = y + Tensor([1])

        while x < y:
            y -= x
        z = Tensor([-1]) - y
        return z

    res = func2312()
    assert res == -2


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_while_after_for_in_while_3():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit
    def func2313():
        x = [1, 2, 3, 4]
        y = Tensor([8])
        while Tensor([sum(x)]) > y:
            for _ in range(1):
                y = Tensor([18])
        while y >= 0:
            y -= Tensor(np.array([x[0]]))
        return Tensor(np.array(x)), y

    res_x, res_y = func2313()
    assert (res_x.asnumpy() == [1, 2, 3, 4]).all()
    assert res_y == -1
