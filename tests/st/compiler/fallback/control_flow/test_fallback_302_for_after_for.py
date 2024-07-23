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
import numpy as np
from mindspore import Tensor, jit, context
from tests.st.compiler.fallback.cases_register import case_register

context.set_context(mode=context.GRAPH_MODE)


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_for_after_for_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit
    def func():
        x = Tensor([0])
        for i in range(3):
            x = x + Tensor([i + 1]).astype("int32")

        y = Tensor([0])
        t = Tensor(np.array([1, 2, 3]))
        for i in range(3):
            y = y + t[i]

        return x, y

    res_x, res_y = func()
    assert res_x == 6
    assert res_y == 6


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_for_after_for_3():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit
    def func():
        y = np.array([5, 6, 7])
        k = 0
        for i in (0, 1.1, 2.2):
            k = k + int(i)
        y = y - k
        z = Tensor(y)
        out = 0
        for i in z:
            out = out + i
        return out

    res = func()
    assert res == 9
