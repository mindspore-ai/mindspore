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
import mindspore
from mindspore import Tensor, jit, context
from tests.st.fallback.cases_register import case_register

context.set_context(mode=context.GRAPH_MODE)


@case_register.level0
@case_register.target_gpu
@case_register.target_ascend
def test_for_after_while_in_if_1():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit
    def func3201():
        x = Tensor([0])
        y = Tensor([0])

        if x == y:
            x = x + 3
            while x < Tensor([5]):
                x = x + Tensor(np.array([min(1, 2)]), dtype=mindspore.int64)

        for _ in range(3):
            y = y + 1
        return x + y

    res = func3201()
    assert res == 8


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_for_after_while_in_if_3():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit
    def func3203():
        x = np.array([1, 2, 3])
        y = np.array([5, 6, 7])
        if x[2] > y[0]:
            y = y + x
        else:
            y = y - x
            while x[2] + 2 > y[0]:
                y = y + np.array([1, 1, 1])

        z = Tensor(y)
        out = 0
        for i in z:
            out = out + i
        return out

    res = func3203()
    assert res == 15
