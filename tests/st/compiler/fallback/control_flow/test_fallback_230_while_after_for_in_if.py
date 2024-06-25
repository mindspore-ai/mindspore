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


@case_register.level0
@case_register.target_gpu
@case_register.target_ascend
def test_while_after_for_in_if_1():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit
    def func2301():
        x = Tensor([1])
        y = Tensor([2])
        if x > y:
            y = y * x
            for _ in range(2):
                y = y + 1
                x = x + Tensor([0])

        z = Tensor([7]) + y
        while y > x and x < z:
            y -= x
        z = z + y
        return y + z

    res = func2301()
    assert res == 11


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_while_after_for_in_if_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit
    def func2302():
        x = Tensor([1])
        y = Tensor([2])
        if x > y:
            y = y * x
        else:
            x = x * y
            for _ in range(2):
                y = y + Tensor(np.array([0]))
                x = x + Tensor([0])

        z = Tensor([7]) - y
        while x < z:
            y += x
            z = z - y
        return x, y, z

    res_x, res_y, res_z = func2302()
    assert res_x == 2
    assert res_y == 4
    assert res_z == 1


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_while_after_for_in_if_3():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit
    def func2303():
        x = np.array([3, 2])
        y = Tensor(np.array([3, 2]))
        if x[0] > x[1]:
            for i in [1, 1, 1]:
                x = x + np.array([i, i])

        else:
            x -= 4
        while (y >= 0).all():
            y -= Tensor(x[0])
        return y

    res = func2303()
    assert (res.asnumpy() == [-3, -4]).all()


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_while_after_for_in_if_4():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit
    def func2304():
        x = [3, 2]
        y = [1, 2, 3, 4]
        if x[0] > x[1]:
            x[0] += 3
            x[1] += 3
            for i in y:
                if not i == 1:
                    break
                x[1] += i
        x = np.array(x)
        z = int(x[1])
        while len(y) < 5:
            y.append(z)
        return Tensor(y)

    res = func2304()
    assert (res.asnumpy() == [1, 2, 3, 4, 6]).all()
