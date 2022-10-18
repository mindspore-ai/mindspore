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

context.set_context(mode=context.GRAPH_MODE)


def test_while_after_while_in_while_numpy_1():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_while_after_while_in_while():
        x = np.array([-1])
        y = np.array([-2])
        z = np.array([0])
        while z < abs(x + y):
            z = abs(x + y + 2)
            y -= 1
            while x < z:
                x = x - y
        while y > x:
            y -= x
            z = z + y
        return Tensor(x), Tensor(y), Tensor(z)
    res = control_flow_while_after_while_in_while()
    assert res == (2, -3, 1)


def test_while_after_while_in_while_numpy_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_while_after_while_in_while():
        x = np.array([-1])
        y = np.array([-2])
        z = np.array([0])
        while abs(z) < abs(x + y):
            z = x + y + 2
            while x * y > z:
                x = x - y
            x = y - 2 * z
        y = y + 1
        while y >= x:
            y -= x
        z = z + y
        return Tensor(x), Tensor(y), Tensor(z)
    res = control_flow_while_after_while_in_while()
    assert res == (2, -1, -3)


def test_while_after_while_in_while_numpy_3():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_while_after_while_in_if():
        x = np.array([-1])
        y = np.array([-2])
        z_sum = Tensor([0])
        output = Tensor([0])
        while abs(x) <= abs(y):
            z = [Tensor([3]), Tensor([4]), Tensor([5])]
            index = 0
            while index < 3:
                z_sum += z[index]
                index += 1
            output = Tensor(x) + z_sum
            x += 1
        while y < x:
            y += 1
        return output + Tensor(y)
    res = control_flow_while_after_while_in_if()
    assert res == 53
