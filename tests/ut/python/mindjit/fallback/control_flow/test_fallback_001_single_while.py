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


def test_single_while_4():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_while():
        z = np.array(0)
        while z <= 3:
            z += 1
        return Tensor(z)

    res = control_flow_while()
    assert res == 4


def test_single_while_builtin_function_abs_numpy():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_while():
        x = np.array(-11)
        y = np.array(0)
        while abs(x) > 2:
            x += np.array(4)
            y += np.array(1)
        return Tensor(x), Tensor(y)
    res_x, res_y = control_flow_while()
    assert res_x == 1
    assert res_y == 3


def test_single_while_builtin_function_abs():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_while():
        x = -11
        y = 0
        while abs(x) > 2:
            x += 4
            y += 1
        return Tensor(x), Tensor(y)
    res_x, res_y = control_flow_while()
    assert res_x == 1
    assert res_y == 3


def test_single_while_builtin_function_max_numpy():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_while():
        x = np.array(3)
        y = np.array(5)
        while max(x, y) > 3:
            x -= np.array(4)
            y -= np.array(1)
        return Tensor(x), Tensor(y)
    res_x, res_y = control_flow_while()
    assert res_x == -5
    assert res_y == 3


def test_single_while_builtin_function_first_in_while_body():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_while():
        i = 0
        while i <= 3:
            i += int(1)
        return i

    res = control_flow_while()
    assert res == 4
