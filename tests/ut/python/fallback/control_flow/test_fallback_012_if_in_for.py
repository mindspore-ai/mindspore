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


def test_if_in_for_tensor_4():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for():
        x = Tensor(7)
        y = Tensor(0.0)
        for _ in range(3):
            x = x + y/2
            if y < Tensor(10) and x < Tensor(20):
                y += x
            y += Tensor(1)
        return x + y
    res = control_flow_for()
    assert res == 42


def test_if_in_for_tensor_5():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for():
        x = Tensor(7)
        y = Tensor(0.0)
        for _ in range(3):
            x = x + y/2
            if y < Tensor(10):
                y += x
            if x < Tensor(15):
                x += y/2
            y += Tensor(1)
        return x + y
    res = control_flow_for()
    assert res == 62


def test_if_in_for_numpy():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for():
        x = np.array([1, 2, 3, 4])
        for _ in range(3):
            if sum(x) <= 15:
                x += 1
        return Tensor(sum(x))
    res = control_flow_for()
    assert res == 18


def test_if_in_for_numpy_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for():
        x = np.array([1, 2, 3, 4])
        for _ in range(3):
            if sum(x) <= 15:
                x += 1
            else:
                x -= 3
        return Tensor(sum(x))
    res = control_flow_for()
    assert res == 6


def test_if_in_for_numpy_3():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for():
        x = np.array([1, 2, 3, 4])
        for _ in range(3):
            if sum(x) <= 15:
                x += 2
            x -= 1
        return Tensor(sum(x))
    res = control_flow_for()
    assert res == 14


def test_if_in_for_numpy_4():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for():
        x = np.array([1, 2, 3, 4])
        y = np.array([1])
        for _ in range(3):
            if sum(x) <= 15:
                y += 1
            x += y
        return Tensor(sum(x) + y)
    res = control_flow_for()
    assert res == 36


def test_if_in_for_with_break():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for():
        x = np.array([1, 2, 3, 4])
        y = (Tensor(1), Tensor(3), Tensor(5))
        a = Tensor(0)
        for e in y:
            if a > 15:
                break
            a += Tensor(sum(x)) + e
        return a
    res = control_flow_for()
    assert res == 24


def test_if_in_for_with_continue():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for():
        x = np.array([1, 2, 3, 4])
        y = (Tensor(1), Tensor(3), Tensor(5))
        a = Tensor(0)
        for e in y:
            if a > 15:
                continue
            a += Tensor(sum(x)) + e
        return a
    res = control_flow_for()
    assert res == 24


def test_if_in_for_with_break_continue():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for():
        x = np.array([1, 2, 3, 4])
        y = (Tensor(1), Tensor(2), Tensor(4), Tensor(5))
        a = Tensor(0)
        for e in y:
            a += Tensor(sum(x)) + e
            if e == Tensor(2):
                continue
            if a > 15:
                a -= Tensor(1)
                break
            a += Tensor(1)
        return a
    res = control_flow_for()
    assert res == 37
