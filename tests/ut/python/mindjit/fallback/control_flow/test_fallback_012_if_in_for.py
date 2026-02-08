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
