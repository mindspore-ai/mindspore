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
""" test graph fallback control flow for after if in for scenario"""
import numpy as np
from mindspore import Tensor, jit, context

context.set_context(mode=context.GRAPH_MODE)


def test_for_after_if_in_for_tensor():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for_after_if_in_for():
        x = Tensor([1])
        y = Tensor([0])
        for _ in range(5):
            x += 2
            y += 3
            if y - x > 2:
                y -= 4
        for _ in range(5):
            x += 1
            y -= 1
        return x - y
    res = control_flow_for_after_if_in_for()
    assert res == 10


def test_for_after_if_in_for_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for_after_if_in_for():
        x = Tensor([1])
        y = Tensor([0])
        for _ in range(5):
            x += 2
            y += 3
            if y > 8:
                break
            y += 1
        for _ in range(5):
            x += 1
        return x - y
    res = control_flow_for_after_if_in_for()
    assert res == 1


def test_for_after_if_in_for_numpy():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for_after_if_in_for():
        x = np.array([1, 2, 3, 4])
        y = np.array([0, 1, 1])
        for _ in range(3):
            x += 2
            y += 1
            if sum(x) > 15:
                break
        for i in y:
            x += i
        return Tensor(max(x))
    res = control_flow_for_after_if_in_for()
    assert res == 11


def test_for_after_if_in_for_numpy_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for_after_if_in_for():
        x = np.array([1, 2, 3, 4])
        y = np.array([0, 1, 2])
        a = 0
        for i in x:
            y += 1
            if max(y) % 2 == 0:
                a += i
            if min(y) > 4:
                break
            y += 2
        for i in range(3):
            a += y[i]
        return Tensor(a)
    res = control_flow_for_after_if_in_for()
    assert res == 26
