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
""" test graph fallback control flow if after while scenario"""
import numpy as np
from mindspore import Tensor, jit, context

context.set_context(mode=context.GRAPH_MODE)


def test_if_after_while_numpy():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_after_while():
        x = np.array([1, 2, 3, 4])
        y = np.array([5, 6])
        while sum(x) < 20:
            x += 1
            y += 1
        if max(y) == 9:
            return Tensor(sum(y))
        y = y - 2
        return Tensor(sum(y))
    res = control_flow_if_after_while()
    assert res == 17


def test_if_after_while_numpy_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_after_while():
        x = np.array([1, 2, 3, 4])
        y = np.array([5, 6])
        while sum(x) < 20:
            x += 1
            y += 1
        if max(y) == 8:
            return Tensor(sum(y))
        y = y - 2
        return Tensor(sum(y))
    res = control_flow_if_after_while()
    assert res == 13


def test_if_after_while_tensor_and_numpy():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_after_while():
        x = np.array([1, 2, 3, 4])
        y = Tensor(5)
        while sum(x) < 20:
            x += 1
            y += 1
        if max(x) == 7:
            return y
        y = y - 2
        return y
    res = control_flow_if_after_while()
    assert res == 8


def test_if_after_while_tensor_and_numpy_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_after_while():
        x = np.array([1, 2, 3, 4])
        y = Tensor(5)
        while sum(x) < 20:
            x += 1
            y += 1
        if max(x) == 6:
            return y
        y = y - 2
        return y
    res = control_flow_if_after_while()
    assert res == 6
