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
from mindspore import context
from mindspore import Tensor, jit

context.set_context(mode=context.GRAPH_MODE)


def test_single_if_1():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if():
        x = Tensor(1)
        if x > Tensor(7):
            return x
        return x * 2
    res = control_flow_if()
    assert res == 2


def test_single_if_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if():
        x = np.array([1, 2, 3, 4, 5])
        y = x % 2
        z = Tensor(y)
        if (x < y).any():
            z = Tensor(x)
        return z
    res = control_flow_if()
    assert np.all(res.asnumpy() == np.array([1, 0, 1, 0, 1]))


def test_single_if_3():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if():
        x = np.array([1])
        if x <= 1:
            x += 1
        return Tensor(x)
    res = control_flow_if()
    assert res == 2


def test_single_if_else_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_else():
        x = np.array([1, 2, 3, 4, 5])
        y = x % 2
        if (x < y).any():
            z = Tensor(x)
        else:
            z = Tensor(y)
        return z
    res = control_flow_if_else()
    assert np.all(res.asnumpy() == np.array([1, 0, 1, 0, 1]))


def test_single_if_change_variable_value():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if():
        x = np.array([1, 2, 3, 4])
        y = np.array([4, 5, 6])
        if max(x) <= min(y):
            x += 3
            return Tensor(x)
        return Tensor(0)
    res = control_flow_if()
    assert np.all(res.asnumpy() == np.array([4, 5, 6, 7]))


def test_single_if_np_all():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if():
        x = np.array([1, 2, 3, 4])
        y = np.array([4, 5, 6])
        if np.all(x == np.array([1, 2, 3, 4])) and np.any(y == np.array([4, 4, 4])):
            x += 3
            return Tensor(x)
        return Tensor(0)
    res = control_flow_if()
    assert np.all(res.asnumpy() == np.array([4, 5, 6, 7]))
