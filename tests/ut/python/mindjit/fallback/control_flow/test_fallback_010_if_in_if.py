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
""" test graph fallback control flow if in if scenario"""
import numpy as np
from mindspore import Tensor, jit, context

context.set_context(mode=context.GRAPH_MODE)


def test_if_in_if_1():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_in_if():
        x = Tensor(1)
        y = Tensor(2)
        if x > Tensor(0):
            if y > Tensor(1):
                return y + 1
            return x + 1
        return x + y
    res = control_flow_if_in_if()
    assert res == 3


def test_if_in_if_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_in_if():
        x = Tensor(1)
        y = Tensor(0)
        if x > Tensor(0):
            if y > Tensor(1):
                return y + 1
            return x + 1
        return x + y
    res = control_flow_if_in_if()
    assert res == 2


def test_if_in_if_3():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_in_if():
        x = Tensor(-2)
        y = Tensor(-3)
        if x > Tensor(0):
            if y > Tensor(1):
                return y + 1
            return x + 1
        return x + y
    res = control_flow_if_in_if()
    assert res == -5


def test_if_else_in_if_else_1():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_in_if():
        x = Tensor(10)
        y = Tensor(7)
        if x - y > Tensor(np.array([0])):
            x = x - Tensor(3)
            if x - y > Tensor(0):
                x = x - Tensor(4)
            else:
                x = x + Tensor(4)
            x = x * 2
        return x - 1
    res = control_flow_if_in_if()
    assert res == 21


def test_if_in_if_multi_conds():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_in_if():
        x = np.array([1, 2, 3, 4])
        y = np.array([4, 5, 6])
        if max(x) <= min(y) and sum(x) == 10:
            x += 3
            if max(x) <= max(y):
                m = Tensor(10)
            elif min(x) != max(y) or x.size > y.size:
                m = Tensor(20)
            else:
                m = Tensor(0)
        else:
            m = Tensor(1)
        return m
    res = control_flow_if_in_if()
    assert res == 20
