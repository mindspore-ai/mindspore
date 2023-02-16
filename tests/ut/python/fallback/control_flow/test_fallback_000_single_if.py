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
from mindspore.nn import Cell
from mindspore import Tensor, jit
from mindspore import dtype as mstype

context.set_context(mode=context.GRAPH_MODE)


def test_single_if_no_else_type():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    class FalseNet(Cell):
        def __init__(self):
            super(FalseNet, self).__init__()
            self.cond = False

        def construct(self):
            x = np.array(1)
            if self.cond:
                return type(2).mro()
            return type(x).mro()

    test_net = FalseNet()
    res = test_net()
    assert str(res) == "[<class 'numpy.ndarray'>, <class 'object'>]"


def test_single_if_no_else_type_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    class TrueNet(Cell):
        def __init__(self):
            super(TrueNet, self).__init__()
            self.cond = True

        def construct(self):
            x = np.array(2)
            y = 2
            if self.cond:
                return type(y).mro()
            return type(x).mro()

    test_net = TrueNet()
    res = test_net()
    assert str(res) == "[<class 'int'>, <class 'object'>]"


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


def test_single_if_else_1():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_else():
        x = Tensor(1)
        if x > Tensor(7):
            return x
        x += Tensor(3)
        return x * 2
    res = control_flow_if_else()
    assert res == 8


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


def test_single_if_builtin_function_sum():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if():
        x = Tensor(-11, mstype.float32)
        y = Tensor(12, mstype.float32)
        if x + y > 0:
            return sum([x, y, 2 * x])
        return x * 2
    res = control_flow_if()
    assert res == -21


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
