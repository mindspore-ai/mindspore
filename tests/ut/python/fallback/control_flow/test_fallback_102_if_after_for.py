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
""" test graph fallback control flow if after for scenario"""
import numpy as np
from mindspore import Tensor, jit, context

context.set_context(mode=context.GRAPH_MODE)


def test_if_after_for_tensor():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_after_for():
        x = Tensor(7)
        y = Tensor(0)
        for _ in range(3):
            x += 1
            y += 3
        if x > y:
            return x + y
        return x - y
    res = control_flow_if_after_for()
    assert res == 19


def test_if_after_for_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_after_for():
        x = Tensor(7)
        y = Tensor(0)
        for i in range(3):
            x += i
            y += 3 * i
        if x == y:
            return x + y
        return x - y
    res = control_flow_if_after_for()
    assert res == 1


def test_if_after_for_tensor_3():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if_after_for():
        x = Tensor(7)
        y = Tensor(10)
        for i in range(3):
            x += i
        if x == y:
            return x + y
        return x - y
    res = control_flow_if_after_for()
    assert res == 20


def test_if_after_for_tensor_zip():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for():
        tuple_x = (Tensor(1), Tensor(3), Tensor(5))
        sum_x = Tensor(0)
        a = Tensor(0)
        for x in zip(tuple_x):
            sum_x += x
            a = x
        if sum_x == 9:
            sum_x += a
        return sum_x

    res = control_flow_for()
    assert res == 14


def test_single_for_numpy():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for():
        x = np.array([1, 3, 5])
        y = np.array([0, 2, 4])
        for _ in range(3):
            x = x + y
        if np.all(x == np.array([1, 9, 17])):
            return Tensor(x)
        return Tensor(x - 1)
    res = control_flow_for()
    assert (res.asnumpy() == [1, 9, 17]).all()


def test_single_for_numpy_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for():
        x = np.array([1, 3, 5])
        y = np.array([0, 2, 4])
        for _ in range(3):
            x = x + y
            y = y + 1
        if np.any(x == np.array([1, 12, 17])):
            return Tensor(x - y)
        return Tensor(x + 1)
    res = control_flow_for()
    assert (res.asnumpy() == [1, 7, 13]).all()
