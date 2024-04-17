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
import os
import pytest
import itertools
import numpy as np
from mindspore import Tensor, jit, context

context.set_context(mode=context.GRAPH_MODE)


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
        return Tensor(x)
    res = control_flow_for()
    assert (res.asnumpy() == [1, 9, 17]).all()


def test_single_for_builtin_function_sum():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for():
        x = np.array([1, 3, 5, 7, 9])
        result = x
        for _ in range(3):
            result = sum(x)
        return Tensor(result)
    res = control_flow_for()
    assert res == 25


def test_single_for_builtin_function_numpy_sum():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for():
        x = np.array([1, 3, 5, 7, 9])
        y = np.array([0, 2, 4, 6, 8])
        result = x
        for _ in range(3):
            x = x + y
            result = sum(x, 1)
        return Tensor(result)
    res = control_flow_for()
    assert res == 86


def test_single_for_builtin_function_tensor_sum():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for():
        x = Tensor(np.array([1, 3, 5, 7, 9]))
        y = Tensor(np.array([0, 2, 4, 6, 8]))
        result = x
        for _ in range(3):
            x = x + y
            result = sum(x, 1)
        return result
    res = control_flow_for()
    assert res == 86


def test_single_for_builtin_function_list():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for():
        x = np.array([1.1, 2.2])
        for _ in range(3):
            x = x + list(x)
        return Tensor(x)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    res = control_flow_for()
    assert (res.asnumpy() == [8.8, 17.6]).all()
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


def test_single_for_x_in_xs():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for():
        x = np.array([1.1, 2.2])
        y = np.array(0)
        for i in x:
            y += i
        return Tensor(y)
    res = control_flow_for()
    assert np.allclose(res.asnumpy(), 3.3)


def test_single_for_x_in_xs_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for():
        y = np.array(0)
        for i in np.array([1.1, 2.2]):
            y += i
        return Tensor(y)
    res = control_flow_for()
    assert np.allclose(res.asnumpy(), 3.3)


def test_single_for_wrong_xs():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for():
        y = np.array(0)
        for i in np.int64(1):
            y += i
        return Tensor(y)

    with pytest.raises(TypeError) as info:
        control_flow_for()
    assert "object is not iterable" in str(info.value)


def test_single_for_wrong_xs_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for():
        x = np.int64(1)
        y = np.array(0)
        for i in x:
            y += i
        return Tensor(y)

    with pytest.raises(TypeError) as info:
        control_flow_for()
    assert "object is not iterable" in str(info.value)


def test_single_for_iter_object():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for():
        a = 0
        m = (1, 2, 3)
        n = (4, 5, 6)
        for i, j in itertools.product(m, n):
            a = a + i * j
        return a

    ret = control_flow_for()
    assert ret == 90
