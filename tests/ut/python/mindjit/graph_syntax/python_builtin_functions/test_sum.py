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
""" test graph fallback buildin python function sum"""
import os
import pytest
import numpy as np
from mindspore import jit, Tensor


def test_fallback_sum_with_x_list_n_default():
    """
    Feature: JIT Fallback
    Description: Test sum() in graph mode with input x list and input n default.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = sum([1, 2, 3])
        return x
    out = foo()
    assert out == 6


def test_fallback_sum_with_x_tuple_n_default():
    """
    Feature: JIT Fallback
    Description: Test sum() in graph mode with input x tuple and input n default.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = sum((1, 2, 3))
        return x
    out = foo()
    assert out == 6


def test_fallback_sum_with_x_numpy_array_n_default():
    """
    Feature: JIT Fallback
    Description: Test sum() in graph mode with input x numpy array and input n default.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = sum(np.array([1, 2, 3]))
        return Tensor(x)
    out = foo()
    assert out.asnumpy() == 6


def test_fallback_sum_with_x_tensor_n_default():
    """
    Feature: JIT Fallback
    Description: Test sum() in graph mode with input x tensor and input n default.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = sum(Tensor([1, 2, 3]))
        return x
    out = foo()
    assert out.asnumpy() == 6


def test_fallback_sum_with_x_numpy_array_n_default_2():
    """
    Feature: JIT Fallback
    Description: Test sum() in graph mode with input x numpy array and input n default.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = sum(np.array([[1, 1], [2, 2]]))
        return Tensor(x)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    out = foo()
    assert np.allclose(out.asnumpy(), np.array([3, 3]))
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


def test_fallback_sum_with_x_list_n_not_default():
    """
    Feature: JIT Fallback
    Description: Test sum() in graph mode with input x list and input n not default.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = sum([1, 2, 3], 10)
        return x
    out = foo()
    assert out == 16


def test_fallback_sum_with_x_tensor_n_not_default_1():
    """
    Feature: JIT Fallback
    Description: Test sum() in graph mode with input x tensor and input n not default.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = sum(Tensor([1, 2, 3]), 10)
        return x
    out = foo()
    assert out == 16


def test_fallback_sum_with_x_tensor_n_not_default_2():
    """
    Feature: JIT Fallback
    Description: Test sum() in graph mode with input x tensor and input n not default.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = sum(Tensor([[1, 2], [3, 4]]), [5, 6])
        return x

    out = foo()
    assert np.allclose(out.asnumpy(), np.array([9, 12]))


def test_fallback_sum_with_x_tuple_n_not_default():
    """
    Feature: JIT Fallback
    Description: Test sum() in graph mode with input x tuple and input n not default.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = sum((1, 2, 3), 10)
        return x
    out = foo()
    assert out == 16


def test_fallback_sum_with_x_numpy_array_n_not_default_1():
    """
    Feature: JIT Fallback
    Description: Test sum() in graph mode with input x numpy array and input n default.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = sum(np.array([[1, 1], [2, 2]]), 5)
        return Tensor(x)
    out = foo()
    assert np.allclose(out.asnumpy(), np.array([8, 8]))


def test_fallback_sum_with_x_numpy_array_n_not_default_2():
    """
    Feature: JIT Fallback
    Description: Test sum() in graph mode with input x numpy array and input n default.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = sum(np.array([[1, 1], [2, 2]]), [3, 4])
        return Tensor(x)

    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    out = foo()
    assert np.allclose(out.asnumpy(), np.array([6, 7]))
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


def test_fallback_sum_with_x_not_iterable_error():
    """
    Feature: JIT Fallback
    Description: Test sum() in graph mode with input x not iterable.
    Expectation: TypeError.
    """
    @jit
    def foo():
        x = sum(1)
        return x
    with pytest.raises(TypeError) as ex:
        foo()
    assert "object is not iterable" in str(ex.value)
