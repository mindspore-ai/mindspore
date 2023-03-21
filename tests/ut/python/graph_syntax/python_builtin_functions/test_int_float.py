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
""" test graph fallback buildin python function int and float"""
import math
import pytest
import numpy as np
from mindspore import jit, context, Tensor

context.set_context(mode=context.GRAPH_MODE)


def test_fallback_int_with_input_tensor():
    """
    Feature: JIT Fallback
    Description: Test int() in graph mode with tensor input.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        return int(x)
    with pytest.raises(ValueError):
        foo(Tensor([1, 2, 4]))


def test_fallback_int_with_input_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test int() in graph mode with tensor input.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = Tensor([10])
        return int(x)

    ret = foo()
    assert isinstance(ret, int)
    assert ret == 10


def test_fallback_int_with_input_tensor_3():
    """
    Feature: JIT Fallback
    Description: Test int() in graph mode with tensor input.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = Tensor([1, 2, 3])
        return int(x)

    with pytest.raises(ValueError) as ex:
        foo()
    assert "Only one element tensors can be" in str(ex.value)


def test_fallback_int_with_input_scalar():
    """
    Feature: JIT Fallback
    Description: Test int() in graph mode with scalar input.
    Expectation: No exception.
    """
    @jit
    def foo():
        return int(10.0)
    ret = foo()
    assert isinstance(ret, int)
    assert ret == 10


def test_fallback_int_with_input_int():
    """
    Feature: JIT Fallback
    Description: Test int() in graph mode with list input.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = [1, 2, 3]
        return int(x)

    with pytest.raises(TypeError) as ex:
        foo()
    assert "int() argument must be a string, a" in str(ex.value)


def test_fallback_int_with_input_string():
    """
    Feature: JIT Fallback
    Description: Test int() in graph mode with string input.
    Expectation: No exception.
    """
    @jit
    def foo():
        return int("1234")
    ret = foo()
    assert isinstance(ret, int)
    assert ret == 1234


def test_fallback_int_with_input_string_2():
    """
    Feature: JIT Fallback
    Description: Test int() in graph mode with string input.
    Expectation: No exception.
    """
    @jit
    def foo():
        return int("abcd")

    with pytest.raises(ValueError) as ex:
        foo()
    assert "invalid literal for int" in str(ex.value)


def test_fallback_int_with_input_numpy():
    """
    Feature: JIT Fallback
    Description: Test int() in graph mode with string input.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = np.array([1, 2, 3, 4])
        return int(x)

    with pytest.raises(TypeError) as ex:
        foo()
    assert "only size-1 arrays can be " in str(ex.value)


def test_fallback_int_with_input_numpy_2():
    """
    Feature: JIT Fallback
    Description: Test int() in graph mode with string input.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = np.array([1,])
        return int(x)

    ret = foo()
    assert isinstance(ret, int)
    assert ret == 1


def test_fallback_int_with_no_input():
    """
    Feature: JIT Fallback
    Description: Test int() in graph mode with string input.
    Expectation: No exception.
    """
    @jit
    def foo():
        return int()

    ret = foo()
    assert isinstance(ret, int)
    assert ret == 0


def test_fallback_int_with_base_input():
    """
    Feature: JIT Fallback
    Description: Test int() in graph mode with string input.
    Expectation: No exception.
    """
    @jit
    def foo():
        x1 = int('12', 16)
        x2 = int('0xa', 16)
        x3 = int('10', 8)
        return x1, x2, x3

    ret = foo()
    assert len(ret) == 3
    assert ret[0] == 18
    assert ret[1] == 10
    assert ret[2] == 8


def test_fallback_float_with_input_tensor():
    """
    Feature: JIT Fallback
    Description: Test float() in graph mode with tensor input.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = Tensor([1])
        return float(x)

    ret = foo()
    assert isinstance(ret, float)
    assert ret == 1.0


def test_fallback_float_with_input_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test int() in graph mode with tensor input.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        return float(x)
    with pytest.raises(ValueError):
        foo(Tensor([1, 2, 4]))


def test_fallback_float_with_input_tensor_3():
    """
    Feature: JIT Fallback
    Description: Test int() in graph mode with tensor input.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = Tensor([1, 2, 3])
        return float(x)

    with pytest.raises(ValueError) as ex:
        foo()
    assert "Only one element tensors can be" in str(ex.value)


def test_fallback_float_with_input_scalar():
    """
    Feature: JIT Fallback
    Description: Test float() in graph mode with scalar input.
    Expectation: No exception.
    """
    @jit
    def foo():
        return float(10.0)
    ret = foo()
    assert isinstance(ret, float)
    assert ret == 10.0


def test_fallback_float_with_input_float():
    """
    Feature: JIT Fallback
    Description: Test float() in graph mode with list input.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = [1, 2, 3]
        return float(x)

    with pytest.raises(TypeError) as ex:
        foo()
    assert "float() argument must be a string or a number" in str(ex.value)


def test_fallback_float_with_input_string():
    """
    Feature: JIT Fallback
    Description: Test int() in graph mode with string input.
    Expectation: No exception.
    """
    @jit
    def foo():
        return int("1234")
    ret = foo()
    assert isinstance(ret, int)
    assert ret == 1234.0


def test_fallback_float_with_input_string_2():
    """
    Feature : JIT Fallback
    Description: Test float(str) in graph mode.
    Expectation: No exception
    """

    @jit
    def foo():
        x1 = float("12.3")
        x2 = float("-12.3")
        x3 = float("1e-003")
        x4 = float("-1234\n")
        x5 = float("-Infinity")
        return x1, x2, x3, x4, x5

    x1, x2, x3, x4, x5 = foo()
    assert math.isclose(x1, 12.3, abs_tol=1e-5) \
           and math.isclose(x2, -12.3, abs_tol=1e-5) \
           and math.isclose(x3, 1e-003, abs_tol=1e-5) \
           and math.isclose(x4, -1234, abs_tol=1e-5) \
           and x5 == float("-Infinity")


def test_fallback_float_with_input_string_3():
    """
    Feature: JIT Fallback
    Description: Test float() in graph mode with string input.
    Expectation: No exception.
    """
    @jit
    def foo():
        return float("abcd")

    with pytest.raises(ValueError) as ex:
        foo()
    assert "could not convert string" in str(ex.value)


def test_fallback_float_with_input_numpy():
    """
    Feature: JIT Fallback
    Description: Test float() in graph mode with string input.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = np.array([1, 2, 3, 4])
        return float(x)

    with pytest.raises(TypeError) as ex:
        foo()
    assert "only size-1 arrays can be " in str(ex.value)


def test_fallback_float_with_input_numpy_2():
    """
    Feature: JIT Fallback
    Description: Test float() in graph mode with string input.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = np.array([1,])
        return float(x)

    ret = foo()
    assert isinstance(ret, float)
    assert ret == 1.0


def test_fallback_float_with_no_input():
    """
    Feature: JIT Fallback
    Description: Test float() in graph mode with string input.
    Expectation: No exception.
    """
    @jit
    def foo():
        return float()

    ret = foo()
    assert isinstance(ret, float)
    assert ret == 0.0
