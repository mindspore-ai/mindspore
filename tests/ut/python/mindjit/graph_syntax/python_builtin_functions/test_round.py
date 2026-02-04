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
""" test graph fallback buildin python function round"""
import math
import pytest
from mindspore import jit, context

context.set_context(mode=context.GRAPH_MODE)


def test_fallback_round_with_x_int_n_default():
    """
    Feature: JIT Fallback
    Description: Test round() in graph mode with input x int and input n default.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = round(10)
        return x
    out = foo()
    assert isinstance(out, int)
    assert out == 10


def test_fallback_round_with_x_float_n_default():
    """
    Feature: JIT Fallback
    Description: Test round() in graph mode with input x float and input n default.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = round(10.123)
        return x
    out = foo()
    assert isinstance(out, int)
    assert out == 10


def test_fallback_round_with_x_float_n_default_2():
    """
    Feature: JIT Fallback
    Description: Test round() in graph mode with input x float and input n default.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = round(10.678)
        return x
    out = foo()
    assert isinstance(out, int)
    assert out == 11


def test_fallback_round_with_n_zero():
    """
    Feature: JIT Fallback
    Description: Test round() in graph mode with input n is zero.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = round(10.678, 0)
        return x
    out = foo()
    assert isinstance(out, float)
    assert math.isclose(out, 11, rel_tol=1e-5)


def test_fallback_round_with_n_none():
    """
    Feature: JIT Fallback
    Description: Test round() in graph mode with input n is None.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = round(10.678, None)
        return x
    out = foo()
    assert isinstance(out, int)
    assert out == 11


def test_fallback_round_with_n_positive_int():
    """
    Feature: JIT Fallback
    Description: Test round() in graph mode with input n is positive int.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = round(10.678, 1)
        return x
    out = foo()
    assert isinstance(out, float)
    assert math.isclose(out, 10.7, rel_tol=1e-5)


def test_fallback_round_with_n_negative_int():
    """
    Feature: JIT Fallback
    Description: Test round() in graph mode with input n is negative int.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = round(10.678, -1)
        return x
    out = foo()
    assert isinstance(out, float)
    assert math.isclose(out, 10, rel_tol=1e-5)


def test_fallback_round_with_n_negative_int_2():
    """
    Feature: JIT Fallback
    Description: Test round() in graph mode with input n is negative int.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = round(17.678, -1)
        return x
    out = foo()
    assert isinstance(out, float)
    assert math.isclose(out, 20, rel_tol=1e-5)


def test_fallback_round_with_input_x_not_number():
    """
    Feature: JIT Fallback
    Description: Test round() in graph mode with input x is not number.
    Expectation: TypeError.
    """
    @jit
    def foo():
        x = round([1, 2, 3])
        return x
    with pytest.raises(TypeError) as ex:
        foo()
    assert "type list doesn't define __round__ method" in str(ex.value)


def test_fallback_round_with_input_n_not_int():
    """
    Feature: JIT Fallback
    Description: Test round() in graph mode with input x is not int.
    Expectation: TypeError.
    """
    @jit
    def foo():
        x = round(10.123, 1.0)
        return x
    with pytest.raises(TypeError) as ex:
        foo()
    assert "cannot be interpreted as an integer" in str(ex.value)
