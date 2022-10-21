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
""" test graph fallback buildin python function str"""
import pytest
import numpy as np
from mindspore import jit, context, Tensor

context.set_context(mode=context.GRAPH_MODE)


def test_fallback_str_with_input_tensor():
    """
    Feature: JIT Fallback
    Description: Test str() in graph mode with tensor input.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        return str(x)
    with pytest.raises(TypeError) as ex:
        foo(Tensor([1, 2, 4]))
    assert "str() does not support non-constant input." in str(ex.value)


def test_fallback_str_with_input_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test str() in graph mode with tensor input.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = Tensor([10])
        return str(x)

    assert foo() == "Tensor(shape=[1], dtype=Int64, value=[10])"


def test_fallback_str_with_input_scalar():
    """
    Feature: JIT Fallback
    Description: Test str() in graph mode with scalar input.
    Expectation: No exception.
    """
    @jit
    def foo():
        return str(10.0)

    assert foo() == "10.0"


def test_fallback_str_with_input_list():
    """
    Feature: JIT Fallback
    Description: Test str() in graph mode with list input.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = [1, 2, 3]
        return str(x)

    assert foo() == "[1, 2, 3]"


def test_fallback_str_with_input_string():
    """
    Feature: JIT Fallback
    Description: Test str() in graph mode with string input.
    Expectation: No exception.
    """
    @jit
    def foo():
        return str("test")

    assert foo() == "test"


def test_fallback_str_with_input_numpy():
    """
    Feature: JIT Fallback
    Description: Test str() in graph mode with numpy input.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = np.array([1, 2, 3])
        return str(x)

    assert foo() == "[1 2 3]"


def test_fallback_str_with_no_input():
    """
    Feature: JIT Fallback
    Description: Test str() in graph mode with no input.
    Expectation: No exception.
    """
    @jit
    def foo():
        return str()

    assert foo() == ""


def test_fallback_str_with_type_input():
    """
    Feature: JIT Fallback
    Description: Test str() in graph mode with type input.
    Expectation: No exception.
    """
    @jit
    def foo():
        return str(int)

    assert foo() == "<class 'int'>"
