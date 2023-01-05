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
""" test graph fallback buildin python function bool"""
import pytest
import numpy as np
import mindspore.nn as nn
from mindspore import ms_function, context, Tensor
context.set_context(mode=context.GRAPH_MODE)


def test_fallback_bool_with_input_tensor():
    """
    Feature: JIT Fallback
    Description: Test bool() in graph mode with tensor input.
    Expectation: No exception.
    """
    @ms_function
    def foo(x):
        return bool(x)
    with pytest.raises(ValueError) as ex:
        foo(Tensor([1, 2, 4]))
    assert "The truth value of an array with" in str(ex.value)


def test_fallback_bool_with_input_tensor_3():
    """
    Feature: JIT Fallback
    Description: Test bool() in graph mode with tensor input.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = Tensor([0])
        return bool(x)

    assert not foo()


def test_fallback_bool_with_input_tensor_4():
    """
    Feature: JIT Fallback
    Description: Test bool() in graph mode with tensor input.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = Tensor([1, 2, 3])
        return bool(x)

    with pytest.raises(ValueError) as ex:
        foo()
    assert "The truth value of an array with" in str(ex.value)


def test_fallback_bool_with_input_scalar():
    """
    Feature: JIT Fallback
    Description: Test bool() in graph mode with scalar input.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        return bool(10.0)

    assert foo()


def test_fallback_bool_with_input_list():
    """
    Feature: JIT Fallback
    Description: Test bool() in graph mode with list input.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = [1, 2, 3]
        return bool(x)

    assert foo()


def test_fallback_bool_with_input_list_2():
    """
    Feature: JIT Fallback
    Description: Test bool() in graph mode with list input.
    Expectation: No exception.
    """
    @ms_function
    def foo(a):
        x = [1, 2, 3, a]
        return bool(x)

    assert foo(Tensor([1, 2, 3]))


def test_fallback_bool_with_input_string():
    """
    Feature: JIT Fallback
    Description: Test bool() in graph mode with string input.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        return bool("test")

    assert foo()


def test_fallback_bool_with_input_string_2():
    """
    Feature: JIT Fallback
    Description: Test bool() in graph mode with string input.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        return bool("")

    assert not foo()


def test_fallback_bool_with_input_numpy():
    """
    Feature: JIT Fallback
    Description: Test bool() in graph mode with numpy input.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = np.array([1, 2, 3, 4])
        return bool(x)

    with pytest.raises(ValueError) as ex:
        foo()
    assert "The truth value of an array" in str(ex.value)


def test_fallback_bool_with_input_numpy_2():
    """
    Feature: JIT Fallback
    Description: Test bool() in graph mode with numpy input.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = np.array([1,])
        return bool(x)

    assert foo()


def test_fallback_bool_with_no_input():
    """
    Feature: JIT Fallback
    Description: Test bool() in graph mode with no input.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        return bool()

    assert not foo()


def test_fallback_bool_with_type_input():
    """
    Feature: JIT Fallback
    Description: Test bool() in graph mode with type input.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        return bool(int)

    assert foo()


class Net(nn.Cell):
    def construct(self):
        return 1


def test_bool_for_cell_object():
    """
    Feature: Bool function.
    Description: Test bool() for cell object input
    Expectation: No exception.
    """
    @ms_function
    def foo():
        net = Net()
        return bool(net)

    assert foo()


def test_bool_for_cell_object_2():
    """
    Feature: Bool function.
    Description: Test bool() for cell object input
    Expectation: No exception.
    """
    @ms_function
    def foo():
        net = Net()
        if net:
            return True
        return False

    assert foo()
