# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
""" test graph fallback """
import numpy as np
from mindspore import ms_function, context, Tensor

context.set_context(mode=context.GRAPH_MODE)


def test_fallback_all_tuple_number():
    """
    Feature: JIT Fallback
    Description: Test all(Tuple) in graph mode
    Expectation: No exception
    """

    @ms_function
    def foo():
        x = (0, 1, 2, 3)
        y = (1, 1)
        return all(x), all(y)

    x, y = foo()
    assert (not x) and y


def test_fallback_all_tuple_string():
    """
    Feature: JIT Fallback
    Description: Test all(Tuple) in graph mode
    Expectation: No exception
    """

    @ms_function
    def foo():
        x = ('a', 'b', '', 'd')
        y = ('a', 'b', 'c', 'd')
        return all(x), all(y)

    x, y = foo()
    assert (not x) and y


def test_fallback_all_list_number():
    """
    Feature: JIT Fallback
    Description: Test all(List) in graph mode
    Expectation: No exception
    """

    @ms_function
    def foo():
        x = [0, 1, 2, 3]
        y = [1, 1]
        return all(x), all(y)

    x, y = foo()
    assert (not x) and y


def test_fallback_all_list_string():
    """
    Feature: JIT Fallback
    Description: Test all(List) in graph mode
    Expectation: No exception
    """

    @ms_function
    def foo():
        x = ['a', 'b', '', 'd']
        y = ['a', 'b', 'c', 'd']
        return all(x), all(y)

    x, y = foo()
    assert (not x) and y


def test_fallback_all_dict():
    """
    Feature: JIT Fallback
    Description: Test all(dict) in graph mode
    Expectation: No exception
    """

    @ms_function
    def foo():
        x = {"": 1, "2": 2}
        y = {"1": 1, "2": 2}
        return all(x), all(y)

    x, y = foo()
    assert (not x) and y


def test_fallback_all_numpy():
    """
    Feature: JIT Fallback
    Description: Test all(numpy.array) in graph mode
    Expectation: No exception
    """

    @ms_function
    def foo():
        x = np.array([0, 1, 2, 3])
        y = np.array([1, 1])
        return all(x), all(y)

    x, y = foo()
    assert (not x) and y


def test_fallback_all_tensor_constant():
    """
    Feature: JIT Fallback
    Description: Test all(Tensor) with a constant tensor in graph mode
    Expectation: No exception
    """

    @ms_function
    def foo():
        x = Tensor(np.array([0, 1, 2, 3]))
        y = Tensor(np.array([1, 1]))
        return all(x), all(y)

    x, y = foo()
    assert (not x) and y


def test_fallback_any_tuple_number():
    """
    Feature: JIT Fallback
    Description: Test any(Tuple) in graph mode
    Expectation: No exception
    """

    @ms_function
    def foo():
        x = (0, 0, 0, 0)
        y = (1, 0)
        return any(x), any(y)

    x, y = foo()
    assert (not x) and y


def test_fallback_any_tuple_string():
    """
    Feature: JIT Fallback
    Description: Test any(Tuple) in graph mode
    Expectation: No exception
    """

    @ms_function
    def foo():
        x = ('a', 'b', '', 'd')
        y = ('a', 'b', 'c', 'd')
        return any(x), any(y)

    x, y = foo()
    assert x and y


def test_fallback_any_list_number():
    """
    Feature: JIT Fallback
    Description: Test any(List) in graph mode
    Expectation: No exception
    """

    @ms_function
    def foo():
        x = [0, 0, 0, 0]
        y = [1, 0]
        return any(x), any(y)

    x, y = foo()
    assert (not x) and y


def test_fallback_any_list_string():
    """
    Feature: JIT Fallback
    Description: Test any(List) in graph mode
    Expectation: No exception
    """

    @ms_function
    def foo():
        x = ['', '', '', '']
        y = ['a', 'b', '', 'd']
        return any(x), any(y)

    x, y = foo()
    assert (not x) and y


def test_fallback_any_dict():
    """
    Feature: JIT Fallback
    Description: Test any(dict) in graph mode
    Expectation: No exception
    """

    @ms_function
    def foo():
        x = {"": 1}
        y = {"1": 1, "2": 2}
        return any(x), any(y)

    x, y = foo()
    assert (not x) and y


def test_fallback_any_numpy():
    """
    Feature: JIT Fallback
    Description: Test any(numpy.array) in graph mode
    Expectation: No exception
    """

    @ms_function
    def foo():
        x = np.array([0, 0, 0])
        y = np.array([1, 0])
        return any(x), any(y)

    x, y = foo()
    assert (not x) and y


def test_fallback_any_tensor_constant():
    """
    Feature: JIT Fallback
    Description: Test any(Tensor) with a constant tensor in graph mode
    Expectation: No exception
    """

    @ms_function
    def foo():
        x = Tensor(np.array([0, 0, 0]))
        y = Tensor(np.array([1, 0]))
        return any(x), any(y)

    x, y = foo()
    assert (not x) and y
