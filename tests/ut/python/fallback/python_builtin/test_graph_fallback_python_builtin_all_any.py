# Copyright 2021 Huawei Technologies Co., Ltd
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
import pytest

from mindspore import ms_function, context, Tensor

context.set_context(mode=context.GRAPH_MODE)


def test_fallback_all_tuple():
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


def test_fallback_all_list():
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


def test_fallback_all_tensor():
    """
    Feature: JIT Fallback
    Description: Test all(Tensor) in graph mode
    Expectation: No exception
    """

    @ms_function
    def foo():
        return all(Tensor(np.array([0, 1, 2, 3]))), all(Tensor(np.array([1, 1])))

    x, y = foo()
    assert (not x) and y


@pytest.mark.skip("Not support yet should convert C++ Tensor to python")
def test_fallback_all_tensor_construct():
    """
    Feature: JIT Fallback
    Description: Test all(numpy.array) in graph mode
    Expectation: No exception
    """

    @ms_function
    def foo():
        x = Tensor(np.array([0, 1, 2, 3]))
        y = Tensor(np.array([1, 1]))
        return all(x), all(y)

    x, y = foo()
    assert (not x) and not y


def test_fallback_any_tuple():
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


def test_fallback_any_list():
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


def test_fallback_any_tensor():
    """
    Feature: JIT Fallback
    Description: Test all(Tensor) in graph mode
    Expectation: No exception
    """

    @ms_function
    def foo():
        return any(Tensor(np.array([0, 0]))), any(Tensor(np.array([1, 0])))

    x, y = foo()
    assert (not x) and y


@pytest.mark.skip("Not support yet should convert C++ Tensor to python")
def test_fallback_any_tensor_construct():
    """
    Feature: JIT Fallback
    Description: Test all(Tensor) in graph mode
    Expectation: No exception
    """

    @ms_function
    def foo():
        x = Tensor(np.array([0, 0, 0]))
        y = Tensor(np.array([1, 0]))
        return any(x), any(y)

    x, y = foo()
    assert (not x) and not y
