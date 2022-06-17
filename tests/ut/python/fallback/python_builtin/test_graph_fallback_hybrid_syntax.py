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
""" test graph fallback hybrid syntax"""
import operator
import numpy as np
from mindspore import ms_function, context, Tensor

context.set_context(mode=context.GRAPH_MODE)


def test_fallback_len_with_numpy():
    """
    Feature: JIT Fallback
    Description: Test len in graph mode with numpy input.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = np.array([1, 2, 3, 4])
        return len(x)

    out = foo()
    assert out == 4


def test_fallback_enumerate_with_numpy():
    """
    Feature: JIT Fallback
    Description: Test enumerate in graph mode with numpy input.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = np.array([1, 2])
        y = enumerate(x)
        return tuple(y)

    out = foo()
    assert operator.eq(out, ((0, 1), (1, 2)))


def test_fallback_zip_with_numpy():
    """
    Feature: JIT Fallback
    Description: Test zip in graph mode with numpy input.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = np.array([1, 2])
        y = np.array([10, 20])
        ret = zip(x, y)
        return tuple(ret)

    out = foo()
    assert operator.eq(out, ((1, 10), (2, 20)))


def test_fallback_zip_with_numpy_and_tensor():
    """
    Feature: JIT Fallback
    Description: Test zip in graph mode with numpy and tensor input.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = np.array([1, 2])
        y = Tensor([10, 20])
        ret = zip(x, y)
        return tuple(ret)

    out = foo()
    assert operator.eq(out, ((1, 10), (2, 20)))


def test_fallback_map_with_numpy():
    """
    Feature: JIT Fallback
    Description: Test map in graph mode with numpy.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = np.array([1, 2, 3, 4])
        y = np.array([1, 1, 1, 1])
        ret = map(lambda x, y: x + y, x, y)
        return tuple(ret)

    out = foo()
    assert operator.eq(out, (2, 3, 4, 5))


def test_fallback_map_with_numpy_and_tensor():
    """
    Feature: JIT Fallback
    Description: Test map in graph mode with numpy.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = np.array([1, 2, 3, 4])
        y = Tensor([1, 1, 1, 1])
        ret = map(lambda x, y: x + y, x, y)
        return tuple(ret)

    out = foo()
    assert operator.eq(out, (2, 3, 4, 5))


def test_fallback_filter_with_numpy_and_tensor():
    """
    Feature: JIT Fallback
    Description: Test filter in graph mode with numpy.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = np.array([1, 2, 3, 4])
        ret = filter(lambda x: x > 2, x)
        return tuple(ret)

    out = foo()
    assert operator.eq(out, (3, 4))
