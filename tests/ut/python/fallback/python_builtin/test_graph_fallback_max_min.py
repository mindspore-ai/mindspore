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
""" test graph fallback buildin python function max and min"""
import operator
import numpy as np
from mindspore import ms_function, context, Tensor

context.set_context(mode=context.GRAPH_MODE)

def test_fallback_max_with_one_input_list():
    """
    Feature: JIT Fallback
    Description: Test max() in graph mode with one input list.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = max([1, 2, 3])
        return x
    out = foo()
    assert out == 3


def test_fallback_max_with_one_input_list_2():
    """
    Feature: JIT Fallback
    Description: Test max() in graph mode with one input list.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = max([(1, 2), (1, 3), (3, 4)])
        return x
    out = foo()
    assert operator.eq(out, (3, 4))


def test_fallback_max_with_one_input_tuple():
    """
    Feature: JIT Fallback
    Description: Test max() in graph mode with one input tuple.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = max((1, 2, 3))
        return x
    out = foo()
    assert out == 3


def test_fallback_max_with_one_input_tuple_2():
    """
    Feature: JIT Fallback
    Description: Test max() in graph mode with one input tuple.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = max((1, 2), (1, 3), (3, 4))
        return x
    out = foo()
    assert operator.eq(out, (3, 4))


def test_fallback_max_with_one_input_dict():
    """
    Feature: JIT Fallback
    Description: Test max() in graph mode with one input dict.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = max({1: 'a', 2: 'b', 3: 'c'})
        return x
    out = foo()
    assert out == 3


def test_fallback_max_with_one_input_numpy_array():
    """
    Feature: JIT Fallback
    Description: Test max() in graph mode with one input numpy array.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = max(np.array([1, 2, 3]))
        return Tensor(x)
    out = foo()
    assert out == 3


def test_fallback_max_with_one_input_tensor():
    """
    Feature: JIT Fallback
    Description: Test max() in graph mode with one input tensor.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = max(Tensor([1, 2, 3]))
        return x
    out = foo()
    assert out == 3


def test_fallback_max_with_two_inputs_list():
    """
    Feature: JIT Fallback
    Description: Test max() in graph mode with two inputs list.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = max([1, 2, 3], [4, 5])
        return x
    out = foo()
    assert operator.eq(out, (4, 5))


def test_fallback_min_with_two_inputs_list():
    """
    Feature: JIT Fallback
    Description: Test min() in graph mode with two inputs list.
    Expectation: No exception.
    """
    @ms_function
    def foo():
        x = min([1, 2, 3], [4, 5])
        return x
    out = foo()
    assert operator.eq(out, (1, 2, 3))
