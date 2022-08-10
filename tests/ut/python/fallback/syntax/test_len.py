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
"""test graph is_instance"""
import numpy as np

from mindspore.common import mutable
from mindspore import Tensor, ms_function, context

context.set_context(mode=context.GRAPH_MODE)


def test_len_tensor():
    """
    Feature: Graph len syntax.
    Description: Graph syntax len support tensor.
    Expectation: No exception.
    """

    @ms_function
    def foo(x):
        return len(x)

    out = foo(Tensor([[1, 2, 3], [4, 5, 6]]))
    assert out == 2


def test_len_tensor_2():
    """
    Feature: Graph len syntax.
    Description: Graph syntax len support tensor.
    Expectation: No exception.
    """

    @ms_function
    def foo():
        x = Tensor([[1, 2, 3], [4, 5, 6]])
        return len(x)

    out = foo()
    assert out == 2


def test_len_tuple():
    """
    Feature: Graph len syntax.
    Description: Graph syntax len support tuple.
    Expectation: No exception.
    """

    @ms_function
    def foo():
        x = [1, 2, 3, 4]
        return len(x)

    out = foo()
    assert out == 4


def test_len_tuple_2():
    """
    Feature: Graph len syntax.
    Description: Graph syntax len support tuple.
    Expectation: No exception.
    """

    @ms_function
    def foo(x):
        a = [1, 2, 3, x, x]
        return len(a)

    out = foo(Tensor([1, 2, 3, 4]))
    assert out == 5


def test_len_tuple_3():
    """
    Feature: Graph len syntax.
    Description: Graph syntax len support tuple.
    Expectation: No exception.
    """

    @ms_function
    def foo(x):
        a = [1, 2, 3, x, np.array([1, 2, 3, 4])]
        return len(a)

    out = foo(Tensor([1, 2, 3, 4]))
    assert out == 5


def test_len_tuple_4():
    """
    Feature: Graph len syntax.
    Description: Graph syntax len support tuple.
    Expectation: No exception.
    """

    @ms_function
    def foo(x):
        return len(x)

    a = mutable((Tensor([1]), Tensor([2]), Tensor([3]), Tensor([4])))
    out = foo(a)
    assert out == 4


def test_len_list():
    """
    Feature: Graph len syntax.
    Description: Graph syntax len support list.
    Expectation: No exception.
    """

    @ms_function
    def foo():
        x = [1, 2, 3, 4]
        return len(x)

    out = foo()
    assert out == 4


def test_len_list_2():
    """
    Feature: Graph len syntax.
    Description: Graph syntax len support list.
    Expectation: No exception.
    """

    @ms_function
    def foo(x):
        return len(x)

    a = mutable((Tensor([1]), Tensor([2]), Tensor([3]), Tensor([4])))
    out = foo(a)
    assert out == 4


def test_len_dict():
    """
    Feature: Graph len syntax.
    Description: Graph syntax len support dict.
    Expectation: No exception.
    """

    @ms_function
    def foo():
        x = {"1": 1, "2": 2}
        return len(x)

    out = foo()
    assert out == 2


def test_len_numpy():
    """
    Feature: Graph len syntax.
    Description: Graph syntax len support numpy ndarray.
    Expectation: No exception.
    """

    @ms_function
    def foo():
        x = np.array([[1, 2, 3], [0, 0, 0]])
        return len(x)

    out = foo()
    assert out == 2
