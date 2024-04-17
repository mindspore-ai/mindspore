# Copyright 2023 Huawei Technologies Co., Ltd
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
import numpy as np
from mindspore import Tensor, context

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


def test_return_constant_list():
    """
    Feature: Return list in graph
    Description: Support return constant list.
    Expectation: No exception.
    """
    def foo():
        return [1, 2, 3, 4]

    res = foo()
    assert res == [1, 2, 3, 4]


def test_return_constant_list_2():
    """
    Feature: Return list in graph
    Description: Support return constant list.
    Expectation: No exception.
    """
    def foo():
        return [True, False, False, True]

    res = foo()
    assert res == [True, False, False, True]


def test_return_constant_list_3():
    """
    Feature: Return list in graph
    Description: Support return constant list.
    Expectation: No exception.
    """
    def foo():
        return [Tensor([1]), Tensor([1, 2, 3]), Tensor([2, 3])]

    res = foo()
    assert len(res) == 3
    assert np.all(res[0].asnumpy() == np.array([1]))
    assert np.all(res[1].asnumpy() == np.array([1, 2, 3]))
    assert np.all(res[2].asnumpy() == np.array([2, 3]))


def test_return_make_list_node():
    """
    Feature: Return list in graph
    Description: Support return make list node.
    Expectation: No exception.
    """
    def foo(x):
        return [x, x+1, x+2, Tensor([4])]

    res = foo(Tensor([1]))
    assert res == [Tensor([1]), Tensor([2]), Tensor([3]), Tensor([4])]


def test_return_list_with_nest():
    """
    Feature: Return list in graph
    Description: Support return make list in nest scene.
    Expectation: No exception.
    """
    def foo():
        return [[1, 2, 3], [4, 5, 6]]

    res = foo()
    assert res == [[1, 2, 3], [4, 5, 6]]


def test_return_make_list_with_nest():
    """
    Feature: Return list in graph
    Description: Support return make list in nest scene.
    Expectation: No exception.
    """
    def foo(x):
        return [[x, x], (x+1, x+2)]

    res = foo(Tensor([0]))
    assert res == [[Tensor([0]), Tensor([0])], (Tensor([1]), Tensor([2]))]


def test_return_buildin_list_func():
    """
    Feature: Return list in graph
    Description: Support return result of list() function.
    Expectation: No exception.
    """
    def foo():
        return list((1, "2", None, Tensor([1])))

    res = foo()
    assert res == [1, "2", None, Tensor([1])]


def test_return_list_from_third_party():
    """
    Feature: Return list in graph
    Description: Support return list from third party.
    Expectation: No exception.
    """
    def foo():
        m = np.array([1, 2, 3, 4])
        x = m.tolist()
        return x

    res = foo()
    assert res == [1, 2, 3, 4]
