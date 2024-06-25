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
"""test graph list comprehension"""
import pytest
import itertools
import numpy as np

from mindspore import Tensor, jit, context
from mindspore.ops import operations as P
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_list_comprehension_with_variable_tensor():
    """
    Feature: Graph isinstance.
    Description: Graph list comprehension syntax with variable input.
    Expectation: No exception.
    """

    @jit
    def foo(a):
        x = [i + 1 for i in a]
        return x

    res = foo(Tensor([1, 2, 3]))
    assert len(res) == 3
    assert res[0] == 2
    assert res[1] == 3
    assert res[2] == 4


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comprehension_with_variable_dict():
    """
    Feature: Graph isinstance.
    Description: Graph list comprehension syntax with variable input.
    Expectation: No exception.
    """

    @jit
    def foo(a):
        m = {"1": a, "2": a+1, "3": a-1}
        x = [m[i]+1 for i in m if i != "1"]
        return x

    res = foo(Tensor([1]))
    assert len(res) == 2
    assert res[0] == 3
    assert res[1] == 1


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_list_comprehension_with_variable_input():
    """
    Feature: Graph isinstance.
    Description: Graph list comprehension syntax with variable input.
    Expectation: No exception.
    """

    @jit
    def foo(a):
        x = [a for i in range(3)]
        return x

    res = foo(Tensor([1, 2, 3]))
    assert len(res) == 3
    assert np.all(res[0].asnumpy() == np.array([1, 2, 3]))
    assert np.all(res[1].asnumpy() == np.array([1, 2, 3]))
    assert np.all(res[2].asnumpy() == np.array([1, 2, 3]))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_list_comprehension_with_variable_input_2():
    """
    Feature: Graph isinstance.
    Description: Graph list comprehension syntax with variable input.
    Expectation: No exception.
    """

    @jit
    def foo(a):
        x = [a + i for i in range(3)]
        return x

    res = foo(Tensor([1, 2, 3]))
    assert len(res) == 3
    assert np.all(res[0].asnumpy() == np.array([1, 2, 3]))
    assert np.all(res[1].asnumpy() == np.array([2, 3, 4]))
    assert np.all(res[2].asnumpy() == np.array([3, 4, 5]))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_list_comprehension_with_variable_input_3():
    """
    Feature: Graph isinstance.
    Description: Graph list comprehension syntax with variable input.
    Expectation: No exception.
    """

    @jit
    def foo(a):
        a = a + 10
        x = [a + i for i in range(3)]
        return x

    res = foo(Tensor([1, 2, 3]))
    assert len(res) == 3
    assert np.all(res[0].asnumpy() == np.array([11, 12, 13]))
    assert np.all(res[1].asnumpy() == np.array([12, 13, 14]))
    assert np.all(res[2].asnumpy() == np.array([13, 14, 15]))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_list_comprehension_with_variable_input_and_condition():
    """
    Feature: Graph isinstance.
    Description: Graph list comprehension syntax with variable input and condition.
    Expectation: No exception.
    """

    @jit
    def foo(a):
        x = [a for i in range(5) if i%2 == 0]
        return x

    res = foo(Tensor([1, 2, 3]))
    assert len(res) == 3
    assert np.all(res[0].asnumpy() == np.array([1, 2, 3]))
    assert np.all(res[1].asnumpy() == np.array([1, 2, 3]))
    assert np.all(res[2].asnumpy() == np.array([1, 2, 3]))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_list_comprehension_with_variable_input_and_condition_2():
    """
    Feature: Graph isinstance.
    Description: Graph list comprehension syntax with variable input and condition.
    Expectation: No exception.
    """

    @jit
    def foo(a):
        x = [a + i for i in range(5) if i%2 == 0]
        return x

    res = foo(Tensor([1, 2, 3]))
    assert len(res) == 3
    assert np.all(res[0].asnumpy() == np.array([1, 2, 3]))
    assert np.all(res[1].asnumpy() == np.array([3, 4, 5]))
    assert np.all(res[2].asnumpy() == np.array([5, 6, 7]))


@pytest.mark.skip(reason="Join error msg change")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_list_comprehension_with_variable_input_and_condition_3():
    """
    Feature: Graph isinstance.
    Description: Graph list comprehension syntax with variable input and condition.
    Expectation: RuntimeError.
    """

    @jit
    def foo(a):
        x = [a + i for i in range(5) if P.ReduceSum()(a + i) > 10]
        return x

    with pytest.raises(RuntimeError) as raise_info:
        foo(Tensor([1, 2, 3]))
    assert "Cannot join the return values of different branches" in str(raise_info.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comprehension_with_iterator_input():
    """
    Feature: Graph syntax list comp.
    Description: Graph list comprehension syntax.
    Expectation: No exception.
    """

    @jit
    def foo():
        m = (1, 2)
        n = (4, 5)
        x = [i for i in itertools.product(m, n)]
        return x

    res = foo()
    assert res == [(1, 4), (1, 5), (2, 4), (2, 5)]


@pytest.mark.skip(reason="AbstractAny cause dynamic length list exist")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_comprehension_with_iterator_input_2():
    """
    Feature: Graph syntax list comp.
    Description: Graph list comprehension syntax.
    Expectation: No exception.
    """

    @jit
    def foo(a, b, c, d):
        m = (a, b)
        n = (c, d)
        x = [i for i in itertools.product(m, n)]
        return x

    foo(Tensor([1]), Tensor([2]), Tensor([3]), Tensor([4]))
