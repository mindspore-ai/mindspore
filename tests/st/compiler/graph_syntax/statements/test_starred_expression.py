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
""" test graph starred expression. """
import pytest
from mindspore import context, jit, Tensor, nn
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_starred_expression_assign_list_input():
    """
    Feature: Support starred expression.
    Description: Support starred expression in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = [1, 2, 3, 4]
        a = *x,  # pylint: disable=R1707
        return a

    ret = foo()
    assert ret == (1, 2, 3, 4)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_assign_list_input():
    """
    Feature: Support assign list.
    Description: Support assign in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = [1, 2, 3, 4]
        a = x,  # pylint: disable=R1707
        return a

    ret = foo()
    assert ret == ([1, 2, 3, 4],)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_starred_expression_assign_tuple_input():
    """
    Feature: Support starred expression.
    Description: Support starred expression in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = (1, 2, 3, 4)
        a = *x,  # pylint: disable=R1707
        return a

    ret = foo()
    assert ret == (1, 2, 3, 4)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_starred_expression_assign_dict_input():
    """
    Feature: Support starred expression.
    Description: Support starred expression in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = {"a": 1, "b": 2}
        out = *x,  # pylint: disable=R1707
        return out

    ret = foo()
    assert ret == ("a", "b")


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_starred_expression_assign_string_input():
    """
    Feature: Support starred expression.
    Description: Support starred expression in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = "abcde"
        out = *x,  # pylint: disable=R1707
        return out

    ret = foo()
    assert ret == ('a', 'b', 'c', 'd', 'e')


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_starred_expression_assign_tensor_input():
    """
    Feature: Support starred expression.
    Description: Support starred expression in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        out = *x,  # pylint: disable=R1707
        return out

    ret = foo(Tensor([1, 2, 3, 4]))
    assert ret == (Tensor(1), Tensor(2), Tensor(3), Tensor(4))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_starred_expression_in_format_string():
    """
    Feature: Support starred expression.
    Description: Support starred expression in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        return f"output is {*x,}"

    ret = foo([1, 2, 3, 4])
    assert ret == "output is (1, 2, 3, 4)"


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_starred_expression_target_tuple():
    """
    Feature: Support starred expression.
    Description: Support starred expression in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = (1, 2, 3, 4)
        *b, = x
        return b

    ret = foo()
    assert ret == [1, 2, 3, 4]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_starred_expression_target_list():
    """
    Feature: Support starred expression.
    Description: Support starred expression in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = [1, 2, 3, 4]
        *b, = x
        return b

    ret = foo()
    assert ret == [1, 2, 3, 4]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_starred_expression_target_dict():
    """
    Feature: Support starred expression.
    Description: Support starred expression in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = {"a": 1, "b": 2}
        *b, = x
        return b

    ret = foo()
    assert ret == ['a', 'b']


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_starred_expression_target_string():
    """
    Feature: Support starred expression.
    Description: Support starred expression in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = "abcde"
        *b, = x
        return b

    ret = foo()
    assert ret == ['a', 'b', 'c', 'd', 'e']


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_starred_expression_target_tensor():
    """
    Feature: Support starred expression.
    Description: Support starred expression in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        *b, c = x
        return b, c

    ret = foo(Tensor([1, 2, 3, 4]))
    assert ret[0] == [Tensor(1), Tensor(2), Tensor(3)]
    assert ret[1] == Tensor(4)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_starred_expression_target_nested_tuple():
    """
    Feature: Support starred expression.
    Description: Support starred expression in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = (1, 2, 3, 4)
        y = (5, 6)
        a, b, *c = x, y
        return a, b, c

    ret = foo()
    assert ret == ((1, 2, 3, 4), (5, 6), [])


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_starred_expression_target_list_2():
    """
    Feature: Support starred expression.
    Description: Support starred expression in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = [1, 2, 3, 4]
        a, *b = x
        return a, b

    ret = foo()
    assert len(ret) == 2
    assert ret[0] == 1
    assert ret[1] == [2, 3, 4]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_starred_expression_target_tuple_2():
    """
    Feature: Support starred expression.
    Description: Support starred expression in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = (1, 2, 3, 4)
        a, *b, c = x
        return a, b, c

    ret = foo()
    assert len(ret) == 3
    assert ret[0] == 1
    assert ret[1] == [2, 3]
    assert ret[2] == 4


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_starred_expression_target_list_tuple():
    """
    Feature: Support starred expression.
    Description: Support starred expression in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = [1, 2, 3, 4]
        y = [5, 6]
        z = (7, 8)
        a, *b = x, y, z
        return a, b

    ret = foo()
    assert len(ret) == 2
    assert ret[0] == [1, 2, 3, 4]
    assert ret[1] == [[5, 6], (7, 8)]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_starred_expression_with_range():
    """
    Feature: Support starred expression.
    Description: Support starred expression in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo():
        a, *b, c = range(5)
        return a, b, c

    ret = foo()
    assert ret == (0, [1, 2, 3], 4)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_starred_expression_for_in():
    """
    Feature: Support starred expression.
    Description: Support starred expression in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo():
        ret = []
        for _, *b in [(1, 2, 3), (4, 5, 6, 7)]:
            ret.append(b)
        return ret

    ret = foo()
    assert ret == [[2, 3], [5, 6, 7]]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_starred_expression_assign_tuple():
    """
    Feature: Support starred expression.
    Description: Support starred expression in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = *[1, 2], *(3, 4), (5, 6)
        return a

    ret = foo()
    assert ret == (1, 2, 3, 4, (5, 6))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_starred_expression_range_tuple():
    """
    Feature: Support starred expression.
    Description: Support starred expression in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = *range(4), 4
        return a

    ret = foo()
    assert ret == (0, 1, 2, 3, 4)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_starred_expression_range_list():
    """
    Feature: Support starred expression.
    Description: Support starred expression in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = [*range(4), 4]
        return a

    ret = foo()
    assert ret == [0, 1, 2, 3, 4]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_starred_expression_return_tuple():
    """
    Feature: Support starred expression.
    Description: Support starred expression in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = (*[1], *[2], 3)
        return a

    ret = foo()
    assert ret == (1, 2, 3)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_starred_expression_dict():
    """
    Feature: Support starred expression.
    Description: Support starred expression in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = {'x': 1, **{'y': 2}}
        return a

    ret = foo()
    assert len(ret) == 2
    assert ret == {'x': 1, 'y': 2}


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_starred_expression_dict_2():
    """
    Feature: Support starred expression.
    Description: Support starred expression in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = {'x': 1, **{'y': 2}, "w": 4, **{'z': 3}}
        return a

    ret = foo()
    assert ret == {'x': 1, 'y': 2, 'w': 4, 'z': 3}


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_starred_expression_dict_3():
    """
    Feature: Support starred expression.
    Description: Support starred expression in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = {'x': 1, **{'y': 2, 'z': 3, **{'w': 4}}}
        return a

    ret = foo()
    assert ret == {'x': 1, 'y': 2, 'z': 3, 'w': 4}


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_starred_expression_dict_4():
    """
    Feature: Support starred expression.
    Description: Support starred expression in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = {'x': 1, 'y': {'z': 3, **{'w': 4}}}
        return a

    ret = foo()
    assert ret == {'x': 1, 'y': {'z': 3, 'w': 4}}


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_starred_expression_dict_key_deduplicate():
    """
    Feature: Support starred expression.
    Description: Support starred expression in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = {'x': 1, **{'x': 2}}
        return a

    ret = foo()
    assert ret == {'x': 2}


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_starred_expression_dict_non_literal():
    """
    Feature: Support starred expression.
    Description: Support starred expression in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, kwarg):
            return {**kwarg, "a": 3, ("d", "e"): 4}

    net = Net()
    arg_dict = {"a": 1, "b": 2}
    out = net(arg_dict)
    assert out == {"a": 3, "b": 2, ("d", "e"): 4}
