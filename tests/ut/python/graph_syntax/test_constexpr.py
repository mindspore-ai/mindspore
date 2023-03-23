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
""" test_dict_get """
import math
from mindspore import Tensor, jit, context, mutable
from mindspore.ops.primitive import constexpr
from mindspore.ops.primitive import _primexpr

context.set_context(mode=context.GRAPH_MODE)


@constexpr
def count_none(arg):
    if arg is None:
        raise ValueError("The arg is None")
    a = 0
    for e in arg:
        if e is None:
            a += 1
        elif isinstance(e, (tuple, list)) and None in e:
            a += 1
    return a


def test_constexpr_input_with_variable_element_tuple():
    """
    Feature: constexpr with variable element tuple input.
    Description: If tuple is used as constexpr input, the variable element will be converted to None.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        arg = (1, 2, x, x+1)
        return count_none(arg)

    out = foo(Tensor([1]))
    assert out == 2


def test_constexpr_input_with_variable_element_tuple_2():
    """
    Feature: constexpr with variable element tuple input.
    Description: If tuple is used as constexpr input, the variable element will be converted to None.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        arg = (1, 2, x, (x, 1, 2))
        return count_none(arg)

    out = foo(Tensor([1]))
    assert out == 2


def test_constexpr_input_with_variable_element_list():
    """
    Feature: constexpr with variable element list input.
    Description: If list is used as constexpr input, the variable element will be converted to None.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        arg = [1, 2, x, x+1]
        return count_none(arg)

    out = foo(Tensor([1]))
    assert out == 2


def test_constexpr_input_with_variable_element_list_2():
    """
    Feature: constexpr with variable element list input.
    Description: If list is used as constexpr input, the variable element will be converted to None.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        arg = [1, 2, x, [x, 1, 2]]
        return count_none(arg)

    out = foo(Tensor([1]))
    assert out == 2


def test_constexpr_input_with_mutable_list():
    """
    Feature: constexpr with mutable list.
    Description: If mutable list is used as constexpr input, all elements will be converted to None.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        arg = mutable([Tensor([1]), Tensor([2]), x])
        return count_none(arg)

    out = foo(Tensor([1]))
    assert out == 3


@_primexpr
def count_none_2(arg):
    if arg is None:
        raise ValueError("The arg is None")
    a = 0
    if not isinstance(a, (list, tuple)):
        if a is None:
            return 1
        return 0
    for e in arg:
        if e is None:
            a += 1
    return a


def test__primexpr_with_mutable_list():
    """
    Feature: constexpr with mutable list.
    Description: If mutable list is used as constexpr input, all elements will be converted to None.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        arg = mutable([Tensor([1]), Tensor([2]), x])
        return count_none_2(arg)

    out = foo(Tensor([1]))
    assert out == 0


def test__primexpr_with_mutable_scalar():
    """
    Feature: constexpr with mutable list.
    Description: If mutable list is used as constexpr input, all elements will be converted to None.
    Expectation: No exception.
    """
    @jit
    def foo():
        arg = mutable(2)
        return count_none_2(arg)

    out = foo()
    assert out == 0


def test_constexpr_with_float_input():
    """
    Feature: constexpr with float input.
    Description: Ensure constexpr with float input has correct value.
    Expectation: No exception.
    """
    @constexpr
    def calculate_floor(x, y):
        return math.floor(x * y)

    @jit
    def foo():
        return calculate_floor(0.9, 20)

    out = foo()
    assert out == 18


def test_constexpr_with_float_input_2():
    """
    Feature: constexpr with float input.
    Description: Ensure constexpr with float input has correct value.
    Expectation: No exception.
    """
    @constexpr
    def calculate_floor(x, y):
        return math.floor(x * y)

    @jit
    def foo(a, b):
        return calculate_floor(a, b)

    out = foo(0.9, 20)
    assert out == 18
