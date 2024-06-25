# Copyright 2023-2024 Huawei Technologies Co., Ltd
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
"""Test graph list inplace operation"""
import pytest
import numpy as np

import mindspore.nn as nn
from mindspore import context, ops
from mindspore import Tensor, jit, jit_class
from mindspore.common import mutable
from mindspore.ops.operations import _sequence_ops as seq
from collections import deque
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)

global_list_1 = [1, 2, 3, 4]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_global_list_used_in_graph():
    """
    Feature: Enable list inplace operation
    Description: List passed as global should not change the python obj
    Expectation: No exception.
    """
    @jit
    def foo():
        return global_list_1

    res = foo()
    assert id(res) == id(global_list_1)


global_float_list_1 = [1.0, 2.0, 3.0, 4.0]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_global_list_used_in_graph_2():
    """
    Feature: Enable list inplace operation
    Description: List passed as global should not change the python obj
    Expectation: No exception.
    """
    @jit
    def foo():
        return global_float_list_1

    res = foo()
    assert id(res) == id(global_float_list_1)


global_numpy_list = [np.array([1, 2, 3]), np.array([4, 5, 6])]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_global_numpy_list_used_in_graph():
    """
    Feature: Enable list inplace operation
    Description: List passed as global should not change the python obj
    Expectation: No exception.
    """
    @jit
    def foo():
        return global_numpy_list

    res = foo()
    assert id(res) == id(global_numpy_list)


global_list_2 = [1, 2, 3, 4, [3, 4], None]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_global_nested_list_getitem_in_graph():
    """
    Feature: Enable list inplace operation
    Description: List passed as global should not change the python obj
    Expectation: No exception.
    """
    @jit
    def foo():
        return global_list_2[4]

    res = foo()
    assert id(res) == id(global_list_2[4])


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_global_nested_list_return_in_graph():
    """
    Feature: Enable list inplace operation
    Description: List passed as global should not change the python obj
    Expectation: No exception.
    """
    @jit
    def foo():
        return global_list_2

    res = foo()
    assert id(res) == id(global_list_2)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_global_nested_list_return_in_graph_2():
    """
    Feature: Enable list inplace operation
    Description: List passed as global should not change the python obj
    Expectation: No exception.
    """
    @jit
    def foo():
        return global_list_2, global_list_2[4]

    res = foo()
    assert len(res) == 2
    assert id(res[0]) == id(global_list_2)
    assert id(res[1]) == id(global_list_2[4])


global_list_3 = [1, 2, 3, (4, [3, 4])]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_global_nested_list_getitem_in_graph_2():
    """
    Feature: Enable list inplace operation
    Description: List passed as global should not change the python obj
    Expectation: No exception.
    """
    @jit
    def foo():
        return global_list_3[3][1]

    res = foo()
    assert id(res) == id(global_list_3[3][1])


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_global_nested_list_return_in_graph_3():
    """
    Feature: Enable list inplace operation
    Description: List passed as global should not change the python obj
    Expectation: No exception.
    """
    @jit
    def foo():
        return global_list_3, global_list_3[3][1]

    res = foo()
    assert len(res) == 2
    assert id(res[0]) == id(global_list_3)
    assert id(res[1]) == id(global_list_3[3][1])


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_local_list():
    """
    Feature: Enable list inplace operation
    Description: List create inside graph should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit
    def foo(a, b):
        x = [1, 2, 3, a, b]
        return x

    input_a = Tensor([1])
    input_b = 2
    ret = foo(input_a, input_b)
    assert isinstance(ret, list)
    assert ret == [1, 2, 3, Tensor([1]), 2]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_local_list_2():
    """
    Feature: Enable list inplace operation
    Description: List create inside graph should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit
    def foo(a, b):
        x = [1, 2, 3, a, b]
        return x

    input_a = Tensor([1])
    input_b = [1, 2]
    ret = foo(input_a, input_b)
    assert isinstance(ret, list)
    assert ret == [1, 2, 3, Tensor([1]), [1, 2]]


global_list_4 = [1, 2]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_local_list_3():
    """
    Feature: Enable list inplace operation
    Description: List create inside graph should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit
    def foo(a):
        x = [1, 2, 3, a, global_list_4]
        return x

    input_a = Tensor([1])
    ret = foo(input_a)
    assert isinstance(ret, list)
    assert ret == [1, 2, 3, Tensor([1]), [1, 2]]
    assert id(ret[4]) == id(global_list_4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_local_list_4():
    """
    Feature: Enable list inplace operation
    Description: List create inside graph should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit
    def foo(a, b):
        x = [1.0, 2, 3.0, a, b]
        return x

    input_a = Tensor([1])
    input_b = 2
    ret = foo(input_a, input_b)
    assert isinstance(ret, list)
    assert ret == [1.0, 2, 3.0, Tensor([1]), 2]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_return_local_list_5():
    """
    Feature: Enable list inplace operation
    Description: List create inside graph should be converted to PyExecute.
    Expectation: No exception.
    """
    @jit
    def foo(a, b):
        x = [1.0, 2, 3.0, a, b]
        return (x,)

    input_a = Tensor([1])
    input_b = 2
    ret = foo(input_a, input_b)
    assert isinstance(ret, tuple)
    assert len(ret) == 1
    assert isinstance(ret[0], list)
    assert ret[0] == [1.0, 2, 3.0, Tensor([1]), 2]


@pytest.mark.skip(reason="No need to convert to PyExecute node. SequenceMul execute fail in Ascend.")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_local_sequence_used_in_graph_with_operator():
    """
    Feature: Enable list inplace operation
    Description: List create inside graph should be converted to PyExecute, should be used as input to ops correctly.
    Expectation: No exception.
    """
    seq_func = seq.SequenceMul()

    @jit
    def foo(x, y):
        list_input = [x, y]
        return seq_func(list_input, 2)

    res = foo(Tensor([1]), Tensor([2]))
    assert isinstance(res, list)
    assert res == [Tensor([1]), Tensor([2]), Tensor([1]), Tensor([2])]


global_list_for_reverse = [1, 2, 3, 4]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_inplace_reverse():
    """
    Feature: list reverse.
    Description: support list reverse.
    Expectation: No exception.
    """
    @jit
    def foo():
        global_list_for_reverse.reverse()
        return global_list_for_reverse

    out = foo()
    assert id(out) == id(global_list_for_reverse)
    assert out == [4, 3, 2, 1]


global_list_for_reverse_2 = [Tensor([1, 2, 3]), Tensor([1, 2])]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_inplace_reverse_2():
    """
    Feature: list reverse.
    Description: support list reverse.
    Expectation: No exception.
    """
    @jit
    def foo():
        global_list_for_reverse_2.reverse()
        return global_list_for_reverse_2

    out = foo()
    assert id(out) == id(global_list_for_reverse_2)
    assert isinstance(out, list)
    assert len(out) == 2
    assert np.all(out[0].asnumpy() == np.array([1, 2]))
    assert np.all(out[1].asnumpy() == np.array([1, 2, 3]))


global_list_for_reverse_3 = ["1", np.array([1, 2, 3]), [1, 2], Tensor([1, 2, 3])]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_inplace_reverse_3():
    """
    Feature: list reverse.
    Description: support list reverse.
    Expectation: No exception.
    """
    @jit
    def foo():
        global_list_for_reverse_3.reverse()
        return global_list_for_reverse_3

    out = foo()
    assert id(out) == id(global_list_for_reverse_3)
    assert isinstance(out, list)
    assert len(out) == 4
    assert np.all(out[0].asnumpy() == np.array([1, 2, 3]))
    assert out[1] == [1, 2]
    assert np.all(out[2] == np.array([1, 2, 3]))
    assert out[3] == "1"


global_list_for_reverse_4 = [[1, 2], [3, 4]]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_inplace_reverse_element():
    """
    Feature: list reverse.
    Description: support list reverse.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = global_list_for_reverse_4[0]
        x.reverse()
        return x

    out = foo()
    assert id(out) == id(global_list_for_reverse_4[0])
    assert out == [2, 1]
    assert global_list_for_reverse_4 == [[2, 1], [3, 4]]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_inplace_reverse_local_list():
    """
    Feature: list reverse.
    Description: support list reverse.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = [[1, 2, 3], [4, 5]]
        y = x[0]
        y.reverse()
        return x, y

    out = foo()
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert out[0] == [[3, 2, 1], [4, 5]]
    assert out[1] == [3, 2, 1]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_inplace_reverse_local_list_2():
    """
    Feature: list reverse.
    Description: support list reverse.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = [[1, 2, 3], [4, 5]]
        y = x[0]
        y.reverse()
        return x

    out = foo()
    assert out == [[3, 2, 1], [4, 5]]


global_list_for_pop = [1, 2, 3, 4]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_inplace_pop():
    """
    Feature: list pop.
    Description: support list reverse.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = global_list_for_pop.pop()
        return x, global_list_for_pop

    out = foo()
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert out[0] == 4
    assert out[1] == [1, 2, 3]
    assert id(out[1]) == id(global_list_for_pop)


global_list_for_pop_2 = [1, [2, 3], "4"]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_inplace_pop_2():
    """
    Feature: list pop.
    Description: support list reverse.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = global_list_for_pop_2.pop()
        return x, global_list_for_pop_2

    out = foo()
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert out[0] == "4"
    assert out[1] == [1, [2, 3]]
    assert id(out[1]) == id(global_list_for_pop_2)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_inplace_pop_3():
    """
    Feature: list pop.
    Description: support list reverse.
    Expectation: No exception.
    """
    class ListNet(nn.Cell):
        def __init__(self, obj):
            super().__init__()
            self.obj = obj

        def construct(self):
            y = self.obj.pop()
            self.obj.pop(1)
            z = self.obj.pop(-1)
            return self.obj, y, z

    obj = [1, 2, Tensor([3]), "x", (3, 4, 5)]
    x, y, z = ListNet(obj)()
    assert id(obj) == id(x)
    assert y == (3, 4, 5)
    assert z == 'x'


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_inplace_pop_local():
    """
    Feature: list pop.
    Description: support list reverse.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = [[1, 2, 3], 4, 5]
        y = x[0]
        z = y.pop()
        return x, y, z

    out = foo()
    assert isinstance(out, tuple)
    assert len(out) == 3
    assert out[0] == [[1, 2], 4, 5]
    assert out[1] == [1, 2]
    assert out[2] == 3


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_inplace_pop_local_2():
    """
    Feature: list pop.
    Description: support list reverse.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = ([1, 2, 3], 4, 5)
        y = x[0]
        z = y.pop()
        return x, y, z

    out = foo()
    assert isinstance(out, tuple)
    assert len(out) == 3
    assert out[0] == ([1, 2], 4, 5)
    assert out[1] == [1, 2]
    assert out[2] == 3


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_inplace_pop_local_3():
    """
    Feature: list pop.
    Description: support list reverse.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = [[1, 2, 3], 4, 5]
        y = x[0]
        y.pop()
        return x

    out = foo()
    assert out == [[1, 2], 4, 5]


global_list_for_pop_extend = [1, 2, 3, 4]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_inplace_extend():
    """
    Feature: list extend.
    Description: support list reverse.
    Expectation: No exception.
    """
    @jit
    def foo():
        global_list_for_pop_extend.extend([1, 2, 3])
        return global_list_for_pop_extend

    out = foo()
    assert out == [1, 2, 3, 4, 1, 2, 3]
    assert id(out) == id(global_list_for_pop_extend)


global_list_for_pop_extend_2 = [1, 2, 3, 4]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_inplace_extend_no_return():
    """
    Feature: list extend.
    Description: support list reverse.
    Expectation: No exception.
    """
    @jit
    def foo():
        global_list_for_pop_extend_2.extend([1, 2, 3])

    foo()
    assert global_list_for_pop_extend_2 == [1, 2, 3, 4, 1, 2, 3]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_inplace_extend_local_list():
    """
    Feature: list extend.
    Description: support list reverse.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = [0, [1, 2, 3], 4, 5]
        y = x[1]
        y.extend(("a", 1))
        return x, y

    out = foo()
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert out[0] == [0, [1, 2, 3, "a", 1], 4, 5]
    assert out[1] == [1, 2, 3, "a", 1]


global_list_for_pop_insert = [1, 2, 3, 4]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_inplace_insert():
    """
    Feature: list insert.
    Description: support list reverse.
    Expectation: No exception.
    """
    @jit
    def foo():
        global_list_for_pop_insert.insert(-2, [1, 2, 3])
        return global_list_for_pop_insert

    out = foo()
    assert out == [1, 2, [1, 2, 3], 3, 4]
    assert id(out) == id(global_list_for_pop_insert)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_inplace_insert_local_list():
    """
    Feature: list extend.
    Description: support list reverse.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = [0, [1, 2, 3], 4, 5]
        y = x[1]
        y.insert(0, ("a", "b"))
        return x, y

    out = foo()
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert out[0] == [0, [("a", "b"), 1, 2, 3], 4, 5]
    assert out[1] == [("a", "b"), 1, 2, 3]


@pytest.mark.skip(reason="When index input is variable, insert will run dynamic shape ListInsert operator.")
@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_inplace_insert_local_list_2():
    """
    Feature: list extend.
    Description: support list reverse.
    Expectation: No exception.
    """
    @jit
    def foo(a):
        x = [0, [1, 2, 3], 4, 5]
        y = x[1]
        y.insert(a, ("a", "b"))
        return x, y

    out = foo(mutable(0))
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert out[0] == [0, [("a", "b"), 1, 2, 3], 4, 5]
    assert out[1] == [("a", "b"), 1, 2, 3]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_inplace_reverse_with_variable():
    """
    Feature: list inplace ops.
    Description: support list inplace ops.
    Expectation: No exception.
    """
    @jit
    def foo(a, b):
        x = [a, b]
        x.reverse()
        return x

    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    out = foo(a, b)
    assert isinstance(out, list)
    assert len(out) == 2
    assert np.all(out[0].asnumpy() == np.array([4, 5, 6]))
    assert np.all(out[1].asnumpy() == np.array([1, 2, 3]))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_inplace_reverse_with_variable_2():
    """
    Feature: list inplace ops.
    Description: support list inplace ops.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = mutable([1, 2, 3])
        x.reverse()
        return x

    out = foo()
    assert out == [3, 2, 1]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_inplace_reverse_with_variable_3():
    """
    Feature: list inplace ops.
    Description: support list inplace ops.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        x.reverse()
        return x

    x = mutable([1, 2, 3])
    out = foo(x)
    assert out == [3, 2, 1]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_len_list_inplace_op():
    """
    Feature: dynamic length list do not run inplace operation.
    Description: support list inplace ops.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = [2, 5, 6, 7]
        y = mutable(2)
        x = mutable(x, True)
        x[y] = 1
        return x

    out = foo()
    assert out == [2, 5, 1, 7]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_len_list_inplace_op_2():
    """
    Feature: dynamic length list do not run inplace operation.
    Description: support list inplace ops.
    Expectation: No exception.
    """
    @jit
    def foo(a):
        x = [1, 2, 3, 4]
        y = x[a::]
        return y

    input_index = Tensor(2)
    out = foo(input_index)
    assert out == [3, 4]


global_list_all_str = ['a', 'b', 'c']


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_inplace_with_all_str():
    """
    Feature: dynamic length list do not run inplace operation.
    Description: support list inplace ops.
    Expectation: No exception.
    """
    @jit
    def foo():
        global_list_all_str.extend(['d', 'e'])
        return global_list_all_str

    out = foo()
    assert id(out) == id(global_list_all_str)
    assert global_list_all_str == ['a', 'b', 'c', 'd', 'e']


global_tuple_with_list_all_str = (['a', 'b', 'c'], 1, 2)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_inplace_with_all_str_2():
    """
    Feature: dynamic length list do not run inplace operation.
    Description: support list inplace ops.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = global_tuple_with_list_all_str[0]
        x.extend(['d', 'e'])
        return x

    out = foo()
    assert id(global_tuple_with_list_all_str[0]) == id(out)
    assert global_tuple_with_list_all_str == (['a', 'b', 'c', 'd', 'e'], 1, 2)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_in_joined_str():
    """
    Feature: dynamic length list do not run inplace operation.
    Description: support list inplace ops.
    Expectation: No exception.
    """
    @jit
    def foo(x, y, z):
        if z >= 1:
            raise TypeError(f"The input is {x}")
        raise TypeError(f"The input is {y}")

    x = mutable([1, 2, 3])
    y = mutable([4, 5])
    z = Tensor([1])
    with pytest.raises(TypeError) as raise_info:
        foo(x, y, z)
    assert "The input is [1, 2, 3]" in str(raise_info.value)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_in_joined_str_2():
    """
    Feature: dynamic length list do not run inplace operation.
    Description: support list inplace ops.
    Expectation: No exception.
    """
    @jit
    def foo(a):
        x = [a, a+1]
        raise TypeError(f"The input is {x}")

    with pytest.raises(TypeError) as raise_info:
        foo(Tensor([1]))
    assert "The input is [Tensor(shape=[1], dtype=Int64, value= [1]), Tensor(shape=[1], dtype=Int64, value= [2])]"\
           in str(raise_info.value)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_in_joined_str_3():
    """
    Feature: dynamic length list do not run inplace operation.
    Description: support list inplace ops.
    Expectation: No exception.
    """
    @jit
    def foo(a):
        x = mutable((a, [a, a+1]))
        raise TypeError(f"The input is {x}")

    with pytest.raises(TypeError) as raise_info:
        foo(Tensor([1]))
    assert "The input is (Tensor(shape=[1], dtype=Int64, value= [1]), [Tensor(shape=[1], dtype=Int64, value= [1])," \
           " Tensor(shape=[1], dtype=Int64, value= [2])])" in str(raise_info.value)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_in_joined_str_4():
    """
    Feature: dynamic length list do not run inplace operation.
    Description: support list inplace ops.
    Expectation: No exception.
    """
    @jit
    def foo(a):
        x = mutable([a, [a, a+1]])
        raise TypeError(f"The input is {x}")

    with pytest.raises(TypeError) as raise_info:
        foo(Tensor([1]))
    assert "The input is [Tensor(shape=[1], dtype=Int64, value= [1]), [Tensor(shape=[1], dtype=Int64, value= [1])," \
           " Tensor(shape=[1], dtype=Int64, value= [2])]]" in str(raise_info.value)


@pytest.mark.skip(reason="PyExecute handle list user data failed.")
@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_in_joined_str_5():
    """
    Feature: dynamic length list do not run inplace operation.
    Description: support list inplace ops.
    Expectation: No exception.
    """
    @jit
    def foo(a):
        x = [a, a+1]
        raise TypeError(f"{x}")

    with pytest.raises(TypeError) as raise_info:
        foo(Tensor([1]))
    assert "[Tensor(shape=[1], dtype=Int64, value= [1]), Tensor(shape=[1], dtype=Int64, value= [2])]" \
           in str(raise_info.value)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_inplace_with_graph_input():
    """
    Feature: Enable list used as graph input do inplace operation.
    Description: support list inplace ops.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        return x

    x = [1, 2, 3, 4]
    ret = foo(x)
    assert id(x) == id(ret)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_inplace_with_graph_input_2():
    """
    Feature: Enable list used as graph input do inplace operation.
    Description: support list inplace ops.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        return x

    x = ([1, 2, 3, 4], 2, 3)
    ret = foo(x)
    assert id(x[0]) == id(ret[0])


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_inplace_with_graph_input_3():
    """
    Feature: Enable list used as graph input do inplace operation.
    Description: support list inplace ops.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        x.reverse()
        return x

    x = mutable([1, 2, 3, 4])
    ret = foo(x)
    assert id(x) != id(ret)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_inplace_with_graph_input_4():
    """
    Feature: Enable list used as graph input do inplace operation.
    Description: support list inplace ops.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        x.reverse()
        return x

    x = [1, 2, 3, 4]
    ret = foo(x)
    assert id(ret) == id(x)
    assert x == [4, 3, 2, 1]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_inplace_with_graph_input_5():
    """
    Feature: Enable list used as graph input do inplace operation.
    Description: support list inplace ops.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        x.reverse()

    x = [1, 2, 3, 4]
    foo(x)
    assert x == [4, 3, 2, 1]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_inplace_with_graph_input_6():
    """
    Feature: Enable list used as graph input do inplace operation.
    Description: support list inplace ops.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        y = x[0]
        y.reverse()
        return y

    x = ([1, 2, 3, 4], 2, 3)
    ret = foo(x)
    assert id(x[0]) == id(ret)
    assert x == ([4, 3, 2, 1], 2, 3)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_inplace_with_attribute():
    """
    Feature: Enable list used as graph input do inplace operation.
    Description: support list inplace ops.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def __init__(self, x):
            super(Net, self).__init__()
            self.x = x

        def construct(self):
            return self.x

    x = [1, 2, 3, 4]
    net = Net(x)
    ret = net()
    assert id(x) == id(ret)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_inplace_with_attribute_2():
    """
    Feature: Enable list used as graph input do inplace operation.
    Description: support list inplace ops.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def __init__(self, x):
            super(Net, self).__init__()
            self.x = x

        def construct(self):
            self.x.reverse()
            return self.x

    x = [1, 2, 3, 4]
    net = Net(x)
    ret = net()
    assert ret == [4, 3, 2, 1]
    assert id(x) == id(ret)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_inplace_with_attribute_3():
    """
    Feature: Enable list used as graph input do inplace operation.
    Description: support list inplace ops.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def __init__(self, x):
            super(Net, self).__init__()
            self.x = x

        def construct(self):
            return self.x[0]

    x = ([1, 2, 3, 4], 5, 6)
    net = Net(x)
    ret = net()
    assert id(x[0]) == id(ret)


@pytest.mark.skip(reason="SetItem with AbstractAny input can not convert to pyexecute")
@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_inplace_with_attribute_4():
    """
    Feature: Enable list used as graph input do inplace operation.
    Description: support list inplace ops.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def __init__(self, x):
            super(Net, self).__init__()
            self.x = x

        def construct(self):
            self.x[0].reverse()
            return self.x[0]

    x = ([1, 2, 3, 4], 5, 6)
    net = Net(x)
    ret = net()
    assert id(x[0]) == id(ret)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_inplace_with_attribute_of_jit_class():
    """
    Feature: Enable list used as graph input do inplace operation.
    Description: support list inplace ops.
    Expectation: No exception.
    """
    @jit_class
    class AttrClass():
        def __init__(self, x):
            self.attr = x

    class Net(nn.Cell):
        def __init__(self, x):
            super(Net, self).__init__()
            self.x = x

        def construct(self):
            return self.x.attr

    x = [1, 2, 3, 4]
    obj = AttrClass(x)
    net = Net(obj)
    ret = net()
    assert id(x) == id(ret)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_inplace_with_attribute_of_jit_class_2():
    """
    Feature: Enable list used as graph input do inplace operation.
    Description: support list inplace ops.
    Expectation: No exception.
    """
    @jit_class
    class AttrClass():
        def __init__(self, x):
            self.attr = x

    class Net(nn.Cell):
        def __init__(self, x):
            super(Net, self).__init__()
            self.x = x

        def construct(self):
            self.x.attr.reverse()

    x = [1, 2, 3, 4]
    obj = AttrClass(x)
    net = Net(obj)
    net()
    assert x == [4, 3, 2, 1]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_inplace_with_attribute_of_jit_class_3():
    """
    Feature: Enable list used as graph input do inplace operation.
    Description: support list inplace ops.
    Expectation: No exception.
    """
    @jit_class
    class AttrClass():
        def __init__(self, x):
            self.attr = x

    class Net(nn.Cell):
        def __init__(self, x):
            super(Net, self).__init__()
            self.x = x

        def construct(self):
            self.x.attr.reverse()
            return self.x.attr

    x = [1, 2, 3, 4]
    obj = AttrClass(x)
    net = Net(obj)
    ret = net()
    assert ret == [4, 3, 2, 1]
    assert id(x) == id(ret)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_inplace_with_attribute_of_jit_class_4():
    """
    Feature: Enable list used as graph input do inplace operation.
    Description: support list inplace ops.
    Expectation: No exception.
    """
    class ListNet(nn.Cell):
        def __init__(self, obj):
            super().__init__()
            self.obj = obj

        def construct(self):
            y = self.obj.pop()
            self.obj.pop(1)
            z = self.obj.pop(-1)
            return self.obj, y, z

    obj = [1, 2, Tensor([3]), "x", (3, 4, 5)]
    x, y, _ = ListNet(obj)()
    assert id(obj) == id(x)
    assert y == (3, 4, 5)


@pytest.mark.skip(reason="The pop operation is not converted to inplace operation")
@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_inplace_with_attribute_of_jit_class_5():
    """
    Feature: Enable list used as graph input do inplace operation.
    Description: support list inplace ops.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, x):
            super(Net, self).__init__()
            self.x = x

        def construct(self):
            a = self.x.pop()
            return a, self.x

    x = [1, 2, 3, 4]
    net = Net(x)
    ret1, ret2 = net()
    assert ret1 == 4
    assert ret2 == [1, 2, 3]
    assert id(ret2) == id(x)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_inplace_with_stub_tensor():
    """
    Feature: Enable list used as graph input do inplace operation.
    Description: support list inplace ops.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self, x):
            super(Net, self).__init__()
            self.x = x

        @jit
        def construct(self, index):
            return self.x[index]

    context.set_context(mode=context.PYNATIVE_MODE)
    x = [ops.add(Tensor(1), 1), ops.add(Tensor(2), 2)]
    index = Tensor(0)
    net = Net(x)
    ret = net(index)
    assert ret == Tensor(2)
    context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_inplace_with_any_input():
    """
    Feature: Enable list used as graph input do inplace operation.
    Description: support list inplace ops.
    Expectation: No exception.
    """
    @jit
    def foo(m):
        x = [[2], [3], [4], [5]]
        x.pop()
        x.extend(m[Tensor([1])])
        return x

    ret = foo([[1], [2], [3], [4]])
    assert ret == [[2], [3], [4], 2]


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_inplace_with_any_input_2():
    """
    Feature: Enable list used as graph input do inplace operation.
    Description: support list inplace ops.
    Expectation: No exception.
    """
    @jit
    def foo(m):
        x = [[2], [3], [4], [5]]
        while x[1] < [4]:
            x.extend([[2]])
            x.insert(2, [6])
            x.reverse()
            x.pop()
            x.extend(m[Tensor(1)])
        return x

    ret = foo([[1], [2], [3], [4]])
    assert ret == [[2], [5], [4], [6], [3], 2]


empty_global_list_1 = []


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_inplace_with_empty_list():
    """
    Feature: Enable list used as graph input do inplace operation.
    Description: support list inplace ops.
    Expectation: No exception.
    """
    @jit
    def foo():
        return empty_global_list_1

    ret = foo()
    assert ret == []
    assert id(ret) == id(empty_global_list_1)


empty_global_list_2 = []


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_inplace_with_empty_list_2():
    """
    Feature: Enable list used as graph input do inplace operation.
    Description: support list inplace ops.
    Expectation: No exception.
    """
    @jit
    def foo():
        empty_global_list_2.extend([3, 4])
        return empty_global_list_2

    ret = foo()
    assert ret == [3, 4]
    assert id(ret) == id(empty_global_list_2)


empty_global_list_3 = []


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_inplace_with_empty_list_3():
    """
    Feature: Enable list used as graph input do inplace operation.
    Description: support list inplace ops.
    Expectation: No exception.
    """
    @jit
    def foo():
        for i in range(3):
            empty_global_list_3.extend([i,])
        return empty_global_list_3

    ret = foo()
    assert ret == [0, 1, 2]
    assert id(ret) == id(empty_global_list_3)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_inplace_with_same_value_list():
    """
    Feature: Enable list used as graph input do inplace operation.
    Description: support list inplace ops.
    Expectation: No exception.
    """
    class InnerNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.y = [1, 2, 3]

        def construct(self, z):
            z.reverse()
            return z

    net = InnerNet()

    @jit
    def list_func(z):
        y = net.y
        y.reverse()
        return net.y, net(z)

    z = [1, 2, 3]
    ret1, ret2 = list_func(z)
    assert id(ret1) == id(net.y)
    assert id(ret2) == id(z)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_inplace_mixed():
    """
    Feature: Enable list used as graph input do inplace operation.
    Description: support list inplace ops.
    Expectation: No exception.
    """
    @jit
    def foo(input1, input2):
        x1 = [[1], [2], [3], [4]]
        for i in range(1, len(x1)):
            y = x1[Tensor([i])]
            y.extend([4])
            x1.insert(1, [5])
            x1.reverse()
            z = x1[input1]
            z.extend(input2[i])
            x1.pop()
        return x1

    input1 = Tensor([2])
    input2 = [Tensor([1]), Tensor([2]), Tensor([3]), Tensor([4])]
    out = foo(input1, input2)
    exp = [[5, 4], [3, Tensor(3)], [2, 4, Tensor(2), 4, Tensor(4)], [5]]
    for i in list(range(len(out))):
        assert out[i] == exp[i]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_inplace_with_attribute_get_twice():
    """
    Feature: Enable list used as graph input do inplace operation.
    Description: support list inplace ops.
    Expectation: No exception.
    """
    class InnerNet():
        def __init__(self):
            self.x = [Tensor([1]), Tensor([2]), Tensor([3])]

    class Net(nn.Cell):
        "Fallback network."
        def __init__(self):
            super(Net, self).__init__()
            obj = InnerNet()
            self.x = obj.x

        def construct(self, y):
            self.x.extend((y,))
            return ops.addn(self.x)

    y = Tensor([5])
    net = Net()
    output = net(y)
    assert output == 11


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_deque_list_cycle():
    """
    Feature: support list deque operation, the graph is correct and does not form cycle.
    Description: support list deque operation.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def construct(self, inputs):
            inputs = inputs.asnumpy()
            inputs = Tensor(inputs)
            queue = deque([inputs])
            outputs = []
            while queue:
                x = queue.popleft()
                if isinstance(x, list):
                    queue.extend(x)
                elif isinstance(x, Tensor):
                    outputs.append(x)
            return outputs

    net = Net()
    x = Tensor([2, 3, 4])
    out = net(x)
    assert (out == x).all()
