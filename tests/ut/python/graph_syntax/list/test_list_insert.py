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
""" test list insert operation """
import pytest
import numpy as np
from mindspore import jit, context, Tensor

context.set_context(mode=context.GRAPH_MODE)


def test_list_insert_1():
    """
    Feature: list insert.
    Description: support list insert.
    Expectation: No exception.
    """
    @jit
    def list_insert():
        x = [1, 3, 4]
        x.insert(0, 2)
        return Tensor(x)

    assert np.all(list_insert().asnumpy() == np.array([2, 1, 3, 4]))


def test_list_insert_2():
    """
    Feature: list insert.
    Description: support list insert.
    Expectation: No exception.
    """
    @jit
    def list_insert():
        x = [1, 3, 4]
        x.insert(5, 2)
        return Tensor(x)

    assert np.all(list_insert().asnumpy() == np.array([1, 3, 4, 2]))


def test_list_insert_3():
    """
    Feature: list insert.
    Description: support list insert.
    Expectation: No exception.
    """
    @jit
    def list_insert():
        x = [1, 3, 4]
        x.insert(-1, 2)
        return Tensor(x)

    assert np.all(list_insert().asnumpy() == np.array([1, 3, 2, 4]))


def test_list_insert_4():
    """
    Feature: list insert.
    Description: support list insert.
    Expectation: No exception.
    """
    @jit
    def list_insert():
        x = [1, 3, 4]
        x.insert(-5, 2)
        return Tensor(x)

    assert np.all(list_insert().asnumpy() == np.array([2, 1, 3, 4]))


def test_list_insert_5():
    """
    Feature: list insert.
    Description: support list insert.
    Expectation: No exception.
    """
    @jit
    def list_insert(x):
        x.insert(-5, 2)
        x.insert(-2, 9)
        return x

    input_x = [Tensor([1]), Tensor([3]), Tensor([4])]
    res = list_insert(input_x)
    assert np.all(res == np.array((2, Tensor([1]), 9, Tensor([3]), Tensor([4]))))


def test_list_insert_pop_1():
    """
    Feature: list insert and pop.
    Description: support list insert and pop.
    Expectation: No exception.
    """
    @jit
    def list_insert_pop(x):
        x.insert(-5, 2)
        x.insert(-2, 9)
        y = x.pop()
        z = x.pop(-2)
        return x, y, z

    input_x = [Tensor([1]), Tensor([3]), Tensor([4])]
    res_x, res_y, res_z = list_insert_pop(input_x)
    assert np.all(res_x == np.array((2, Tensor([1]), Tensor([3]))))
    assert res_y == Tensor([4])
    assert res_z == 9


def test_list_insert_pop_2():
    """
    Feature: list insert.
    Description: support list insert.
    Expectation: No exception.
    """
    @jit
    def list_insert_pop(index):
        x = [1, 3, 4]
        y = x.pop(index)
        x.insert(0, y)
        return x, y

    res_x, res_y = list_insert_pop(-2)
    assert np.all(res_x == [3, 1, 4])
    assert res_y == 3


def test_list_insert_pop_append_1():
    """
    Feature: list insert, pop and append.
    Description: support list insert, pop and append.
    Expectation: No exception.
    """
    @jit
    def list_insert_pop_append(x):
        x.insert(-5, 2)
        x.insert(-2, 9)
        y = x.pop()
        x.append(10)
        z = x.pop(-2)
        x.append(5)
        return x, y, z

    input_x = [Tensor([1]), Tensor([3]), Tensor([5])]
    res_x, res_y, res_z = list_insert_pop_append(input_x)
    assert np.all(res_x == np.array((2, Tensor([1]), 9, 10, 5)))
    assert res_y == Tensor([5])
    assert res_z == 3


def test_list_insert_type_error():
    """
    Feature: list insert.
    Description: support list insert.
    Expectation: No exception.
    """
    @jit
    def list_insert():
        x = [1, 2, 3]
        x.insert(1.0, 9)
        return x

    with pytest.raises(TypeError) as error_info:
        res = list_insert()
        print("res:", res)
    assert "Integer argument expected, but got FP32Imm type value: 1.000000" in str(error_info)
