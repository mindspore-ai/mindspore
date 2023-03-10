# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
""" test list pop operation """
import pytest
import numpy as np
import mindspore as ms
from mindspore import jit, context, Tensor, nn

context.set_context(mode=context.GRAPH_MODE)


def test_list_pop_1():
    """
    Feature: list pop.
    Description: support list pop.
    Expectation: No exception.
    """
    @jit
    def list_pop():
        x = [1, 2, 3, 4]
        y = x.pop()
        return x, y

    res_x, res_y = list_pop()
    assert np.all(res_x == [1, 2, 3])
    assert res_y == 4


def test_list_pop_2():
    """
    Feature: list pop.
    Description: support list pop.
    Expectation: No exception.
    """
    @jit
    def list_pop():
        x = [1, 2, 3, 4]
        y = x.pop(-2)
        return x, y

    res_x, res_y = list_pop()
    assert np.all(res_x == [1, 2, 4])
    assert res_y == 3


def test_list_pop_3():
    """
    Feature: list pop.
    Description: support list pop.
    Expectation: No exception.
    """
    @jit
    def list_pop():
        x = [1, 2, 3, 4]
        y = x.pop(1)
        return x, y

    res_x, res_y = list_pop()
    assert np.all(res_x == [1, 3, 4])
    assert res_y == 2


def test_list_pop_4():
    """
    Feature: list pop.
    Description: support list pop.
    Expectation: No exception.
    """
    @jit
    def list_pop():
        x = [1, 2, 3, 4]
        y = x.pop(4)
        return x, y

    with pytest.raises(IndexError, match="The pop index out of range."):
        res_x, res_y = list_pop()
        print("res_x:", res_x)
        print("res_y:", res_y)


def test_list_pop_5():
    """
    Feature: list pop.
    Description: support list pop.
    Expectation: No exception.
    """
    @jit
    def list_insert():
        x = [1, 2, 3, 4]
        y = x.pop(-5)
        return x, y

    with pytest.raises(IndexError, match="The pop index out of range."):
        res_x, res_y = list_insert()
        print("res_x:", res_x)
        print("res_y:", res_y)


def test_list_pop_6():
    """
    Feature: list pop.
    Description: support list pop.
    Expectation: No exception.
    """
    @jit
    def list_insert():
        x = []
        y = x.pop()
        return x, y

    with pytest.raises(IndexError, match="The pop index out of range."):
        res_x, res_y = list_insert()
        print("res_x:", res_x)
        print("res_y:", res_y)


def test_list_pop_7():
    """
    Feature: list pop.
    Description: support list pop.
    Expectation: No exception.
    """
    @jit
    def list_pop():
        x1 = [1, 2, 3, 4]
        x2 = [5, 6, 7, 8]
        y1 = x1.pop(1)
        y2 = x2.pop(2)
        return x1, x2, y1 + y2

    res_x1, res_x2, res_y = list_pop()
    assert np.all(res_x1 == [1, 3, 4])
    assert np.all(res_x2 == [5, 6, 8])
    assert res_y == 9


def test_list_pop_8():
    """
    Feature: list pop.
    Description: support list pop.
    Expectation: No exception.
    """
    @jit
    def list_pop(index):
        x = [Tensor([1]), Tensor([2]), Tensor([3])]
        y = x.pop(index)
        return x, y

    res_x, res_y = list_pop(2)
    assert res_x == [Tensor([1]), Tensor([2])]
    assert res_y == Tensor([3])


def test_list_pop_9():
    """
    Feature: list pop.
    Description: support list pop.
    Expectation: No exception.
    """
    @jit
    def list_pop(x, index):
        y = x.pop(index)
        return x, y

    input_x = [Tensor([1]), Tensor([2]), Tensor([3])]
    res_x, res_y = list_pop(input_x, 2)
    assert res_x == [Tensor([1]), Tensor([2])]
    assert res_y == Tensor([3])


def test_list_pop_type_error():
    """
    Feature: list pop.
    Description: support list pop.
    Expectation: No exception.
    """
    @jit
    def list_pop():
        x = [1, 2, 3]
        x.pop(1.0)
        return x

    with pytest.raises(TypeError) as error_info:
        res = list_pop()
        print("res:", res)
    assert "Integer argument expected, but got FP32Imm type value: 1.000000" in str(error_info)


def test_list_pop_no_return():
    """
    Feature: list pop has no return.
    Description: support list pop.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            shp1 = x.shape
            shp1 = list(shp1)
            shp1.insert(2, 3)
            shp1.pop()
            return shp1

    net = Net()
    np1 = np.random.randint(6, size=(2, 4, 3, 4, 5))
    data1 = Tensor(np1, dtype=ms.float32)
    out = net(data1)
    assert out == [2, 4, 3, 3, 4]
