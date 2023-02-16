# Copyright 2020 Huawei Technologies Co., Ltd
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
""" test_dictionary """
import numpy as np
import pytest

from mindspore import Tensor, context, jit
from mindspore.nn import Cell

context.set_context(mode=context.GRAPH_MODE)


class Net1(Cell):
    def construct(self, x):
        dic = {'x': 0, 'y': 1}
        output = []
        for i in dic.keys():
            output.append(i)
        for j in dic.values():
            output.append(j)
        return output


class Net2(Cell):
    def construct(self, x):
        dic = {'x': x, 'y': 1}
        output = []
        for i in dic.keys():
            output.append(i)
        for j in dic.values():
            output.append(j)
        return output


class Net3(Cell):
    def construct(self, x):
        dic = {'x': 0}
        dic['y'] = (0, 1)
        output = []
        for i in dic.keys():
            output.append(i)
        for j in dic.values():
            output.append(j)
        return output


def test_dict1():
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)
    net = Net1()
    out_me = net(input_me)
    assert out_me == ['x', 'y', 0, 1]


def test_dict2():
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)
    net = Net2()
    net(input_me)


def test_dict3():
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)
    net = Net3()
    out_me = net(input_me)
    assert out_me == ['x', 'y', 0, (0, 1)]


def test_dict4():
    class Net(Cell):
        def construct(self, tuple_x):
            output = tuple_x + tuple_x
            return output

    x = (1, Tensor([1, 2, 3]), (Tensor([1, 2, 3]), 1))
    net = Net()
    out_me = net(x)
    assert out_me == x + x


def test_dict_with_multitype_keys_1():
    """
    Feature: dict with multitype keys
    Description: Test dict with multitype keys in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo():
        ele_num = 3.14
        ele_str = "str"
        ele_tensor = Tensor([1, 2])
        ele_tuple = (ele_num, (ele_str, ele_tensor))
        dic = {ele_num: ele_tuple, ele_str: ele_tensor, ele_tensor: ele_str, ele_tuple: ele_num}
        return dic.get(Tensor([1, 2]), 0)

    result = foo()
    assert result == 0


def test_dict_with_multitype_keys_2():
    """
    Feature: dict with multitype keys
    Description: Test dict with multitype keys in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo():
        constant_tensor1 = Tensor(np.arange(6).reshape(1, 2, 3))
        constant_tensor2 = Tensor([1, 2])
        dic = {"str": 10, 2: 20, ("str_in_tuple", 3.14159): 30, constant_tensor1: 40.5,
               (1, (constant_tensor2, "3")): {1: 2}}
        sum_values = dic.get("str") + dic.get(2) + dic.get(("str_in_tuple", 3.14159)) + dic.get(constant_tensor1)
        return sum_values

    result = foo()
    assert result == 100.5


def test_dict_with_multitype_keys_3():
    """
    Feature: dict with multitype keys
    Description: Test dict with multitype keys in graph mode.
    Expectation: No exception.
    """

    @jit
    def foo():
        constant_tensor1 = Tensor([3, 4])
        dic = {True: -1, ((3.14, constant_tensor1), "i", 0): 2}
        dic[True] = 1
        if dic.has_key(True) and ((3.14, constant_tensor1), "i", 0) in dic:
            dic.clear()
        dic = dict.fromkeys((2, Tensor([1, 2])), [3, 4, 5])
        dic.update({Tensor([1, 2]): Tensor([3, 4, 5]),
                    2: "str_val"})  # for Dict's key, 'Tensor([1,2]) is not equal to Tensor([1,2])'
        return len(dic)

    result = foo()
    assert result == 3


def test_dict_error_1():
    """
    Feature: dict with multitype keys
    Description: do not support list for dict's key.
    Expectation: RuntimeError.
    """

    @jit
    def foo():
        x = [1, 2]
        dic = {x: 3}
        return dic.values()

    with pytest.raises(RuntimeError) as ex:
        foo()
    assert "key only supports string, number, constant tensor and tuple, but got" in str(ex.value)


def test_dict_error_2():
    """
    Feature: dict with multitype keys
    Description: do not support tuple with variables for dict's key.
    Expectation: RuntimeError.
    """

    @jit
    def foo():
        x = [1, 2]
        dic = {((3, x), 3): 4}
        return dic.values()

    with pytest.raises(RuntimeError) as ex:
        foo()
    assert "key should not be tuple that contains variables, but got" in str(ex.value)


def test_dict_error_3():
    """
    Feature: dict with multitype keys
    Description: do not support variables for dict's key.
    Expectation: RuntimeError.
    """

    @jit
    def foo(x):
        dic = {x: 10}
        return dic.values()

    x = Tensor([1, 2])
    with pytest.raises(RuntimeError) as ex:
        foo(x)
    assert "key only supports string, number, constant tensor and tuple, but got" in str(ex.value)
