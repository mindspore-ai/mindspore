# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.2 (the "License");
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

import pytest
import numpy as np
from mindspore import Tensor, jit, context
from collections import Iterator
import mindspore.nn as nn
from tests.mark_utils import arg_mark


context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.skip(reason="No support yet.")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_list_tuple_string_dict_tensor_iter():
    """
    Feature: iterator
    Description: Test iterator
    Expectation: No exception
    """

    @jit
    def foo():
        list_x = [1, 2, 3]
        tuple_y = (4, 5, 6)
        string_z = "789"
        dict_w = {"one": 1, "two": 2}
        return iter(list_x), iter(tuple_y), iter(string_z), iter(dict_w)

    res_list, res_tuple, res_str, res_dict = foo()
    assert isinstance(res_list, Iterator)
    assert isinstance(res_tuple, Iterator)
    assert isinstance(res_str, Iterator)
    assert isinstance(res_dict, Iterator)


@pytest.mark.skip(reason="No support yet.")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_tensor_iter():
    """
    Feature: iterator
    Description: Test iterator
    Expectation: No exception
    """

    @jit
    def foo(x):
        return iter(x)

    x = Tensor(np.array([1, 2, 3, 4]))
    res = foo(x)
    print("res:", res)
    assert isinstance(res, Iterator)


@pytest.mark.skip(reason="No support yet.")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_list_iter_in_for():
    """
    Feature: iterator
    Description: Test iterator
    Expectation: No exception
    """

    @jit
    def foo():
        my_list = [1, 2, 3, 4, 5]
        my_iterator = iter(my_list)
        my_sum = 0
        for i in my_iterator:
            my_sum += i
        return my_sum, my_iterator

    res = foo()
    assert res[0] == 15
    assert isinstance(res[1], Iterator)


@pytest.mark.skip(reason="No support yet.")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_iter_asnumpy():
    """
    Feature: iterator
    Description: Test iterator
    Expectation: No exception
    """

    @jit
    def foo():
        x = Tensor(np.array([1, 2, 3, 4])).asnumpy()
        y = 0
        for i in x:
            y += i
        return iter(x), y

    res = foo()
    assert isinstance(res[0], Iterator)
    assert res[1] == 10


@pytest.mark.skip(reason="No support yet.")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_tuple_iter_next():
    """
    Feature: iterator
    Description: Test iterator
    Expectation: No exception
    """

    @jit
    def foo():
        my_tuple = (1, 2, 3, 4, 5)
        my_iterator = iter(my_tuple)
        my_sum = 0
        my_sum += next(my_iterator)
        my_sum += next(my_iterator)
        my_sum += next(my_iterator)
        return my_iterator, my_sum

    res = foo()
    assert isinstance(res[0], Iterator)
    assert res[1] == 6


@pytest.mark.skip(reason="No support yet.")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_dict_iter_next():
    """
    Feature: iterator
    Description: Test iterator
    Expectation: No exception
    """

    @jit
    def foo():
        my_dict = {"one": 1, "two": 2, "three": 3}
        my_iterator = iter(my_dict)
        my_sum = ""
        my_sum += next(my_iterator) + "_"
        my_sum += next(my_iterator) + "_"
        my_sum += next(my_iterator)
        return my_iterator, my_sum

    res = foo()
    assert isinstance(res[0], Iterator)
    assert res[1] == "one_two_three"


class MyIterator(nn.Cell):
    def __init__(self, start, end):
        self.current = start
        self.end = end

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.end:
            raise StopIteration
        value = self.current
        self.current += 1
        return value


@pytest.mark.skip(reason="No support yet.")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_custom_self_iterator_1():
    """
    Feature: iterator
    Description: Test iterator
    Expectation: No exception
    """

    @jit
    def foo():
        my_iterator = MyIterator(1, 5)
        my_sum = 0
        for i in my_iterator:
            my_sum += i
        return my_iterator, my_sum

    res = foo()
    assert isinstance(res[0], Iterator)
    assert res[1] == 10


@pytest.mark.skip(reason="No support yet.")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_custom_self_iterator_2():
    """
    Feature: iterator
    Description: Test iterator
    Expectation: No exception
    """

    @jit
    def foo():
        my_iterator = MyIterator(1, 5)
        my_sum = 0
        for i in my_iterator:
            my_sum += i
        my_sum += next(my_iterator)
        return my_iterator, my_sum

    with pytest.raises(StopIteration, match=""):
        res = foo()
        print("res:", res)


@pytest.mark.skip(reason="No support yet.")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_custom_self_iterator_3():
    """
    Feature: iterator
    Description: Test iterator
    Expectation: No exception
    """

    @jit
    def foo():
        my_iterator = MyIterator(1, 5)
        my_sum = 0
        for i in my_iterator:
            my_sum += i
        my_sum_2 = 0
        for i in my_iterator:
            my_sum_2 += i
        return my_iterator, my_sum, my_sum_2

    res = foo()
    assert isinstance(res[0], Iterator)
    assert res[1] == 10
    assert res[2] == 0
