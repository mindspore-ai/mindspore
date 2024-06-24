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
from tests.mark_utils import arg_mark
import pytest
import numpy as np

from mindspore import context, Tensor
from mindspore.common import mutable
from mindspore.nn import Cell
from mindspore.ops.composite import GradOperation
from mindspore._extends.parse.standard_method import list_insert
from sequence_help import context_prepare

context.set_context(mode=context.GRAPH_MODE)
context_prepare()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_list_insert1():
    """
    Feature: test sequence getitem op
    Description: setitem operation on tuple type
    Expectation: the behavior is matched to python style
    """
    class Net(Cell):
        def construct(self, x, idx, y):
            return list_insert(x, idx, y)

    net_ms = Net()
    input_x = mutable([2], True)
    idx = mutable(0)
    input_y = mutable(3)
    res = net_ms(input_x, idx, input_y)
    expect = [3, 2]
    assert res == expect


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_insert2():
    """
    Feature: test sequence getitem op
    Description: setitem operation on tuple type
    Expectation: the behavior is matched to python style
    """
    class Net(Cell):
        def construct(self, x, idx, y):
            return list_insert(x, idx, y)

    net_ms = Net()
    input_x = mutable([Tensor(2)], True)
    idx = 1
    input_y = Tensor(3)
    res = net_ms(input_x, idx, input_y)
    expect = [Tensor(2), Tensor(3)]
    for i in range(2):
        assert np.all(res[i].asnumpy() == expect[i].asnumpy())


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_insert3():
    """
    Feature: test sequence getitem op
    Description: setitem operation on tuple type
    Expectation: the behavior is matched to python style
    """
    class Net(Cell):
        def construct(self, x, idx, y):
            return list_insert(x, idx, y)

    net_ms = Net()
    input_x = mutable([Tensor([[2, 3], [4, 5]])], True)
    idx = 1
    input_y = Tensor([[4, 5], [5, 6]])
    res = net_ms(input_x, idx, input_y)
    expect = [Tensor([[2, 3], [4, 5]]), Tensor([[4, 5], [5, 6]])]
    for i in range(2):
        assert np.all(res[i].asnumpy() == expect[i].asnumpy())


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_insert_grad():
    """
    Feature: test sequence getitem grad op
    Description: setitem operation on tuple type
    Expectation: the behavior is matched to python style
    """
    class Net(Cell):
        def construct(self, x, idx, y):
            return list_insert(x, idx, y)

    net_ms = Net()
    seq = mutable((1, 2, 3, 4, 5, 6), True)
    idx = 3
    value = 8
    dout = (1, 2, 3, 8, 4, 5, 6)
    grad_func = GradOperation(get_all=True, sens_param=True)(net_ms)
    print("grad out1 = ", grad_func(seq, idx, value, dout))
