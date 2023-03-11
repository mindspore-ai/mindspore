# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
""" test enumerate"""
import numpy as np
import pytest
import mindspore.nn as nn
from mindspore.nn import Cell
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore import Tensor, jit
from mindspore import context


def test_list_index_1d():
    """
    Feature: List index assign
    Description: Test list assign in pynative mode
    Expectation: No exception.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    class Net(nn.Cell):
        def construct(self):
            list_ = [[1], [2, 2], [3, 3, 3]]
            list_[0] = [100]
            return list_

    net = Net()
    out = net()
    assert list(out[0]) == [100]
    assert list(out[1]) == [2, 2]
    assert list(out[2]) == [3, 3, 3]

    context.set_context(mode=context.GRAPH_MODE)
    net = Net()
    out = net()
    assert list(out[0]) == [100]
    assert list(out[1]) == [2, 2]
    assert list(out[2]) == [3, 3, 3]



def test_list_neg_index_1d():
    """
    Feature: List index assign
    Description: Test list assign in pynative mode
    Expectation: No exception.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    class Net(nn.Cell):
        def construct(self):
            list_ = [[1], [2, 2], [3, 3, 3]]
            list_[-3] = [100]
            return list_

    net = Net()
    out = net()
    assert list(out[0]) == [100]
    assert list(out[1]) == [2, 2]
    assert list(out[2]) == [3, 3, 3]

    context.set_context(mode=context.GRAPH_MODE)
    out = net()
    assert list(out[0]) == [100]
    assert list(out[1]) == [2, 2]
    assert list(out[2]) == [3, 3, 3]


def test_list_index_2d():
    """
    Feature: List index assign
    Description: Test list assign in pynative mode
    Expectation: No exception.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    class Net(nn.Cell):
        def construct(self):
            list_ = [[1], [2, 2], [3, 3, 3]]
            list_[1][0] = 200
            list_[1][1] = 201
            return list_

    net = Net()
    out = net()
    assert list(out[0]) == [1]
    assert list(out[1]) == [200, 201]
    assert list(out[2]) == [3, 3, 3]

    context.set_context(mode=context.GRAPH_MODE)
    out = net()
    assert list(out[0]) == [1]
    assert list(out[1]) == [200, 201]
    assert list(out[2]) == [3, 3, 3]


def test_list_neg_index_2d():
    """
    Feature: List index assign
    Description: Test list assign in pynative mode
    Expectation: No exception.
    """

    context.set_context(mode=context.PYNATIVE_MODE)
    class Net(nn.Cell):
        def construct(self):
            list_ = [[1], [2, 2], [3, 3, 3]]
            list_[1][-2] = 20
            list_[1][-1] = 21
            return list_

    net = Net()
    out = net()
    assert list(out[0]) == [1]
    assert list(out[1]) == [20, 21]
    assert list(out[2]) == [3, 3, 3]

    context.set_context(mode=context.GRAPH_MODE)
    out = net()
    assert list(out[0]) == [1]
    assert list(out[1]) == [20, 21]
    assert list(out[2]) == [3, 3, 3]


def test_list_index_3d():
    """
    Feature: List index assign
    Description: Test list assign in pynative mode
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def construct(self):
            list_ = [[1], [2, 2], [[3, 3, 3]]]
            list_[2][0][0] = 300
            list_[2][0][1] = 301
            list_[2][0][2] = 302
            return list_

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    out = net()
    assert list(out[0]) == [1]
    assert list(out[1]) == [2, 2]
    assert list(out[2][0]) == [300, 301, 302]

    context.set_context(mode=context.GRAPH_MODE)
    out = net()
    assert list(out[0]) == [1]
    assert list(out[1]) == [2, 2]
    assert list(out[2][0]) == [300, 301, 302]


def test_list_neg_index_3d():
    """
    Feature: List index assign
    Description: Test list assign in pynative mode
    Expectation: No exception.
    """
    context.set_context(mode=context.PYNATIVE_MODE)

    class Net(nn.Cell):
        def construct(self):
            list_ = [[1], [2, 2], [[3, 3, 3]]]
            list_[2][0][-3] = 30
            list_[2][0][-2] = 31
            list_[2][0][-1] = 32
            return list_

    net = Net()
    out = net()
    assert list(out[0]) == [1]
    assert list(out[1]) == [2, 2]
    assert list(out[2][0]) == [30, 31, 32]

    context.set_context(mode=context.GRAPH_MODE)
    out = net()
    assert list(out[0]) == [1]
    assert list(out[1]) == [2, 2]
    assert list(out[2][0]) == [30, 31, 32]




def test_list_index_1D_parameter():
    context.set_context(mode=context.GRAPH_MODE)
    class Net(nn.Cell):
        def construct(self, x):
            list_ = [x]
            list_[0] = 100
            return list_

    net = Net()
    net(Tensor(0))


def test_list_index_2D_parameter():
    context.set_context(mode=context.GRAPH_MODE)
    class Net(nn.Cell):
        def construct(self, x):
            list_ = [[x, x]]
            list_[0][0] = 100
            return list_

    net = Net()
    net(Tensor(0))


def test_list_index_3D_parameter():
    context.set_context(mode=context.GRAPH_MODE)
    class Net(nn.Cell):
        def construct(self, x):
            list_ = [[[x, x]]]
            list_[0][0][0] = 100
            return list_

    net = Net()
    net(Tensor(0))


def test_const_list_index_3D_bprop():
    context.set_context(mode=context.GRAPH_MODE)
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = [[1], [2, 2], [[3, 3], [3, 3]]]
            self.relu = P.ReLU()

        def construct(self, input_x):
            list_x = self.value
            list_x[2][0][1] = input_x
            return self.relu(list_x[2][0][1])

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.grad_all_with_sens = C.GradOperation(get_all=True, sens_param=True)

        def construct(self, x, sens):
            return self.grad_all_with_sens(self.net)(x, sens)

    net = Net()
    grad_net = GradNet(net)
    x = Tensor(np.arange(2 * 3).reshape(2, 3))
    sens = Tensor(np.arange(2 * 3).reshape(2, 3))
    grad_net(x, sens)


def test_parameter_list_index_3D_bprop():
    context.set_context(mode=context.GRAPH_MODE)
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.value = [[1], [2, 2], [[3, 3], [3, 3]]]
            self.relu = P.ReLU()

        def construct(self, x, value):
            list_value = [[x], [x, x], [[x, x], [x, x]]]
            list_value[2][0][1] = value
            return self.relu(list_value[2][0][1])

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.grad_all_with_sens = C.GradOperation(get_all=True, sens_param=True)

        def construct(self, x, value, sens):
            return self.grad_all_with_sens(self.net)(x, value, sens)

    net = Net()
    grad_net = GradNet(net)
    x = Tensor(np.arange(2 * 3).reshape(2, 3))
    value = Tensor(np.ones((2, 3), np.int64))
    sens = Tensor(np.arange(2 * 3).reshape(2, 3))
    grad_net(x, value, sens)



class Net1(Cell):
    def construct(self, a, b, start=None, stop=None, step=None):
        a[start:stop:step] = b[start:stop:step]
        return tuple(a)


def compare_func1(a, b, start=None, stop=None, step=None):
    a[start:stop:step] = b[start:stop:step]
    return tuple(a)



def test_list_slice_length_equal():
    """
    Feature: List assign
    Description: Test list assign the size is equal
    Expectation: No exception.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    a = [1, 2, 3, 4]
    b = [5, 6, 7, 8]
    python_out = compare_func1(a, b, 0, None, 2)

    a = [1, 2, 3, 4]
    b = [5, 6, 7, 8]
    net = Net1()
    pynative_mode_out = net(a, b, 0, None, 2)
    assert pynative_mode_out == python_out

    context.set_context(mode=context.GRAPH_MODE)
    graph_out = net(a, b, 0, None, 2)
    assert graph_out == python_out


def test_list_slice_length_error():
    """
    Feature: List assign
    Description: Test list assign the size is not equal
    Expectation: ValueError.
    """
    context.set_context(mode=context.GRAPH_MODE)
    a = [1, 2, 3, 4, 5]
    b = [5, 6, 7, 8]
    net = Net1()
    with pytest.raises(ValueError) as err:
        net(a, b, 0, None, 2)
    assert "attempt to assign sequence of size 2 to extended slice of size 3" in str(err.value)

    context.set_context(mode=context.PYNATIVE_MODE)
    with pytest.raises(ValueError) as err:
        net(a, b, 0, None, 2)
    assert "attempt to assign sequence of size 2 to extended slice of size 3" in str(err.value)


def compare_func2(a, b, start=None, stop=None, step=None):
    a[start:stop:step] = b
    return tuple(a)


class Net2(Cell):
    def construct(self, a, b, start=None, stop=None, step=None):
        a[start:stop:step] = b
        return tuple(a)


def test_list_slice_shrink():
    """
    Feature: List assign
    Description: Test list slice shrink assign
    Expectation: No exception.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = [11, 22, 33]
    python_out = compare_func2(a, b, 0, 5)
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = [11, 22, 33]
    net = Net2()
    pynative_out = net(a, b, 0, 5)
    assert pynative_out == python_out

    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = [11, 22, 33]
    context.set_context(mode=context.GRAPH_MODE)
    graph_out = net(a, b, 0, 5)
    assert graph_out == python_out


def test_list_slice_insert():
    """
    Feature: List assign
    Description: Test list slice insert assign
    Expectation: No exception.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = [11, 22, 33, 44, 55]
    python_out = compare_func2(a, b, 0, 1)
    net = Net2()
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = [11, 22, 33, 44, 55]
    pynative_out = net(a, b, 0, 1)
    assert pynative_out == python_out

    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = [11, 22, 33, 44, 55]
    context.set_context(mode=context.GRAPH_MODE)
    graph_out = net(a, b, 0, 1)
    assert graph_out == python_out


def test_list_slice_assign():
    """
    Feature: List assign
    Description: Test list slice start and stop is larger than size
    Expectation: No exception.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = [11, 22, 33, 44, 55]
    python_out = compare_func2(a, b, -12, 456)

    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = [11, 22, 33, 44, 55]
    net = Net2()
    pynative_out = net(a, b, -12, 456)
    assert pynative_out == python_out

    context.set_context(mode=context.GRAPH_MODE)
    graph_out = net(a, b, -12, 456)
    assert graph_out == python_out


def test_list_slice_extend():
    """
    Feature: List assign
    Description: Test list slice extend
    Expectation: No exception.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = [11, 22, 33, 44, 55]
    net = Net2()
    python_out = compare_func2(a, b, 1234, 0)

    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = [11, 22, 33, 44, 55]
    pynative_out = net(a, b, 1234, 0)
    assert pynative_out == python_out

    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = [11, 22, 33, 44, 55]
    context.set_context(mode=context.GRAPH_MODE)
    graph_out = net(a, b, 1234, 0)
    assert graph_out == python_out


def test_list_slice_extend_front():
    """
    Feature: List assign
    Description: Test list slice extend
    Expectation: No exception.
    """
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = [11, 22, 33, 44, 55]
    python_out = compare_func2(a, b, 0, 0)

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net2()
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = [11, 22, 33, 44, 55]
    pynative_out = net(a, b, 0, 0)
    assert pynative_out == python_out

    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = [11, 22, 33, 44, 55]
    context.set_context(mode=context.GRAPH_MODE)
    graph_out = net(a, b, 0, 0)
    assert graph_out == python_out


def test_list_slice_extend_inner():
    """
    Feature: List assign
    Description: Test list slice extend
    Expectation: No exception.
    """
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = [11, 22, 33, 44, 55]
    python_out = compare_func2(a, b, 5, 5)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = [11, 22, 33, 44, 55]
    net = Net2()
    pynative_out = net(a, b, 5, 5)
    assert pynative_out == python_out

    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = [11, 22, 33, 44, 55]
    context.set_context(mode=context.GRAPH_MODE)
    graph_out = net(a, b, 5, 5)
    assert graph_out == python_out


def test_list_slice_erase():
    """
    Feature: List assign
    Description: Test list slice erase
    Expectation: No exception.
    """
    a = [1, 2, 3, 4, 5, 6, 7]
    python_out = compare_func2(a, [], 1, 3)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = [1, 2, 3, 4, 5, 6, 7]
    net = Net2()
    pynative_out = net(a, [], 1, 3)
    assert pynative_out == python_out

    context.set_context(mode=context.GRAPH_MODE)
    a = [1, 2, 3, 4, 5, 6, 7]
    graph_out = net(a, [], 1, 3)
    assert graph_out == python_out


def test_list_slice_tuple_without_step():
    """
    Feature: List assign
    Description: Test list slice assign with tuple
    Expectation: No exception.
    """
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = (11, 22, 33)
    python_out = compare_func2(a, b, 0, 4, None)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = (11, 22, 33)
    net = Net2()
    pynative_out = net(a, b, 0, 4, None)
    assert pynative_out == python_out

    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = (11, 22, 33)
    context.set_context(mode=context.GRAPH_MODE)
    graph_out = net(a, b, 0, 4, None)
    assert graph_out == python_out


def test_list_slice_tuple_with_step():
    """
    Feature: List assign
    Description: Test list slice assign with tuple
    Expectation: No exception.
    """

    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = (11, 22, 33)
    python_out = compare_func2(a, b, 1, None, 3)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = (11, 22, 33)
    net = Net2()
    pynative_out = net(a, b, 1, None, 3)
    assert pynative_out == python_out

    context.set_context(mode=context.GRAPH_MODE)
    graph_out = net(a, b, 1, None, 3)
    assert graph_out == python_out


def test_list_double_slice():
    """
    Feature: List assign
    Description: Test list double slice assign
    Expectation: ValueError
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    @jit
    def foo(a, b, start1, stop1, step1, start2, stop2, step2):
        a[start1:stop1:step1][start2: stop2: step2] = b
        return a

    class NetInner(Cell):
        def construct(self, a, b, start1, stop1, step1, start2, stop2, step2):
            a[start1:stop1:step1][start2: stop2: step2] = b
            return a

    net = NetInner()
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = [11, 22, 33]
    assert foo(a, b, 0, None, 1, 0, None, 3) == net(a, b, 0, None, 1, 0, None, 3)


def convert_tuple(a):
    result = tuple()
    for i in a:
        if isinstance(i, list):
            result += (tuple(i),)
            continue
        result += (i,)
    return result


def test_list_in_list_slice():
    """
    Feature: List assign
    Description: Test high dimension list slice assign
    Expectation: No exception.
    """

    class TestNet(Cell):
        def construct(self, a, b, index, start=None, stop=None, step=None):
            a[index][start:stop:step] = b
            return tuple(a)

    def com_func3(a, b, index, start=None, stop=None, step=None):
        a[index][start:stop:step] = b
        return convert_tuple(a)

    a = [1, 2, [1, 2, 3, 4, 5, 6, 7], 8, 9]
    b = [1111, 2222]
    python_out = com_func3(a, b, 2, 1, None, 3)

    context.set_context(mode=context.PYNATIVE_MODE)
    net = TestNet()
    a = [1, 2, [1, 2, 3, 4, 5, 6, 7], 8, 9]
    b = [1111, 2222]
    pynative_out = convert_tuple(net(a, b, 2, 1, None, 3))
    assert pynative_out == python_out

    context.set_context(mode=context.GRAPH_MODE)
    graph_out = convert_tuple(net(a, b, 2, 1, None, 3))
    assert graph_out == python_out


def test_list_slice_negative_step():
    """
    Feature: List assign
    Description: Test negative step list slice assign
    Expectation: No exception.
    """
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = [33, 44, 55]
    python_out = compare_func2(a, b, -1, -9, -3)

    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net2()
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = [33, 44, 55]
    pynative_out = net(a, b, -1, -9, -3)
    assert pynative_out == python_out

    context.set_context(mode=context.GRAPH_MODE)
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = [33, 44, 55]
    graph_out = net(a, b, -1, -9, -3)
    assert graph_out == python_out


def test_list_slice_negetive_error():
    """
    Feature: List assign
    Description: Test negative step list slice assign
    Expectation: ValueError
    """
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = [33, 44, 55]
    net = Net2()
    context.set_context(mode=context.PYNATIVE_MODE)
    with pytest.raises(ValueError) as err:
        net(a, b, -1, -3, -3)
    assert "attempt to assign sequence of size 3 to extended slice of size 1" in str(err.value)

    context.set_context(mode=context.GRAPH_MODE)
    with pytest.raises(ValueError) as err:
        net(a, b, -1, -3, -3)
    assert "attempt to assign sequence of size 3 to extended slice of size 1" in str(err.value)


def test_list_slice_negetive_step():
    """
    Feature: List assign
    Description: Test negative step list slice assign
    Expectation: ValueError
    """
    @jit
    def ms_func():
        a = [1, 2, 3, 4, 5]
        b = [11, 22, 33, 44, 55]
        a[-1:-4:-1] = b[-1:-4:-1]
        return a

    def py_func():
        a = [1, 2, 3, 4, 5]
        b = [11, 22, 33, 44, 55]
        a[-1:-4:-1] = b[-1:-4:-1]
        return a

    x = py_func()
    y = ms_func()
    assert x == y


def test_list_double_slice_assign_error():
    """
    Feature: List assign
    Description: Test negative step list slice assign
    Expectation: ValueError
    """
    @jit
    def ms_func():
        a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        b = [11, 22, 33, 44]
        a[::2][:3:2] = b
        return a

    with pytest.raises(ValueError) as err:
        ms_func()
    assert "attempt to assign sequence of size 4 to extended slice of size 2" in str(err.value)


def test_list_slice_only_with_step():
    """
    Feature: List assign
    Description: Test negative step list slice assign
    Expectation: ValueError
    """

    @jit
    def ms_func():
        a = [1, 2, 3, 4]
        b = [11, 22]
        a[::2] = b
        return a

    def py_func():
        a = [1, 2, 3, 4]
        b = [11, 22]
        a[::2] = b
        return a

    assert ms_func() == py_func()
