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
"""Test list assignment"""
import pytest
from mindspore.nn import Cell

from mindspore import Tensor
from mindspore import context
from mindspore import jit
from mindspore.graph.api import _pynative_executor
from tests.mark_utils import arg_mark


class Net2(Cell):
    def construct(self, a, b, start=None, stop=None, step=None):
        a[start:stop:step] = b
        return tuple(a)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_pynative_list_slice_tensor_no_step():
    """
    Feature: List assign
    Description: Test list slice assign with tensor
    Expectation: No exception.
    """

    class NetInner(Cell):
        def construct(self, start=None, stop=None, step=None):
            a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            b = Tensor([11, 22, 33])
            a[start:stop:step] = b
            return tuple(a)

    context.set_context(mode=context.PYNATIVE_MODE)
    net = NetInner()
    python_out = (Tensor(11), Tensor(22), Tensor(33), 4, 5, 6, 7, 8, 9)
    pynative_out = net(0, 3, None)
    assert pynative_out == python_out

    context.set_context(mode=context.GRAPH_MODE)
    graph_out = net(0, 3, None)
    assert graph_out == python_out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_pynative_list_slice_tensor_with_step():
    """
    Feature: List assign
    Description: Test list slice assign with tensor
    Expectation: No exception.
    """

    class NetInner(Cell):
        def construct(self, start=None, stop=None, step=None):
            a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            b = Tensor([11, 22, 33])
            a[start:stop:step] = b
            return tuple(a)

    context.set_context(mode=context.PYNATIVE_MODE)
    net = NetInner()
    python_out = (Tensor(11), 2, 3, Tensor(22), 5, 6, Tensor(33), 8, 9)
    pynative_out = net(0, None, 3)
    assert python_out == pynative_out

    context.set_context(mode=context.GRAPH_MODE)
    graph_out = net(0, None, 3)
    assert python_out == graph_out


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_graph_list_slice_assign_extended_number():
    """
    Feature: List assign
    Description: Test negative step list slice assign
    Expectation: No exception.
    """
    a = [1, 2, 3, 4, 5, 6]
    b = 1

    net = Net2()
    context.set_context(mode=context.PYNATIVE_MODE)
    with pytest.raises(TypeError) as err:
        net(a, b, 0, None, 2)
    assert "must assign iterable to extended slice" in str(err.value)

    context.set_context(mode=context.GRAPH_MODE)
    with pytest.raises(TypeError) as err:
        net(a, b, 0, None, 2)
    # pylint: disable=condition-evals-to-constant
    assert "None object is not iterable" or \
           "must assign iterable to extended slice" in str(err.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_graph_list_slice_assign_number():
    """
    Feature: List assign
    Description: Test negative step list slice assign
    Expectation: No exception.
    """
    a = [1, 2, 3, 4, 5, 6]
    b = 1
    net = Net2()
    context.set_context(mode=context.PYNATIVE_MODE)
    with pytest.raises(TypeError) as err:
        net(a, b, 0, None, 1)
    assert "can only assign an iterable" in str(err.value)

    context.set_context(mode=context.GRAPH_MODE)
    with pytest.raises(TypeError) as err:
        net(a, b, 0, None, 1)
    # pylint: disable=condition-evals-to-constant
    assert "None object is not iterable" or \
           "can only assign an iterable" in str(err.value)


def convert_tuple(a):
    """Convert nested list to nested tuple for comparison."""
    result = tuple()
    for i in a:
        if isinstance(i, list):
            i = convert_tuple(i)
            result += (tuple(i), )
            continue
        result += (i, )
    return result


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_constant_list_assign_list():
    """
    Feature: List slice assignment.
    Description: Test list slice assignment with constant list and list value.
    Expectation: Assignment result matches expected list.
    Migrated from: test_parser_list_slice_assignment.py::test_a_is_constant_list_b_is_list
    """
    class Net(Cell):
        @jit
        def construct(self, start=None, end=None, step=None):
            a = [1, 2, 3, 4, 5]
            b = [11, 22, 33]
            a[start:end:step] = b
            return a

    net = Net()
    ma = net(start=None, end=1, step=None)
    pa = [11, 22, 33, 2, 3, 4, 5]
    assert ma == pa

    ma = net(start=2, end=3, step=None)
    pa = [1, 2, 11, 22, 33, 4, 5]
    assert ma == pa

    ma = net(start=-1, end=None, step=None)
    pa = [1, 2, 3, 4, 11, 22, 33]
    assert ma == pa

    ma = net(start=None, end=None, step=2)
    pa = [11, 2, 22, 4, 33]
    assert ma == pa

    ma = net(start=None, end=None, step=-2)
    pa = [33, 2, 22, 4, 11]
    assert ma == pa


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_constant_list_assign_tensor():
    """
    Feature: List slice assignment.
    Description: Test list slice assignment with constant list and tensor value.
    Expectation: Assignment result matches expected list with tensor elements.
    Migrated from: test_parser_list_slice_assignment.py::test_a_is_constant_list_b_is_tensor
    """
    class Net(Cell):
        @jit
        def construct(self, start=None, end=None, step=None):
            a = [1, 2, 3, 4, 5]
            b = Tensor([11, 22, 33])
            a[start:end:step] = b
            return a

    net = Net()
    ma = net(start=None, end=1, step=None)
    pa = [Tensor(11), Tensor(22), Tensor(33), 2, 3, 4, 5]
    assert ma == pa

    ma = net(start=2, end=3, step=None)
    pa = [1, 2, Tensor(11), Tensor(22), Tensor(33), 4, 5]
    assert ma == pa

    ma = net(start=-1, end=None, step=None)
    pa = [1, 2, 3, 4, Tensor(11), Tensor(22), Tensor(33)]
    assert ma == pa

    ma = net(start=None, end=None, step=2)
    pa = [Tensor(11), 2, Tensor(22), 4, Tensor(33)]
    assert ma == pa

    ma = net(start=None, end=None, step=-2)
    pa = [Tensor(33), 2, Tensor(22), 4, Tensor(11)]
    assert ma == pa


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_constant_list_assign_tuple():
    """
    Feature: List slice assignment.
    Description: Test list slice assignment with constant list and tuple value.
    Expectation: Assignment result matches expected list.
    Migrated from: test_parser_list_slice_assignment.py::test_a_is_constant_list_b_is_tuple
    """
    class Net(Cell):
        @jit
        def construct(self, start=None, end=None, step=None):
            a = [1, 2, 3, 4, 5]
            b = (11, 22, 33)
            a[start:end:step] = b
            return a

    net = Net()
    ma = net(start=None, end=1, step=None)
    pa = [11, 22, 33, 2, 3, 4, 5]
    assert ma == pa

    ma = net(start=2, end=3, step=None)
    pa = [1, 2, 11, 22, 33, 4, 5]
    assert ma == pa

    ma = net(start=-1, end=None, step=None)
    pa = [1, 2, 3, 4, 11, 22, 33]
    assert ma == pa

    ma = net(start=None, end=None, step=2)
    pa = [11, 2, 22, 4, 33]
    assert ma == pa

    ma = net(start=None, end=None, step=-2)
    pa = [33, 2, 22, 4, 11]
    assert ma == pa


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_constant_list_extend():
    """
    Feature: List slice assignment.
    Description: Test list slice assignment that extends the list (extend operation).
    Expectation: Assignment result matches expected extended list.
    Migrated from: test_parser_list_slice_assignment.py::test_a_is_constant_list_extend
    """
    class Net1(Cell):
        @jit
        def construct(self, a, b, start=None, end=None, step=None):
            a[start:end:step] = b
            return a

    def list_slice_assignment_python(c, d, start=None, end=None, step=None):
        c[start:end:step] = d
        return c

    net = Net1()
    ma = net(a=[1, 2, 3, 4, 5], b=[11, 22, 33], start=5, end=0, step=None)
    pa = list_slice_assignment_python(c=[1, 2, 3, 4, 5], d=[11, 22, 33], start=5, end=0, step=None)
    assert ma == pa


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_constant_list_front():
    """
    Feature: List slice assignment.
    Description: Test list slice assignment at the front of the list.
    Expectation: Assignment result matches expected list.
    Migrated from: test_parser_list_slice_assignment.py::test_a_is_constant_list_front
    """
    class Net1(Cell):
        @jit
        def construct(self, a, b, start=None, end=None, step=None):
            a[start:end:step] = b
            return a

    def list_slice_assignment_python(c, d, start=None, end=None, step=None):
        c[start:end:step] = d
        return c

    net = Net1()
    ma = net(a=[1, 2, 3, 4, 5], b=[11, 22, 33], start=None, end=-6, step=None)
    pa = list_slice_assignment_python(c=[1, 2, 3, 4, 5],
                                      d=[11, 22, 33],
                                      start=None,
                                      end=-6,
                                      step=None)
    assert ma == pa

    net = Net1()
    ma = net(a=[1, 2, 3, 4, 5], b=[11, 22, 33], start=None, end=0, step=None)
    pa = list_slice_assignment_python(c=[1, 2, 3, 4, 5],
                                      d=[11, 22, 33],
                                      start=None,
                                      end=0,
                                      step=None)
    assert ma == pa


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_constant_list_insert():
    """
    Feature: List slice assignment.
    Description: Test list slice assignment that inserts elements (insert operation).
    Expectation: Assignment result matches expected list.
    Migrated from: test_parser_list_slice_assignment.py::test_a_is_constant_list_insert
    """
    class Net1(Cell):
        @jit
        def construct(self, a, b, start=None, end=None, step=None):
            a[start:end:step] = b
            return a

    def list_slice_assignment_python(c, d, start=None, end=None, step=None):
        c[start:end:step] = d
        return c

    net = Net1()
    ma = net(a=[1, 2, 3, 4, 5], b=[11, 22, 33], start=-3, end=-3, step=None)
    pa = list_slice_assignment_python(c=[1, 2, 3, 4, 5],
                                      d=[11, 22, 33],
                                      start=-3,
                                      end=-3,
                                      step=None)
    assert ma == pa

    net = Net1()
    ma = net(a=[1, 2, 3, 4, 5], b=[11, 22, 33], start=4, end=0, step=None)
    pa = list_slice_assignment_python(c=[1, 2, 3, 4, 5], d=[11, 22, 33], start=4, end=0, step=None)
    assert ma == pa


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_constant_list_remove():
    """
    Feature: List slice assignment.
    Description: Test list slice assignment that removes elements (remove operation).
    Expectation: Assignment result matches expected list.
    Migrated from: test_parser_list_slice_assignment.py::test_a_is_constant_list_remove
    """
    class Net1(Cell):
        @jit
        def construct(self, a, b, start=None, end=None, step=None):
            a[start:end:step] = b
            return a

    def list_slice_assignment_python(c, d, start=None, end=None, step=None):
        c[start:end:step] = d
        return c

    net = Net1()
    ma = net(a=[1, 2, 3, 4, 5], b=[], start=-4, end=-3, step=None)
    pa = list_slice_assignment_python(c=[1, 2, 3, 4, 5], d=[], start=-4, end=-3, step=None)
    assert ma == pa


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_constant_list_slice_equal_size():
    """
    Feature: List slice assignment.
    Description: Test list slice assignment when slice sizes are equal.
    Expectation: Assignment result matches expected list.
    Migrated from: test_parser_list_slice_assignment.py::test_a_is_constant_list_and_slice_size_equal_b_slice_size
    """
    class Net(Cell):
        @jit
        def construct(self, start, end, step):
            a = [1, 2, 3, 4, 5]
            b = [11, 22, 33, 44, 55]
            a[start:end:step] = b[start:end:step]
            return a

    def list_slice_assignment_at_python(start, end, step):
        c = [1, 2, 3, 4, 5]
        d = [11, 22, 33, 44, 55]
        c[start:end:step] = d[start:end:step]
        return c

    net = Net()
    ma = net(start=None, end=None, step=None)
    pa = list_slice_assignment_at_python(start=None, end=None, step=None)
    assert ma == pa

    net = Net()
    ma = net(start=0, end=3, step=None)
    pa = list_slice_assignment_at_python(start=0, end=3, step=None)
    assert ma == pa

    net = Net()
    ma = net(start=-1, end=-4, step=-1)
    pa = list_slice_assignment_at_python(start=-1, end=-4, step=-1)
    assert ma == pa

    net = Net()
    ma = net(start=None, end=None, step=2)
    pa = list_slice_assignment_at_python(start=None, end=None, step=2)
    assert ma == pa

    net = Net()
    ma = net(start=0, end=4, step=2)
    pa = list_slice_assignment_at_python(start=0, end=4, step=2)
    assert ma == pa

    net = Net()
    ma = net(start=-1, end=None, step=-2)
    pa = list_slice_assignment_at_python(start=-1, end=None, step=-2)
    assert ma == pa


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_constant_list_slice_unequal_size():
    """
    Feature: List slice assignment.
    Description: Test list slice assignment when slice sizes are unequal.
    Expectation: Assignment result matches expected list or raises ValueError.
    Migrated from: test_parser_list_slice_assignment.py::test_a_is_constant_list_and_slice_size_no_equal_b_slice_size
    """
    class Net(Cell):
        @jit
        def construct(self, a, b, start, end, step):
            a[start:end:step] = b[start:end:step]
            return a

    def list_slice_assignment_at_python(a, b, start, end, step):
        a[start:end:step] = b[start:end:step]
        return a

    a = [1, 2, 3, 4, 5]
    b = [11, 22, 33, 44]
    net = Net()
    ma = net(a, b, start=0, end=3, step=None)
    a = [1, 2, 3, 4, 5]
    b = [11, 22, 33, 44]
    pa = list_slice_assignment_at_python(a, b, start=0, end=3, step=None)
    assert ma == pa

    a = [1, 2, 3, 4, 5]
    b = [11, 22, 33, 44, 55, 66]
    net = Net()
    ma = net(a, b, start=-1, end=-4, step=-1)
    a = [1, 2, 3, 4, 5]
    b = [11, 22, 33, 44, 55, 66]
    pa = list_slice_assignment_at_python(a, b, start=-1, end=-4, step=-1)
    assert ma == pa

    a = [1, 2, 3, 4, 5]
    b = [11, 22, 33, 44, 55, 66, 77]
    net = Net()
    with pytest.raises(ValueError):
        net(a, b, start=None, end=None, step=2)
        _pynative_executor.sync()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_constant_list_slice_many_times():
    """
    Feature: List slice assignment.
    Description: Test multiple slice assignments on nested list.
    Expectation: Assignment result matches expected list or raises ValueError.
    Migrated from: test_parser_list_slice_assignment.py::test_a_slice_many_times
    """
    class Net(Cell):
        @jit
        def construct(self, a, b, start, end, step):
            a[start[0]:end[0]:step[0]][start[1]:end[1]:step[1]] = b
            return a

    def list_slice_assignment_at_python(a, b, start, end, step):
        a[start[0]:end[0]:step[0]][start[1]:end[1]:step[1]] = b
        return a

    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = [11, 22, 33, 44]
    net = Net()
    ma = net(a, b, start=(None, None), end=(None, 3), step=(2, None))
    assert ma == a
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = [11, 22, 33, 44]
    pa = list_slice_assignment_at_python(a, b, start=(None, None), end=(None, 3), step=(2, None))
    assert pa == a
    assert ma == pa

    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = [11, 22, 33, 44]
    net = Net()
    ma = net(a, b, start=(None, None), end=(None, 4), step=(2, None))
    assert ma == a
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = [11, 22, 33, 44]
    pa = list_slice_assignment_at_python(a, b, start=(None, None), end=(None, 4), step=(2, None))
    assert pa == a
    assert ma == pa

    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = [11, 22, 33, 44]
    net = Net()
    ma = net(a, b, start=(None, None), end=(None, 1), step=(2, None))
    assert ma == a
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = [11, 22, 33, 44]
    pa = list_slice_assignment_at_python(a, b, start=(None, None), end=(None, 1), step=(2, None))
    assert pa == a
    assert ma == pa

    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = [11, 22, 33, 44]
    net = Net()
    with pytest.raises(ValueError):
        net(a, b, start=(None, None), end=(None, 3), step=(2, 2))
        _pynative_executor.sync()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_variable_list_assign_tensor_with_list():
    """
    Feature: List slice assignment.
    Description: Test list slice assignment with variable list and tensor value stored as attribute.
    Expectation: Assignment result matches expected list.
    Migrated from: test_parser_list_slice_assignment.py::test_a_is_variable_list_b_is_tensor_with_list
    """
    class Net(Cell):
        def __init__(self, b, start=None, end=None, step=None):
            super().__init__()
            self.b = b
            self.start = start
            self.end = end
            self.step = step

        @jit
        def construct(self):
            a = [1, 2, 3, 4, 5]
            a[self.start:self.end:self.step] = self.b
            return a

    def list_slice_assignment_at_python(d, start=None, end=None, step=None):
        c = [1, 2, 3, 4, 5]
        c[start:end:step] = d
        return c

    b = Tensor([11, 22, 33])
    net = Net(b, start=2, end=3, step=None)
    ma = net()

    d = Tensor([11, 22, 33])
    pa = list_slice_assignment_at_python(d, start=2, end=3, step=None)
    assert ma == pa


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_variable_list_assign_list_or_tuple():
    """
    Feature: List slice assignment.
    Description: Test list slice assignment with variable list containing tensor elements.
    Expectation: Assignment result matches expected list or raises ValueError.
    Migrated from: test_parser_list_slice_assignment.py::test_a_is_variable_list_b_is_list_or_tuple
    """
    class Net1(Cell):
        @jit
        def construct(self, a, b, start=None, end=None, step=None):
            a[start:end:step] = b
            return a

    net = Net1()
    x = Tensor(1)
    ma = net(a=[x, x, x, x, x], b=[11, 22, 33], start=None, end=1, step=None)
    pa = [11, 22, 33, Tensor(1), Tensor(1), Tensor(1), Tensor(1)]
    assert ma == pa
    x = Tensor(1)
    ma = net(a=[x, x, x, x, x], b=[11, 22, 33], start=2, end=3, step=None)
    pa = [Tensor(1), Tensor(1), Tensor(11), Tensor(22), Tensor(33), Tensor(1), Tensor(1)]
    assert ma == pa

    ma = net(a=[x, x, x, x, x], b=(11, 22, 33), start=-1, end=None, step=None)
    pa = [Tensor(1), Tensor(1), Tensor(1), Tensor(1), 11, 22, 33]
    assert ma == pa

    ma = net(a=[x, x, x, x, x],
             b=[Tensor(11), Tensor(22), Tensor(33)],
             start=None,
             end=None,
             step=2)
    pa = [Tensor(11), Tensor(1), Tensor(22), Tensor(1), Tensor(33)]
    assert ma == pa

    ma = net(a=[x, x, x, x, x], b=[11, 22, 33], start=None, end=None, step=-2)
    pa = [33, Tensor(1), 22, Tensor(1), 11]
    assert ma == pa

    with pytest.raises(ValueError):
        net(a=[x, x, x, x, x], b=(11, 22, 33), start=-1, end=None, step=-3)
        _pynative_executor.sync()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_variable_list_slice_equal_size():
    """
    Feature: List slice assignment.
    Description: Test list slice assignment with variable list when slice sizes are equal.
    Expectation: Assignment result matches expected list.
    Migrated from: test_parser_list_slice_assignment.py::test_a_is_variable_list_and_slice_size_equal_b_slice_size
    """
    class Net(Cell):
        @jit
        def construct(self, x, start, end, step):
            a = [x, x, x, x, x]
            b = [11, 22, 33, 44, 55]
            a[start:end:step] = b[start:end:step]
            return a

    def list_slice_assignment_at_python(x, start, end, step):
        c = [x, x, x, x, x]
        d = [11, 22, 33, 44, 55]
        c[start:end:step] = d[start:end:step]
        return c

    net = Net()
    ma = net(x=Tensor(1), start=None, end=None, step=None)
    pa = list_slice_assignment_at_python(x=Tensor(1), start=None, end=None, step=None)
    assert ma == pa

    net = Net()
    ma = net(x=Tensor(1), start=0, end=3, step=None)
    pa = list_slice_assignment_at_python(x=Tensor(1), start=0, end=3, step=None)
    assert ma == pa

    net = Net()
    ma = net(x=Tensor(1), start=-1, end=-4, step=-1)
    pa = list_slice_assignment_at_python(x=Tensor(1), start=-1, end=-4, step=-1)
    assert ma == pa

    net = Net()
    ma = net(x=Tensor(1), start=None, end=None, step=2)
    pa = list_slice_assignment_at_python(x=Tensor(1), start=None, end=None, step=2)
    assert ma == pa

    net = Net()
    ma = net(x=Tensor(1), start=0, end=4, step=2)
    pa = list_slice_assignment_at_python(x=Tensor(1), start=0, end=4, step=2)
    assert ma == pa

    net = Net()
    ma = net(x=Tensor(1), start=-1, end=None, step=-2)
    pa = list_slice_assignment_at_python(x=Tensor(1), start=-1, end=None, step=-2)
    assert ma == pa


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_variable_list_slice_unequal_size():
    """
    Feature: List slice assignment.
    Description: Test list slice assignment with variable list when slice sizes are unequal.
    Expectation: Assignment result matches expected list or raises ValueError.
    Migrated from: test_parser_list_slice_assignment.py::test_a_is_variable_list_and_slice_size_no_equal_b_slice_size
    """
    class Net(Cell):
        @jit
        def construct(self, a, b, start, end, step):
            a[start:end:step] = b[start:end:step]
            return a

    def list_slice_assignment_at_python(c, d, start, end, step):
        c[start:end:step] = d[start:end:step]
        return c

    x = Tensor(1)
    net = Net()
    ma = net(a=[x, x, x, x, x], b=[11, 22, 33, 44], start=0, end=3, step=None)
    pa = list_slice_assignment_at_python(c=[x, x, x, x, x],
                                         d=[11, 22, 33, 44],
                                         start=0,
                                         end=3,
                                         step=None)
    assert ma == pa

    net = Net()
    ma = net(a=[x, x, x, x, x], b=[11, 22, 33, 44, 55, 66], start=-1, end=-4, step=-1)
    pa = list_slice_assignment_at_python(c=[x, x, x, x, x],
                                         d=[11, 22, 33, 44, 55, 66],
                                         start=-1,
                                         end=-4,
                                         step=-1)
    assert ma == pa

    with pytest.raises(ValueError):
        net(a=[x, x, x, x, x], b=(11, 22, 33), start=None, end=None, step=3)
        _pynative_executor.sync()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_nested_list():
    """
    Feature: List slice assignment.
    Description: Test list slice assignment on nested list.
    Expectation: Assignment result matches expected nested list or raises ValueError.
    Migrated from: test_parser_list_slice_assignment.py::test_a_is_nested_list
    """
    class Net(Cell):
        @jit
        def construct(self, index, b, start=None, end=None, step=None):
            a = [[1, 2, 3, 4, 5], [1, 3, 5, 7], 3]
            a[index][start:end:step] = b
            return a

    def list_slice_assignment_at_python(index, d, start, end, step):
        c = [[1, 2, 3, 4, 5], [1, 3, 5, 7], 3]
        c[index][start:end:step] = d
        return convert_tuple(c)

    net = Net()
    ma = net(index=0, b=[11, 22, 33], start=None, end=1, step=None)
    pa = list_slice_assignment_at_python(index=0, d=[11, 22, 33], start=None, end=1, step=None)
    assert convert_tuple(ma) == pa

    ma = net(index=1, b=[Tensor(11), Tensor(22), Tensor(33)], start=1, end=2, step=None)
    pa = list_slice_assignment_at_python(index=1,
                                         d=[Tensor(11), Tensor(22),
                                            Tensor(33)],
                                         start=1,
                                         end=2,
                                         step=None)
    assert convert_tuple(ma) == pa

    ma = net(index=0, b=(11, 22, 33), start=-1, end=None, step=None)
    pa = list_slice_assignment_at_python(index=0, d=(11, 22, 33), start=-1, end=None, step=None)
    assert convert_tuple(ma) == pa

    ma = net(index=0, b=[11, 22, 33], start=None, end=None, step=-2)
    pa = list_slice_assignment_at_python(index=0, d=[11, 22, 33], start=None, end=None, step=-2)
    assert convert_tuple(ma) == pa

    with pytest.raises(ValueError) as e:
        net(index=1, b=Tensor([11, 22, 33]), start=None, end=None, step=2)
        _pynative_executor.sync()
        assert "the inputs should be constant, but found variable 'value' to be nonconstant" in str(e.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_multi_level_nested_list():
    """
    Feature: List slice assignment.
    Description: Test list slice assignment on multi-level nested list.
    Expectation: Assignment result matches expected nested list or raises ValueError.
    Migrated from: test_parser_list_slice_assignment.py::test_a_is_multi_level_nested_list
    """
    class Net(Cell):
        @jit
        def construct(self, index, b, start=None, end=None, step=None):
            a = [[1, 2, [11, 22, 33, 44], 4, 5], [1, 3, 5, 7], 3]
            a[index[0]][index[1]][start:end:step] = b
            return a

    def list_slice_assignment_at_python(index, d, start, end, step):
        c = [[1, 2, [11, 22, 33, 44], 4, 5], [1, 3, 5, 7], 3]
        c[index[0]][index[1]][start:end:step] = d
        return convert_tuple(c)

    net = Net()
    ma = net(index=(0, 2), b=[11, 22, 33], start=None, end=1, step=None)
    pa = list_slice_assignment_at_python(index=(0, 2), d=[11, 22, 33], start=None, end=1, step=None)
    assert convert_tuple(ma) == pa

    ma = net(index=(0, 2), b=[Tensor(11), Tensor(22), Tensor(33)], start=2, end=3, step=None)
    pa = list_slice_assignment_at_python(index=(0, 2),
                                         d=[Tensor(11), Tensor(22),
                                            Tensor(33)],
                                         start=2,
                                         end=3,
                                         step=None)
    assert convert_tuple(ma) == pa

    ma = net(index=(-3, 2), b=(11, 22, 33), start=-1, end=None, step=None)
    pa = list_slice_assignment_at_python(index=(0, 2),
                                         d=(11, 22, 33),
                                         start=-1,
                                         end=None,
                                         step=None)
    assert convert_tuple(ma) == pa

    ma = net(index=(0, -3), b=[11, 22], start=None, end=None, step=-2)
    pa = list_slice_assignment_at_python(index=(0, 2), d=[11, 22], start=None, end=None, step=-2)
    assert convert_tuple(ma) == pa

    with pytest.raises(ValueError) as e:
        net(index=(0, 2), b=Tensor([11, 22, 33]), start=None, end=None, step=2)
        _pynative_executor.sync()
        assert "the inputs should be constant, but found variable 'value' to be nonconstant" in str(e.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_multi_level_nested_list_with_tensor_or_tuple():
    """
    Feature: List slice assignment.
    Description: Test list slice assignment on multi-level nested list containing tensor or tuple elements.
    Expectation: Assignment result matches expected nested list.
    Migrated from: test_parser_list_slice_assignment.py::test_a_is_multi_level_nested_list_element_is_tensor_or_tuple
    """
    class Net(Cell):
        @jit
        def construct(self, a, index, b, start=None, end=None, step=None):
            a[index[0]][index[1]][start:end:step] = b
            return a

    def list_slice_assignment_at_python(c, index, d, start, end, step):
        c[index[0]][index[1]][start:end:step] = d
        return convert_tuple(c)

    net = Net()
    a = [[Tensor(1), 2, [11, Tensor(22), 33, 44], (4, 5)], (1, 3, 5, 7), 3]
    ma = net(a, index=(0, 2), b=[11, 22, 33], start=-3, end=-2, step=None)
    c = [[Tensor(1), 2, [11, Tensor(22), 33, 44], (4, 5)], (1, 3, 5, 7), 3]
    pa = list_slice_assignment_at_python(c,
                                         index=(0, 2),
                                         d=[11, 22, 33],
                                         start=-3,
                                         end=-2,
                                         step=None)
    assert convert_tuple(ma) == pa

    a = [[Tensor(1), 2, 11, [Tensor(22), 33, 44, (4, 5)]], (1, 3, 5, 7), 3]
    ma = net(a, index=(0, 3), b=[Tensor(11), Tensor(22), Tensor(33)], start=3, end=None, step=None)
    c = [[Tensor(1), 2, 11, [Tensor(22), 33, 44, (4, 5)]], (1, 3, 5, 7), 3]
    pa = list_slice_assignment_at_python(c,
                                         index=(0, 3),
                                         d=[Tensor(11), Tensor(22),
                                            Tensor(33)],
                                         start=3,
                                         end=None,
                                         step=None)
    assert convert_tuple(ma) == pa

    a = [1, 2, [[Tensor(1), 2, 11, Tensor(22), 33, 44], (4, 5)], (1, 3, 5, 7), 3]
    ma = net(a, index=(2, -2), b=(11, 22, 33), start=2, end=None, step=-1)
    c = [1, 2, [[Tensor(1), 2, 11, Tensor(22), 33, 44], (4, 5)], (1, 3, 5, 7), 3]
    pa = list_slice_assignment_at_python(c,
                                         index=(2, -2),
                                         d=(11, 22, 33),
                                         start=2,
                                         end=None,
                                         step=-1)
    assert convert_tuple(ma) == pa

    a = [[Tensor(1), 2, [11, Tensor(22), 33, 44], (4, 5)], (1, 3, 5, 7), 3]
    ma = net(a, index=(0, 2), b=[11, 22], start=None, end=None, step=-2)
    c = [[Tensor(1), 2, [11, Tensor(22), 33, 44], (4, 5)], (1, 3, 5, 7), 3]
    pa = list_slice_assignment_at_python(c, index=(0, 2), d=[11, 22], start=None, end=None, step=-2)
    assert convert_tuple(ma) == pa


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_assign_number_or_variable_list():
    """
    Feature: List slice assignment.
    Description: Test list slice assignment with number or variable list as value.
    Expectation: Raises TypeError for number or succeeds for variable list.
    Migrated from: test_parser_list_slice_assignment.py::test_a_is_list_b_type_is_number_or_variable_list
    """
    class Net1(Cell):
        @jit
        def construct(self, a, b, start=None, end=None, step=None):
            a[start:end:step] = b
            return a

    net = Net1()
    for b in (5, 2.0, True):
        with pytest.raises((TypeError, RuntimeError)):
            net(a=[Tensor(1), Tensor(1), 3, 4, Tensor(1)], b=b, start=None, end=1, step=None)
            _pynative_executor.sync()

    b = Tensor([1, 1, 1])
    c = net(a=[Tensor(1), Tensor(1), 3, 4, Tensor(1)], b=b, start=None, end=1, step=None)
    assert len(c) == 7


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_nested_list_element_assignment():
    # pylint: disable=line-too-long
    """
    Feature: List slice assignment.
    Description: Test list slice assignment on nested list element.
    Expectation: Assignment result matches expected list.
    Migrated from: test_parser_list_slice_assignment.py::test_a_is_nested_list_and_the_element_with_type_list_is_assigned_a_value
    """
    class Net3(Cell):
        @jit
        def construct(self, a, c, index):
            b = a[index[0]]
            b[index[1]] = c
            return a

    # pylint: disable=redefined-outer-name
    class Net2(Cell):
        @jit
        def construct(self, a, c, index, start, end, step):
            b = [a[index[0]], a[index[1]]]
            b[start:end:step] = c
            return a

    def list_slice_assignment_at_python(d, f, index):
        e = d[index[0]]
        e[index[1]] = f
        return d

    net = Net2()
    a = [[1, 2, 3, 4], 5, 6, 7]
    ma = net(a, c=[123321, 4], index=(0, 2), start=None, end=None, step=None)
    assert ma == [[1, 2, 3, 4], 5, 6, 7]

    net = Net3()
    a = [[1, 2, 3, 4], 5, 6, 7]
    ma = net(a, c=123321, index=(0, 2))
    assert ma == [[1, 2, 3, 4], 5, 6, 7]

    d = [[1, 2, 3, 4], 5, 6, 7]
    pa = list_slice_assignment_at_python(d, f=123321, index=(0, 2))
    assert pa == [[1, 2, 123321, 4], 5, 6, 7]
