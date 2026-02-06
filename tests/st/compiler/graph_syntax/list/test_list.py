# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Test list operations"""

import numpy as np
import pytest
from mindspore import Tensor, jit
from mindspore.graph.api import _pynative_executor
from mindspore.nn import Cell, ReLU
from mindspore.ops import operations as P
from mindspore.train.model import Model
from tests.mark_utils import arg_mark
from tests.st.compiler.utils import assert_equal
from tests.st.pi_jit.share.grad import compute_grad_of_net_inputs


class Net1(Cell):
    def __init__(self, lista):
        super().__init__()
        self.lista = lista
        self.fla = ReLU()

    def construct(self, x):
        for _ in self.lista:
            x = self.fla(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_with_for():
    """
    Feature: List iteration in for loop.
    Description: Test iterating over a list in a for loop within Cell construct.
    Expectation: JIT result and gradient match pynative result.
    Migrated from: test_parser_list.py::test_parser_list_with_for_001
    """
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = Net1((1, 2))
    pynative_net.set_grad()
    pynative_result = pynative_net(input_me)

    sens = Tensor(np.random.randn(*pynative_result.shape).astype(np.float32))
    pynative_grad = compute_grad_of_net_inputs(pynative_net, input_me, sens=sens)

    # JIT mode execution
    jit_net = Net1((1, 2))
    jit_net.set_grad()
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    jit_grad = compute_grad_of_net_inputs(jit_net, input_me, sens=sens)

    # Compare forward results and gradients
    assert_equal(pynative_result, jit_result)
    assert_equal(pynative_grad, jit_grad, decimal=5)


class Net2(Cell):
    def __init__(self, lista):
        super().__init__()
        self.lista = lista
        self.fla = ReLU()

    def construct(self, x):
        if self.lista[0]:
            x = self.fla(x)
        if self.lista[1]:
            x = self.fla(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_using_index():
    """
    Feature: List indexing operation.
    Description: Test accessing list elements using index in Cell construct.
    Expectation: JIT result and gradient match pynative result.
    Migrated from: test_parser_list.py::test_parser_list_using_index_002
    """
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = Net2([1, 2])
    pynative_net.set_grad()
    pynative_result = pynative_net(input_me)

    sens = Tensor(np.random.randn(*pynative_result.shape).astype(np.float32))
    pynative_grad = compute_grad_of_net_inputs(pynative_net, input_me, sens=sens)

    # JIT mode execution
    jit_net = Net2([1, 2])
    jit_net.set_grad()
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    jit_grad = compute_grad_of_net_inputs(jit_net, input_me, sens=sens)

    # Compare forward results and gradients
    assert_equal(pynative_result, jit_result)
    assert_equal(pynative_grad, jit_grad, decimal=5)


class Net4(Cell):
    def __init__(self, lista):
        super().__init__()
        self.lista = lista
        self.fla = P.Flatten()

    @jit
    def construct(self, x):
        for _ in self.lista[0:2]:
            x = self.fla(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sliced_list_using_index():
    """
    Feature: List slice operation.
    Description: Test slicing a list and iterating over the slice in Cell construct.
    Expectation: The network executes successfully.
    Migrated from: test_parser_list.py::test_parser_sliced_list_using_index_003
    """
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)
    net = Net4([1, 2, 3])
    model = Model(net)
    model.predict(input_me)


class Net3(Cell):
    def __init__(self):
        super().__init__()
        self.list = []
        self.list.append(ReLU())
        self.list.append(P.Flatten())

    @jit
    def construct(self, x):
        for i in self.list:
            x = i(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_cell_list_with_for():
    """
    Feature: List of cells iteration in for loop.
    Description: Test iterating over a list of cells in a for loop within Cell construct.
    Expectation: The network executes successfully.
    Migrated from: test_parser_list.py::test_parser_cell_list_with_for_002
    """
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)
    net = Net3()
    model = Model(net)
    model.predict(input_me)


class NetGet0001(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[0] == 1:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_get_0001():
    """
    Feature: List element access using index 0.
    Description: Test accessing list element at index 0 and using it in conditional logic.
    Expectation: JIT result and gradient match pynative result.
    Migrated from: test_parser_list.py::test_parser_list_get_0001
    """
    l = [1, [20.5, "2020"], 3, 4, 5, 6]
    input_np = np.random.randn(2, 2, 1, 1).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetGet0001(l)
    pynative_net.set_grad()
    pynative_result = pynative_net(input_me)

    sens = Tensor(np.random.randn(*pynative_result.shape).astype(np.float32))
    pynative_grad = compute_grad_of_net_inputs(pynative_net, input_me, sens=sens)

    # JIT mode execution
    jit_net = NetGet0001(l)
    jit_net.set_grad()
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    jit_grad = compute_grad_of_net_inputs(jit_net, input_me, sens=sens)

    # Compare forward results and gradients
    assert_equal(pynative_result, jit_result)
    assert_equal(pynative_grad, jit_grad, decimal=5)


class NetGet0002(Cell):
    def __init__(self, l):
        super().__init__()
        self.list = l
        self.relu = ReLU()

    def construct(self, x):
        if self.list[-1] == 6:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_get_0002():
    """
    Feature: List element access using negative index.
    Description: Test accessing list element using negative index (-1) and using it in conditional logic.
    Expectation: JIT result and gradient match pynative result.
    Migrated from: test_parser_list.py::test_parser_list_get_0002
    """
    l = [1, [20.5, "2020"], 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetGet0002(l)
    pynative_net.set_grad()
    pynative_result = pynative_net(input_me)

    sens = Tensor(np.random.randn(*pynative_result.shape).astype(np.float32))
    pynative_grad = compute_grad_of_net_inputs(pynative_net, input_me, sens=sens)

    # JIT mode execution
    jit_net = NetGet0002(l)
    jit_net.set_grad()
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    jit_grad = compute_grad_of_net_inputs(jit_net, input_me, sens=sens)

    # Compare forward results and gradients
    assert_equal(pynative_result, jit_result)
    assert_equal(pynative_grad, jit_grad, decimal=5)


class NetGet0003(Cell):
    @jit
    def construct(self):
        l = [1, [20, 30], 3, 4, 5, 6]
        return l[1]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_get_0003():
    """
    Feature: List element access returning nested list.
    Description: Test accessing list element that contains a nested list and returning it.
    Expectation: Result matches expected constant value.
    Migrated from: test_parser_list.py::test_parser_list_get_0003
    """
    net = NetGet0003()
    out_me = net()
    assert out_me == [20, 30]


class NetGet0004(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista

    @jit
    def construct(self):
        return self.list[1]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_get_0004():
    """
    Feature: List element access returning nested list with float elements.
    Description: Test accessing list element that contains a nested list with float elements.
    Expectation: Result matches expected constant value.
    Migrated from: test_parser_list.py::test_parser_list_get_0004
    """
    l = [1, [20.5, 30.5], 3, 4, 5, 6]
    net = NetGet0004(l)
    out_me = net()
    assert out_me == [20.5, 30.5]


class NetGet0005(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[1] == ["2020", "2020"]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_get_0005():
    """
    Feature: List element access with string comparison.
    Description: Test accessing list element that contains a nested list with string elements and comparing it.
    Expectation: JIT result and gradient match pynative result.
    Migrated from: test_parser_list.py::test_parser_list_get_0005
    """
    l = [1, ["2020", "2020"], 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetGet0005(l)
    pynative_net.set_grad()
    pynative_result = pynative_net(input_me)

    sens = Tensor(np.random.randn(*pynative_result.shape).astype(np.float32))
    pynative_grad = compute_grad_of_net_inputs(pynative_net, input_me, sens=sens)

    # JIT mode execution
    jit_net = NetGet0005(l)
    jit_net.set_grad()
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    jit_grad = compute_grad_of_net_inputs(jit_net, input_me, sens=sens)

    # Compare forward results and gradients
    assert_equal(pynative_result, jit_result)
    assert_equal(pynative_grad, jit_grad, decimal=5)


class NetGet0006(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[1] == ["2020", 20, 30.5]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_get_0006():
    """
    Feature: List element access with mixed type comparison.
    Description: Test accessing list element that contains a nested list with
    mixed types (string, int, float) and comparing it.
    Expectation: JIT result and gradient match pynative result.
    Migrated from: test_parser_list.py::test_parser_list_get_0006
    """
    l = [1, ["2020", 20, 30.5], 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetGet0006(l)
    pynative_net.set_grad()
    pynative_result = pynative_net(input_me)

    sens = Tensor(np.random.randn(*pynative_result.shape).astype(np.float32))
    pynative_grad = compute_grad_of_net_inputs(pynative_net, input_me, sens=sens)

    # JIT mode execution
    jit_net = NetGet0006(l)
    jit_net.set_grad()
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    jit_grad = compute_grad_of_net_inputs(jit_net, input_me, sens=sens)

    # Compare forward results and gradients
    assert_equal(pynative_result, jit_result)
    assert_equal(pynative_grad, jit_grad, decimal=5)


class NetGet0007(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[1][1] == "2020":
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_get_0007():
    """
    Feature: Nested list element access.
    Description: Test accessing nested list element using double index and using it in conditional logic.
    Expectation: JIT result and gradient match pynative result.
    Migrated from: test_parser_list.py::test_parser_list_get_0007
    """
    l = [1, [20.5, "2020"], 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetGet0007(l)
    pynative_net.set_grad()
    pynative_result = pynative_net(input_me)

    sens = Tensor(np.random.randn(*pynative_result.shape).astype(np.float32))
    pynative_grad = compute_grad_of_net_inputs(pynative_net, input_me, sens=sens)

    # JIT mode execution
    jit_net = NetGet0007(l)
    jit_net.set_grad()
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    jit_grad = compute_grad_of_net_inputs(jit_net, input_me, sens=sens)

    # Compare forward results and gradients
    assert_equal(pynative_result, jit_result)
    assert_equal(pynative_grad, jit_grad, decimal=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_get_0008():
    """
    Feature: Nested list element access with tensor append.
    Description: Test accessing nested list element where list contains tensors appended at runtime.
    Expectation: The network executes successfully.
    Migrated from: test_parser_list.py::test_parser_list_get_0008
    """

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.relu = ReLU()

        @jit
        def construct(self, x, y):
            l = [[]]
            l[0].append(x)
            l[0].append(y)
            return l[0][0]

    input_np_x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me_x = Tensor(input_np_x)
    input_np_y = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me_y = Tensor(input_np_y)
    net = Net()
    net(input_me_x, input_me_y)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_get_0009():
    """
    Feature: Nested list element access with int32 tensor append.
    Description: Test accessing nested list element where list contains int32 tensors appended at runtime.
    Expectation: The network executes successfully.
    Migrated from: test_parser_list.py::test_parser_list_get_0009
    """

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.relu = ReLU()

        @jit
        def construct(self, x, y):
            l = [[]]
            l[0].append(x)
            l[0].append(y)
            return l[0][0]

    input_np_x = np.random.randn(2, 3, 4, 5).astype(np.int32)
    input_me_x = Tensor(input_np_x)
    input_np_y = np.random.randn(2, 3, 4, 5).astype(np.int32)
    input_me_y = Tensor(input_np_y)
    net = Net()
    net(input_me_x, input_me_y)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_get_0010():
    """
    Feature: Nested list element access with mixed tensor types.
    Description: Test accessing nested list element where list contains mixed
    tensor types (float32 and int32) appended at runtime.
    Expectation: The network executes successfully.
    Migrated from: test_parser_list.py::test_parser_list_get_0010
    """

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.relu = ReLU()

        @jit
        def construct(self, x, y):
            l = [[]]
            l[0].append(x)
            l[0].append(y)
            return l[0][0]

    input_np_x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me_x = Tensor(input_np_x)
    input_np_y = np.random.randn(2, 3, 4, 5).astype(np.int32)
    input_me_y = Tensor(input_np_y)
    net = Net()
    net(input_me_x, input_me_y)


class NetGetException0001(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    @jit
    def construct(self, x):
        if self.list[2020] == "2020":
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_get_exception():
    """
    Feature: List index out of range exception.
    Description: Test accessing list element with index that exceeds list size.
    Expectation: Exception is raised with correct error message.
    Migrated from: test_parser_list.py::test_parser_list_get_exception_0001
    """
    l = [1, [20.5, "2020"], 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)
    net = NetGetException0001(l)
    with pytest.raises(Exception):
        net(input_me)
        _pynative_executor.sync()


# Slice test cases
class NetSlice0004(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[1:5:2] == [2, 4]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0004():
    """
    Feature: List slice operation with step.
    Description: Test slicing list with start < end and step <= list size.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0004
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0004(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0004(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0005(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[1:-2:2] == [2, 4]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0005():
    """
    Feature: List slice operation with negative end index.
    Description: Test slicing list with start < end (using negative index) and step <= list size.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0005
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0005(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0005(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0006(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[-6:4:2] == [1, 3]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0006():
    """
    Feature: List slice operation with negative start index.
    Description: Test slicing list with negative start index, start < end, and step <= list size.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0006
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0006(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0006(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0007(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[-6:-2:2] == [1, 3]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0007():
    """
    Feature: List slice operation with negative indices and step.
    Description: Test slicing list[-6:-2:2] with start < end and step <= list size.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0007
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0007(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0007(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0008(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[1:5:10] == [2]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0008():
    """
    Feature: List slice operation with step greater than list size.
    Description: Test slicing list[1:5:10] with start < end and step > list size.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0008
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0008(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0008(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0009(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[1:-2:10] == [2]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0009():
    """
    Feature: List slice operation with negative end index and large step.
    Description: Test slicing list[1:-2:10] with start < end and step > list size.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0009
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0009(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0009(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0010(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[-6:4:10] == [1]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0010():
    """
    Feature: List slice operation with negative start index and large step.
    Description: Test slicing list[-6:4:10] with start < end and step > list size.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0010
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0010(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0010(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0013(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[-6:-2:10] == [1]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0013():
    """
    Feature: List slice operation with negative indices and large step.
    Description: Test slicing list[-6:-2:10] with start < end and step > list size.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0013
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0013(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0013(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0014(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if not self.list[5:2:2]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0014():
    """
    Feature: List slice operation with start > end.
    Description: Test slicing list[5:2:2] where start > end.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0014
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0014(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0014(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0016(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if not self.list[5:-6:2]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0016():
    """
    Feature: List slice operation with start > end using negative index.
    Description: Test slicing list[5:-6:2] where start > end.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0016
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0016(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0016(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0017(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if not self.list[5:5:2]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0017():
    """
    Feature: List slice operation with start equal to end.
    Description: Test slicing list[5:5:2] where start = end.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0017
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0017(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0017(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0018(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if not self.list[-5:-5:2]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0018():
    """
    Feature: List slice operation with negative indices where start equals end.
    Description: Test slicing list[-5:-5:2] where start = end.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0018
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0018(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0018(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0019(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[:2:2] == [1]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0019():
    """
    Feature: List slice operation without start index.
    Description: Test slicing list[:2:2] without start index, end in range, step <= list size.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0019
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0019(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0019(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0020(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[:-2:2] == [1, 3]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0020():
    """
    Feature: List slice operation without start index and negative end index.
    Description: Test slicing list[:-2:2] without start index, end in range, step <= list size.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0020
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0020(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0020(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0021(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[:2:10] == [1]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0021():
    """
    Feature: List slice operation without start index and large step.
    Description: Test slicing list[:2:10] without start index, end in range, step > list size.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0021
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0021(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0021(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0022(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[:-2:10] == [1]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0022():
    """
    Feature: List slice operation without start index, negative end index and large step.
    Description: Test slicing list[:-2:10] without start index, end in range, step > list size.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0022
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0022(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0022(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0023(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[:10:2] == [1, 3, 5]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0023():
    """
    Feature: List slice operation without start index and end out of range.
    Description: Test slicing list[:10:2] without start index, end out of range, step <= list size.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0023
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0023(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0023(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0024(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[:-10:2] == []:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0024():
    """
    Feature: List slice operation without start index, negative end out of range.
    Description: Test slicing list[:-10:2] without start index, end out of range, step <= list size.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0024
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0024(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0024(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0025(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[:10:10] == [1]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0025():
    """
    Feature: List slice operation without start index, end out of range and large step.
    Description: Test slicing list[:10:10] without start index, end out of range, step > list size.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0025
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0025(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0025(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0026(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if not self.list[:-10:10]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0026():
    """
    Feature: List slice operation without start index, negative end out of range and large step.
    Description: Test slicing list[:-10:10] without start index, end out of range, step > list size.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0026
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0026(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0026(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0027(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[2::2] == [3, 5]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0027():
    """
    Feature: List slice operation without end index.
    Description: Test slicing list[2::2] without end index, start in range, step <= list size.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0027
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0027(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0027(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0028(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[-2::2] == [5]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0028():
    """
    Feature: List slice operation without end index and negative start index.
    Description: Test slicing list[-2::2] without end index, start in range, step <= list size.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0028
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0028(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0028(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0029(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[2::10] == [3]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0029():
    """
    Feature: List slice operation without end index and large step.
    Description: Test slicing list[2::10] without end index, start in range, step > list size.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0029
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0029(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0029(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0030(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[-2::10] == [5]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0030():
    """
    Feature: List slice operation without end index, negative start index and large step.
    Description: Test slicing list[-2::10] without end index, start in range, step > list size.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0030
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0030(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0030(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0031(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[10::2] == []:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0031():
    """
    Feature: List slice operation without end index and start out of range.
    Description: Test slicing list[10::2] without end index, start out of range, step <= list size.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0031
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0031(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0031(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0032(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[-10::2] == [1, 3, 5]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0032():
    """
    Feature: List slice operation without end index and negative start out of range.
    Description: Test slicing list[-10::2] without end index, start out of range, step <= list size.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0032
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0032(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0032(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0033(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[10::10] == []:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0033():
    """
    Feature: List slice operation without end index, start out of range and large step.
    Description: Test slicing list[10::10] without end index, start out of range, step > list size.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0033
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0033(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0033(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0034(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[-10::10] == [1]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0034():
    """
    Feature: List slice operation without end index, negative start out of range and large step.
    Description: Test slicing list[-10::10] without end index, start out of range, step > list size.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0034
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0034(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0034(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0035(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[1:5] == [2, 3, 4, 5]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0035():
    """
    Feature: List slice operation without step parameter.
    Description: Test slicing list[1:5] without step parameter, start in range, end in range.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0035
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0035(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0035(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0036(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[1:5:] == [2, 3, 4, 5]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0036():
    """
    Feature: List slice operation with empty step parameter.
    Description: Test slicing list[1:5:] without step parameter, start in range, end in range.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0036
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0036(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0036(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0078(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[1:-1] == [2, 3, 4, 5]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0078():
    """
    Feature: List slice operation with negative end index.
    Description: Test slicing list[1:-1] without step parameter, start in range, end in range.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0078
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0078(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0078(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0037(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[-5:5] == [2, 3, 4, 5]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0037():
    """
    Feature: List slice operation with negative start index.
    Description: Test slicing list[-5:5] without step parameter, start in range, end in range.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0037
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0037(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0037(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0038(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[-5:1] == []:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0038():
    """
    Feature: List slice operation with start > end.
    Description: Test slicing list[-5:1] without step parameter, start in range, end in range.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0038
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0038(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0038(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0039(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[1:10] == [2, 3, 4, 5, 6]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0039():
    """
    Feature: List slice operation with end out of range.
    Description: Test slicing list[1:10] without step parameter, start in range, end out of range.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0039
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0039(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0039(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0040(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[-1:10] == [6]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0040():
    """
    Feature: List slice operation with negative start and end out of range.
    Description: Test slicing list[-1:10] without step parameter, start in range, end out of range.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0040
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0040(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0040(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0041(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[-1:10] == [6]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0041():
    """
    Feature: List slice operation with negative start and end out of range.
    Description: Test slicing list[-1:10] without step parameter, start in range, end out of range.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0041
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0041(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0041(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0042(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[1:-10] == []:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0042():
    """
    Feature: List slice operation with negative end out of range.
    Description: Test slicing list[1:-10] without step parameter, start in range, end out of range.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0042
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0042(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0042(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0043(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[10:1] == []:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0043():
    """
    Feature: List slice operation with start out of range.
    Description: Test slicing list[10:1] without step parameter, start out of range, end in range.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0043
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0043(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0043(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0044(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[-10:1] == [1]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0044():
    """
    Feature: List slice operation with negative start out of range.
    Description: Test slicing list[-10:1] without step parameter, start out of range, end in range.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0044
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0044(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0044(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0045(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[10:-1] == []:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0045():
    """
    Feature: List slice operation with start out of range and negative end.
    Description: Test slicing list[10:-1] without step parameter, start out of range, end in range.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0045
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0045(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0045(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0046(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[-10:-8] == []:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0046():
    """
    Feature: List slice operation with both indices out of range.
    Description: Test slicing list[-10:-8] without step parameter, start out of range, end out of range.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0046
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0046(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0046(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0047(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[-10:-18] == []:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0047():
    """
    Feature: List slice operation with both negative indices out of range.
    Description: Test slicing list[-10:-18] without step parameter, start out of range, end out of range.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0047
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0047(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0047(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0048(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[-10:8] == [1, 2, 3, 4, 5, 6]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0048():
    """
    Feature: List slice operation with negative start and positive end out of range.
    Description: Test slicing list[-10:8] without step parameter, start out of range, end out of range.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0048
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0048(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0048(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0049(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[-10:8] == [1, 2, 3, 4, 5, 6]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0049():
    """
    Feature: List slice operation with negative start and positive end out of range.
    Description: Test slicing list[-10:8] without step parameter, start out of range, end out of range.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0049
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0049(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0049(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0050(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[8:10] == []:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0050():
    """
    Feature: List slice operation with both positive indices out of range.
    Description: Test slicing list[8:10] without step parameter, start out of range, end out of range.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0050
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0050(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0050(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0051(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[10:8] == []:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0051():
    """
    Feature: List slice operation with start > end, both out of range.
    Description: Test slicing list[10:8] without step parameter, start out of range, end out of range.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0051
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0051(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0051(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0052(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[:5:] == [1, 2, 3, 4, 5]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0052():
    """
    Feature: List slice operation without start and step parameters.
    Description: Test slicing list[:5:] without start and step parameters, end in range.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0052
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0052(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0052(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0053(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[:-2:] == [1, 2, 3, 4]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0053():
    """
    Feature: List slice operation without start and step parameters, negative end.
    Description: Test slicing list[:-2:] without start and step parameters, end in range.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0053
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0053(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0053(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0054(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[:10:] == [1, 2, 3, 4, 5, 6]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0054():
    """
    Feature: List slice operation without start and step parameters, end out of range.
    Description: Test slicing list[:10:] without start and step parameters, end out of range.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0054
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0054(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0054(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0055(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[:-10:] == []:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0055():
    """
    Feature: List slice operation without start and step parameters, negative end out of range.
    Description: Test slicing list[:-10:] without start and step parameters, end out of range.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0055
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0055(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0055(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0056(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[5::] == [6]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0056():
    """
    Feature: List slice operation without end and step parameters.
    Description: Test slicing list[5::] without end and step parameters, start in range.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0056
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0056(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0056(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0057(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[-2::] == [5, 6]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0057():
    """
    Feature: List slice operation without end and step parameters, negative start.
    Description: Test slicing list[-2::] without end and step parameters, start in range.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0057
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0057(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0057(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0058(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[10::] == []:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0058():
    """
    Feature: List slice operation without end and step parameters, start out of range.
    Description: Test slicing list[10::] without end and step parameters, start out of range.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0058
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0058(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0058(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0059(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[::-10] == [6]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0059():
    """
    Feature: List slice operation with negative step, no start and end.
    Description: Test slicing list[::-10] without start and end, negative step >= list size.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0059
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0059(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0059(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0060(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[::-2] == [6, 4, 2]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0060():
    """
    Feature: List slice operation with negative step, no start and end.
    Description: Test slicing list[::-2] without start and end, negative step < list size.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0060
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0060(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0060(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0061(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[::10] == [1]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0061():
    """
    Feature: List slice operation without start and end, large positive step.
    Description: Test slicing list[::10] without start and end, step > 0, step >= list size.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0061
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0061(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0061(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0062(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[::2] == [1, 3, 5]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0062():
    """
    Feature: List slice operation without start and end, positive step.
    Description: Test slicing list[::2] without start and end, step > 0, step < list size.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0062
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0062(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0062(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0063(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[::] == [1, 2, 3, 4, 5, 6]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0063():
    """
    Feature: List slice operation without all parameters.
    Description: Test slicing list[::] without start, end and step parameters.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0063
    """
    l = [1, 2, 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0063(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0063(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


# list in list
class NetSlice0064(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[1] == [10, 20, 30, 40, 50, 60]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0064():
    """
    Feature: List index operation to get list from list.
    Description: Test getting list from list using index list[1].
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0064
    """
    l = [1, [10, 20, 30, 40, 50, 60], 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0064(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0064(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0065(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[1][-6:4:2] == [10, 30]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0065():
    """
    Feature: List slice operation on nested list.
    Description: Test slicing list[1][-6:4:2] to slice list from nested list.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0065
    """
    l = [1, [10, 20, 30, 40, 50, 60], 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0065(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0065(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0066(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[1][5:2:2] == []:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0066():
    """
    Feature: List slice operation on nested list with start > end.
    Description: Test slicing list[1][5:2:2] to slice list from nested list.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0066
    """
    l = [1, [10, 20, 30, 40, 50, 60], 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0066(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0066(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0067(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[1][:2:10] == [10]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0067():
    """
    Feature: List slice operation on nested list without start.
    Description: Test slicing list[1][:2:10] to slice list from nested list.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0067
    """
    l = [1, [10, 20, 30, 40, 50, 60], 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0067(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0067(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0068(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[1][:2:10] == [10]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0068():
    """
    Feature: List slice operation on nested list without start.
    Description: Test slicing list[1][:2:10] to slice list from nested list.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0068
    """
    l = [1, [10, 20, 30, 40, 50, 60], 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0068(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0068(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0069(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[1][1:5] == [20, 30, 40, 50]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0069():
    """
    Feature: List slice operation on nested list without step.
    Description: Test slicing list[1][1:5] to slice list from nested list.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0069
    """
    l = [1, [10, 20, 30, 40, 50, 60], 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0069(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0069(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0070(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[1][10:-1] == []:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0070():
    """
    Feature: List slice operation on nested list with start out of range.
    Description: Test slicing list[1][10:-1] to slice list from nested list.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0070
    """
    l = [1, [10, 20, 30, 40, 50, 60], 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0070(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0070(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0071(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[1][:-10:] == []:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0071():
    """
    Feature: List slice operation on nested list without start and step.
    Description: Test slicing list[1][:-10:] to slice list from nested list.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0071
    """
    l = [1, [10, 20, 30, 40, 50, 60], 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0071(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0071(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0072(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[1][::] == [10, 20, 30, 40, 50, 60]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0072():
    """
    Feature: List slice operation on nested list without all parameters.
    Description: Test slicing list[1][::] to slice list from nested list.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0072
    """
    l = [1, [10, 20, 30, 40, 50, 60], 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0072(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0072(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0073(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[1][0] == 10:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0073():
    """
    Feature: List index operation on nested list with single element.
    Description: Test indexing list[1][0] where size of list[1] is 1.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0073
    """
    l = [1, [10], 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0073(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0073(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0074(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[1][:] == [10]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0074():
    """
    Feature: List slice operation on nested list with single element.
    Description: Test slicing list[1][:] where size of list[1] is 1.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0074
    """
    l = [1, [10], 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0074(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0074(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0075(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[1][1:] == []:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0075():
    """
    Feature: List slice operation on nested list with single element.
    Description: Test slicing list[1][1:] where size of list[1] is 1.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0075
    """
    l = [1, [10], 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0075(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0075(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0076(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[1][:1] == [10]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0076():
    """
    Feature: List slice operation on nested list with single element.
    Description: Test slicing list[1][:1] where size of list[1] is 1.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0076
    """
    l = [1, [10], 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0076(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0076(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetSlice0077(Cell):
    def __init__(self, lista):
        super().__init__()
        self.list = lista
        self.relu = ReLU()

    def construct(self, x):
        if self.list[1][:1:2] == [10]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_0077():
    """
    Feature: List slice operation on nested list with single element.
    Description: Test slicing list[1][:1:2] where size of list[1] is 1.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_slice_0077
    """
    l = [1, [10], 3, 4, 5, 6]
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetSlice0077(l)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetSlice0077(l)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


# Append test cases
class NetAppend0001(Cell):
    def __init__(self):
        super().__init__()
        self.relu = ReLU()

    def construct(self, x):
        l = [10, 20]
        l.append(1)
        if l[2] == 1:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_append_0001():
    """
    Feature: List append operation with int.
    Description: Test appending an integer to a list and using it in conditional logic.
    Expectation: JIT result and gradient match pynative result.
    Migrated from: test_parser_list.py::test_parser_list_append_0001
    """
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetAppend0001()
    pynative_net.set_grad()
    pynative_result = pynative_net(input_me)

    sens = Tensor(np.random.randn(*pynative_result.shape).astype(np.float32))
    pynative_grad = compute_grad_of_net_inputs(pynative_net, input_me, sens=sens)

    # JIT mode execution
    jit_net = NetAppend0001()
    jit_net.set_grad()
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    jit_grad = compute_grad_of_net_inputs(jit_net, input_me, sens=sens)

    # Compare forward results and gradients
    assert_equal(pynative_result, jit_result)
    assert_equal(pynative_grad, jit_grad, decimal=5)


class NetAppend0002(Cell):
    def __init__(self):
        super().__init__()
        self.relu = ReLU()

    def construct(self, x):
        l = []
        l.append(1.5)
        if l[0] == 1.5:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_append_0002():
    """
    Feature: List append operation with float.
    Description: Test appending a float to a list and using it in conditional logic.
    Expectation: JIT result and gradient match pynative result.
    Migrated from: test_parser_list.py::test_parser_list_append_0002
    """
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetAppend0002()
    pynative_net.set_grad()
    pynative_result = pynative_net(input_me)

    sens = Tensor(np.random.randn(*pynative_result.shape).astype(np.float32))
    pynative_grad = compute_grad_of_net_inputs(pynative_net, input_me, sens=sens)

    # JIT mode execution
    jit_net = NetAppend0002()
    jit_net.set_grad()
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    jit_grad = compute_grad_of_net_inputs(jit_net, input_me, sens=sens)

    # Compare forward results and gradients
    assert_equal(pynative_result, jit_result)
    assert_equal(pynative_grad, jit_grad, decimal=5)


class NetAppend0003(Cell):
    def __init__(self):
        super().__init__()
        self.relu = ReLU()

    def construct(self, x):
        l = []
        l.append('2020')
        if l[0] == '2020':
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_append_0003():
    """
    Feature: List append operation with string.
    Description: Test appending a string to a list and using it in conditional logic.
    Expectation: JIT result and gradient match pynative result.
    Migrated from: test_parser_list.py::test_parser_list_append_0003
    """
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetAppend0003()
    pynative_net.set_grad()
    pynative_result = pynative_net(input_me)

    sens = Tensor(np.random.randn(*pynative_result.shape).astype(np.float32))
    pynative_grad = compute_grad_of_net_inputs(pynative_net, input_me, sens=sens)

    # JIT mode execution
    jit_net = NetAppend0003()
    jit_net.set_grad()
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    jit_grad = compute_grad_of_net_inputs(jit_net, input_me, sens=sens)

    # Compare forward results and gradients
    assert_equal(pynative_result, jit_result)
    assert_equal(pynative_grad, jit_grad, decimal=5)


class NetAppend0004(Cell):
    def __init__(self):
        super().__init__()
        self.relu = ReLU()

    def construct(self, x):
        l = []
        l.append(True)
        if l[0]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_append_0004():
    """
    Feature: List append operation with bool.
    Description: Test appending a boolean to a list and using it in conditional logic.
    Expectation: JIT result and gradient match pynative result.
    Migrated from: test_parser_list.py::test_parser_list_append_0004
    """
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetAppend0004()
    pynative_net.set_grad()
    pynative_result = pynative_net(input_me)

    sens = Tensor(np.random.randn(*pynative_result.shape).astype(np.float32))
    pynative_grad = compute_grad_of_net_inputs(pynative_net, input_me, sens=sens)

    # JIT mode execution
    jit_net = NetAppend0004()
    jit_net.set_grad()
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    jit_grad = compute_grad_of_net_inputs(jit_net, input_me, sens=sens)

    # Compare forward results and gradients
    assert_equal(pynative_result, jit_result)
    assert_equal(pynative_grad, jit_grad, decimal=5)


class NetAppend0005(Cell):
    def __init__(self):
        super().__init__()
        self.relu = ReLU()

    def construct(self, x):
        l = []
        l.append([10])
        if l[0] == [10]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_append_0005():
    """
    Feature: List append operation with list.
    Description: Test appending a list to a list and using it in conditional logic.
    Expectation: JIT result and gradient match pynative result.
    Migrated from: test_parser_list.py::test_parser_list_append_0005
    """
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetAppend0005()
    pynative_net.set_grad()
    pynative_result = pynative_net(input_me)

    sens = Tensor(np.random.randn(*pynative_result.shape).astype(np.float32))
    pynative_grad = compute_grad_of_net_inputs(pynative_net, input_me, sens=sens)

    # JIT mode execution
    jit_net = NetAppend0005()
    jit_net.set_grad()
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    jit_grad = compute_grad_of_net_inputs(jit_net, input_me, sens=sens)

    # Compare forward results and gradients
    assert_equal(pynative_result, jit_result)
    assert_equal(pynative_grad, jit_grad, decimal=5)


class NetAppend0006(Cell):
    def __init__(self):
        super().__init__()
        self.relu = ReLU()

    def construct(self, x):
        l = []
        l.append((10,))
        if l[0] == (10,):
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_append_0006():
    """
    Feature: List append operation with tuple.
    Description: Test appending a tuple to a list and using it in conditional logic.
    Expectation: JIT result and gradient match pynative result.
    Migrated from: test_parser_list.py::test_parser_list_append_0006
    """
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetAppend0006()
    pynative_net.set_grad()
    pynative_result = pynative_net(input_me)

    sens = Tensor(np.random.randn(*pynative_result.shape).astype(np.float32))
    pynative_grad = compute_grad_of_net_inputs(pynative_net, input_me, sens=sens)

    # JIT mode execution
    jit_net = NetAppend0006()
    jit_net.set_grad()
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    jit_grad = compute_grad_of_net_inputs(jit_net, input_me, sens=sens)

    # Compare forward results and gradients
    assert_equal(pynative_result, jit_result)
    assert_equal(pynative_grad, jit_grad, decimal=5)


# Assign test cases
class NetAssign0001(Cell):
    def __init__(self):
        super().__init__()
        self.relu = ReLU()

    def construct(self, x):
        l = [1, 2, 3]
        l[0] = 10
        if l[0] == 10:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_assign_0001():
    """
    Feature: List assignment operation with int.
    Description: Test assigning an integer to a list element and using it in conditional logic.
    Expectation: JIT result and gradient match pynative result.
    Migrated from: test_parser_list.py::test_parser_list_assign_0001
    """
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetAssign0001()
    pynative_net.set_grad()
    pynative_result = pynative_net(input_me)

    sens = Tensor(np.random.randn(*pynative_result.shape).astype(np.float32))
    pynative_grad = compute_grad_of_net_inputs(pynative_net, input_me, sens=sens)

    # JIT mode execution
    jit_net = NetAssign0001()
    jit_net.set_grad()
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    jit_grad = compute_grad_of_net_inputs(jit_net, input_me, sens=sens)

    # Compare forward results and gradients
    assert_equal(pynative_result, jit_result)
    assert_equal(pynative_grad, jit_grad, decimal=5)


class NetAssign0002(Cell):
    def __init__(self):
        super().__init__()
        self.relu = ReLU()

    def construct(self, x):
        l = [[10, 20], 2]
        l[0][0] = 100
        if l[0][0] == 100:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_assign_0002():
    """
    Feature: Nested list assignment operation.
    Description: Test assigning a value to a nested list element and using it in conditional logic.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_assign_0002
    """
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)

    # Pynative mode execution
    pynative_net = NetAssign0002()
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = NetAssign0002()
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


class NetAssign0003(Cell):
    @jit
    def construct(self):
        l = [1, 2, 3]
        l[0] = True
        return l[0]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_assign_0003():
    """
    Feature: List assignment operation with bool.
    Description: Test assigning a boolean to a list element and returning it.
    Expectation: Result is True.
    Migrated from: test_parser_list.py::test_parser_list_assign_0003
    """
    net = NetAssign0003()
    out_me = net()
    assert out_me


class NetAssign0004(Cell):
    def __init__(self):
        super().__init__()
        self.relu = ReLU()

    @jit
    def construct(self):
        l = [1, 2, 3]
        l[0] = "2020"
        return l[0]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_assign_0004():
    """
    Feature: List assignment operation with string.
    Description: Test assigning a string to a list element and returning it.
    Expectation: Result matches expected constant value.
    Migrated from: test_parser_list.py::test_parser_list_assign_0004
    """
    net = NetAssign0004()
    out_me = net()
    assert out_me == "2020"


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_assign_0005():
    """
    Feature: List assignment with tensor unpacking.
    Description: Test unpacking a list containing tensors and using the first element.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_list.py::test_parser_list_assign_0005
    """

    class Net(Cell):
        def __init__(self, t):
            super().__init__()
            self.tuple = t
            self.relu = P.ReLU()

        def construct(self):
            x, _ = self.tuple
            return self.relu(x)

    input_np_x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me_x = Tensor(input_np_x)
    input_np_y = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me_y = Tensor(input_np_y)
    t = [input_me_x, input_me_y]

    # Pynative mode execution
    pynative_net = Net(t)
    pynative_result = pynative_net()

    # JIT mode execution
    jit_net = Net(t)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net()

    # Compare accuracy
    assert_equal(pynative_result, jit_result)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_assign_0006():
    """
    Feature: List assignment with scalar unpacking.
    Description: Test unpacking a list containing scalars and using the first element in conditional logic.
    Expectation: JIT result and gradient match pynative result.
    Migrated from: test_parser_list.py::test_parser_list_assign_0006
    """

    class Net(Cell):
        def __init__(self, l):
            super().__init__()
            self.l = l
            self.relu = P.ReLU()

        def construct(self, x):
            a, _ = self.l
            if a == 1:
                x = self.relu(x)
            return x

    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)
    l = [1, 2]

    # Pynative mode execution
    pynative_net = Net(l)
    pynative_net.set_grad()
    pynative_result = pynative_net(input_me)

    sens = Tensor(np.random.randn(*pynative_result.shape).astype(np.float32))
    pynative_grad = compute_grad_of_net_inputs(pynative_net, input_me, sens=sens)

    # JIT mode execution
    jit_net = Net(l)
    jit_net.set_grad()
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    jit_grad = compute_grad_of_net_inputs(jit_net, input_me, sens=sens)

    # Compare forward results and gradients
    assert_equal(pynative_result, jit_result)
    assert_equal(pynative_grad, jit_grad, decimal=5)
