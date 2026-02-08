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
"""Test tuple operations"""

import numpy as np
from mindspore import Tensor, jit
from mindspore.nn import Cell, ReLU
from mindspore.ops import operations as P
from mindspore.train.model import Model
from tests.mark_utils import arg_mark
from tests.st.compiler.utils import match_array


class Net1(Cell):
    def __init__(self, tuple_a):
        super().__init__()
        self.tuple = tuple_a
        self.relu = P.ReLU()

    @jit
    def construct(self, x):
        for _ in self.tuple:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_tuple_with_for():
    """
    Feature: Tuple iteration in for loop.
    Description: Test iterating over a tuple in a for loop within Cell construct.
    Expectation: The network executes successfully with tuple iteration.
    Migrated from: test_parser_tuple.py::test_parser_tuple_with_for_001
    """
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)
    net = Net1((1, 2))
    model = Model(net)
    model.predict(input_me)


class Net2(Cell):
    def __init__(self, tuple_a):
        super().__init__()
        self.tuple = tuple_a
        self.relu = P.ReLU()

    @jit
    def construct(self, x):
        if self.tuple[0]:
            x = self.relu(x)
        if self.tuple[1]:
            x = self.relu(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_tuple_using_index():
    """
    Feature: Tuple indexing operation.
    Description: Test accessing tuple elements using index in Cell construct.
    Expectation: The network executes successfully with tuple indexing.
    Migrated from: test_parser_tuple.py::test_parser_tuple_using_index_002
    """
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)
    net = Net2([1, 2])
    model = Model(net)
    model.predict(input_me)


class Net3(Cell):
    def __init__(self):
        super().__init__()
        self.relu = ReLU()
        self.flatten = P.Flatten()
        self.tuple = (self.relu, self.flatten)

    @jit
    def construct(self, x):
        for i in self.tuple:
            x = i(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_cell_tuple_with_for():
    """
    Feature: Tuple of Cell objects iteration in for loop.
    Description: Test iterating over a tuple containing Cell objects in a for loop within Cell construct.
    Expectation: The network executes successfully with tuple of Cell objects iteration.
    Migrated from: test_parser_tuple.py::test_parser_cell_tuple_with_for_003
    """
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)
    net = Net3()
    model = Model(net)
    model.predict(input_me)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_tuple_assign_tensor():
    """
    Feature: Tuple unpacking assignment with tensor operations.
    Description: Test tuple unpacking assignment where tuple contains tensors.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_tuple.py::test_parser_tuple_assign_0001
    """

    class Net(Cell):
        def __init__(self, tx):
            super().__init__()
            self.tuple = tx
            self.relu = P.ReLU()

        def construct(self):
            x, _ = self.tuple
            return self.relu(x)

    input_np_x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me_x = Tensor(input_np_x)
    input_np_y = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me_y = Tensor(input_np_y)
    t = (input_me_x, input_me_y)

    # Pynative mode execution
    pynative_net = Net(t)
    pynative_result = pynative_net()

    # JIT mode execution
    jit_net = Net(t)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net()

    # Compare accuracy
    match_array(pynative_result, jit_result)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_tuple_assign_created_in_construct():
    """
    Feature: Tuple unpacking assignment with tuple created in construct.
    Description: Test tuple unpacking assignment where tuple is created inside construct method.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_tuple.py::test_parser_tuple_assign_0002
    """

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.relu = P.ReLU()

        def construct(self, x, y):
            t = (x, y)
            a, _ = t
            return self.relu(a)

    input_np_x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me_x = Tensor(input_np_x)
    input_np_y = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me_y = Tensor(input_np_y)

    # Pynative mode execution
    pynative_net = Net()
    pynative_result = pynative_net(input_me_x, input_me_y)

    # JIT mode execution
    jit_net = Net()
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me_x, input_me_y)

    # Compare accuracy
    match_array(pynative_result, jit_result)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_tuple_assign_scalar():
    """
    Feature: Tuple unpacking assignment with scalar operations.
    Description: Test tuple unpacking assignment where tuple contains scalars used in conditional logic.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_tuple.py::test_parser_tuple_assign_0003
    """

    class Net(Cell):
        def __init__(self, tx):
            super().__init__()
            self.tuple = tx
            self.relu = P.ReLU()

        def construct(self, x):
            a, _ = self.tuple
            if a == 1:
                x = self.relu(x)
            return x

    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)
    t = (1, 2)

    # Pynative mode execution
    pynative_net = Net(t)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = Net(t)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    match_array(pynative_result, jit_result)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_scalar_tuple_add():
    """
    Feature: Scalar tuple addition operation.
    Description: Test adding two scalar tuples and using the result in conditional logic.
    Expectation: JIT result matches pynative result.
    Migrated from: test_parser_tuple.py::test_parser_tuple_add_0001
    """

    class Net(Cell):
        def __init__(self, tx, ty):
            super().__init__()
            self.tuple1 = tx
            self.tuple2 = ty
            self.relu = P.ReLU()

        def construct(self, x):
            t = self.tuple1 + self.tuple2
            if t == (1, 2, 3, 4):
                x = self.relu(x)
            return x

    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)
    t1 = (1, 2)
    t2 = (3, 4)

    # Pynative mode execution
    pynative_net = Net(t1, t2)
    pynative_result = pynative_net(input_me)

    # JIT mode execution
    jit_net = Net(t1, t2)
    jit_net.construct = jit(jit_net.construct)
    jit_result = jit_net(input_me)

    # Compare accuracy
    match_array(pynative_result, jit_result)


class Net4(Cell):
    def __init__(self, tx, ty):
        super().__init__()
        self.tuple1 = tx
        self.tuple2 = ty
        self.relu = P.ReLU()

    @jit
    def construct(self):
        t = self.tuple1 + self.tuple2
        return t


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_tensor_tuple_add():
    """
    Feature: Tensor tuple addition operation.
    Description: Test adding two tuples containing tensors.
    Expectation: The network executes successfully with tensor tuple addition.
    Migrated from: test_parser_tuple.py::test_parser_tuple_add_0002
    """
    input_np_x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me_x = Tensor(input_np_x)
    input_np_y = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me_y = Tensor(input_np_y)
    t1 = (input_me_x,)
    t2 = (input_me_y,)
    net = Net4(t1, t2)
    net()


class Net5(Cell):
    def __init__(self):
        super().__init__()
        self.relu = P.ReLU()

    @jit
    def construct(self, x, y):
        t1 = (x,)
        t2 = (y,)
        t = t1 + t2
        x = self.relu(t[0])
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_tensor_tuple_add_in_construct():
    """
    Feature: Tensor tuple addition operation in construct.
    Description: Test adding two tuples containing tensors created inside construct method.
    Expectation: The network executes successfully with tensor tuple addition in construct.
    Migrated from: test_parser_tuple.py::test_parser_tuple_add_0003
    """
    input_np_x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me_x = Tensor(input_np_x)
    input_np_y = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me_y = Tensor(input_np_y)
    net = Net5()
    net(input_me_x, input_me_y)
