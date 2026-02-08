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
""" test string operations """

import numpy as np
from mindspore import Tensor, jit
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.train.model import Model
from tests.mark_utils import arg_mark


class Net1(Cell):
    def __init__(self, string=""):
        super().__init__()
        self.string = string
        self.fla = P.Flatten()

    @jit
    def construct(self, x):
        for _ in self.string:
            x = self.fla(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_string_with_for():
    """
    Feature: String iteration in for loop.
    Description: Test iterating over a string in a for loop within Cell construct.
    Expectation: The network executes successfully with string iteration.
    Migrated from: test_parser_string.py::test_parser_string_with_for_001
    """
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)
    net = Net1("12")
    model = Model(net)
    model.predict(input_me)


class Net2(Cell):
    def __init__(self, string=""):
        super().__init__()
        self.string = string
        self.fla = P.Flatten()

    @jit
    def construct(self, x):
        if self.string[0] == "1":
            x = self.fla(x)
        if self.string[1] == "2":
            x = self.fla(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_string_using_index():
    """
    Feature: String indexing operation.
    Description: Test accessing string elements using index in Cell construct.
    Expectation: The network executes successfully with string indexing.
    Migrated from: test_parser_string.py::test_parser_string_using_index_002
    """
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)
    net = Net2("12")
    model = Model(net)
    model.predict(input_me)


class Net4(Cell):
    def __init__(self, string=""):
        super().__init__()
        self.string = string
        self.fla = P.Flatten()

    @jit
    def construct(self, x):
        for _ in self.string[0:2]:
            x = self.fla(x)
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sliced_string_with_for():
    """
    Feature: String slicing in for loop.
    Description: Test iterating over a sliced string in a for loop within Cell construct.
    Expectation: The network executes successfully with sliced string iteration.
    Migrated from: test_parser_string.py::test_parser_sliced_string_with_for_003
    """
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)
    net = Net4("123")
    model = Model(net)
    model.predict(input_me)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_string_add_operator():
    """
    Feature: String addition operator.
    Description: Test string concatenation using addition operator in Cell construct.
    Expectation: The result matches the expected concatenated string.
    Migrated from: test_parser_string.py::test_parser_string_add_operator
    """
    class Net(Cell):
        def __init__(self, s1, s2):
            super().__init__()
            self.s1 = s1
            self.s2 = s2

        @jit
        def construct(self):
            return self.s1 + self.s2

    s1 = "123"
    s2 = "456string"
    net = Net(s1, s2)
    out_me = net()
    assert out_me == "123456string"
