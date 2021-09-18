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
""" test_operator """
import numpy as np

from mindspore import Tensor, Model, context
from mindspore.nn import Cell
from mindspore.nn import ReLU
from mindspore.ops import operations as P
from ...ut_filter import non_graph_engine


class arithmetic_Net(Cell):
    """ arithmetic_Net definition """

    def __init__(self, symbol, loop_count=(1, 3)):
        super().__init__()
        self.symbol = symbol
        self.loop_count = loop_count
        self.relu = ReLU()

    def construct(self, x):
        a, b = self.loop_count
        y = self.symbol
        if y == 1:
            a += b
            for _ in (b, a):
                x = self.relu(x)
        elif y == 2:
            b -= a
            for _ in (a, b):
                x = self.relu(x)
        elif y == 3:
            z = a + b
            for _ in (b, z):
                x = self.relu(x)
        elif y == 4:
            z = b - a
            for _ in (z, b):
                x = self.relu(x)
        elif y == 5:
            z = a * b
            for _ in (a, z):
                x = self.relu(x)
        elif y == 6:
            z = b / a
            for _ in (a, z):
                x = self.relu(x)
        elif y == 7:
            z = b % a + 1
            for _ in (a, z):
                x = self.relu(x)
        else:
            if not a:
                x = self.relu(x)
        return x


class logical_Net(Cell):
    """ logical_Net definition """

    def __init__(self, symbol, loop_count=(1, 3)):
        super().__init__()
        self.symbol = symbol
        self.loop_count = loop_count
        self.fla = P.Flatten()
        self.relu = ReLU()

    def construct(self, x):
        a, b = self.loop_count
        y = self.symbol
        if y == 1:
            if b and a:
                x = self.relu(x)
            else:
                x = self.fla(x)
        else:
            if b or a:
                x = self.relu(x)
            else:
                x = self.fla(x)
        return x


def arithmetic_operator_base(symbol):
    """ arithmetic_operator_base """
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)
    logical_operator = {"++": 1, "--": 2, "+": 3, "-": 4, "*": 5, "/": 6, "%": 7, "not": 8}
    x = logical_operator[symbol]
    net = arithmetic_Net(x)
    context.set_context(mode=context.GRAPH_MODE)
    model = Model(net)
    model.predict(input_me)


def logical_operator_base(symbol):
    """ logical_operator_base """
    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me = Tensor(input_np)
    logical_operator = {"and": 1, "or": 2}
    x = logical_operator[symbol]
    net = logical_Net(x)
    context.set_context(mode=context.GRAPH_MODE)
    model = Model(net)
    model.predict(input_me)


@non_graph_engine
def test_ME_arithmetic_operator_0080():
    """ test_ME_arithmetic_operator_0080 """
    arithmetic_operator_base('not')


@non_graph_engine
def test_ME_arithmetic_operator_0070():
    """ test_ME_arithmetic_operator_0070 """
    logical_operator_base('and')


@non_graph_engine
def test_ME_logical_operator_0020():
    """ test_ME_logical_operator_0020 """
    logical_operator_base('or')


def test_ops():
    class OpsNet(Cell):
        """ OpsNet definition """

        def __init__(self, x, y):
            super(OpsNet, self).__init__()
            self.x = x
            self.y = y
            self.int = 4
            self.float = 3.2
            self.str_a = "hello"
            self.str_b = "world"

        def construct(self, x, y):
            h = x // y
            m = x ** y
            n = x % y
            r = self.x // self.y
            s = self.x ** self.y
            t = self.x % self.y
            p = h + m + n
            q = r + s + t
            ret_pow = p ** q + q ** p
            ret_mod = p % q + q % p
            ret_floor = p // q + q // p
            ret = ret_pow + ret_mod + ret_floor
            if self.int > self.float:
                if [1, 2, 3] is not None:
                    if self.str_a + self.str_b == "helloworld":
                        if q == 86:
                            return ret
            return x

    net = OpsNet(9, 2)
    x = Tensor(np.random.randint(low=1, high=10, size=(2, 3, 4), dtype=np.int32))
    y = Tensor(np.random.randint(low=10, high=20, size=(2, 3, 4), dtype=np.int32))
    context.set_context(mode=context.GRAPH_MODE)
    net(x, y)


def test_in_dict():
    class InDictNet(Cell):
        """ InDictNet definition """

        def __init__(self, key_in, key_not_in):
            super(InDictNet, self).__init__()
            self.key_in = key_in
            self.key_not_in = key_not_in

        def construct(self, x, y, z):
            d = {"a": x, "b": y}
            ret_in = 1
            ret_not_in = 2
            if self.key_in in d:
                ret_in = d[self.key_in]
            if self.key_not_in not in d:
                ret_not_in = z
            ret = ret_in + ret_not_in
            return ret

    net = InDictNet("a", "c")
    x = Tensor(np.random.randint(low=1, high=10, size=(2, 3, 4), dtype=np.int32))
    y = Tensor(np.random.randint(low=10, high=20, size=(2, 3, 4), dtype=np.int32))
    z = Tensor(np.random.randint(low=20, high=30, size=(2, 3, 4), dtype=np.int32))
    context.set_context(mode=context.GRAPH_MODE)
    net(x, y, z)
