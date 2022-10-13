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
""" test_cont_break """
import numpy as np

import mindspore as ms
from mindspore import Tensor, context, nn, ms_function
from mindspore.nn import Cell
from mindspore.ops import operations as P


class WhileSubGraphParam(Cell):
    def __init__(self):
        super().__init__()
        self.update = ms.Parameter(Tensor(1, ms.float32), "update")

    def construct(self, x, y, z):
        out1 = z
        while x < y:
            self.update = self.update + 1
            out1 = out1 + 1
            x = x + 1
        return out1, self.update


def test_while_loop_phi():
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(0, ms.float32)
    y = Tensor(10, ms.float32)
    z = Tensor(100, ms.float32)

    net = WhileSubGraphParam()
    net(x, y, z)


class WhileSubGraphParam2(Cell):
    def __init__(self):
        super().__init__()
        self.update = ms.Parameter(Tensor(1, ms.float32), "update")

    def construct(self, x, y, z):
        out1 = z
        i = self.update
        while x < y:
            i = i + 1
            out1 = out1 + 1
            x = x + 1
        return out1, self.update


def test_while_loop_phi_2():
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(0, ms.float32)
    y = Tensor(10, ms.float32)
    z = Tensor(100, ms.float32)

    net = WhileSubGraphParam2()
    net(x, y, z)


class WhileSubGraphParam3(Cell):
    def __init__(self, initial_input_x):
        super().__init__()
        self.initial_input_x = initial_input_x
        self.x = ms.Parameter(initial_input_x, name="parameter_x")
        self.y = ms.Parameter(self.initial_input_x, name="parameter_y")

    def construct(self):
        a = 0
        while a < 3:
            self.x = self.x + self.y
            a += 1
        return self.x


def test_while_loop_phi_3():
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(0, ms.float32)

    net = WhileSubGraphParam3(x)
    net()


class ControlMixedWhileIf(nn.Cell):
    def __init__(self):
        super().__init__()
        self.assign = P.Assign()
        self.var = ms.Parameter(ms.Tensor([1], ms.float32), name="var")

    @ms_function
    def construct(self, x, y, z, c2, c4):
        out = self.assign(self.var, c4)
        while x < c2:
            self.assign(self.var, c4)
            y = self.var
            while y < c2 and x < c2:
                if 2 * y < c2:
                    y = y + 2
                else:
                    y = y + 1
            out = out + y
            self.assign(self.var, c4)
            z = self.var
            while z < c2:
                z = z + 1
            out = out + z
            x = x + 1
        out = out + x
        while x < 2 * c2:
            self.assign(self.var, c4)
            y = self.var
            x = x + 1
            while y < c2:
                self.assign(self.var, c4)
                z = self.var
                while z < c2:
                    z = z + 1
                if x < c2:
                    y = y - 1
                else:
                    y = y + 1
                out = out + z
            out = out + y
        out = out + x
        return out


def test_mixed_while_if():
    context.set_context(mode=context.PYNATIVE_MODE)
    x = np.array(2).astype(np.int32)
    y = np.array(14).astype(np.int32)
    z = np.array(1).astype(np.int32)
    c2 = Tensor([14], ms.int32)
    c4 = Tensor([0], ms.int32)
    net = ControlMixedWhileIf()
    output = net(Tensor(x), Tensor(y), Tensor(z), c2, c4)
    expect = np.array(3318).astype(np.int32)
    assert np.allclose(expect, output.asnumpy(), 0.0001, 0.0001)
    context.set_context(mode=context.GRAPH_MODE)
