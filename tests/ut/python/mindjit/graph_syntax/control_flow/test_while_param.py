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

import mindspore as ms
from mindspore import Tensor, context
from mindspore.nn import Cell


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
