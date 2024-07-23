# Copyright 2024 Huawei Technologies Co., Ltd
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

import pytest
from mindspore.nn import Cell
from mindspore.common import Tensor, Parameter
from mindspore import context, ops, lazy_inline, nn, no_inline, jit
from tests.mark_utils import arg_mark


class Grad(Cell):
    def __init__(self, net):
        super(Grad, self).__init__()
        self.grad = ops.GradOperation()
        self.net = net

    def construct(self, x):
        grad_net = self.grad(self.net)
        return grad_net(x)


class TestBlock(Cell):
    def __init__(self):
        super(TestBlock, self).__init__()
        self.y = Parameter(Tensor(5))

    def construct(self, x):
        x = x + self.y
        x = x + self.y * 2
        x = x - 9
        return x


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_nest():
    """
    Feature: Nest reusing cell with lazy inline.
    Description: Nest reusing cell with lazy inline.
    Expectation: Run successfully.
    """

    class MyBlock(Cell):
        @lazy_inline(policy="front")
        def __init__(self):
            super(MyBlock, self).__init__()
            self.block = TestBlock()

        def construct(self, x):
            x = x + 3
            x = self.block(x)
            x = x + 4
            return x

    class InnerBlock(Cell):
        @lazy_inline(policy="front")
        def __init__(self):
            super(InnerBlock, self).__init__()
            self.blocks = nn.SequentialCell()
            for _ in range(5):
                b = MyBlock()
                self.blocks.append(b)

        def construct(self, x):
            x = x + 1
            x = self.blocks(x)
            return x

    class OuterBlock(Cell):
        @lazy_inline
        def __init__(self):
            super(OuterBlock, self).__init__()
            self.blocks = nn.SequentialCell()
            for _ in range(5):
                b = InnerBlock()
                self.blocks.append(b)

        def construct(self, x):
            out = x + 2
            out = self.blocks(out)
            return out

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.blocks = nn.SequentialCell()
            for _ in range(3):
                b = OuterBlock()
                self.blocks.append(b)

        def construct(self, x):
            out = x
            out = self.blocks(out)
            out = out + 0.1
            out = self.blocks(out)
            return out

    class Net1(Cell):
        def __init__(self):
            super(Net1, self).__init__()
            self.blocks = nn.SequentialCell()
            for _ in range(3):
                b = OuterBlock()
                self.blocks.append(b)

        def construct(self, x):
            out = x
            out = self.blocks(out)
            out = out + x
            out = self.blocks(out)
            return out

    context.set_context(mode=context.GRAPH_MODE, save_graphs=0, save_graphs_path="./lazy")
    x = Tensor(10)
    net = Net1()
    net(x)
    net = Grad(net)
    net(x)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_no_inline():
    """
    Feature: make reusing function with no inline.
    Description: reusing function with no inline.
    Expectation: Run successfully.
    """

    @no_inline
    def no_inline_fun(val):
        x = val * 3 + 2
        return x

    @jit
    def call_no_inline_fun(val):
        for _ in range(100):
            val = no_inline_fun(val)
        return val

    x = Tensor(1)
    x = call_no_inline_fun(x)
