# Copyright 2024-2025 Huawei Technologies Co., Ltd
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
"""Test ms.lazy_inline and ms.no_inline"""

import torch
import numpy as np
from mindspore.nn import Cell
from mindspore.common import Tensor, Parameter
from mindspore import context, ops, lazy_inline, nn, no_inline, jit
from tests.mark_utils import arg_mark


class Grad(Cell):
    def __init__(self, net):
        super().__init__()
        self.grad_net = ops.grad(net)

    def construct(self, *inputs):
        return self.grad_net(*inputs)


class TestBlock(Cell):
    def __init__(self):
        super().__init__()
        self.y = Parameter(Tensor(5))

    def construct(self, x):
        x = x + self.y
        x = x + self.y * 2
        x = x - 9
        return x


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.01)
def test_nest():
    """
    Feature: Nest reusing cell with lazy inline.
    Description: Nest reusing cell with lazy inline.
    Expectation: Run successfully.
    """

    class MyBlock(Cell):
        @lazy_inline(policy="front")
        def __init__(self):
            super().__init__()
            self.block = TestBlock()

        def construct(self, x):
            x = x + 3
            x = self.block(x)
            x = x + 4
            return x

    class InnerBlock(Cell):
        @lazy_inline(policy="front")
        def __init__(self):
            super().__init__()
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
            super().__init__()
            self.blocks = nn.SequentialCell()
            for _ in range(5):
                b = InnerBlock()
                self.blocks.append(b)

        def construct(self, x):
            out = x + 2
            out = self.blocks(out)
            return out

    class Net1(Cell):
        def __init__(self):
            super().__init__()
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
    context.set_context(jit_config={"jit_level": "O0"})
    x = Tensor(10)
    net = Net1()
    net(x)
    net = Grad(net)
    net(x)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=1024.01)
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


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_lazy_inline_block_if_in_if():
    """
    Feature: lazy inline
    Description: Test lazy inline with control flow
    Expectation: No exception.
    """
    class IfInIf(nn.Cell):
        @lazy_inline
        def __init__(self):
            super().__init__()
            self.a = 1
            self.b = 2

        def construct(self, x, y):
            out = x + x
            if y > self.a:
                if y > self.b:
                    out = x * out
                else:
                    out = out * out
            return out


    class PtBlock2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = 1
            self.b = 2

        def forward(self, x, y):
            out = x + x
            if y > self.a:
                if y > self.b:
                    out = x * out
                else:
                    out = out * out
            return out

    class OuterNet(nn.Cell):
        def __init__(self, block):
            super().__init__()
            self.block = block

        def construct(self, x, y):
            for _ in range(2):
                x = self.block(x, y)
            return x

    class OuterMod(torch.nn.Module):
        def __init__(self, block):
            super().__init__()
            self.block = block

        def forward(self, x, y):
            for _ in range(2):
                x = self.block(x, y)
            return x

    context.set_context(mode=context.GRAPH_MODE, jit_level='O0')
    npx = np.ones([2, 3], np.float32)
    npy = np.array([4], np.int32)
    ms_x = Tensor(npx)
    ms_y = Tensor(npy)
    pt_x = torch.tensor(npx, dtype=torch.float)
    pt_y = torch.tensor(npy, dtype=torch.float)

    ms_block = OuterNet(IfInIf())
    pt_block = OuterMod(PtBlock2())
    ms_out = ms_block(ms_x, ms_y)
    pt_out = pt_block(pt_x, pt_y)
    assert np.allclose(ms_out.asnumpy(), pt_out.detach().numpy(), 0.0001, 0.0001)

    ms_grad = Grad(ms_block)(Tensor(npx), Tensor(npy))
    pt_x.requires_grad = True
    pt_y.requires_grad = True
    out = pt_block(pt_x, pt_y)
    sens = torch.ones_like(out)
    out.backward(sens)
    pt_grad = pt_x.grad
    assert np.allclose(pt_grad.detach().numpy(), ms_grad.asnumpy(), 0.0001, 0.0001)
