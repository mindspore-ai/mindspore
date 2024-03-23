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
import numpy as np
from mindspore.nn import Cell
from mindspore.common import Tensor, Parameter
import mindspore.ops.operations as P
from mindspore import context, ops, lazy_inline, nn


class Grad(Cell):
    def __init__(self, net):
        super(Grad, self).__init__()
        self.grad = ops.GradOperation()
        self.net = net

    def construct(self, x):
        grad_net = self.grad(self.net)
        return grad_net(x)


class Block(Cell):
    def __init__(self):
        super(Block, self).__init__()
        self.transpose1 = P.Transpose()
        self.transpose2 = P.Transpose()
        self.transpose3 = P.Transpose()
        self.transpose4 = P.Transpose()
        self.real_div1 = P.RealDiv()
        self.real_div2 = P.RealDiv()
        self.batch_matmul1 = P.BatchMatMul()
        self.batch_matmul2 = P.BatchMatMul()
        self.add = P.Add()
        self.softmax = P.Softmax(-1)
        self.dropout = P.Dropout(0.9)
        self.expand_dims = P.ExpandDims()
        self.sub = P.Sub()
        self.mul = P.Mul()
        self.y = Parameter(Tensor(np.ones((8, 128, 128)).astype(np.float32)))

    def construct(self, x):
        transpose1 = self.transpose1(x, (0, 2, 1, 3))
        real_div1 = self.real_div1(transpose1, Tensor(2.37891))
        transpose2 = self.transpose2(x, (0, 2, 3, 1))
        real_div2 = self.real_div2(transpose2, Tensor(2.37891))
        batch_matmul1 = self.batch_matmul1(real_div1, real_div2)
        expand_dims = self.expand_dims(self.y, 1)
        sub = self.sub(Tensor([1.0]), expand_dims)
        mul = self.mul(sub, Tensor([-0.0001]))
        add = self.add(mul, batch_matmul1)
        soft_max = self.softmax(add)
        dropout = self.dropout(soft_max)
        transpose3 = self.transpose3(x, (0, 2, 1, 3))
        batch_matmul2 = self.batch_matmul2(dropout[0], transpose3)
        transpose4 = self.transpose4(batch_matmul2, (0, 2, 1, 3))
        return transpose4


class TestBlock(Cell):
    def __init__(self):
        super(TestBlock, self).__init__()
        self.y = Parameter(Tensor(5))

    def construct(self, x):
        x = x + self.y
        x = x + self.y * 2
        x = x - 9
        return x


class TestIfBlock(Cell):
    def __init__(self):
        super(TestIfBlock, self).__init__()
        self.y = Parameter(Tensor(5))

    def construct(self, x):
        if x > 10:
            x = x + self.y * 2
        else:
            x = x + self.y
        x = x - 9
        return x


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_basic():
    """
    Feature: lazy inline.
    Description: The function construct of the cell decorated by lazy inline
    Expectation: Run successfully
    """

    class MyBlock(Cell):
        @lazy_inline
        def __init__(self):
            super(MyBlock, self).__init__()
            self.block = TestBlock()

        def construct(self, x):
            x = self.block(x)
            return x

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.blocks = nn.SequentialCell()
            for _ in range(3):
                b = MyBlock()
                self.blocks.append(b)

        def construct(self, x):
            out = self.blocks(x)
            return out

    context.set_context(mode=context.GRAPH_MODE, save_graphs=0, save_graphs_path="./lazy")
    x = Tensor(10)
    net = Net()
    net(x)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_multi_nest_cell():
    """
    Feature: Nest reusing cell with lazy inline.
    Description: Nest reusing cell with lazy inline.
    Expectation: Run successfully.
    """

    class MyBlock(Cell):
        @lazy_inline
        def __init__(self):
            super(MyBlock, self).__init__()
            self.block = TestBlock()
            self.if_block = TestIfBlock()
            self.if_block.recompute()

        def construct(self, x):
            y = x + 3
            if x > 3:
                x = self.if_block(x)
            else:
                x = self.block(y)
            x = x + 4
            return x

    class InnerBlock(Cell):
        @lazy_inline
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

    context.set_context(mode=context.GRAPH_MODE, save_graphs=0, save_graphs_path="./lazy")
    x = Tensor(10)
    net = Net()
    net(x)

    net = Grad(net)
    net(x)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_lazy_inline_default_value():
    """
    Feature: reusing cell with lazy inline.
    Description: The function construct of the cell decorated by lazy inline has default value.
    Expectation: Run successfully
    """

    class MyBlock(Cell):
        @lazy_inline
        def __init__(self):
            super(MyBlock, self).__init__()
            self.block = TestBlock()
            self.if_block = TestIfBlock()

        def construct(self, x, condition=None):
            y = x + 3
            if condition is not None:
                x = self.if_block(x)
            else:
                x = self.block(y)
            x = x + 4
            return x

    class InnerBlock(Cell):
        @lazy_inline
        def __init__(self):
            super(InnerBlock, self).__init__()
            self.blocks = nn.SequentialCell()
            for _ in range(5):
                b = MyBlock()
                self.blocks.append(b)

        def construct(self, *inputs):
            x = inputs[0]
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

        def construct(self, x, x1=None, x2=2):
            out = x + 2
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
