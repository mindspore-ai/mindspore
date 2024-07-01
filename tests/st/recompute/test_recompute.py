# Copyright 2023 Huawei Technologies Co., Ltd
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
import numpy as np
from mindspore.nn import Cell
from mindspore.common import Tensor, Parameter
import mindspore.ops.operations as P
from mindspore import context, ops, lazy_inline, nn

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
context.set_context(jit_level='O2')


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


def test_recompute_block_recompute():
    """
    Feature: Recompute with lazy inline.
    Description: Each block is set recompute by the cell recompute api.
    Expectation: Run successfully and the memory usage is reduced.
    """

    class OuterBlock(Cell):
        @lazy_inline
        def __init__(self):
            super(OuterBlock, self).__init__()
            self.block = Block()

        def construct(self, x):
            return self.block(x)

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.blocks = nn.CellList()
            for _ in range(3):
                b = OuterBlock()
                b.recompute()
                self.blocks.append(b)

        def construct(self, x):
            out = x
            for i in range(3):
                out = self.blocks[i](out)
            return out

    x = Tensor(np.ones((8, 128, 16, 32)).astype(np.float32))
    net = Net()
    grad_net = Grad(net)
    grad_net(x)


def test_recompute_op_recompute1():
    """
    Feature: Recompute with lazy inline.
    Description: Each block is set recompute by the primitive recompute api.
    Expectation: Run successfully and the memory usage is reduced.
    """

    class OuterBlock(Cell):
        @lazy_inline
        def __init__(self):
            super(OuterBlock, self).__init__()
            self.block = Block()
            self.block.real_div1.recompute()
            self.block.batch_matmul1.recompute()
            self.block.add.recompute()
            self.block.softmax.recompute()

        def construct(self, x):
            return self.block(x)

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.blocks = nn.CellList()
            for _ in range(3):
                b = OuterBlock()
                self.blocks.append(b)

        def construct(self, x):
            out = x
            for i in range(3):
                out = self.blocks[i](out)
            return out

    x = Tensor(np.ones((8, 128, 16, 32)).astype(np.float32))
    net = Net()
    grad_net = Grad(net)
    grad_net(x)


def test_recompute_op_recompute2():
    """
    Feature: Recompute with lazy inline.
    Description: Each block is set recompute by the primitive recompute api.
    Expectation: Run successfully and the memory usage is reduced.
    """

    class OuterBlock(Cell):
        @lazy_inline
        def __init__(self):
            super(OuterBlock, self).__init__()
            self.block = Block()
            self.block.transpose1.recompute()
            self.block.transpose2.recompute()
            self.block.real_div1.recompute()
            self.block.real_div2.recompute()
            self.block.batch_matmul1.recompute()
            self.block.add.recompute()
            self.block.softmax.recompute()
            self.block.dropout.recompute()

        def construct(self, x):
            return self.block(x)

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.blocks = nn.CellList()
            for _ in range(3):
                b = OuterBlock()
                self.blocks.append(b)

        def construct(self, x):
            out = x
            for i in range(3):
                out = self.blocks[i](out)
            return out

    x = Tensor(np.ones((8, 128, 16, 32)).astype(np.float32))
    net = Net()
    grad_net = Grad(net)
    grad_net(x)


def test_recompute_op_recompute3():
    """
    Feature: Recompute with lazy inline.
    Description: Each block is set recompute by the primitive recompute api.
    Expectation: Run successfully and the memory usage is reduced.
    """

    class Block1(Cell):
        def __init__(self):
            super(Block1, self).__init__()
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
            self.sub1 = P.Sub()
            self.sub2 = P.Sub()
            self.mul = P.Mul()
            self.y = Parameter(Tensor(np.ones((8, 16, 128, 128)).astype(np.float32)))

        def construct(self, x):
            transpose1 = self.transpose1(x, (0, 2, 1, 3))
            real_div1 = self.real_div1(transpose1, Tensor(2.37891))
            sub1 = self.sub1(Tensor([1.0]), transpose1)
            sub2 = self.sub2(Tensor([1.0]), sub1)
            mul = self.mul(sub2, Tensor([-0.0001]))
            add = self.add(mul, real_div1)
            soft_max = self.softmax(add)
            dropout = self.dropout(soft_max)
            transpose3 = self.transpose3(x, (0, 2, 1, 3))
            batch_matmul2 = self.batch_matmul2(dropout[0], transpose3)
            transpose4 = self.transpose4(batch_matmul2, (0, 2, 1, 3))
            return transpose4

    class OuterBlock(Cell):
        @lazy_inline
        def __init__(self):
            super(OuterBlock, self).__init__()
            self.block = Block1()
            self.block.mul.recompute()
            self.block.real_div1.recompute()
            self.block.transpose1.recompute()
            self.block.sub1.recompute()
            self.block.add.recompute()
            self.block.softmax.recompute()

        def construct(self, x):
            return self.block(x)

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.blocks = nn.CellList()
            for _ in range(3):
                b = OuterBlock()
                self.blocks.append(b)

        def construct(self, x):
            out = x
            for i in range(3):
                out = self.blocks[i](out)
            return out

    x = Tensor(np.ones((8, 128, 16, 128)).astype(np.float32))
    net = Net()
    grad_net = Grad(net)
    grad_net(x)


def test_recompute_cell_and_op_recompute1():
    """
    Feature: Recompute with lazy inline.
    Description: Each block is set recompute by both the primitive and cell recompute api.
    Expectation: Run successfully and the memory usage is reduced.
    """

    class Net1(Cell):
        def __init__(self):
            super(Net1, self).__init__()
            self.transpose2 = P.Transpose()
            self.real_div2 = P.RealDiv()

        def construct(self, x):
            transpose2 = self.transpose2(x, (0, 2, 3, 1))
            real_div2 = self.real_div2(transpose2, Tensor(2.37891))
            return real_div2

    class Block1(Cell):
        def __init__(self):
            super(Block1, self).__init__()
            self.transpose1 = P.Transpose()
            self.transpose2 = P.Transpose()
            self.transpose3 = P.Transpose()
            self.transpose4 = P.Transpose()
            self.real_div1 = P.RealDiv()
            self.real_div1.recompute()
            self.real_div2 = P.RealDiv()
            self.batch_matmul1 = P.BatchMatMul()
            self.batch_matmul1.recompute()
            self.batch_matmul2 = P.BatchMatMul()
            self.add = P.Add()
            self.add.recompute()
            self.softmax = P.Softmax(-1)
            self.softmax.recompute()
            self.dropout = P.Dropout(0.9)
            self.expand_dims = P.ExpandDims()
            self.sub = P.Sub()
            self.mul = P.Mul()
            self.net1 = Net1()
            self.net1.recompute()
            self.y = Parameter(Tensor(np.ones((8, 128, 128)).astype(np.float32)))

        def construct(self, x):
            transpose1 = self.transpose1(x, (0, 2, 1, 3))
            real_div1 = self.real_div1(transpose1, Tensor(2.37891))
            real_div2 = self.net1(x)
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

    class OuterBlock(Cell):
        @lazy_inline
        def __init__(self):
            super(OuterBlock, self).__init__()
            self.block = Block1()

        def construct(self, x):
            return self.block(x)

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.blocks = nn.CellList()
            for _ in range(3):
                b = OuterBlock()
                self.blocks.append(b)

        def construct(self, x):
            out = x
            for i in range(3):
                out = self.blocks[i](out)
            return out

    x = Tensor(np.ones((8, 128, 16, 32)).astype(np.float32))
    net = Net()
    grad_net = Grad(net)
    grad_net(x)


def test_recompute_cell_and_op_recompute2():
    """
    Feature: Recompute with lazy inline.
    Description: Each block is set recompute by both the primitive and cell recompute api.
    Expectation: Run successfully and the memory usage is reduced.
    """

    class Net1(Cell):
        def __init__(self):
            super(Net1, self).__init__()
            self.transpose2 = P.Transpose()
            self.real_div2 = P.RealDiv()

        def construct(self, x):
            transpose2 = self.transpose2(x, (0, 2, 3, 1))
            real_div2 = self.real_div2(transpose2, Tensor(2.37891))
            return real_div2

    class Block1(Cell):
        def __init__(self):
            super(Block1, self).__init__()
            self.transpose1 = P.Transpose()
            self.transpose2 = P.Transpose()
            self.transpose3 = P.Transpose()
            self.transpose4 = P.Transpose()
            self.real_div1 = P.RealDiv()
            self.real_div1.recompute()
            self.real_div2 = P.RealDiv()
            self.batch_matmul1 = P.BatchMatMul()
            self.batch_matmul1.recompute()
            self.batch_matmul2 = P.BatchMatMul()
            self.add = P.Add()
            self.add.recompute()
            self.softmax = P.Softmax(-1)
            self.softmax.recompute()
            self.dropout = P.Dropout(0.9)
            self.expand_dims = P.ExpandDims()
            self.sub = P.Sub()
            self.mul = P.Mul()
            self.depend = ops.Depend()
            self.net1 = Net1()
            self.net1.recompute()
            self.y = Parameter(Tensor(np.ones((8, 128, 128)).astype(np.float32)))

        def construct(self, x):
            real_div2 = self.net1(x)
            depend = self.depend(x, real_div2)
            transpose1 = self.transpose1(depend, (0, 2, 1, 3))
            real_div1 = self.real_div1(transpose1, Tensor(2.37891))
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

    class OuterBlock(Cell):
        @lazy_inline
        def __init__(self):
            super(OuterBlock, self).__init__()
            self.block = Block1()

        def construct(self, x):
            return self.block(x)

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.blocks = nn.CellList()
            for _ in range(3):
                b = OuterBlock()
                self.blocks.append(b)

        def construct(self, x):
            out = x
            for i in range(3):
                out = self.blocks[i](out)
            return out

    x = Tensor(np.ones((8, 128, 16, 32)).astype(np.float32))
    net = Net()
    grad_net = Grad(net)
    grad_net(x)


def test_recompute_cell_and_op_recompute_with_tuple_outputs1():
    """
    Feature: Recompute with lazy inline.
    Description: Each block is set recompute by both the primitive and cell recompute api and return a tuple.
    Expectation: Run successfully and the memory usage is reduced.
    """

    class Net1(Cell):
        def __init__(self):
            super(Net1, self).__init__()
            self.transpose2 = P.Transpose()
            self.real_div2 = P.RealDiv()

        def construct(self, x):
            transpose2 = self.transpose2(x, (0, 2, 3, 1))
            real_div2 = self.real_div2(transpose2, Tensor(2.37891))
            return real_div2

    class Block1(Cell):
        def __init__(self):
            super(Block1, self).__init__()
            self.transpose1 = P.Transpose()
            self.transpose2 = P.Transpose()
            self.transpose3 = P.Transpose()
            self.transpose4 = P.Transpose()
            self.transpose4.recompute()
            self.real_div1 = P.RealDiv()
            self.real_div1.recompute()
            self.real_div2 = P.RealDiv()
            self.batch_matmul1 = P.BatchMatMul()
            self.batch_matmul1.recompute()
            self.batch_matmul2 = P.BatchMatMul()
            self.add = P.Add()
            self.add.recompute()
            self.add1 = P.Add()
            self.softmax = P.Softmax(-1)
            self.softmax.recompute()
            self.dropout = P.Dropout(0.9)
            self.expand_dims = P.ExpandDims()
            self.sub = P.Sub()
            self.mul = P.Mul()
            self.net1 = Net1()
            self.net1.recompute()
            self.y = Parameter(Tensor(np.ones((8, 128, 128)).astype(np.float32)))

        def construct(self, x, z):
            transpose1 = self.transpose1(x, (0, 2, 1, 3))
            real_div1 = self.real_div1(transpose1, Tensor(2.37891))
            real_div2 = self.net1(x)
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
            add1 = self.add1(transpose4, z)
            return add1, transpose4

    class OuterBlock(Cell):
        @lazy_inline
        def __init__(self):
            super(OuterBlock, self).__init__()
            self.block = Block1()

        def construct(self, x, z):
            return self.block(x, z)

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.blocks = nn.CellList()
            for _ in range(3):
                b = OuterBlock()
                self.blocks.append(b)

        def construct(self, x):
            out1, out2 = x, x
            for i in range(3):
                out1, out2 = self.blocks[i](out1, out2)
            return out1, out2

    x = Tensor(np.ones((8, 128, 16, 32)).astype(np.float32))
    net = Net()
    grad_net = Grad(net)
    grad_net(x)


def test_recompute_cell_and_op_recompute_with_tuple_outputs2():
    """
    Feature: Recompute with lazy inline.
    Description: Each block is set recompute by both the primitive and cell recompute api and return a tuple.
    Expectation: Run successfully and the memory usage is reduced.
    """

    class Net1(Cell):
        def __init__(self):
            super(Net1, self).__init__()
            self.transpose2 = P.Transpose()
            self.real_div2 = P.RealDiv()

        def construct(self, x):
            transpose2 = self.transpose2(x, (0, 2, 3, 1))
            real_div2 = self.real_div2(transpose2, Tensor(2.37891))
            return real_div2

    class Block1(Cell):
        def __init__(self):
            super(Block1, self).__init__()
            self.transpose1 = P.Transpose()
            self.transpose2 = P.Transpose()
            self.transpose3 = P.Transpose()
            self.transpose4 = P.Transpose()
            self.transpose4.recompute()
            self.real_div1 = P.RealDiv()
            self.real_div1.recompute()
            self.real_div2 = P.RealDiv()
            self.batch_matmul1 = P.BatchMatMul()
            self.batch_matmul1.recompute()
            self.batch_matmul2 = P.BatchMatMul()
            self.add = P.Add()
            self.add.recompute()
            self.add1 = P.Add()
            self.add1.recompute()
            self.softmax = P.Softmax(-1)
            self.softmax.recompute()
            self.dropout = P.Dropout(0.9)
            self.expand_dims = P.ExpandDims()
            self.sub = P.Sub()
            self.mul = P.Mul()
            self.net1 = Net1()
            self.net1.recompute()
            self.y = Parameter(Tensor(np.ones((8, 128, 128)).astype(np.float32)))

        def construct(self, x, z):
            transpose1 = self.transpose1(x, (0, 2, 1, 3))
            real_div1 = self.real_div1(transpose1, Tensor(2.37891))
            real_div2 = self.net1(x)
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
            add1 = self.add1(transpose4, z)
            return add1, transpose4

    class OuterBlock(Cell):
        @lazy_inline
        def __init__(self):
            super(OuterBlock, self).__init__()
            self.block = Block1()

        def construct(self, x, z):
            return self.block(x, z)

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.blocks = nn.CellList()
            for _ in range(3):
                b = OuterBlock()
                self.blocks.append(b)

        def construct(self, x):
            out1, out2 = x, x
            for i in range(3):
                out1, out2 = self.blocks[i](out1, out2)
            return out1, out2

    x = Tensor(np.ones((8, 128, 16, 32)).astype(np.float32))
    net = Net()
    grad_net = Grad(net)
    grad_net(x)
