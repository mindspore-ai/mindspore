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
import numpy as np
from mindspore.nn import Cell
from mindspore.common import Tensor, Parameter
import mindspore.ops.operations as P
from mindspore import context, ops, lazy_inline, nn
from tests.mark_utils import arg_mark

# pylint: disable=W0212
# W0212: protected-access

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
context.set_context(jit_level='O0')
context.set_context(max_device_memory='50GB')


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
        transpose3 = self.transpose3(x, (0, 2, 1, 3))
        batch_matmul2 = self.batch_matmul2(soft_max, transpose3)
        transpose4 = self.transpose4(batch_matmul2, (0, 2, 1, 3))
        return transpose4


def run_offload_cell_offload(net, folder_path):
    np.random.seed(10)
    context.set_context(save_graphs=False)
    x = Tensor(np.ones((8, 128, 16, 32)).astype(np.float32))
    grad_net = Grad(net)
    forward_output = net(x)
    backward_output = grad_net(x)
    return forward_output, backward_output


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard",
        essential_mark="essential", mem_peak=74.0)
def test_offload_op_offload1():
    """
    Feature: Offload with lazy inline.
    Description: Each block is set offload by the primitive offload api.
    Expectation: Run successfully and the memory usage is reduced.
    """

    class OuterBlock(Cell):
        @lazy_inline
        def __init__(self):
            super(OuterBlock, self).__init__()
            self.block = Block()
            self.block.real_div1._offload()
            self.block.batch_matmul1._offload()
            self.block.add._offload()
            self.block.softmax._offload()

        def construct(self, x):
            return self.block(x)

    class Offload_Net(Cell):
        def __init__(self):
            super(Offload_Net, self).__init__()
            self.blocks = nn.CellList()
            for _ in range(3):
                b = OuterBlock()
                self.blocks.append(b)

        def construct(self, x):
            out = x
            for i in range(3):
                out = self.blocks[i](out)
            return out

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.blocks = nn.CellList()
            for _ in range(3):
                b = Block()
                self.blocks.append(b)

        def construct(self, x):
            out = x
            for i in range(3):
                out = self.blocks[i](out)
            return out

    net = Net()
    offload_net = Offload_Net()
    folder_path_off = "./offload_graph_off"
    folder_path_on = "./offload_graph_on"

    forward_output, backward_output = run_offload_cell_offload(net, folder_path_off)
    offload_forward_output, offload_backward_output = run_offload_cell_offload(offload_net, folder_path_on)
    assert np.all(forward_output.asnumpy() == offload_forward_output.asnumpy())
    assert np.all(backward_output.asnumpy() == offload_backward_output.asnumpy())


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard",
        essential_mark="essential", mem_peak=74.0)
def test_offload_op_offload2():
    """
    Feature: Offload with lazy inline.
    Description: Each block is set offload by the primitive offload api.
    Expectation: Run successfully and the memory usage is reduced.
    """

    class OuterBlock(Cell):
        @lazy_inline
        def __init__(self):
            super(OuterBlock, self).__init__()
            self.block = Block()
            self.block.transpose1._offload()
            self.block.transpose2._offload()
            self.block.real_div1._offload()
            self.block.real_div2._offload()
            self.block.batch_matmul1._offload()
            self.block.add._offload()
            self.block.softmax._offload()

        def construct(self, x):
            return self.block(x)

    class Offload_Net(Cell):
        def __init__(self):
            super(Offload_Net, self).__init__()
            self.blocks = nn.CellList()
            for _ in range(3):
                b = OuterBlock()
                self.blocks.append(b)

        def construct(self, x):
            out = x
            for i in range(3):
                out = self.blocks[i](out)
            return out

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.blocks = nn.CellList()
            for _ in range(3):
                b = Block()
                self.blocks.append(b)

        def construct(self, x):
            out = x
            for i in range(3):
                out = self.blocks[i](out)
            return out

    net = Net()
    offload_net = Offload_Net()
    folder_path_off = "./offload_graph_off"
    folder_path_on = "./offload_graph_on"

    forward_output, backward_output = run_offload_cell_offload(net, folder_path_off)
    offload_forward_output, offload_backward_output = run_offload_cell_offload(offload_net, folder_path_on)
    assert np.all(forward_output.asnumpy() == offload_forward_output.asnumpy())
    assert np.all(backward_output.asnumpy() == offload_backward_output.asnumpy())


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard",
        essential_mark="essential", mem_peak=74.0)
def test_offload_op_with_recompute1():
    """
    Feature: Offload with recompute.
    Description: Each block is set offload/recompute by the primitive offload/recompute api.
    Expectation: Run successfully and the memory usage is reduced.
    """

    class OuterBlock(Cell):
        @lazy_inline
        def __init__(self):
            super(OuterBlock, self).__init__()
            self.block = Block()
            self.block.real_div1._offload()
            self.block.batch_matmul1._offload()
            self.block.add.recompute()
            self.block.softmax.recompute()

        def construct(self, x):
            return self.block(x)

    class Offload_Net(Cell):
        def __init__(self):
            super(Offload_Net, self).__init__()
            self.blocks = nn.CellList()
            for _ in range(3):
                b = OuterBlock()
                self.blocks.append(b)

        def construct(self, x):
            out = x
            for i in range(3):
                out = self.blocks[i](out)
            return out

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.blocks = nn.CellList()
            for _ in range(3):
                b = Block()
                self.blocks.append(b)

        def construct(self, x):
            out = x
            for i in range(3):
                out = self.blocks[i](out)
            return out

    net = Net()
    offload_net = Offload_Net()
    folder_path_off = "./offload_graph_off"
    folder_path_on = "./offload_graph_on"

    forward_output, backward_output = run_offload_cell_offload(net, folder_path_off)
    offload_forward_output, offload_backward_output = run_offload_cell_offload(offload_net, folder_path_on)
    assert np.all(forward_output.asnumpy() == offload_forward_output.asnumpy())
    assert np.all(backward_output.asnumpy() == offload_backward_output.asnumpy())


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard",
        essential_mark="essential", mem_peak=74.0)
def test_offload_op_with_recompute2():
    """
    Feature: Offload with recompute.
    Description: Each block is set offload/recompute by the offload/recompute offload api.
    Expectation: Run successfully and the memory usage is reduced.
    """

    class OuterBlock(Cell):
        @lazy_inline
        def __init__(self):
            super(OuterBlock, self).__init__()
            self.block = Block()
            self.block.real_div1.recompute()
            self.block.batch_matmul1.recompute()
            self.block.add._offload()
            self.block.softmax._offload()

        def construct(self, x):
            return self.block(x)

    class Offload_Net(Cell):
        def __init__(self):
            super(Offload_Net, self).__init__()
            self.blocks = nn.CellList()
            for _ in range(3):
                b = OuterBlock()
                self.blocks.append(b)

        def construct(self, x):
            out = x
            for i in range(3):
                out = self.blocks[i](out)
            return out

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.blocks = nn.CellList()
            for _ in range(3):
                b = Block()
                self.blocks.append(b)

        def construct(self, x):
            out = x
            for i in range(3):
                out = self.blocks[i](out)
            return out

    net = Net()
    offload_net = Offload_Net()
    folder_path_off = "./offload_graph_off"
    folder_path_on = "./offload_graph_on"

    forward_output, backward_output = run_offload_cell_offload(net, folder_path_off)
    offload_forward_output, offload_backward_output = run_offload_cell_offload(offload_net, folder_path_on)
    assert np.all(forward_output.asnumpy() == offload_forward_output.asnumpy())
    assert np.all(backward_output.asnumpy() == offload_backward_output.asnumpy())


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard",
        essential_mark="essential", mem_peak=74.0)
def test_offload_op_with_recompute3():
    """
    Feature: Offload with lazy inline.
    Description: Each block is set offload/recompute by the primitive offload/recompute api.
    Expectation: Run successfully and the memory usage is reduced.
    """

    class OuterBlock(Cell):
        @lazy_inline
        def __init__(self):
            super(OuterBlock, self).__init__()
            self.block = Block()
            self.block.transpose1._offload()
            self.block.transpose2.recompute()
            self.block.real_div1._offload()
            self.block.real_div2.recompute()
            self.block.batch_matmul1.recompute()
            self.block.add._offload()
            self.block.softmax.recompute()

        def construct(self, x):
            return self.block(x)

    class Offload_Net(Cell):
        def __init__(self):
            super(Offload_Net, self).__init__()
            self.blocks = nn.CellList()
            for _ in range(3):
                b = OuterBlock()
                self.blocks.append(b)

        def construct(self, x):
            out = x
            for i in range(3):
                out = self.blocks[i](out)
            return out

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.blocks = nn.CellList()
            for _ in range(3):
                b = Block()
                self.blocks.append(b)

        def construct(self, x):
            out = x
            for i in range(3):
                out = self.blocks[i](out)
            return out

    net = Net()
    offload_net = Offload_Net()
    folder_path_off = "./offload_graph_off"
    folder_path_on = "./offload_graph_on"

    forward_output, backward_output = run_offload_cell_offload(net, folder_path_off)
    offload_forward_output, offload_backward_output = run_offload_cell_offload(offload_net, folder_path_on)
    assert np.all(forward_output.asnumpy() == offload_forward_output.asnumpy())
    assert np.all(backward_output.asnumpy() == offload_backward_output.asnumpy())


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard",
        essential_mark="essential", mem_peak=102.0)
def test_offload_op_with_recompute4():
    """
    Feature: Offload with lazy inline.
    Description: Each block is set offload/recompute by the primitive offload/recompute api.
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
            self.batch_matmul2 = P.BatchMatMul(transpose_b=True)
            self.add = P.Add()
            self.softmax = P.Softmax(-1)
            self.dropout = P.Dropout(1.0)
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
            self.block.real_div1._offload()
            self.block.transpose1._offload()
            self.block.sub1.recompute()
            self.block.add.recompute()
            self.block.softmax._offload()
            self.block.dropout._offload()

        def construct(self, x):
            return self.block(x)

    class Offload_Net(Cell):
        def __init__(self):
            super(Offload_Net, self).__init__()
            self.blocks = nn.CellList()
            for _ in range(3):
                b = OuterBlock()
                self.blocks.append(b)

        def construct(self, x):
            out = x
            for i in range(3):
                out = self.blocks[i](out)
            return out

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.blocks = nn.CellList()
            for _ in range(3):
                b = Block1()
                self.blocks.append(b)

        def construct(self, x):
            out = x
            for i in range(3):
                out = self.blocks[i](out)
            return out

    net = Net()
    offload_net = Offload_Net()
    folder_path_off = "./offload_graph_off"
    folder_path_on = "./offload_graph_on"

    forward_output, backward_output = run_offload_cell_offload(net, folder_path_off)
    offload_forward_output, offload_backward_output = run_offload_cell_offload(offload_net, folder_path_on)
    assert np.all(forward_output.asnumpy() == offload_forward_output.asnumpy())
    assert np.all(backward_output.asnumpy() == offload_backward_output.asnumpy())


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="onecard",
        essential_mark="essential", mem_peak=88.0)
def test_offload_op_with_recompute5():
    """
    Feature: Offload with lazy inline.
    Description: Each block is set offload/recompute by the primitive offload/recompute api.
    Expectation: Run successfully and the memory usage is reduced.
    """

    class Block1(Cell):
        def __init__(self):
            super(Block1, self).__init__()
            self.transpose1 = P.Transpose().recompute()
            self.transpose2 = P.Transpose()._offload()
            self.transpose3 = P.Transpose()._offload()
            self.transpose4 = P.Transpose()
            self.real_div1 = P.RealDiv().recompute()
            self.real_div2 = P.RealDiv()._offload()
            self.batch_matmul1 = P.BatchMatMul()._offload()
            self.batch_matmul2 = P.BatchMatMul(transpose_b=True)._offload()
            self.add = P.Add()._offload()
            self.softmax = P.Softmax(-1)._offload()
            self.dropout = P.Dropout(1.0).recompute()
            self.expand_dims = P.ExpandDims()._offload()
            self.sub1 = P.Sub()._offload()
            self.sub2 = P.Sub()._offload()
            self.mul = P.Mul().recompute()
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

        def construct(self, x):
            return self.block(x)

    class Offload_Net(Cell):
        def __init__(self):
            super(Offload_Net, self).__init__()
            self.blocks = nn.CellList()
            for _ in range(3):
                b = OuterBlock()
                self.blocks.append(b)

        def construct(self, x):
            out = x
            for i in range(3):
                out = self.blocks[i](out)
            return out

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.blocks = nn.CellList()
            for _ in range(3):
                b = Block1()
                self.blocks.append(b)

        def construct(self, x):
            out = x
            for i in range(3):
                out = self.blocks[i](out)
            return out

    net = Net()
    offload_net = Offload_Net()
    folder_path_off = "./offload_graph_off"
    folder_path_on = "./offload_graph_on"

    forward_output, backward_output = run_offload_cell_offload(net, folder_path_off)
    offload_forward_output, offload_backward_output = run_offload_cell_offload(offload_net, folder_path_on)
    assert np.all(forward_output.asnumpy() == offload_forward_output.asnumpy())
    assert np.all(backward_output.asnumpy() == offload_backward_output.asnumpy())
