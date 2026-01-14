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
"""Test view operators on Ascend platform"""

from tests.mark_utils import arg_mark
import numpy as np
import pytest
from mindspore import context
import mindspore as ms
from mindspore import nn
from mindspore import Tensor
from mindspore.ops.auto_generate import BroadcastToView, ExpandDimsView, NarrowView, SelectExtView, SplitTensorView, \
    ChunkView, DiagonalView, SliceExtView, SplitWithSizeView, UnstackExtView
import mindspore.ops as P


context.set_context(mode=context.GRAPH_MODE)

class Net(nn.Cell):
    def __init__(self, tanspose_a=False, transpose_b=False):
        super().__init__()
        self.transpose = P.TransposeView()
        self.matmul = P.MatMul(tanspose_a, transpose_b)

    def construct(self, x, perm, mat):
        out = self.transpose(x, perm)
        out = self.matmul(out, mat)
        return out


class NetSplit(nn.Cell):
    def __init__(self):
        super().__init__()
        self.split = P.Split(0, 2)
        self.matmul = P.MatMul()

    def construct(self, x):
        a, b = self.split(x)
        out = self.matmul(a, b)
        return out


class NetCat(nn.Cell):
    def __init__(self):
        super().__init__()
        self.cat = P.Concat(axis=0)
        self.matmul = P.MatMul()

    def construct(self, x, y, z):
        a = self.matmul(x, y)
        b = self.matmul(x, z)
        c = self.cat((a, b))
        out = c + c
        out = out / 2.0
        return out

def chunk_view_kbk():
    class NetChunk(nn.Cell):
        def __init__(self):
            super().__init__()
            self.chunk = ChunkView()

        @ms.jit
        def construct(self, x, chunks):
            out = self.chunk(x, chunks, 0)
            return out

    x = np.random.rand(16, 16).astype(np.float32)
    net = NetChunk()
    out = net(Tensor(x), 16)
    for i, row in enumerate(out):
        assert np.allclose(row.asnumpy(), x[i], rtol=10e-4, atol=10e-4)

def diagonal_view_kbk():
    class NetDiagonal(nn.Cell):
        def __init__(self):
            super().__init__()
            self.diagonal = DiagonalView()

        @ms.jit
        def construct(self, x):
            out = self.diagonal(x)
            return out

    x = np.random.rand(3, 3, 3).astype(np.float32)
    net = NetDiagonal()
    out = net(Tensor(x))
    out_np = np.diagonal(x)
    assert np.allclose(out.asnumpy(), out_np, rtol=10e-4, atol=10e-4)

def slice_ext_view_kbk():
    class NetSlice(nn.Cell):
        def __init__(self):
            super().__init__()
            self.slice_view = SliceExtView()

        @ms.jit
        def construct(self, x, dim, start, end, step):
            output = self.slice_view(x, dim, start, end, step)
            return output

    x = Tensor([[[1, 1, 1], [2, 2, 2]],
                [[3, 3, 3], [4, 4, 4]],
                [[5, 5, 5], [6, 6, 6]]], ms.float32)
    net = NetSlice()
    output = net(x, 0, 0, 1, 1)
    expect_output = [[[1, 1, 1], [2, 2, 2]]]
    assert np.allclose(output.asnumpy(), expect_output, rtol=10e-4, atol=10e-4)

def split_with_size_view_kbk():
    class NetSplitWithSize(nn.Cell):
        def __init__(self):
            super().__init__()
            self.split = SplitWithSizeView()

        @ms.jit
        def construct(self, x, split_size):
            output = self.split(x, split_size, 0)
            return output

    x = Tensor(np.arange(10, dtype=np.float32))
    net = NetSplitWithSize()
    output = net(x, [3, 3, 4])
    expect_output = [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]
    for i, value in enumerate(output):
        assert np.allclose(value.asnumpy(), expect_output[i], rtol=10e-4, atol=10e-4)

def unstack_ext_view_kbk():
    class NetUnstackExtView(nn.Cell):
        def __init__(self):
            super().__init__()
            self.unstack_ext_view = UnstackExtView()

        @ms.jit
        def construct(self, x, dim):
            output = self.unstack_ext_view(x, dim)
            return output

    x = np.random.rand(2, 3, 4).astype(np.float32)
    net = NetUnstackExtView()
    output = net(Tensor(x), 0)

    assert np.allclose(output[0].asnumpy(), x[0], rtol=10e-4, atol=10e-4)
    assert np.allclose(output[1].asnumpy(), x[1], rtol=10e-4, atol=10e-4)

def transpose_view_matmul_kbk():
    x = np.random.rand(1280, 256).astype(np.float32) / 10
    mat = np.random.rand(1280, 3840).astype(np.float32) / 10
    perm = (1, 0)

    net = Net()
    out = net(Tensor(x), perm, Tensor(mat))
    out_np = np.matmul(x.T, mat)
    assert np.allclose(out.asnumpy(), out_np, rtol=10e-4, atol=10e-4)

def broadcast_to_view_kbk():
    class BroadcastToViewNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.broadcast_to_view = BroadcastToView()

        @ms.jit
        def construct(self, x):
            output = self.broadcast_to_view(x, (2, 3))
            return output

    x = Tensor(np.array([1, 2, 3]).astype(np.float32))
    net = BroadcastToViewNet()
    graph_output = net(x)

    output = BroadcastToView()(x, (2, 3))
    assert (graph_output.asnumpy() == output.asnumpy()).all()

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def transpose_ext_view_kbk():
    class TransposeExtViewNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.transpose_ext_view = P.TransposeExtView()

        def construct(self, x):
            output = self.transpose_ext_view(x, 0, 2)
            return output

    context.set_context(jit_config={"jit_level": "O0"})
    x = Tensor(np.ones((2, 3, 4), dtype=np.float32))
    net = TransposeExtViewNet()
    output = net(x)
    assert output.shape == (4, 3, 2)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_chunk_view_ascend():
    """
    Feature: Chunk view operation
    Description: test the Chunk kernel, with view operation.
    Expectation: the output is same as expected
    """
    chunk_view_kbk()

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_chunk_view_cpu():
    """
    Feature: Chunk view operation with MS_OP_PLUGIN_PATH
    Description: test the Chunk kernel, with view operation.
    Expectation: the output is same as expected
    """
    chunk_view_kbk()

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_diagonal_view_ascend():
    """
    Feature: Diagonal view operation
    Description: test the Diagonal kernel, with view operation.
    Expectation: the output matches each row of the input
    """
    diagonal_view_kbk()

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_diagonal_view_cpu():
    """
    Feature: Diagonal view operation with MS_OP_PLUGIN_PATH
    Description: test the Diagonal kernel, with view operation.
    Expectation: the output matches each row of the input.
    """
    diagonal_view_kbk()

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_slice_ext_view_ascend():
    """
    Feature: Slice view operation
    Description: test the Slice kernel, with view operation.
    Expectation: the output is same as expected.
    """
    slice_ext_view_kbk()

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_slice_ext_view_cpu():
    """
    Feature: Slice view operation with MS_OP_PLUGIN_PATH
    Description: test the Slice kernel, with view operation.
    Expectation: the output is same as expected
    """
    slice_ext_view_kbk()

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_split_with_size_view_ascend():
    """
    Feature: SplitWithSize view operation
    Description: test the SplitWithSize kernel, with view operation.
    Expectation: the output is same as expected
    """
    split_with_size_view_kbk()

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_split_with_size_view_cpu():
    """
    Feature: SplitWithSize view operation with MS_OP_PLUGIN_PATH
    Description: test the SplitWithSize kernel, with view operation.
    Expectation: the output is same as expected
    """
    split_with_size_view_kbk()

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_unstack_view_ascend():
    """
    Feature: UnstackExtView view operation
    Description: test the UnstackExtView kernel with view operation
    Expectation: the output is same as numpy
    """
    unstack_ext_view_kbk()

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_unstack_ext_view_cpu():
    """
    Feature: UnstackExtView view operation with MS_OP_PLUGIN_PATH
    Description: test the UnstackExtView kernel using view operation
    Expectation: the output is same with numpy
    """
    unstack_ext_view_kbk()

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=26.00)
def test_transpose_view_ascend():
    """
    Feature: Transpose view operation
    Description: test the Transpose kernel, with view operation.
    Expectation: the output is same with numpy
    """
    transpose_view_matmul_kbk()

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_transpose_view_cpu():
    """
    Feature: Transpose view operation with MS_OP_PLUGIN_PATH
    Description: test the Transpose kernel, with view operation.
    Expectation: the output is same with numpy
    """
    transpose_view_matmul_kbk()

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_broadcast_to_view_ascend():
    """
    Feature: BroadcastTo view operation
    Description: test the BroadcastTo kernel, with view operation.
    Expectation: No exception.
    """
    broadcast_to_view_kbk()

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_broadcast_to_view_cpu():
    """
    Feature: BroadcastTo view operation with MS_OP_PLUGIN_PATH
    Description: test the BroadcastTo kernel, with view operation.
    Expectation: No exception.
    """
    broadcast_to_view_kbk()

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_transpose_ext_view_ascend():
    """
    Feature: TransposeExt view operation
    Description: test the TransposeExt kernel, with view operation.
    Expectation: No exception.
    """
    transpose_ext_view_kbk()

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_transpose_ext_view_cpu():
    """
    Feature: TransposeExt view operation with MS_OP_PLUGIN_PATH
    Description: test the TransposeExt kernel, with view operation.
    Expectation: No exception.
    """
    transpose_ext_view_kbk()

class ViewOut(nn.Cell):
    '''net with view out'''
    def __init__(self):
        super().__init__()
        self.transpose = P.TransposeView()

    @ms.jit
    def construct(self, x):
        x = self.transpose(x, (0, 1, 2, 4, 3))
        res = ms.mint.select(x, 1, 2)
        return res


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_graph_view_out():
    """
    Feature: Runtime view graph mode.
    Description: view op as graph output.
    Expectation: the output is same as pynative
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x = ms.Tensor(np.random.randn(2, 3, 4, 5, 6).astype(np.float32))
    net = ViewOut()
    out_graph = net(x)
    x = x.transpose((0, 1, 2, 4, 3))
    out_pynative = ms.mint.select(x, 1, 2)
    assert np.allclose(out_graph.asnumpy(), out_pynative.asnumpy(), rtol=10e-4, atol=10e-4)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_pynative_view_to_graph():
    """
    Feature: Runtime view graph mode.
    Description: view input from pynative.
    Expectation: the output is same as pynative
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x = ms.Tensor(np.random.randn(2, 3, 4, 5, 6).astype(np.float32))
    x = x.transpose((0, 1, 2, 4, 3))
    net = ViewOut()
    out_graph = net(x)
    x = x.transpose((0, 1, 2, 4, 3))
    out_pynative = ms.mint.select(x, 1, 2)
    assert np.allclose(out_graph.asnumpy(), out_pynative.asnumpy(), rtol=10e-4, atol=10e-4)


class MakeContiguous(nn.Cell):
    '''net with view to aclop'''
    def __init__(self):
        super().__init__()
        self.transpose = P.TransposeView()

    @ms.jit
    def construct(self, x):
        x = self.transpose(x, (0, 1, 2, 4, 3))
        res = ms.mint.select(x, 1, 2)
        res = res[:0] # StridedSlice is aclop
        return res


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_graph_view_to_aclop():
    """
    Feature: Runtime view graph mode.
    Description: view input from pynative.
    Expectation: the output is same as pynative
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x = ms.Tensor(np.random.randn(2, 3, 4, 5, 6).astype(np.float32))
    net = MakeContiguous()
    out_graph = net(x)

    x = x.transpose((0, 1, 2, 4, 3))
    out_pynative = ms.mint.select(x, 1, 2)
    out_pynative = out_pynative[:0]
    assert np.allclose(out_graph.asnumpy(), out_pynative.asnumpy(), rtol=10e-4, atol=10e-4)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_expand_dims_view():
    """
    Feature: Runtime view graph mode.
    Description: Runtime view graph mode.
    Expectation: No exception.
    """
    class ExpandDimsViewNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.expand_dims_view = ExpandDimsView()

        def construct(self, x):
            output = self.expand_dims_view(x, 0)
            return output

    x = Tensor(np.array([[2, 2], [2, 2]]), ms.float32)
    net = ExpandDimsViewNet()
    graph_output = net(x)

    pynative_output = ExpandDimsView()(x, 0)
    assert (graph_output.asnumpy() == pynative_output.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_narrow_view():
    """
    Feature: Runtime view graph mode.
    Description: Runtime view graph mode.
    Expectation: No exception.
    """
    class NarrowViewNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.narrow_view = NarrowView()

        def construct(self, x):
            output = self.narrow_view(x, 0, 0, 2)
            return output

    x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], ms.int32)
    net = NarrowViewNet()
    graph_output = net(x)

    pynative_output = NarrowView()(x, 0, 0, 2)
    assert (graph_output.asnumpy() == pynative_output.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_view_and_inplace_nested_ctrl_dynamic_rank():
    """
    Feature: Runtime view graph mode.
    Description: Runtime view graph mode.
    Expectation: No exception.
    """
    class DynamicNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.reducesum = P.ReduceSum()
            self.expanddimsview = ExpandDimsView()
            self.selectview = SelectExtView()
            self.splittensorview = SplitTensorView()

        def construct(self, x, y):
            if self.reducesum(x) < 3 * self.reducesum(y):
                x.add_(y)
            else:
                y = self.expanddimsview(y, 1)
            if x.shape == (2, 4, 8):
                x = self.selectview(x, 0, 1)
                y = self.splittensorview(y, 2, 0)
            return x, y

    context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')
    with pytest.raises(ValueError) as raise_info:
        x_np = np.ones([2, 4, 8]).astype(np.int32)
        input_x = Tensor(x_np)
        y_np = 2 * np.ones([2, 4, 8]).astype(np.int32)
        input_y = Tensor(y_np)
        net = DynamicNet()
        out = net(input_x, input_y)
        print("out: ", out)
    assert "Unsupported dynamic shape for graph mode." in str(raise_info.value)
