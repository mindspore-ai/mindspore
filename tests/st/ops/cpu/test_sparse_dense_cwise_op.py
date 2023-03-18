# Copyright 2022 Huawei Technologies Co., Ltd
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
import mindspore.context as context
import mindspore as ms
from mindspore.ops.operations.sparse_ops import SparseDenseCwiseAdd, SparseDenseCwiseMul, SparseDenseCwiseDiv
from mindspore import nn, Tensor


class SparseDenseCwiseAddNet(nn.Cell):

    def __init__(self) -> None:
        super(SparseDenseCwiseAddNet, self).__init__()
        self.op = SparseDenseCwiseAdd()

    def construct(self, x1_indices, x1_values, x1_shape, x2):
        return self.op(x1_indices, x1_values, x1_shape, x2)


class SparseDenseCwiseMulNet(nn.Cell):

    def __init__(self) -> None:
        super(SparseDenseCwiseMulNet, self).__init__()
        self.op = SparseDenseCwiseMul()

    def construct(self, x1_indices, x1_values, x1_shape, x2):
        return self.op(x1_indices, x1_values, x1_shape, x2)


class SparseDenseCwiseDivNet(nn.Cell):

    def __init__(self) -> None:
        super(SparseDenseCwiseDivNet, self).__init__()
        self.op = SparseDenseCwiseDiv()

    def construct(self, x1_indices, x1_values, x1_shape, x2):
        return self.op(x1_indices, x1_values, x1_shape, x2)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sparse_dense_add_dyn():
    """
    Feature: test SparseDenseCwiseAdd op in cpu.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

    net = SparseDenseCwiseAddNet()
    x1_indices_dyn = Tensor(shape=[None, 2], dtype=ms.int64)
    x1_values_dyn = Tensor(shape=[None], dtype=ms.int32)
    x1_shape = Tensor([3, 3], dtype=ms.int64)
    x2_dyn = Tensor(shape=[None], dtype=ms.int32)
    net.set_inputs(x1_indices_dyn, x1_values_dyn, x1_shape, x2_dyn)

    x1_indices = Tensor([[0, 0], [2, 2]], dtype=ms.int64)
    x1_values = Tensor([1, 2], dtype=ms.int32)
    x2 = Tensor([1, 2, 3], dtype=ms.int32)

    out = net(x1_indices, x1_values, x1_shape, x2)
    expect_out_shape = (2,)
    assert out.asnumpy().shape == expect_out_shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sparse_dense_mul_dyn():
    """
    Feature: test SparseDenseCwiseMul op in cpu.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

    net = SparseDenseCwiseMulNet()
    x1_indices_dyn = Tensor(shape=[None, 2], dtype=ms.int64)
    x1_values_dyn = Tensor(shape=[None], dtype=ms.int32)
    x1_shape = Tensor([3, 3], dtype=ms.int64)
    x2_dyn = Tensor(shape=[None], dtype=ms.int32)
    net.set_inputs(x1_indices_dyn, x1_values_dyn, x1_shape, x2_dyn)

    x1_indices = Tensor([[0, 0], [2, 2]], dtype=ms.int64)
    x1_values = Tensor([1, 2], dtype=ms.int32)
    x2 = Tensor([1, 2, 3], dtype=ms.int32)

    out = net(x1_indices, x1_values, x1_shape, x2)
    expect_out_shape = (2,)
    assert out.asnumpy().shape == expect_out_shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.skip(reason="Have issues")
def test_sparse_dense_div_dyn():
    """
    Feature: test SparseDenseCwiseDiv op in cpu.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

    net = SparseDenseCwiseDivNet()
    x1_indices_dyn = Tensor(shape=[None, 2], dtype=ms.int64)
    x1_values_dyn = Tensor(shape=[None], dtype=ms.int32)
    x1_shape = Tensor([3, 3], dtype=ms.int64)
    x2_dyn = Tensor(shape=[None], dtype=ms.int32)
    net.set_inputs(x1_indices_dyn, x1_values_dyn, x1_shape, x2_dyn)

    x1_indices = Tensor([[0, 0], [2, 2]], dtype=ms.int64)
    x1_values = Tensor([4, 2], dtype=ms.int32)
    x2 = Tensor([1, 2, 2], dtype=ms.int32)

    out = net(x1_indices, x1_values, x1_shape, x2)
    expect_out_shape = (2,)
    assert out.asnumpy().shape == expect_out_shape
