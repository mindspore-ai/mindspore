# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
import pytest

import mindspore.context as context
import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor
from mindspore.ops import composite as C

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class SparseDenseMatmulNet(nn.Cell):

    def __init__(self, adjoint_st=False, adjoint_dt=False):
        super(SparseDenseMatmulNet, self).__init__()
        self.matmul = nn.SparseTensorDenseMatmul(adjoint_st, adjoint_dt)

    def construct(self, indices, values, dens_shape, dense):
        return self.matmul(indices, values, dens_shape, dense)


class GradNet(nn.Cell):

    def __init__(self, network):
        super(GradNet, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=False)
        self.network = network

    def construct(self, indices, values, dens_shape, dense):
        return self.grad(self.network)(indices, values, dens_shape, dense)


def judge_result_correct(result, expect):
    assert result.dtype == expect.dtype
    assert result.shape == expect.shape
    assert np.allclose(result, expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sparse_tensor_dense_mul_dyn():
    """
    Feature: test SparseTensorDenseMul op in cpu.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = SparseDenseMatmulNet()

    x1_indices_dyn = Tensor(shape=[None, 2], dtype=ms.int64)
    x1_values_dyn = Tensor(shape=[None], dtype=ms.float32)
    x1_shape = Tensor([3, 4], dtype=ms.int64)
    x2_dyn = Tensor(shape=[4, None], dtype=ms.float32)
    net.set_inputs(x1_indices_dyn, x1_values_dyn, x1_shape, x2_dyn)

    x1_indices = Tensor([[0, 1], [1, 2]], dtype=ms.int64)
    x1_values = Tensor([1, 2], dtype=ms.float32)
    x2 = Tensor([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=ms.float32)
    out = net(x1_indices, x1_values, x1_shape, x2)

    expect_out_shape = (3, 2)
    assert out.asnumpy().shape == expect_out_shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sparse_tensor_dense_matmul_no_transpose():
    indices_np = np.array([[0, 0], [1, 1], [2, 2], [2, 3]], np.int64)
    values_np = np.array([2, 3, 4, 5], np.float16)
    dense_shape = (3, 4)
    sparse_np = np.array([[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 4, 5]],
                         dtype=np.float16)
    dense_np = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
                        dtype=np.float16)

    sparse_dense_matmul_net = SparseDenseMatmulNet()
    indices = Tensor(indices_np)
    values = Tensor(values_np)
    dense = Tensor(dense_np)
    out_ms = sparse_dense_matmul_net(indices, values, dense_shape, dense)
    out_np = np.matmul(sparse_np, dense_np)
    judge_result_correct(out_ms.asnumpy(), out_np)

    grad_net = GradNet(sparse_dense_matmul_net)
    grad_ms = grad_net(indices, values, dense_shape, dense)
    expect_values_grad = np.array([3., 12., 21., 30.], dtype=np.float16)
    judge_result_correct(grad_ms[1].asnumpy(), expect_values_grad)
    expect_dense_grad = np.array(
        [[2., 2., 2.], [3., 3., 3.], [4., 4., 4.], [5., 5., 5.]],
        dtype=np.float16)
    judge_result_correct(grad_ms[2].asnumpy(), expect_dense_grad)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sparse_tensor_dense_matmul_transpose_a():
    indices_np = np.array([[0, 0], [1, 1], [2, 0], [2, 2], [3, 1], [3, 2]],
                          np.int32)
    values_np = np.array([1, 2, 3, 4, 5, 6], np.float64)
    dense_shape = (4, 3)
    sparse_np = np.array([[1, 0, 0], [0, 2, 0], [3, 0, 4], [0, 5, 6]],
                         dtype=np.float64)
    dense_np = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
                        dtype=np.float64)

    sparse_dense_matmul_net = SparseDenseMatmulNet(adjoint_st=True)
    indices = Tensor(indices_np)
    values = Tensor(values_np)
    dense = Tensor(dense_np)
    out_ms = sparse_dense_matmul_net(indices, values, dense_shape, dense)
    perm = (1, 0)
    out_np = np.matmul(np.transpose(sparse_np, perm), dense_np)
    judge_result_correct(out_ms.asnumpy(), out_np)

    grad_net = GradNet(sparse_dense_matmul_net)
    grad_ms = grad_net(indices, values, dense_shape, dense)
    expect_values_grad = np.array([3., 12., 21., 21., 30., 30.],
                                  dtype=np.float64)
    judge_result_correct(grad_ms[1].asnumpy(), expect_values_grad)
    expect_dense_grad = np.array(
        [[1., 1., 1.], [2., 2., 2.], [7., 7., 7.], [11., 11., 11.]],
        dtype=np.float64)
    judge_result_correct(grad_ms[2].asnumpy(), expect_dense_grad)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sparse_tensor_dense_matmul_transpose_b():
    indices_np = np.array([[0, 0], [1, 1], [2, 0], [2, 2], [3, 1], [3, 2]],
                          np.int64)
    values_np = np.array([1, 2, 3, 4, 5, 6], np.int32)
    dense_shape = (4, 3)
    sparse_np = np.array([[1, 0, 0], [0, 2, 0], [3, 0, 4], [0, 5, 6]],
                         dtype=np.int32)
    dense_np = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
                        dtype=np.int32)

    sparse_dense_matmul_net = SparseDenseMatmulNet(adjoint_dt=True)
    indices = Tensor(indices_np)
    values = Tensor(values_np)
    dense = Tensor(dense_np)
    out_ms = sparse_dense_matmul_net(indices, values, dense_shape, dense)
    perm = (1, 0)
    out_np = np.matmul(sparse_np, np.transpose(dense_np, perm))
    judge_result_correct(out_ms.asnumpy(), out_np)

    grad_net = GradNet(sparse_dense_matmul_net)
    grad_ms = grad_net(indices, values, dense_shape, dense)
    expect_values_grad = np.array([18., 22., 18., 26., 22., 26.],
                                  dtype=np.int32)
    judge_result_correct(grad_ms[1].asnumpy(), expect_values_grad)
    expect_dense_grad = np.array(
        [[4, 7, 10], [4, 7, 10], [4, 7, 10], [4, 7, 10]], dtype=np.int32)
    judge_result_correct(grad_ms[2].asnumpy(), expect_dense_grad)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sparse_tensor_dense_matmul_transpose_all():
    indices_np = np.array([[0, 0], [1, 1], [2, 2], [2, 3]], np.int64)
    values_np = np.array([2, 3, 4, 5], np.int64)
    dense_shape = (3, 4)
    sparse_np = np.array([[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 4, 5]],
                         dtype=np.int64)
    dense_np = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
                        dtype=np.int64)

    sparse_dense_matmul_net = SparseDenseMatmulNet(adjoint_st=True,
                                                   adjoint_dt=True)
    indices = Tensor(indices_np)
    values = Tensor(values_np)
    dense = Tensor(dense_np)
    out_ms = sparse_dense_matmul_net(indices, values, dense_shape, dense)
    perm = (1, 0)
    out_np = np.matmul(np.transpose(sparse_np, perm),
                       np.transpose(dense_np, perm))
    judge_result_correct(out_ms.asnumpy(), out_np)

    grad_net = GradNet(sparse_dense_matmul_net)
    grad_ms = grad_net(indices, values, dense_shape, dense)
    expect_values_grad = np.array([18, 22, 26, 26], dtype=np.int64)
    judge_result_correct(grad_ms[1].asnumpy(), expect_values_grad)
    expect_dense_grad = np.array([[2, 3, 9], [2, 3, 9], [2, 3, 9], [2, 3, 9]],
                                 dtype=np.int64)
    judge_result_correct(grad_ms[2].asnumpy(), expect_dense_grad)
