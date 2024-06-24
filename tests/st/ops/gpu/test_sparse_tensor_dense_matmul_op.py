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
from tests.mark_utils import arg_mark

# This example should be run with multiple processes.

# Please refer to Learning > Tutorials > Experts > Distributed Parallel Startup Methods on mindspore.cn. Pick a

# supported startup method for your hardware and get more detail information on the corresponding page.

"""smoke tests for SparseTensorDenseMatmul"""
import pytest
import numpy as np

from mindspore import context
from mindspore import Tensor
from mindspore.ops.composite import GradOperation

import mindspore.nn as nn
import mindspore.ops.operations.sparse_ops as sp_ops


class NetSparseTensorDenseMatmul(nn.Cell):
    def __init__(self, adjoint_st=False, adjoint_dt=False):
        super(NetSparseTensorDenseMatmul, self).__init__()
        self.sparse_tensor_dense_matmul = sp_ops.SparseTensorDenseMatmul(adjoint_st=adjoint_st, adjoint_dt=adjoint_dt)

    def construct(self, indices, values, sparse_shape, dense):
        output = self.sparse_tensor_dense_matmul(indices, values, sparse_shape, dense)
        return output


class GradNet(nn.Cell):
    def __init__(self, network):
        super(GradNet, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=False)
        self.network = network

    def construct(self, indices, values, dens_shape, dense):
        return self.grad(self.network)(indices, values, dens_shape, dense)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sparse_tensor_dense_matmul_int32_int32():
    """
    Feature: Test sparse tensor dense matmul ops.
    Description: Test type_int32_int32 sparse tensor dense add ops.
    Expectation: Success.
    """
    for mode in [context.PYNATIVE_MODE, context.GRAPH_MODE]:
        context.set_context(mode=mode, device_target="GPU")
        indices_np = np.array([[0, 0], [1, 1], [2, 2], [2, 3]], np.int32)
        values_np = np.array([2, 3, 4, 5], np.int32)
        dense_shape = np.array([3, 4], np.int64)
        sparse_np = np.array([[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 4, 5]], dtype=np.int32)
        dense_np = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=np.int32)
        sparse_dense_matmul_net = NetSparseTensorDenseMatmul()
        indices = Tensor(indices_np)
        values = Tensor(values_np)
        sparse_shape = Tensor(dense_shape)
        dense = Tensor(dense_np)
        out_ms = sparse_dense_matmul_net(indices, values, sparse_shape, dense)
        out_np = np.matmul(sparse_np, dense_np)
        assert (out_ms.asnumpy() == out_np).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sparse_tensor_dense_matmul_int32_int64():
    """
    Feature: Test sparse tensor dense matmul ops.
    Description: Test type_int32_int64 sparse tensor dense add ops.
    Expectation: Success.
    """
    for mode in [context.PYNATIVE_MODE, context.GRAPH_MODE]:
        context.set_context(mode=mode, device_target="GPU")
        indices_np = np.array([[0, 0], [1, 1], [2, 2], [2, 3]], np.int32)
        values_np = np.array([2, 3, 4, 5], np.int64)
        dense_shape = np.array([3, 4], np.int64)
        sparse_np = np.array([[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 4, 5]], dtype=np.int64)
        dense_np = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=np.int64)
        sparse_dense_matmul_net = NetSparseTensorDenseMatmul()
        indices = Tensor(indices_np)
        values = Tensor(values_np)
        sparse_shape = Tensor(dense_shape)
        dense = Tensor(dense_np)
        out_ms = sparse_dense_matmul_net(indices, values, sparse_shape, dense)
        out_np = np.matmul(sparse_np, dense_np)
        assert (out_ms.asnumpy() == out_np).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sparse_tensor_dense_matmul_int32_float16():
    """
    Feature: Test sparse tensor dense matmul ops.
    Description: Test type_int32_float16 sparse tensor dense add ops.
    Expectation: Success.
    """
    for mode in [context.PYNATIVE_MODE, context.GRAPH_MODE]:
        context.set_context(mode=mode, device_target="GPU")
        indices_np = np.array([[0, 0], [1, 1], [2, 2], [2, 3]], np.int32)
        values_np = np.array([2, 3, 4, 5], np.float16)
        dense_shape = np.array([3, 4], np.int64)
        sparse_np = np.array([[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 4, 5]], dtype=np.float16)
        dense_np = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=np.float16)
        sparse_dense_matmul_net = NetSparseTensorDenseMatmul()
        indices = Tensor(indices_np)
        values = Tensor(values_np)
        sparse_shape = Tensor(dense_shape)
        dense = Tensor(dense_np)
        out_ms = sparse_dense_matmul_net(indices, values, sparse_shape, dense)
        out_np = np.matmul(sparse_np, dense_np)
        assert (out_ms.asnumpy() == out_np).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sparse_tensor_dense_matmul_int32_float():
    """
    Feature: Test sparse tensor dense matmul ops.
    Description: Test type_int32_float sparse tensor dense add ops.
    Expectation: Success.
    """
    for mode in [context.PYNATIVE_MODE, context.GRAPH_MODE]:
        context.set_context(mode=mode, device_target="GPU")
        indices_np = np.array([[0, 0], [1, 1], [2, 2], [2, 3]], np.int32)
        values_np = np.array([2, 3, 4, 5], np.float32)
        dense_shape = np.array([3, 4], np.int64)
        sparse_np = np.array([[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 4, 5]], dtype=np.float32)
        dense_np = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=np.float32)
        sparse_dense_matmul_net = NetSparseTensorDenseMatmul()
        indices = Tensor(indices_np)
        values = Tensor(values_np)
        sparse_shape = Tensor(dense_shape)
        dense = Tensor(dense_np)
        out_ms = sparse_dense_matmul_net(indices, values, sparse_shape, dense)
        out_np = np.matmul(sparse_np, dense_np)
        assert (out_ms.asnumpy() == out_np).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sparse_tensor_dense_matmul_int32_double():
    """
    Feature: Test sparse tensor dense matmul ops.
    Description: Test type_int32_double sparse tensor dense add ops.
    Expectation: Success.
    """
    for mode in [context.PYNATIVE_MODE, context.GRAPH_MODE]:
        context.set_context(mode=mode, device_target="GPU")
        indices_np = np.array([[0, 0], [1, 1], [2, 2], [2, 3]], np.int32)
        values_np = np.array([2, 3, 4, 5], np.float64)
        dense_shape = np.array([3, 4], np.int64)
        sparse_np = np.array([[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 4, 5]], dtype=np.float64)
        dense_np = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=np.float64)
        sparse_dense_matmul_net = NetSparseTensorDenseMatmul()
        indices = Tensor(indices_np)
        values = Tensor(values_np)
        sparse_shape = Tensor(dense_shape)
        dense = Tensor(dense_np)
        out_ms = sparse_dense_matmul_net(indices, values, sparse_shape, dense)
        out_np = np.matmul(sparse_np, dense_np)
        assert (out_ms.asnumpy() == out_np).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sparse_tensor_dense_matmul_int64_int32():
    """
    Feature: Test sparse tensor dense matmul ops.
    Description: Test type_int64_int32 sparse tensor dense add ops.
    Expectation: Success.
    """
    for mode in [context.PYNATIVE_MODE, context.GRAPH_MODE]:
        context.set_context(mode=mode, device_target="GPU")
        indices_np = np.array([[0, 0], [1, 1], [2, 2], [2, 3]], np.int32)
        values_np = np.array([2, 3, 4, 5], np.int32)
        dense_shape = np.array([3, 4], np.int64)
        sparse_np = np.array([[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 4, 5]], dtype=np.int32)
        dense_np = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=np.int32)
        sparse_dense_matmul_net = NetSparseTensorDenseMatmul()
        indices = Tensor(indices_np)
        values = Tensor(values_np)
        sparse_shape = Tensor(dense_shape)
        dense = Tensor(dense_np)
        out_ms = sparse_dense_matmul_net(indices, values, sparse_shape, dense)
        out_np = np.matmul(sparse_np, dense_np)
        assert (out_ms.asnumpy() == out_np).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sparse_tensor_dense_matmul_int64_int64():
    """
    Feature: Test sparse tensor dense matmul ops.
    Description: Test type_int64_int64 sparse tensor dense add ops.
    Expectation: Success.
    """
    for mode in [context.PYNATIVE_MODE, context.GRAPH_MODE]:
        context.set_context(mode=mode, device_target="GPU")
        indices_np = np.array([[0, 0], [1, 1], [2, 2], [2, 3]], np.int64)
        values_np = np.array([2, 3, 4, 5], np.int64)
        dense_shape = np.array([3, 4], np.int64)
        sparse_np = np.array([[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 4, 5]], dtype=np.int64)
        dense_np = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=np.int64)
        sparse_dense_matmul_net = NetSparseTensorDenseMatmul()
        indices = Tensor(indices_np)
        values = Tensor(values_np)
        sparse_shape = Tensor(dense_shape)
        dense = Tensor(dense_np)
        out_ms = sparse_dense_matmul_net(indices, values, sparse_shape, dense)
        out_np = np.matmul(sparse_np, dense_np)
        assert (out_ms.asnumpy() == out_np).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sparse_tensor_dense_matmul_int64_float16():
    """
    Feature: Test sparse tensor dense matmul ops.
    Description: Test type_int64_float16 sparse tensor dense add ops.
    Expectation: Success.
    """
    for mode in [context.PYNATIVE_MODE, context.GRAPH_MODE]:
        context.set_context(mode=mode, device_target="GPU")
        indices_np = np.array([[0, 0], [1, 1], [2, 2], [2, 3]], np.int64)
        values_np = np.array([2, 3, 4, 5], np.float16)
        dense_shape = np.array([3, 4], np.int64)
        sparse_np = np.array([[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 4, 5]], dtype=np.float16)
        dense_np = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=np.float16)
        sparse_dense_matmul_net = NetSparseTensorDenseMatmul()
        indices = Tensor(indices_np)
        values = Tensor(values_np)
        sparse_shape = Tensor(dense_shape)
        dense = Tensor(dense_np)
        out_ms = sparse_dense_matmul_net(indices, values, sparse_shape, dense)
        out_np = np.matmul(sparse_np, dense_np)
        assert (out_ms.asnumpy() == out_np).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sparse_tensor_dense_matmul_int64_float():
    """
    Feature: Test sparse tensor dense matmul ops.
    Description: Test type_int64_float sparse tensor dense add ops.
    Expectation: Success.
    """
    for mode in [context.PYNATIVE_MODE, context.GRAPH_MODE]:
        context.set_context(mode=mode, device_target="GPU")
        indices_np = np.array([[0, 0], [1, 1], [2, 2], [2, 3]], np.int64)
        values_np = np.array([2, 3, 4, 5], np.float32)
        dense_shape = np.array([3, 4], np.int64)
        sparse_np = np.array([[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 4, 5]], dtype=np.float32)
        dense_np = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=np.float32)
        sparse_dense_matmul_net = NetSparseTensorDenseMatmul()
        indices = Tensor(indices_np)
        values = Tensor(values_np)
        sparse_shape = Tensor(dense_shape)
        dense = Tensor(dense_np)
        out_ms = sparse_dense_matmul_net(indices, values, sparse_shape, dense)
        out_np = np.matmul(sparse_np, dense_np)
        assert (out_ms.asnumpy() == out_np).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sparse_tensor_dense_matmul_int64_double():
    """
    Feature: Test sparse tensor dense matmul ops.
    Description: Test type_int64_double sparse tensor dense add ops.
    Expectation: Success.
    """
    for mode in [context.PYNATIVE_MODE, context.GRAPH_MODE]:
        context.set_context(mode=mode, device_target="GPU")
        indices_np = np.array([[0, 0], [1, 1], [2, 2], [2, 3]], np.int64)
        values_np = np.array([2, 3, 4, 5], np.float64)
        dense_shape = np.array([3, 4], np.int64)
        sparse_np = np.array([[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 4, 5]], dtype=np.float64)
        dense_np = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=np.float64)
        sparse_dense_matmul_net = NetSparseTensorDenseMatmul()
        indices = Tensor(indices_np)
        values = Tensor(values_np)
        sparse_shape = Tensor(dense_shape)
        dense = Tensor(dense_np)
        out_ms = sparse_dense_matmul_net(indices, values, sparse_shape, dense)
        out_np = np.matmul(sparse_np, dense_np)
        assert (out_ms.asnumpy() == out_np).all()
