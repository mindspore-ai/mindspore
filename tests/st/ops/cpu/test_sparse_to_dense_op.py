# Copyright 2021 Huawei Technologies Co., Ltd
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
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True, device_target="CPU")


class SparseToDenseNet(Cell):
    def __init__(self):
        super(SparseToDenseNet, self).__init__()
        self.sparse_to_dense = P.SparseToDense()

    def construct(self, indices, values, dense_shape):
        return self.sparse_to_dense(indices, values, dense_shape)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu_training
@pytest.mark.env_onecard
def test_sparse_to_dense_A():
    np.random.seed(0)
    indices = np.array([[0, 1], [1, 2]]).astype(np.int32)
    values = np.array([7, 8]).astype(np.int32)
    dense_shape = (3, 4)
    net = SparseToDenseNet()
    result = net(Tensor(indices), Tensor(values), dense_shape)
    expect_output = np.array([[0, 7, 0, 0],
                              [0, 0, 8, 0],
                              [0, 0, 0, 0]]).astype(np.int32)
    assert np.allclose(result.asnumpy(), expect_output, rtol=1.e-4, atol=1.e-8, equal_nan=True)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu_training
@pytest.mark.env_onecard
def test_sparse_to_dense_B():
    np.random.seed(0)
    indices = np.array([[0, 1], [1, 2], [2, 3]]).astype(np.int32)
    values = np.array([6.5, 7.5, 9.5]).astype(np.float64)
    dense_shape = (3, 4)
    net = SparseToDenseNet()
    result = net(Tensor(indices), Tensor(values), dense_shape)
    expect_output = np.array([[0, 6.5, 0, 0],
                              [0, 0, 7.5, 0],
                              [0, 0, 0, 9.5]]).astype(np.float64)
    assert np.allclose(result.asnumpy(), expect_output, rtol=1.e-4, atol=1.e-8, equal_nan=True)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu_training
@pytest.mark.env_onecard
def test_sparse_to_dense_C():
    np.random.seed(0)
    indices = np.array([[0, 1, 0, 0],
                        [1, 0, 0, 2],
                        [2, 0, 3, 0],
                        [4, 2, 3, 5]]).astype(np.int32)
    values = np.array([26.5, 17.5, 39.5, 11.5]).astype(np.float16)
    dense_shape = (10, 8, 5, 10)
    net = SparseToDenseNet()
    result = net(Tensor(indices), Tensor(values), dense_shape)
    expect_output = np.zeros([10, 8, 5, 10]).astype(np.float16)
    for i in range(0, indices.shape[0]):
        j = indices[i][0]
        k = indices[i][1]
        l = indices[i][2]
        m = indices[i][3]
        expect_output[j][k][l][m] = values[i]
    assert np.allclose(result.asnumpy(), expect_output, rtol=1.e-4, atol=1.e-8, equal_nan=True)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu_training
@pytest.mark.env_onecard
def test_sparse_to_dense_D():
    np.random.seed(0)
    indices = np.array([[0, 1, 0, 0, 2, 1],
                        [9, 0, 0, 8, 0, 0],
                        [2, 0, 4, 0, 1, 1],
                        [4, 2, 3, 5, 0, 2],
                        [7, 4, 3, 9, 0, 1]]).astype(np.int32)
    values = np.array([1, 1, 1, 1, 1]).astype(np.bool)
    dense_shape = (10, 5, 5, 10, 3, 3)
    net = SparseToDenseNet()
    result = net(Tensor(indices), Tensor(values), dense_shape)
    expect_output = np.zeros([10, 5, 5, 10, 3, 3]).astype(np.bool)
    for i in range(0, indices.shape[0]):
        j = indices[i][0]
        k = indices[i][1]
        l = indices[i][2]
        m = indices[i][3]
        u = indices[i][4]
        v = indices[i][5]
        expect_output[j][k][l][m][u][v] = values[i]
    assert np.allclose(result.asnumpy(), expect_output, rtol=1.e-4, atol=1.e-8, equal_nan=True)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu_training
@pytest.mark.env_onecard
def test_sparse_to_dense_E():
    indices = np.array([2, 5, 7]).astype(np.int32)
    values = np.array([17, 18, 19]).astype(np.int8)
    dense_shape = ([10])
    expect_output = np.zeros([10]).astype(np.int8)
    for i in range(0, indices.shape[0]):
        j = indices[i]
        expect_output[j] = values[i]

    net = SparseToDenseNet()
    result = net(Tensor(indices), Tensor(values), dense_shape)
    assert np.allclose(result.asnumpy(), expect_output, rtol=1.e-4, atol=1.e-8, equal_nan=True)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu_training
@pytest.mark.env_onecard
def test_sparse_to_dense_F():
    indices = np.array([2, 4, 18]).astype(np.int32)
    values = np.array([-23, 18, -1]).astype(np.int16)
    dense_shape = ([20])
    expect_output = np.zeros([20]).astype(np.int16)
    for i in range(0, indices.shape[0]):
        j = indices[i]
        expect_output[j] = values[i]

    net = SparseToDenseNet()
    result = net(Tensor(indices), Tensor(values), dense_shape)
    assert np.allclose(result.asnumpy(), expect_output, rtol=1.e-4, atol=1.e-8, equal_nan=True)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu_training
@pytest.mark.env_onecard
def test_sparse_to_dense_G():
    indices = np.array([2, 5, 7]).astype(np.int32)
    values = np.array([17, 18, 19]).astype(np.uint8)
    dense_shape = ([10])
    expect_output = np.zeros([10]).astype(np.uint8)
    for i in range(0, indices.shape[0]):
        j = indices[i]
        expect_output[j] = values[i]

    net = SparseToDenseNet()
    result = net(Tensor(indices), Tensor(values), dense_shape)
    assert np.allclose(result.asnumpy(), expect_output, rtol=1.e-4, atol=1.e-8, equal_nan=True)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu_training
@pytest.mark.env_onecard
def test_sparse_to_dense_H():
    indices = np.array([2, 5, 7]).astype(np.int32)
    values = np.array([17, 18, 19]).astype(np.uint16)
    dense_shape = ([10])
    expect_output = np.zeros([10]).astype(np.uint16)
    for i in range(0, indices.shape[0]):
        j = indices[i]
        expect_output[j] = values[i]

    net = SparseToDenseNet()
    result = net(Tensor(indices), Tensor(values), dense_shape)
    assert np.allclose(result.asnumpy(), expect_output, rtol=1.e-4, atol=1.e-8, equal_nan=True)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu_training
@pytest.mark.env_onecard
def test_sparse_to_dense_I():
    indices = np.array([2, 5, 7]).astype(np.int64)
    values = np.array([17, 18, 19]).astype(np.float16)
    dense_shape = ([10])
    expect_output = np.zeros([10]).astype(np.float16)
    for i in range(0, indices.shape[0]):
        j = indices[i]
        expect_output[j] = values[i]

    net = SparseToDenseNet()
    result = net(Tensor(indices), Tensor(values), dense_shape)
    assert np.allclose(result.asnumpy(), expect_output, rtol=1.e-4, atol=1.e-8, equal_nan=True)
