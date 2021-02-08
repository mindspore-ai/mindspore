# Copyright 2020 Huawei Technologies Co., Ltd
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
"""
@File  : test_sparse_tensor.py
@Author:
@Date  : 2020-07-16
@Desc  : test mindspore sparse_tensor's operation
"""
import numpy as np
import pytest

import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import composite as C
from mindspore import Tensor, SparseTensor, context

@pytest.fixture(scope="module", autouse=True)
def setup_teardown():
    context.set_context(mode=context.GRAPH_MODE, enable_sparse=True)
    yield
    context.set_context(enable_sparse=False)


grad_op = C.GradOperation(get_all=True)

class MakeSparseTensor(nn.Cell):
    def __init__(self, dense_shape):
        super(MakeSparseTensor, self).__init__()
        self.dense_shape = dense_shape
    def construct(self, indices, values):
        ret = (SparseTensor(indices, values, self.dense_shape),)
        return ret[0]


def test_sparse_tensor_make_sparse_tensor():
    indices = Tensor([[0, 1], [1, 2]])
    values = Tensor([1, 2], dtype=ms.float32)
    MakeSparseTensor((3, 4))(indices, values)


def test_sparse_tensor_attr():
    class SparseTensorGetAttr(nn.Cell):
        def __init__(self):
            super(SparseTensorGetAttr, self).__init__()
            self.dense_shape = (3, 4)
        def construct(self, indices, values):
            x = SparseTensor(indices, values, self.dense_shape)
            return x.values, x.indices, x.dense_shape

    indices = Tensor([[0, 1], [1, 2]])
    values = Tensor([1, 2], dtype=ms.float32)
    SparseTensorGetAttr()(indices, values)
    grad_op(SparseTensorGetAttr())(indices, values)


def test_sparse_tensor_indices_dim_greater_than_dense_shape_dim():
    indices = Tensor(np.array([[0, 0, 0], [0, 0, 1]], dtype=np.int32))
    values = Tensor(np.array([100, 200], dtype=np.float32))
    dense_shape = (2, 2)
    with pytest.raises(TypeError):
        MakeSparseTensor(dense_shape)(indices, values)


def test_sparse_tensor_indices_dim_less_than_dense_shape_dim():
    indices = Tensor(np.array([[0, 0], [0, 1]], dtype=np.int32))
    values = Tensor(np.array([100, 200], dtype=np.float32))
    dense_shape = (2, 2, 2)
    with pytest.raises(TypeError):
        MakeSparseTensor(dense_shape)(indices, values)


def test_sparse_tensor_to_tensor():
    class SparseToDenseCell(nn.Cell):
        def __init__(self, dense_shape):
            super(SparseToDenseCell, self).__init__()
            self.dense_shape = dense_shape
            self.sparse_to_dense = nn.SparseToDense()
        def construct(self, indices, values):
            sparse = SparseTensor(indices, values, self.dense_shape)
            return self.sparse_to_dense(sparse)

    indices = Tensor([[0, 1], [1, 2]])
    values = Tensor([1, 2], dtype=ms.float32)
    dense_shape = (3, 4)
    SparseToDenseCell(dense_shape)(indices, values)
    grad_op(SparseToDenseCell(dense_shape))(indices, values)
