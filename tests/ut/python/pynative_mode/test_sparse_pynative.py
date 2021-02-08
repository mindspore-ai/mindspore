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
@File  : test_sparse_pynative.py
@Author:
@Date  : 2020-08-04
@Desc  : test mindspore sparse pynative
"""
import pytest
import mindspore as ms
import mindspore.nn as nn
from mindspore import context, Tensor, RowTensor, SparseTensor
from mindspore.ops import composite as C

@pytest.fixture(scope="module", autouse=True)
def setup_teardown():
    context.set_context(mode=context.PYNATIVE_MODE, enable_sparse=True)
    yield
    context.set_context(mode=context.GRAPH_MODE, enable_sparse=False)


grad_all = C.GradOperation(get_all=True)
class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network
    def construct(self, *args):
        grad = grad_all(self.network)(*args)
        return grad


def test_row_tensor_attr():
    class RowTensorGetAttr(nn.Cell):
        def __init__(self, dense_shape):
            super(RowTensorGetAttr, self).__init__()
            self.dense_shape = dense_shape
        def construct(self, indices, values):
            x = RowTensor(indices, values, self.dense_shape)
            return x.values, x.indices, x.dense_shape
    indices = Tensor([0])
    values = Tensor([[1, 2]], dtype=ms.float32)
    RowTensorGetAttr((3, 2))(indices, values)
    GradWrap(RowTensorGetAttr((3, 2)))(indices, values)


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
    GradWrap(SparseTensorGetAttr())(indices, values)
