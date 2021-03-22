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
import mindspore as ms
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import SparseTensor

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

class NetSparseDenseMatmul(nn.Cell):
    def __init__(self):
        super(NetSparseDenseMatmul, self).__init__()
        self.matmul = nn.SparseTensorDenseMatmul()

    def construct(self, indices, values, dens_shape, dt):
        return self.matmul(indices, values, dens_shape, dt)

class NetSparseTensor(nn.Cell):
    def __init__(self, dense_shape):
        super(NetSparseTensor, self).__init__()
        self.dense_shape = dense_shape
    def construct(self, indices, values):
        x = SparseTensor(indices, values, self.dense_shape)
        return x.values, x.indices, x.dense_shape

def test_sparse_tensor_dense_matmul():
    indices = Tensor([[0, 1], [1, 1]])
    values = Tensor([5, 5], dtype=ms.float32)
    dens_shape = (3, 3)
    spMatrix = np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]], dtype=np.float32)
    dsMatrix = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32)
    test_SparseDenseMatmul = NetSparseDenseMatmul()

    out_ms = test_SparseDenseMatmul(indices, values, dens_shape, Tensor(dsMatrix))
    out_np = np.matmul(spMatrix, dsMatrix)
    error = np.ones(shape=dsMatrix.shape) * 10e-6
    diff = out_ms.asnumpy() - out_np
    assert np.all(diff < error)
