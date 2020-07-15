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
import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import composite as C
from mindspore import Tensor, SparseTensor, context

context.set_context(mode=context.GRAPH_MODE, enable_sparse=True)

def test_sparse_tensor_make_sparse_tensor():
    class MakeSparseTensor(nn.Cell):
        def __init__(self):
            super(MakeSparseTensor, self).__init__()
            self.dense_shape = (3, 4)
        def construct(self, indices, values):
            ret = (SparseTensor(indices, values, self.dense_shape),)
            return ret[0]
    indices = Tensor([[0, 1], [1, 2]])
    values = Tensor([1, 2], dtype=ms.float32)
    MakeSparseTensor()(indices, values)


def test_sparse_tensor_attr():
    grad_op = C.GradOperation('get_all', get_all=True)
    class GradWrap(nn.Cell):
        def __init__(self, network):
            super(GradWrap, self).__init__()
            self.network = network
        def construct(self, input1, input2):
            gout = grad_op(self.network)(input1, input2)
            return gout

    class SparseTensorGetAttr(nn.Cell):
        def __init__(self):
            super(SparseTensorGetAttr, self).__init__()
            self.dense_shape = (3, 4)
        def construct(self, indices, values):
            x = SparseTensor(indices, values, self.dense_shape)
            return x.values(), x.indices(), x.dense_shape()

    indices = Tensor([[0, 1], [1, 2]])
    values = Tensor([1, 2], dtype=ms.float32)
    SparseTensorGetAttr()(indices, values)
