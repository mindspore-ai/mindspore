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

import mindspore as ms
from mindspore import nn, context, Tensor
from mindspore.ops.operations.sparse_ops import SparseTensorDenseMatmul
from .test_grad_of_dynamic import TestDynamicGrad


class NetSparseTensorDenseMatmul(nn.Cell):
    def __init__(self):
        super(NetSparseTensorDenseMatmul, self).__init__()
        self.sparse_tensor_dense_matmul = SparseTensorDenseMatmul()

    def construct(self, x1_indices, x1_values, x1_shape, x2):
        return self.sparse_tensor_dense_matmul(x1_indices, x1_values, x1_shape, x2)


def grad_dyn_case(is_dynamic_rank):
    test_dynamic = TestDynamicGrad(NetSparseTensorDenseMatmul())
    x1_indices = Tensor([[0, 1], [1, 2]], dtype=ms.int32)
    x1_values = Tensor([1, 2], dtype=ms.float32)
    x1_shape = (3, 4)
    x2 = Tensor([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=ms.float32)
    test_dynamic.test_dynamic_grad_net((x1_indices, x1_values, x1_shape, x2), is_dynamic_rank)


def test_cpu_grad_dynamic_shape():
    """
    Feature: test SparseTensorDenseMatmul dynamic shape on CPU.
    Description: input is dynamic shape.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    grad_dyn_case(False)


def test_cpu_grad_dynamic_rank():
    """
    Feature: test SparseTensorDenseMatmul dynamic rank on CPU.
    Description: input is dynamic rank.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    grad_dyn_case(True)
