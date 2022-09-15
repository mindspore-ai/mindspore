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
import mindspore as ms
from mindspore import nn, context, Tensor
from mindspore.ops.operations.sparse_ops import SparseTensorDenseAdd
from .test_grad_of_dynamic import TestDynamicGrad


class NetSparseTensorDenseAdd(nn.Cell):
    def __init__(self):
        super(NetSparseTensorDenseAdd, self).__init__()
        self.sparse_tensor_dense_add = SparseTensorDenseAdd()

    def construct(self, x1_indices, x1_values, x1_shape, x2):
        return self.sparse_tensor_dense_add(x1_indices, x1_values, x1_shape, x2)


def grad_dyn_case(is_dynamic_rank):
    test_dynamic = TestDynamicGrad(NetSparseTensorDenseAdd())
    x1_indices = Tensor([[0, 0], [0, 1]], dtype=ms.int64)
    x1_values = Tensor([1, 1], dtype=ms.float32)
    x1_shape = Tensor([3, 3], dtype=ms.int64)
    x2 = Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=ms.float32)
    test_dynamic.test_dynamic_grad_net((x1_indices, x1_values, x1_shape, x2), is_dynamic_rank)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cpu_grad_dynamic_shape():
    """
    Feature: test SparseTensorDenseAdd dynamic shape on CPU.
    Description: input is dynamic shape.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    grad_dyn_case(False)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cpu_grad_dynamic_rank():
    """
    Feature: test SparseTensorDenseAdd dynamic rank on CPU.
    Description: input is dynamic rank.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    grad_dyn_case(True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gpu_grad_dynamic_shape():
    """
    Feature: test SparseTensorDenseAdd dynamic shape on GPU.
    Description: input is dynamic shape.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    grad_dyn_case(False)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gpu_grad_dynamic_rank():
    """
    Feature: test SparseTensorDenseAdd dynamic rank on GPU.
    Description: input is dynamic rank.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    grad_dyn_case(True)
