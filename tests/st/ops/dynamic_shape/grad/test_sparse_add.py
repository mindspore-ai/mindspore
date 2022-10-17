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
from mindspore.ops.operations.sparse_ops import SparseAdd
from .test_grad_of_dynamic import TestDynamicGrad


class NetSparseAdd(nn.Cell):
    def __init__(self):
        super(NetSparseAdd, self).__init__()
        self.sparse_add = SparseAdd()

    def construct(self, a_indices, a_values, a_shape, b_indices, b_values, b_shape, thresh):
        return self.sparse_add(a_indices, a_values, a_shape, b_indices, b_values, b_shape, thresh)


def grad_dyn_case(is_dynamic_rank):
    test_dynamic = TestDynamicGrad(NetSparseAdd())
    value_type = ms.float32
    thresh_type = ms.float32
    thresh_value = 0
    a_indices = Tensor([[0, 1], [1, 2]], ms.int64)
    a_values = Tensor([1, 2], value_type)
    a_shape = Tensor([3, 4], ms.int64)
    b_indices = Tensor([[0, 1], [1, 2]], ms.int64)
    b_values = Tensor([1, 2], value_type)
    b_shape = Tensor([3, 4], ms.int64)
    thresh = Tensor(thresh_value, thresh_type)
    test_dynamic.test_dynamic_grad_net([a_indices, a_values, a_shape, b_indices, b_values, b_shape, thresh],
                                       is_dynamic_rank)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_grad_dynamic_shape():
    """
    Feature: test SparseAdd dynamic shape.
    Description: input is dynamic shape.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    grad_dyn_case(False)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_grad_dynamic_rank():
    """
    Feature: test SparseAdd dynamic rank.
    Description: input is dynamic rank.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    grad_dyn_case(True)
