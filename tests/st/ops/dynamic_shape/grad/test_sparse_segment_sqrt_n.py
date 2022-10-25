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
import mindspore.ops.operations.sparse_ops as P
from mindspore import nn, context, Tensor
from .test_grad_of_dynamic import TestDynamicGrad

context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')


class NetSparseSegmentSqrtN(nn.Cell):

    def __init__(self):
        super(NetSparseSegmentSqrtN, self).__init__()
        self.op = P.SparseSegmentSqrtN()

    def construct(self, x, indices, segment_ids):
        return self.op(x, indices, segment_ids)


def sparse_segment_sqrt_n_test(is_dyn_rank):
    x = Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=ms.float32)
    indices = Tensor([0, 1, 2], dtype=ms.int32)
    segment_ids = Tensor([0, 1, 2], dtype=ms.int32)
    tester = TestDynamicGrad(NetSparseSegmentSqrtN())
    tester.test_dynamic_grad_net([x, indices, segment_ids], is_dyn_rank)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
def test_sparse_segment_sqrt_n_dyn_shape():
    """
    Feature: SparseSegmentMean Grad Dynamic Shape.
    Description: Test case of dynamic shape for SparseSegmentMean grad operator.
    Expectation: success.
    """
    sparse_segment_sqrt_n_test(False)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
def test_sparse_segment_sqrt_n_dyn_rank():
    """
    Feature: SparseSegmentMean Grad Dynamic Rank.
    Description: Test case of dynamic rank for SparseSegmentMean grad operator.
    Expectation: success.
    """
    sparse_segment_sqrt_n_test(True)
