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
import numpy as np
from mindspore import nn, context, Tensor
from mindspore.ops.operations.sparse_ops import SparseSegmentSqrtN
from .test_grad_of_dynamic import TestDynamicGrad


class NetSparseSegmentSqrtN(nn.Cell):
    def __init__(self):
        super(NetSparseSegmentSqrtN, self).__init__()
        self.sparse_seg_sqrt_n = SparseSegmentSqrtN()

    def construct(self, x, indices, seg):
        return self.sparse_seg_sqrt_n(x, indices, seg)


def grad_dyn_case(is_dynamic_rank):
    test_dynamic = TestDynamicGrad(NetSparseSegmentSqrtN())
    x = Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]).astype(np.float32))
    indices = Tensor(np.array([0, 1, 2]).astype(np.int32))
    segment_ids = Tensor(np.array([0, 1, 2]).astype(np.int32))
    test_dynamic.test_dynamic_grad_net((x, indices, segment_ids), is_dynamic_rank)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cpu_grad_dynamic_shape():
    """
    Feature: test SparseSegmentSqrtN dynamic shape on CPU.
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
    Feature: test SparseSegmentSqrtN dynamic rank on CPU.
    Description: input is dynamic rank.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    grad_dyn_case(True)
