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
from mindspore.ops.operations.sparse_ops import SparseSegmentSqrtNWithNumSegments
from .test_grad_of_dynamic import TestDynamicGrad


class NetSparseSegmentSqrtNWithNumSegments(nn.Cell):
    def __init__(self):
        super(NetSparseSegmentSqrtNWithNumSegments, self).__init__()
        self.sparse_seg_sqrt_n_with_n_seg = SparseSegmentSqrtNWithNumSegments()

    def construct(self, x, indices, seg, num):
        return self.sparse_seg_sqrt_n_with_n_seg(x, indices, seg, num)


def grad_dyn_case(is_dynamic_rank):
    test_dynamic = TestDynamicGrad(NetSparseSegmentSqrtNWithNumSegments(), skip_convert_out_ids=[0])
    x = Tensor([[0, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 0]], dtype=ms.float16)
    indices = Tensor([0, 2, 1], dtype=ms.int32)
    segment_ids = Tensor([0, 1, 2], dtype=ms.int32)
    num_segments = Tensor([4], dtype=ms.int32)
    test_dynamic.test_dynamic_grad_net((x, indices, segment_ids, num_segments), is_dynamic_rank)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_cpu_grad_dynamic_shape():
    """
    Feature: test SparseSegmentSqrtNWithNumSegments dynamic shape on CPU.
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
    Feature: test SparseSegmentSqrtNWithNumSegments dynamic rank on CPU.
    Description: input is dynamic rank.
    Expectation: the result match with static shape
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    grad_dyn_case(True)
