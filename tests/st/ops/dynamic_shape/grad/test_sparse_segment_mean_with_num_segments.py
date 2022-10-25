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
import mindspore
import mindspore.ops.operations.sparse_ops as S
from mindspore import nn, context, Tensor
from .test_grad_of_dynamic import TestDynamicGrad


class NetSparseSegmentMeanWithNumSegments(nn.Cell):

    def __init__(self):
        super(NetSparseSegmentMeanWithNumSegments, self).__init__()
        self.sparse_segment_mean_with_num_segments = S.SparseSegmentMeanWithNumSegments(
        )

    def construct(self, x, indices, seg_ids, num_segments):
        return self.sparse_segment_mean_with_num_segments(
            x, indices, seg_ids, num_segments)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
def test_sparse_segment_mean_with_num_segments_dynamic_shape():
    """
    Feature: SparseSegmentMeanWithNumSegments Grad DynamicShape.
    Description: Test case of dynamic shape for SparseSegmentMeanWithNumSegments grad operator on CPU.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    test_dynamic = TestDynamicGrad(NetSparseSegmentMeanWithNumSegments(),
                                   skip_convert_out_ids=[0])
    x = Tensor([[0, 2, 0, 0], [0, 1, 1, 0], [2, 0, 2, 0]],
               dtype=mindspore.float32)
    indices = Tensor([0, 2, 1], dtype=mindspore.int32)
    segment_ids = Tensor([0, 0, 2], dtype=mindspore.int32)
    num_segments = Tensor([4], dtype=mindspore.int32)
    inputs = [x, indices, segment_ids, num_segments]
    test_dynamic.test_dynamic_grad_net(inputs, False)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
def test_sparse_segment_mean_with_num_segments_dynamic_rank():
    """
    Feature: SparseSegmentMeanWithNumSegments Grad DynamicShape.
    Description: Test case of dynamic rank for SparseSegmentMeanWithNumSegments grad operator on CPU.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    x = Tensor([[0, 2, 0, 0], [0, 1, 1, 0], [2, 0, 2, 0]],
               dtype=mindspore.float32)
    indices = Tensor([0, 2, 1], dtype=mindspore.int32)
    segment_ids = Tensor([0, 0, 2], dtype=mindspore.int32)
    num_segments = Tensor([4], dtype=mindspore.int32)
    test_dynamic = TestDynamicGrad(NetSparseSegmentMeanWithNumSegments(),
                                   skip_convert_out_ids=[0])
    inputs = [x, indices, segment_ids, num_segments]
    test_dynamic.test_dynamic_grad_net(inputs, True)
