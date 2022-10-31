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

import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops.operations.sparse_ops as P
import mindspore as ms
from mindspore import Tensor


class NetSparseSegmentMeanWithNumSegments(nn.Cell):

    def __init__(self) -> None:
        super(NetSparseSegmentMeanWithNumSegments, self).__init__()
        self.op = P.SparseSegmentMeanWithNumSegments()

    def construct(self, x, indices, segment_ids, num_segments):
        return self.op(x, indices, segment_ids, num_segments)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sparse_segment_mean_with_num_segments_dyn():
    """
    Feature: test SparseSegmentMeanWithNumSegments ops in cpu.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = NetSparseSegmentMeanWithNumSegments()

    x_dyn = Tensor(shape=[None, None], dtype=ms.float32)
    indices_dyn = Tensor(shape=[None], dtype=ms.int32)
    segment_ids_dyn = Tensor(shape=[None], dtype=ms.int32)
    num_segments_dyn = Tensor(shape=[None], dtype=ms.int32)
    net.set_inputs(x_dyn, indices_dyn, segment_ids_dyn, num_segments_dyn)

    x = Tensor([[0, 2, 0, 0], [0, 1, 1, 0], [2, 0, 2, 0]], dtype=ms.float32)
    indices = Tensor([0, 2, 1], dtype=ms.int32)
    segment_ids = Tensor([0, 0, 2], dtype=ms.int32)
    num_segments = Tensor([4], dtype=ms.int32)
    output = net(x, indices, segment_ids, num_segments)

    expect_shape = (4, 4)
    assert output.asnumpy().shape == expect_shape
