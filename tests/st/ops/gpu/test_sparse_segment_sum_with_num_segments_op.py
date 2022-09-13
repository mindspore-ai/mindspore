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
import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops.operations.sparse_ops as P
from mindspore import Tensor
from mindspore.common.api import ms_function


class SparseSegmentSumWithNumSegmentsNet(nn.Cell):
    def __init__(self):
        super(SparseSegmentSumWithNumSegmentsNet, self).__init__()
        self.net = P.SparseSegmentSumWithNumSegments()

    @ms_function
    def construct(self, x, indices, segment_ids, num_segments):
        return self.net(x, indices, segment_ids, num_segments)


def sparse_segment_sum_with_num_segments(loss):
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    x_np = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.float32)
    indices_np = np.array([0, 0, 1, 2, 2, 3], dtype=np.int32)
    segment_ids_np = np.array([0, 3, 3, 5, 7, 7], dtype=np.int32)
    num_segments_np = np.array(8, dtype=np.int32)
    x_ms = Tensor(x_np)
    indices_ms = Tensor(indices_np)
    segment_ids_ms = Tensor(segment_ids_np)
    num_segments_ms = Tensor(num_segments_np)
    net_ms = SparseSegmentSumWithNumSegmentsNet()
    out_ms = net_ms(x_ms, indices_ms, segment_ids_ms, num_segments_ms)
    expected = np.array([[1, 2, 3, 4],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [6, 8, 10, 12],
                         [0, 0, 0, 0],
                         [9, 10, 11, 12],
                         [0, 0, 0, 0],
                         [22, 24, 26, 28]], dtype=np.float32)
    assert np.allclose(out_ms.asnumpy(), expected, loss, loss)


def sparse_segment_sum_with_num_segments_pynative(loss):
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    x_np = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.float64)
    indices_np = np.array([0, 1, 1, 2, 3, 3], dtype=np.int64)
    segment_ids_np = np.array([0, 0, 3, 5, 5, 7], dtype=np.int64)
    num_segments_np = np.array(8, dtype=np.int64)
    x_ms = Tensor(x_np)
    indices_ms = Tensor(indices_np)
    segment_ids_ms = Tensor(segment_ids_np)
    num_segments_ms = Tensor(num_segments_np)
    net_ms = SparseSegmentSumWithNumSegmentsNet()
    out_ms = net_ms(x_ms, indices_ms, segment_ids_ms, num_segments_ms)
    expected = np.array([[6, 8, 10, 12],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [5, 6, 7, 8],
                         [0, 0, 0, 0],
                         [22, 24, 26, 28],
                         [0, 0, 0, 0],
                         [13, 14, 15, 16]], dtype=np.float64)
    assert np.allclose(out_ms.asnumpy(), expected, loss, loss)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sparse_segment_sum_with_num_segments_graph_float32_int32_int32_int32():
    """
    Feature: ALL To ALL
    Description: test cases for SparseSegmentSumWithNumSegments
    Expectation: the result match to tensorflow
    """
    sparse_segment_sum_with_num_segments(loss=1.0e-4)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sparse_segment_sum_with_num_segments_pynative_float64_int64_int64_int64():
    """
    Feature: ALL To ALL
    Description: test cases for SparseSegmentSumWithNumSegments
    Expectation: the result match to tensorflow
    """
    sparse_segment_sum_with_num_segments_pynative(loss=1.0e-5)
