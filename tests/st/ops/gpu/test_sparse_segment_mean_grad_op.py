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
from tests.mark_utils import arg_mark
import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops.operations._grad_ops as P
from mindspore import Tensor
from mindspore.common.api import ms_function


class SparseSegmentMeanGradNet(nn.Cell):
    def __init__(self):
        super(SparseSegmentMeanGradNet, self).__init__()
        self.net = P.SparseSegmentMeanGrad()

    @ms_function
    def construct(self, grad, indices, segment_ids, output_dim0):
        return self.net(grad, indices, segment_ids, output_dim0)


def sparse_segment_mean_grad_graph_mode(loss):
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    grad_np = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.float32)
    indices_np = np.array([0, 0, 1, 2, 2, 3], dtype=np.int32)
    segment_ids_np = np.array([0, 1, 2, 2, 3, 3], dtype=np.int32)
    output_dim0_np = np.array(8, dtype=np.int32)
    grad_ms = Tensor(grad_np)
    indices_ms = Tensor(indices_np)
    segment_ids_ms = Tensor(segment_ids_np)
    output_dim0_ms = Tensor(output_dim0_np)
    net_ms = SparseSegmentMeanGradNet()
    out_ms = net_ms(grad_ms, indices_ms, segment_ids_ms, output_dim0_ms)
    expected = np.array([[6, 8, 10, 12],
                         [4.5, 5, 5.5, 6],
                         [11, 12, 13, 14],
                         [6.5, 7, 7.5, 8],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0]], dtype=np.float32)
    assert np.allclose(out_ms.asnumpy(), expected, loss, loss)


def sparse_segment_mean_grad_pynative_mode(loss):
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    grad_np = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.float64)
    indices_np = np.array([0, 1, 1, 2, 3, 3], dtype=np.int64)
    segment_ids_np = np.array([0, 0, 1, 2, 2, 3], dtype=np.int64)
    output_dim0_np = np.array(8, dtype=np.int32)
    grad_ms = Tensor(grad_np)
    indices_ms = Tensor(indices_np)
    segment_ids_ms = Tensor(segment_ids_np)
    output_dim0_ms = Tensor(output_dim0_np)
    net_ms = SparseSegmentMeanGradNet()
    out_ms = net_ms(grad_ms, indices_ms, segment_ids_ms, output_dim0_ms)
    expected = np.array([[0.5, 1, 1.5, 2],
                         [5.5, 7, 8.5, 10],
                         [4.5, 5, 5.5, 6],
                         [17.5, 19, 20.5, 22],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0]], dtype=np.float64)
    assert np.allclose(out_ms.asnumpy(), expected, loss, loss)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sparse_segment_mean_grad_graph_float32_int32_int32():
    """
    Feature: ALL To ALL
    Description: test cases for SparseSegmentMeanGrad
    Expectation: the result match to tensorflow
    """
    sparse_segment_mean_grad_graph_mode(loss=1.0e-4)



@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sparse_segment_mean_grad_pynative_float64_int64_int64():
    """
    Feature: ALL To ALL
    Description: test cases for SparseSegmentMeanGrad
    Expectation: the result match to tensorflow
    """
    sparse_segment_mean_grad_pynative_mode(loss=1.0e-5)
