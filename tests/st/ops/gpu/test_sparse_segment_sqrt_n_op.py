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
import mindspore as ms
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops.operations.sparse_ops as P
from mindspore import Tensor
from mindspore.common.api import jit


class SparseSegmentSqrtNNet(nn.Cell):

    def __init__(self):
        super(SparseSegmentSqrtNNet, self).__init__()
        self.net = P.SparseSegmentSqrtN()

    @jit
    def construct(self, x, indices, segment_ids):
        return self.net(x, indices, segment_ids)


def sparse_segment_sqrt_n(loss):
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    x_np = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        dtype=np.float32)
    indices_np = np.array([0, 0, 1, 2, 2, 3], dtype=np.int32)
    segment_ids_np = np.array([0, 1, 2, 2, 3, 3], dtype=np.int32)
    x_ms = Tensor(x_np)
    indices_ms = Tensor(indices_np)
    segment_ids_ms = Tensor(segment_ids_np)
    net_ms = SparseSegmentSqrtNNet()
    out_ms = net_ms(x_ms, indices_ms, segment_ids_ms)
    expected = np.array([[1, 2, 3, 4], [1, 2, 3, 4],
                         [9.899495, 11.313708, 12.727922, 14.142136],
                         [15.556349, 16.970562, 18.384777, 19.79899]],
                        dtype=np.float32)
    assert np.allclose(out_ms.asnumpy(), expected, loss, loss)


def sparse_segment_sqrt_n_pynative(loss):
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    x_np = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        dtype=np.float64)
    indices_np = np.array([0, 1, 1, 2, 3, 3], dtype=np.int64)
    segment_ids_np = np.array([0, 0, 1, 2, 2, 3], dtype=np.int64)
    x_ms = Tensor(x_np)
    indices_ms = Tensor(indices_np)
    segment_ids_ms = Tensor(segment_ids_np)
    net_ms = SparseSegmentSqrtNNet()
    out_ms = net_ms(x_ms, indices_ms, segment_ids_ms)
    expected = np.array(
        [[4.24264069, 5.65685425, 7.07106781, 8.48528137], [5, 6, 7, 8],
         [15.55634919, 16.97056275, 18.38477631, 19.79898987],
         [13, 14, 15, 16]],
        dtype=np.float64)
    assert np.allclose(out_ms.asnumpy(), expected, loss, loss)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sparse_segment_sqrt_n_graph_float32_int32_int32():
    """
    Feature: ALL To ALL
    Description: test cases for SparseSegmentSqrtN
    Expectation: the result match to tensorflow
    """
    sparse_segment_sqrt_n(loss=1.0e-4)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sparse_segment_sqrt_n_pynative_float64_int64_int64():
    """
    Feature: ALL To ALL
    Description: test cases for SparseSegmentSqrtN
    Expectation: the result match to tensorflow
    """
    sparse_segment_sqrt_n_pynative(loss=1.0e-5)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu
@pytest.mark.env_onecard
def test_sparse_segment_sqrt_n_dyn():
    """
    Feature: test SparseSegmentSqrtN ops in gpu.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = SparseSegmentSqrtNNet()

    x_dyn = Tensor(shape=[None, None], dtype=ms.float32)
    indices_dyn = Tensor(shape=[None], dtype=ms.int32)
    segment_ids_dyn = Tensor(shape=[None], dtype=ms.int32)
    net.set_inputs(x_dyn, indices_dyn, segment_ids_dyn)

    x = Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=ms.float32)
    indices = Tensor([0, 1, 2], dtype=ms.int32)
    segment_ids = Tensor([0, 1, 2], dtype=ms.int32)
    output = net(x, indices, segment_ids)

    expect_shape = (3, 4)
    assert output.asnumpy().shape == expect_shape
