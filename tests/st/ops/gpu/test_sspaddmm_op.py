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
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.ops.operations.sparse_ops import Sspaddmm


class SspaddmmNet(nn.Cell):
    def __init__(self):
        super(SspaddmmNet, self).__init__()
        self.sspaddmm = Sspaddmm()

    def construct(self, x1_indices, x1_values, x1_shape, x2_indices, x2_values, x2_shape, x3_dense, alpha, beta):
        return self.sspaddmm(x1_indices, x1_values, x1_shape, x2_indices, x2_values, x2_shape, x3_dense, alpha, beta)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sspaddmm_input_int32():
    """
    Feature: Sspaddmm gpu TEST.
    Description: 2d int32 test case for Sspaddmm
    Expectation: The value and shape of output are the expected values.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    x1_indices = Tensor(np.array([[0, 1], [0, 1]]), mstype.int32)
    x1_values = Tensor(np.array([1, 2]), mstype.int32)
    x1_shape = Tensor(np.array([3, 3]), mstype.int32)
    x2_indices = Tensor(np.array([[0, 1], [2, 2]]), mstype.int32)
    x2_values = Tensor(np.array([3, 4]), mstype.int32)
    x2_shape = Tensor(np.array([3, 3]), mstype.int32)
    x3_dense = Tensor(np.array([[1, 2, 3], [1, 3, 2], [3, 2, 1]]), mstype.int32)
    alpha = Tensor(np.array([1]), mstype.int32)
    beta = Tensor(np.array([1]), mstype.int32)
    net = SspaddmmNet()
    y_indices, y_values, y_shape = net(x1_indices, x1_values, x1_shape, x2_indices, x2_values, x2_shape, x3_dense,
                                       alpha, beta)
    y_indices_expect = np.array([[0, 1, 0, 0, 0, 1, 1, 1], [0, 1, 0, 1, 2, 0, 1, 2]], dtype=np.int64)
    y_values_expect = np.array([1, 2, 9, 6, 3, 12, 8, 4], dtype=np.int32)
    y_shape_expect = np.array([3, 3], dtype=np.int64)

    assert np.allclose(y_indices.asnumpy(), y_indices_expect.astype(np.int64), 0.0001, 0.0001)
    assert np.allclose(y_values.asnumpy(), y_values_expect.astype(np.int32), 0.0001, 0.0001)
    assert np.allclose(y_shape.asnumpy(), y_shape_expect.astype(np.int64), 0.0001, 0.0001)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sspaddmm_input_int64():
    """
    Feature: Sspaddmm gpu TEST.
    Description: 2d int64 test case for Sspaddmm
    Expectation: The value and shape of output are the expected values.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

    x1_indices = Tensor(np.array([[0, 1], [0, 1]]), mstype.int32)
    x1_values = Tensor(np.array([7, 6]), mstype.int32)
    x1_shape = Tensor(np.array([3, 3]), mstype.int32)
    x2_indices = Tensor(np.array([[0, 1], [2, 2]]), mstype.int32)
    x2_values = Tensor(np.array([11, 23]), mstype.int32)
    x2_shape = Tensor(np.array([3, 3]), mstype.int32)
    x3_dense = Tensor(np.array([[1, 2, 3], [1, 3, 2], [3, 2, 1]]), mstype.int32)
    alpha = Tensor(np.array([2]), mstype.int32)
    beta = Tensor(np.array([2]), mstype.int32)
    net = SspaddmmNet()
    y_indices, y_values, y_shape = net(x1_indices, x1_values, x1_shape, x2_indices, x2_values, x2_shape, x3_dense,
                                       alpha, beta)
    y_indices_expect = np.array([[0, 1, 0, 0, 0, 1, 1, 1], [0, 1, 0, 1, 2, 0, 1, 2]])
    y_values_expect = np.array([14, 12, 66, 44, 22, 138, 92, 46])
    y_shape_expect = np.array([3, 3])

    assert np.allclose(y_indices.asnumpy(), y_indices_expect.astype(np.int64), 0.0001, 0.0001)
    assert np.allclose(y_values.asnumpy(), y_values_expect.astype(np.int32), 0.0001, 0.0001)
    assert np.allclose(y_shape.asnumpy(), y_shape_expect.astype(np.int64), 0.0001, 0.0001)
