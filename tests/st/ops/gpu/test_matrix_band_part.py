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

import mindspore.nn as nn
from mindspore import Tensor, context
from mindspore.ops import functional as F
from mindspore.ops.operations import _inner_ops as inner


class MatrixBandPartDynamicShapeNet(nn.Cell):
    def __init__(self):
        super(MatrixBandPartDynamicShapeNet, self).__init__()
        self.test_dynamic = inner.GpuConvertToDynamicShape()

    def construct(self, x, lower, upper):
        x = self.test_dynamic(x)
        return F.matrix_band_part(x, lower, upper)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [np.int32, np.float16, np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize('batch_shape, rows, cols',
                         [([], 1, 1), ([], 1, 7), ([], 7, 1), ([], 7, 7),
                          ([2], 1, 1), ([2], 1, 7), ([2], 7, 1), ([2], 7, 7),
                          ([1, 3, 2], 1, 1), ([1, 3, 2], 1, 7), ([1, 3, 2], 7, 1), ([1, 3, 2], 7, 7)])
def test_matrix_band_part(mode, dtype, batch_shape, rows, cols):
    """
    Feature: ALL TO ALL
    Description: test general matrix cases for matrix_band_diag
    Expectation: the result match numpy.
    """
    context.set_context(mode=mode, device_target="GPU")
    input_x = np.ones(batch_shape + [rows, cols]).astype(dtype)
    for lower in (-1, 0, 1, rows - 1):
        for upper in (-1, 0, 1, cols - 1):
            np_output = input_x
            if lower >= 0:
                np_output = np.triu(np_output, -lower)
            if upper >= 0:
                np_output = np.tril(np_output, upper)
            ms_output = F.matrix_band_part(Tensor(input_x), lower, upper)
            np.testing.assert_array_almost_equal(ms_output.asnumpy(), np_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_matrix_band_part_vmap(mode):
    """
    Feature: test matrix_band_part vmap feature.
    Description: test matrix_band_part vmap feature.
    Expectation: Success.
    """
    context.set_context(mode=mode, device_target="GPU")
    x = Tensor(np.ones((2, 2, 3, 5)).astype(np.float32))
    # Case 1
    lower = 1
    upper = 1
    output = F.vmap(F.matrix_band_part, (0, None, None), 0)(x, lower, upper)
    expect_output = np.array([[[[1., 1., 0., 0., 0.],
                                [1., 1., 1., 0., 0.],
                                [0., 1., 1., 1., 0.]],
                               [[1., 1., 0., 0., 0.],
                                [1., 1., 1., 0., 0.],
                                [0., 1., 1., 1., 0.]]],
                              [[[1., 1., 0., 0., 0.],
                                [1., 1., 1., 0., 0.],
                                [0., 1., 1., 1., 0.]],
                               [[1., 1., 0., 0., 0.],
                                [1., 1., 1., 0., 0.],
                                [0., 1., 1., 1., 0.]]]], dtype=np.float32)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)

    # Case 2
    lower = 1
    upper = 1
    output = F.vmap(F.matrix_band_part, (-1, None, None), -1)(x, lower, upper)
    expect_output = np.array([[[[1., 1., 1., 1., 1.],
                                [1., 1., 1., 1., 1.],
                                [0., 0., 0., 0., 0.]],
                               [[1., 1., 1., 1., 1.],
                                [1., 1., 1., 1., 1.],
                                [1., 1., 1., 1., 1.]]],
                              [[[1., 1., 1., 1., 1.],
                                [1., 1., 1., 1., 1.],
                                [0., 0., 0., 0., 0.]],
                               [[1., 1., 1., 1., 1.],
                                [1., 1., 1., 1., 1.],
                                [1., 1., 1., 1., 1.]]]], dtype=np.float32)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)

    # Case 3
    lower = Tensor(np.array([1, 0]).astype(np.int64))
    upper = 1
    output = F.vmap(F.matrix_band_part, (0, 0, None), 0)(x, lower, upper)
    expect_output = np.array([[[[1., 1., 0., 0., 0.],
                                [1., 1., 1., 0., 0.],
                                [0., 1., 1., 1., 0.]],
                               [[1., 1., 0., 0., 0.],
                                [1., 1., 1., 0., 0.],
                                [0., 1., 1., 1., 0.]]],
                              [[[1., 1., 0., 0., 0.],
                                [0., 1., 1., 0., 0.],
                                [0., 0., 1., 1., 0.]],
                               [[1., 1., 0., 0., 0.],
                                [0., 1., 1., 0., 0.],
                                [0., 0., 1., 1., 0.]]]], dtype=np.float32)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)

    # Case 4
    lower = Tensor(np.array([1, 0]).astype(np.int32))
    upper = Tensor(np.array([1, 0]).astype(np.int32))
    output = F.vmap(F.matrix_band_part, (0, 0, 0), 0)(x, lower, upper)
    expect_output = np.array([[[[1., 1., 0., 0., 0.],
                                [1., 1., 1., 0., 0.],
                                [0., 1., 1., 1., 0.]],
                               [[1., 1., 0., 0., 0.],
                                [1., 1., 1., 0., 0.],
                                [0., 1., 1., 1., 0.]]],
                              [[[1., 0., 0., 0., 0.],
                                [0., 1., 0., 0., 0.],
                                [0., 0., 1., 0., 0.]],
                               [[1., 0., 0., 0., 0.],
                                [0., 1., 0., 0., 0.],
                                [0., 0., 1., 0., 0.]]]], dtype=np.float32)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)

    # Case 5
    lower = Tensor(np.array([1, 0]).astype(np.int64))
    upper = Tensor(np.array([1, -1]).astype(np.int64))
    output = F.vmap(F.matrix_band_part, (0, 0, 0), 0)(x, lower, upper)
    expect_output = np.array([[[[1., 1., 0., 0., 0.],
                                [1., 1., 1., 0., 0.],
                                [0., 1., 1., 1., 0.]],
                               [[1., 1., 0., 0., 0.],
                                [1., 1., 1., 0., 0.],
                                [0., 1., 1., 1., 0.]]],
                              [[[1., 1., 1., 1., 1.],
                                [0., 1., 1., 1., 1.],
                                [0., 0., 1., 1., 1.]],
                               [[1., 1., 1., 1., 1.],
                                [0., 1., 1., 1., 1.],
                                [0., 0., 1., 1., 1.]]]], dtype=np.float32)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_matrix_band_part_dynamic_shape(mode):
    """
        Feature: test matrix_band_part dynamic_shape feature.
        Description: test matrix_band_part dynamic_shape feature.
        Expectation: Success.
    """
    context.set_context(mode=mode, device_target="GPU")
    x = Tensor(np.ones((2, 2, 3, 5)).astype(np.float32))
    lower = 1
    upper = 0
    output = MatrixBandPartDynamicShapeNet()(x, lower, upper)
    expect_output = np.array([[[[1., 0., 0., 0., 0.],
                                [1., 1., 0., 0., 0.],
                                [0., 1., 1., 0., 0.]],
                               [[1., 0., 0., 0., 0.],
                                [1., 1., 0., 0., 0.],
                                [0., 1., 1., 0., 0.]]],
                              [[[1., 0., 0., 0., 0.],
                                [1., 1., 0., 0., 0.],
                                [0., 1., 1., 0., 0.]],
                               [[1., 0., 0., 0., 0.],
                                [1., 1., 0., 0., 0.],
                                [0., 1., 1., 0., 0.]]]], dtype=np.float32)
    np.testing.assert_almost_equal(output.asnumpy(), expect_output)
