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

from mindspore import Tensor, context
from mindspore.ops import functional as F


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [np.int32, np.float16, np.float32, np.float64])
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
            if batch_shape:
                np_output = np.tile(np_output, batch_shape + [1, 1])
            ms_output = F.matrix_band_part(Tensor(np_output), lower, upper)
            np.testing.assert_array_almost_equal(ms_output.asnumpy(), np_output)
