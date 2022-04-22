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
import mindspore.context as context
import mindspore.ops as ops
from mindspore import Tensor


def cummin_compare(x, expected, axis, data_type):
    x = np.array(x).astype(data_type)
    expected = (np.array(expected[0]).astype(data_type), np.array(expected[1]).astype(data_type))

    # Pynative
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    output = ops.cummin(Tensor(x), axis=axis)
    assert np.allclose(output[0].asnumpy(), expected[0], equal_nan=True)
    assert np.allclose(output[1].asnumpy(), expected[1])

    # Graph
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    output = ops.cummin(Tensor(x), axis=axis)
    assert np.allclose(output[0].asnumpy(), expected[0], equal_nan=True)
    assert np.allclose(output[1].asnumpy(), expected[1])


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize("data_type", [np.int32, np.float16, np.float32])
def test_cummin_multi_dims(data_type):
    """
    Feature: Op Cummin
    Description: test Cummin operator with multiple dimension.
    Expectation: the result match expectation.
    """
    axis = 1
    x = [[[14, 19, 18, 11, 6], [1, 4, 18, 6, 1], [15, 13, 12, 9, 19]],
         [[16, 16, 17, 10, 15], [9, 7, 10, 9, 4], [6, 14, 16, 3, 2]],
         [[1, 13, 15, 1, 6], [20, 6, 8, 19, 19], [3, 14, 20, 18, 19]],
         [[20, 1, 14, 9, 3], [13, 11, 2, 17, 14], [0, 15, 13, 7, 10]]]
    cummin_output = (
        [[[14, 19, 18, 11, 6], [1, 4, 18, 6, 1], [1, 4, 12, 6, 1]],
         [[16, 16, 17, 10, 15], [9, 7, 10, 9, 4], [6, 7, 10, 3, 2]],
         [[1, 13, 15, 1, 6], [1, 6, 8, 1, 6], [1, 6, 8, 1, 6]], [[20, 1, 14, 9, 3], [13, 1, 2, 9, 3], [0, 1, 2, 7, 3]]],
        [[[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 2, 1, 1]], [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [2, 1, 1, 2, 2]],
         [[0, 0, 0, 0, 0], [0, 1, 1, 0, 0], [0, 1, 1, 0, 0]], [[0, 0, 0, 0, 0], [1, 0, 1, 0, 0], [2, 0, 1, 2, 0]]])

    cummin_compare(x, cummin_output, axis, data_type)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize("data_type", [np.float16, np.float32])
def test_cummin_nan(data_type):
    """
    Feature: Op Cummin
    Description: test Cummin operator with nan input.
    Expectation: the result match expectation.
    """
    inf = float('inf')
    nan = float('nan')
    axis = 0
    x = [4, inf, 1.5, -inf, 0, nan, 1]
    cummin_output = ([4, 4, 1.5, -inf, -inf, nan, nan], [0, 0, 2, 3, 3, 5, 5])

    cummin_compare(x, cummin_output, axis, data_type)
