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
from mindspore import Tensor
from mindspore.ops.operations.image_ops import ResizeLinear1D

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_resize_linear_1d_align_corners(dtype):
    """
    Feature: ResizeLinear1D cpu kernel align_corners mode
    Description: test the rightness of ResizeLinear1D cpu kernel.
    Expectation: the output is same as expect.
    """
    x = Tensor(np.array([[[1, 2, 3],
                          [4, 5, 6]]], dtype=dtype))
    size = Tensor(np.array([6], dtype=np.int64))
    output = ResizeLinear1D()(x, size)
    expect = np.array([[[1., 1.4, 1.8, 2.2, 2.6, 3.],
                        [4., 4.4, 4.8, 5.2, 5.6, 6.]]]).astype(dtype)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_resize_linear_1d_half_pixel(dtype):
    """
    Feature: ResizeLinear1D cpu kernel half_pixel mode
    Description: test the rightness of ResizeLinear1D cpu kernel.
    Expectation: the output is same as expect.
    """
    x = Tensor(np.array([[[1, 2, 3],
                          [4, 5, 6]]], dtype=dtype))
    size = Tensor(np.array([6], dtype=np.int64))
    output = ResizeLinear1D(
        coordinate_transformation_mode="half_pixel")(x, size)
    expect = np.array([[[1., 1.25, 1.75, 2.25, 2.75, 3.],
                        [4., 4.25, 4.75, 5.25, 5.75, 6.]]]).astype(dtype)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_resize_linear_1d_size_not_change(dtype):
    """
    Feature: ResizeLinear1D cpu kernel same input shape
    Description: test the rightness of ResizeLinear1D cpu kernel.
    Expectation: the output is same as expect.
    """
    x = Tensor(np.array([[[1, 2, 3],
                          [4, 5, 6]]], dtype=dtype))
    size = Tensor(np.array([3], dtype=np.int64))
    output = ResizeLinear1D()(x, size)
    expect = np.array([[[1., 2., 3.],
                        [4., 5., 6.]]]).astype(dtype)
    assert np.allclose(output.asnumpy(), expect)
