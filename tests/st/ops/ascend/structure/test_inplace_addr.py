# Copyright 2026 Huawei Technologies Co., Ltd
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
""" test inplace_addr"""
import numpy as np
from tests.mark_utils import arg_mark
import mindspore as ms


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard', mem_peak=0.01)
def test_inplace_addr_basic():
    """
    Feature: Structured ops.
    Description: Test inplace addr ops.
    Expectation: Success.
    """
    x = ms.mint.zeros([2, 3])
    vec1 = ms.Tensor([1., 2.])
    vec2 = ms.Tensor([3., 4., 5.])

    ms.mint.addr_(x, vec1, vec2)
    assert np.allclose(x.asnumpy(), np.array([[3., 4., 5.], [6., 8., 10.]]), 1e-5, 1e-5)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard', mem_peak=0.01)
def test_inplace_alpha_beta():
    """
    Feature: Structured ops.
    Description: Test inplace addr ops with alpha and beta.
    Expectation: Success.
    """
    x = ms.mint.ones((2, 3))

    vec1 = ms.Tensor([1., 2.])
    vec2 = ms.Tensor([3., 4., 5.])

    ms.mint.addr_(x, vec1, vec2, beta=2, alpha=0.5)

    expected = np.array([
        [3.5, 4.0, 4.5],
        [5.0, 6.0, 7.0]
    ])
    assert np.allclose(x.asnumpy(), expected, 1e-5, 1e-5)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard', mem_peak=0.01)
def test_addr_inplace_broadcast():
    """
    Feature: Structured ops.
    Description: Test inplace addr ops with vec1 and vec2 broadcast.
    Expectation: Success.
    """
    x = ms.mint.ones((1, 3))

    vec1 = ms.Tensor([2.])
    vec2 = ms.Tensor([3., 4., 5.])

    ms.mint.addr_(x, vec1, vec2)

    expected = np.array([
        [7., 9., 11.]
    ])
    assert np.allclose(x.asnumpy(), expected, 1e-5, 1e-5)
