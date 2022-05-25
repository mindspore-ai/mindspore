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
import mindspore.nn as nn
import mindspore.context as context
from mindspore.ops import functional as F
from mindspore.common.tensor import Tensor


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_clip_by_norm_graph():
    """
    Feature: ClipByNorm Operation function verification in GRAPH mode.
    Description: The calculation results of 'ops.ClipByNorm' should be same with the 'nn.ClipByNorm'.
    Expectation: Normal output without assert wrong.
    """
    context.set_context(mode=context.GRAPH_MODE)
    # test input arg with data type float32 and float32
    x1 = np.random.rand(2, 3, 6, 16) * 10 - 5
    x1 = Tensor(x1, ms.float32)
    clip_norm_g = Tensor(np.array([1.0]).astype(np.float32))
    actual_out1 = F.clip_by_norm(x1, clip_norm_g)
    expected_out1 = nn.ClipByNorm()(x1, clip_norm_g)
    assert np.allclose(actual_out1.asnumpy(), expected_out1.asnumpy(), 0.0001, 0.0001)
    # test input arg with data type float16 and float16
    x2 = np.random.rand(2, 3, 6, 16) * 10 - 5
    x2 = Tensor(x2, ms.float16)
    clip_norm_g = Tensor(np.array([1.0]).astype(np.float16))
    actual_out2 = F.clip_by_norm(x2, clip_norm_g)
    expected_out2 = nn.ClipByNorm()(x2, clip_norm_g)
    assert np.allclose(actual_out2.asnumpy(), expected_out2.asnumpy(), 0.001, 0.001)
    # test input arg with data type float32 and float16
    x3 = np.random.rand(2, 3, 6, 16) * 10 - 5
    x3 = Tensor(x3, ms.float32)
    clip_norm_g = Tensor(np.array([2.0]).astype(np.float16))
    actual_out3 = F.clip_by_norm(x3, clip_norm_g)
    expected_out3 = nn.ClipByNorm()(x3, clip_norm_g)
    assert np.allclose(actual_out3.asnumpy(), expected_out3.asnumpy(), 0.0001, 0.0001)
    # test input arg with data type float16 and float32
    x4 = np.random.rand(2, 3, 6, 16) * 10 - 5
    x4 = Tensor(x4, ms.float16)
    clip_norm_g = Tensor(np.array([2.0]).astype(np.float32))
    actual_out4 = F.clip_by_norm(x4, clip_norm_g)
    expected_out4 = nn.ClipByNorm()(x4, clip_norm_g)
    assert np.allclose(actual_out4.asnumpy(), expected_out4.asnumpy(), 0.001, 0.001)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_clip_by_norm_pynative():
    """
    Feature: ClipByNorm Operation function verification in PyNative mode.
    Description: The calculation results of 'ops.ClipByNorm' should be same with the 'nn.ClipByNorm'.
    Expectation: Normal output without assert wrong.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    # test input arg with data type float32 and float32
    x5 = np.random.rand(2, 3, 6, 16) * 10 - 5
    x5 = Tensor(x5, ms.float32)
    clip_norm_y = Tensor(np.array([1.0]).astype(np.float32))
    actual_out5 = F.clip_by_norm(x5, clip_norm_y)
    expected_out5 = nn.ClipByNorm()(x5, clip_norm_y)
    assert np.allclose(actual_out5.asnumpy(), expected_out5.asnumpy(), 0.0001, 0.0001)
    # test input arg with data type float16 and float16
    x6 = np.random.rand(2, 3, 6, 16) * 10 - 5
    x6 = Tensor(x6, ms.float16)
    clip_norm_y = Tensor(np.array([1.0]).astype(np.float16))
    actual_out6 = F.clip_by_norm(x6, clip_norm_y)
    expected_out6 = nn.ClipByNorm()(x6, clip_norm_y)
    assert np.allclose(actual_out6.asnumpy(), expected_out6.asnumpy(), 0.001, 0.001)
    # test input arg with data type float32 and float16
    x7 = np.random.rand(2, 3, 6, 16) * 10 - 5
    x7 = Tensor(x7, ms.float32)
    clip_norm_y = Tensor(np.array([2.0]).astype(np.float16))
    actual_out7 = F.clip_by_norm(x7, clip_norm_y)
    expected_out7 = nn.ClipByNorm()(x7, clip_norm_y)
    assert np.allclose(actual_out7.asnumpy(), expected_out7.asnumpy(), 0.0001, 0.0001)
    # test input arg with data type float16 and float32
    x8 = np.random.rand(2, 3, 6, 16) * 10 - 5
    x8 = Tensor(x8, ms.float16)
    clip_norm_y = Tensor(np.array([2.0]).astype(np.float32))
    actual_out8 = F.clip_by_norm(x8, clip_norm_y)
    expected_out8 = nn.ClipByNorm()(x8, clip_norm_y)
    assert np.allclose(actual_out8.asnumpy(), expected_out8.asnumpy(), 0.001, 0.001)
