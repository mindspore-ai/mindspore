# Copyright 2021 Huawei Technologies Co., Ltd
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
from mindspore.ops import operations as P
from mindspore.ops.operations import _inner_ops
from mindspore.common.tensor import Tensor


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_clip_grad_norm():
    """
    Feature: ClipGradNorm Operation function verification.
    Description: The calculation results of 'ops.ClipGradNorm' should be same with the 'nn.ClipByNorm'.
    Expectation: Normal output without assert wrong.
    """
    clip_norm = Tensor(np.array([1.0]), ms.float16)
    scaling_factor = Tensor(np.array([65536]), ms.float32)

    # test input arg with shape(32, 3, 224, 224)
    x = np.random.rand(32, 3, 224, 224) * 100 - 50
    x = Tensor(x, ms.float16)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    scaling_out = x * P.Reciprocal()(scaling_factor)
    clip_by_norm_out = nn.ClipByNorm()(scaling_out, clip_norm)
    clip_grad_norm_out = _inner_ops.ClipGradNorm()(x, clip_norm, scaling_factor)
    assert np.allclose(clip_by_norm_out.asnumpy(), clip_grad_norm_out.asnumpy(), 0.00000001, 0.00000001)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    scaling_out = x * P.Reciprocal()(scaling_factor)
    clip_by_norm_out = nn.ClipByNorm()(scaling_out, clip_norm)
    clip_grad_norm_out = _inner_ops.ClipGradNorm()(x, clip_norm, scaling_factor)
    assert np.allclose(clip_by_norm_out.asnumpy(), clip_grad_norm_out.asnumpy(), 0.00000001, 0.00000001)

    # test input arg with shape(60, 224, 224)
    x = np.random.rand(60, 224, 224) * 100 - 50
    x = Tensor(x, ms.float16)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    scaling_out = x * P.Reciprocal()(scaling_factor)
    clip_by_norm_out = nn.ClipByNorm()(scaling_out, clip_norm)
    clip_grad_norm_out = _inner_ops.ClipGradNorm()(x, clip_norm, scaling_factor)
    assert np.allclose(clip_by_norm_out.asnumpy(), clip_grad_norm_out.asnumpy(), 0.00000001, 0.00000001)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    scaling_out = x * P.Reciprocal()(scaling_factor)
    clip_by_norm_out = nn.ClipByNorm()(scaling_out, clip_norm)
    clip_grad_norm_out = _inner_ops.ClipGradNorm()(x, clip_norm, scaling_factor)
    assert np.allclose(clip_by_norm_out.asnumpy(), clip_grad_norm_out.asnumpy(), 0.00000001, 0.00000001)

    # test input arg with shape(21128, 60)
    x = np.random.rand(21128, 60) * 100 - 50
    x = Tensor(x, ms.float16)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    scaling_out = x * P.Reciprocal()(scaling_factor)
    clip_by_norm_out = nn.ClipByNorm()(scaling_out, clip_norm)
    clip_grad_norm_out = _inner_ops.ClipGradNorm()(x, clip_norm, scaling_factor)
    assert np.allclose(clip_by_norm_out.asnumpy(), clip_grad_norm_out.asnumpy(), 0.00000001, 0.00000001)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    scaling_out = x * P.Reciprocal()(scaling_factor)
    clip_by_norm_out = nn.ClipByNorm()(scaling_out, clip_norm)
    clip_grad_norm_out = _inner_ops.ClipGradNorm()(x, clip_norm, scaling_factor)
    assert np.allclose(clip_by_norm_out.asnumpy(), clip_grad_norm_out.asnumpy(), 0.00000001, 0.00000001)

    # test input args with shape(60)
    x = np.random.rand(60) * 100 - 50
    x = Tensor(x, ms.float16)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    scaling_out = x * P.Reciprocal()(scaling_factor)
    clip_by_norm_out = nn.ClipByNorm()(scaling_out, clip_norm)
    clip_grad_norm_out = _inner_ops.ClipGradNorm()(x, clip_norm, scaling_factor)
    assert np.allclose(clip_by_norm_out.asnumpy(), clip_grad_norm_out.asnumpy(), 0.00000001, 0.00000001)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    scaling_out = x * P.Reciprocal()(scaling_factor)
    clip_by_norm_out = nn.ClipByNorm()(scaling_out, clip_norm)
    clip_grad_norm_out = _inner_ops.ClipGradNorm()(x, clip_norm, scaling_factor)
    assert np.allclose(clip_by_norm_out.asnumpy(), clip_grad_norm_out.asnumpy(), 0.00000001, 0.00000001)
