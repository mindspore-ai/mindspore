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
"""Tests for ops.cov."""
import numpy as np
import pytest
from tests.mark_utils import arg_mark
from tests.st.utils import test_utils

import mindspore as ms
from mindspore import Tensor, ops


@test_utils.run_with_cell
def cov_forward_func(x, correction, fweights):
    return ops.cov(x, correction=correction, fweights=fweights)

def _numpy_cov(x_np, correction, fweights_np):
    """Compute covariance using numpy as reference (always in float32+)."""
    x_fp32 = x_np.astype(np.float32) if x_np.dtype == np.float16 else x_np
    return np.cov(x_fp32, aweights=None, fweights=fweights_np,
                  bias=correction == 0).astype(np.float32)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu'],
          level_mark='level1', card_mark='onecard',
          mem_peak=1024.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_cov_float16_forward(context_mode):
    """
    Feature: ops.cov
    Description: Verify ops.cov forward with float16 input produces float32 output.
    Expectation: output dtype is float32 and values match numpy reference.
    """
    ms.context.set_context(mode=context_mode)
    np.random.seed(0)
    x_np = np.random.randn(6, 8).astype(np.float16)
    correction = 1
    fweights_np = np.random.randint(1, 10, (8,))

    x = Tensor(x_np)
    fw = Tensor(fweights_np)
    output = cov_forward_func(x, correction, fw)

    assert output.dtype == ms.float32, (
        f"Expected float32 output for float16 input, got {output.dtype}")
    expect = _numpy_cov(x_np, correction, fweights_np)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)
