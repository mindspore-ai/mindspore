# Copyright 2023 Huawei Technologies Co., Ltd
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
from mindspore import ops
import mindspore as ms
import test_utils


@test_utils.run_with_cell
def rsqrt_forward_func(x):
    return ops.auto_generate.rsqrt(x)


@test_utils.run_with_cell
def rsqrt_backward_func(x):
    return ops.grad(rsqrt_forward_func, (0,))(x)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_rsqrt(mode):
    """
    Feature: Ops.
    Description: test op rsqrt.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = ms.Tensor(np.array([0.0370, 0.2970, 1.5420, 0.9105]).astype(np.float32))
    out = rsqrt_forward_func(x)
    expect_out = np.array([5.1987524, 1.8349396, 0.80530024, 1.047997]).astype(np.float32)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-3)

    grad = rsqrt_backward_func(x)
    expect_grad = np.array([-70.25341, -3.089124, -0.26112202, -0.5755063]).astype(np.float32)
    np.testing.assert_allclose(grad.asnumpy(), expect_grad, rtol=1e-3)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_rsqrt_vmap(mode):
    """
    Feature: test vmap function.
    Description: test rsqrt op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    axes = -1
    x = ms.Tensor(np.random.rand(4, 3, 2).astype(np.float32))
    net_vmap = ops.vmap(ops.vmap(rsqrt_forward_func, in_axes=axes, out_axes=axes), in_axes=axes, out_axes=axes)
    out = net_vmap(x)
    expect_out = rsqrt_forward_func(x)
    assert (out.asnumpy() == expect_out.asnumpy()).all()
