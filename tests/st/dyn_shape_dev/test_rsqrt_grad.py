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
import mindspore as ms
from mindspore import ops
from mindspore.ops.auto_generate.gen_ops_def import _rsqrt_grad
import test_utils

ms.context.set_context(ascend_config={"precision_mode": "force_fp32"})

@test_utils.run_with_cell
def rsqrt_grad_func(dy, x):
    return _rsqrt_grad(dy, x)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_func
def test_rsqrt_grad(mode):
    """
    Feature: Ops.
    Description: test op rsqrt grad.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    dy = ms.Tensor(np.array([[[[-1, 1, 10],
                               [5.9, 6.1, 6],
                               [10, 1, -1]]]]).astype(np.float32))
    x = ms.Tensor(np.array([[[[1, 1, 1],
                              [2, 2, 2],
                              [3, 3, 3]]]]).astype(np.float32))
    out = rsqrt_grad_func(dy, x)
    expect_out = np.array([[[[0.5, -0.5, -500],
                             [-205.37901, -226.98099, -216],
                             [-1500, -1.5, 1.5]]]]).astype(np.float32)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_func
def test_rsqrt_grad_vmap(mode):
    """
    Feature: test vmap function.
    Description: test rsqrt op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    axes = (-1, -1)
    dy = ms.Tensor(np.random.rand(4, 3, 2).astype(np.float32))
    x = ms.Tensor(np.random.rand(4, 3, 2).astype(np.float32))
    net_vmap = ops.vmap(ops.vmap(rsqrt_grad_func, in_axes=axes, out_axes=-1), in_axes=axes, out_axes=-1)
    out = net_vmap(dy, x)
    expect_out = rsqrt_grad_func(dy, x)
    assert (out.asnumpy() == expect_out.asnumpy()).all()
