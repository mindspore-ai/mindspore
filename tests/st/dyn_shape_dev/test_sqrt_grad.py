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
import test_utils

from mindspore import ops
import mindspore as ms


@test_utils.run_with_cell
def sqrt_grad_forward_func(dy, x):
    return ops.auto_generate.sqrt_grad(dy, x)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE])
def test_sqrt_grad_forward(mode):
    """
    Feature: Ops.
    Description: test op sqrt_grad.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x_np = np.array([[0.02595769, 0.25027096]]).astype(np.float32)
    dy = ms.Tensor(x_np)
    x = ms.Tensor(x_np * x_np)
    expect_out = np.array([[0.0129776, 0.12512207]]).astype(np.float32)
    out = sqrt_grad_forward_func(dy, x)
    print("out:", out)
    assert np.allclose(out.asnumpy(), expect_out, 1e-04, 1e-04)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE])
def test_sqrt_grad_vmap(mode):
    """
    Feature: test vmap function.
    Description: test sqrt_grad op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    in_axes = (0, 0)
    x_np = np.array([[[0.21901467, 1.9256916]]]).astype(np.float32)
    dy = ms.Tensor(x_np)
    x = ms.Tensor(x_np * x_np)
    expect_out = np.array([[[0.10955811, 0.9628906]]]).astype(np.float32)
    nest_vmap = ops.vmap(ops.vmap(
        sqrt_grad_forward_func, in_axes=in_axes, out_axes=0), in_axes=in_axes, out_axes=0)
    out = nest_vmap(dy, x)
    print("out:", out)
    assert np.allclose(out.asnumpy(), expect_out, 1e-04, 1e-04)
