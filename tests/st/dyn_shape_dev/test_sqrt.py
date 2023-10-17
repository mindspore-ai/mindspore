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
def sqrt_forward_func(x):
    return ops.auto_generate.sqrt(x)


@test_utils.run_with_cell
def sqrt_backward_func(x):
    return ops.grad(sqrt_forward_func, (0,))(x)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_sqrt_forward(mode):
    """
    Feature: Ops.
    Description: test op sqrt.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = ms.Tensor(np.array([[0.2948122, 0.49372014]]).astype(np.float32))
    expect_out = np.array([[0.5429661, 0.7026522]]).astype(np.float32)
    out = sqrt_forward_func(x)
    print("out:", out)
    assert np.allclose(out.asnumpy(), expect_out, 1e-04, 1e-04)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_sqrt_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op sqrt.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = ms.Tensor(np.array([[0.02595769, 0.25027096]]).astype(np.float32))
    expect_out = np.array([[3.1033945, 0.9994585]]).astype(np.float32)
    grads = sqrt_backward_func(x)
    print("grads:", grads)
    assert np.allclose(grads.asnumpy(), expect_out, 1e-04, 1e-04)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_sqrt_vmap(mode):
    """
    Feature: test vmap function.
    Description: test sqrt op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    in_axes = -1
    x = ms.Tensor(np.array([[[0.21901467, 1.9256916]]]).astype(np.float32))
    expect_out = np.array([[[0.46801758]], [[1.3876953]]]).astype(np.float32)
    nest_vmap = ops.vmap(ops.vmap(
        sqrt_forward_func, in_axes=in_axes, out_axes=0), in_axes=in_axes, out_axes=0)
    out = nest_vmap(x)
    print("out:", out)
    assert np.allclose(out.asnumpy(), expect_out, 1e-04, 1e-04)
