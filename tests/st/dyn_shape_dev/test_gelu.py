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
from mindspore import Tensor
import mindspore as ms


@test_utils.run_with_cell
def gelu_forward_func(x):
    return ops.auto_generate.gelu(x)


@test_utils.run_with_cell
def gelu_backward_func(x):
    return ops.grad(gelu_forward_func, (0,))(x)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE])
def test_gelu_forward(mode):
    """
    Feature: Ops.
    Description: test op gelu.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    np_array = np.array([1.0, 2.0, 3.0]).astype('float32')
    x = Tensor(np_array)
    out = gelu_forward_func(x)
    expect = np.array([0.841192, 1.9545976, 2.9963627]).astype('float32')
    assert np.allclose(out.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE])
def test_gelu_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op gelu.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    np_array = np.array([1.0, 2.0, 3.0]).astype('float32')
    x = Tensor(np_array)
    grads = gelu_backward_func(x)
    print("grads: ", grads)
    expect = np.array([1.0829641, 1.0860993, 1.0115843]).astype('float32')
    assert np.allclose(grads.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE])
def test_gelu_vmap(mode):
    """
    Feature: test vmap function.
    Description: test avgpool op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    np_array = np.array([[0.5, 0.4, -0.3, -0.2]]).astype('float32')
    x = Tensor(np_array)
    nest_vmap = ops.vmap(ops.vmap(gelu_forward_func, in_axes=0), in_axes=0)
    out = nest_vmap(x)
    print("vmap out: ", out)
    expect = np.array(
        [[0.345714, 0.26216117, -0.11462908, -0.08414857]]).astype(np.float32)
    assert np.allclose(out.asnumpy(), expect, rtol=1e-4, atol=1e-4)
