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

import pytest
import numpy as np
import mindspore as ms
from mindspore import ops, Tensor


@ms.jit
def expand_dims_forward_func(x, axis):
    return ops.auto_generate.expand_dims(x, axis)


@ms.jit
def expand_dims_backward_func(x, axis):
    return ops.grad(expand_dims_forward_func, (0,))(x, axis)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
def test_expand_dims_forward():
    """
    Feature: Ops.
    Description: test op expand_dims.
    Expectation: expect correct result.
    """
    x = Tensor(np.array([[2, 2], [2, 2]]), ms.float32)
    axis = 0
    output = expand_dims_forward_func(x, axis)
    expect = [[[2., 2.], [2., 2.]]]
    assert np.allclose(output.asnumpy(), expect, rtol=0.001)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
def test_expand_dims_backward():
    """
    Feature: Auto grad.
    Description: test auto grad of op expand_dims.
    Expectation: expect correct result.
    """
    x = Tensor(np.array([[2, 2], [2, 2]]), ms.float32)
    axis = 0
    output = expand_dims_backward_func(x, axis)
    expect = [[1., 1.], [1., 1.]]
    assert np.allclose(output.asnumpy(), expect, rtol=0.001)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
def test_expand_dims_vmap():
    """
    Feature: test vmap function.
    Description: test expand_dims op vmap.
    Expectation: expect correct result.
    """
    x = Tensor(np.array([[[[2, 2], [2, 2]]]]), ms.float32)
    axis = 0
    nest_vmap = ops.vmap(ops.vmap(expand_dims_forward_func, in_axes=(0, None)), in_axes=(0, None))
    output = nest_vmap(x, axis)
    expect = [[[[[2., 2.], [2., 2.]]]]]
    assert np.allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)
