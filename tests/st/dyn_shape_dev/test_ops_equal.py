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
import test_utils


@test_utils.run_with_cell
def equal_forward_func(x, y):
    return ops.auto_generate.equal(x, y)


@test_utils.run_with_cell
def equal_backward_func(x, y):
    return ops.grad(equal_forward_func, (0,))(x, y)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE])
def test_equal_forward(mode):
    """
    Feature: Ops.
    Description: test op equal.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(np.array([0.0, 1.0, 2.0, -1]), ms.float32)
    y = Tensor(np.array([0.0, 1.0, 2.0, -2]), ms.float32)
    output = equal_forward_func(x, y)
    expect = [True, True, True, False]
    assert np.allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE])
def test_equal_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op equal.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(np.array([0.0, 1.0, 2.0, -1]), ms.float32)
    y = Tensor(np.array([0.0, 1.0, 2.0, -2]), ms.float32)
    output = equal_backward_func(x, y)
    expect = [0, 0, 0, 0]
    assert np.allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)
