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


@ms.jit
def round_forward_func(x):
    return ops.auto_generate.round(x)


@ms.jit
def round_backward_func(x):
    return ops.grad(round_forward_func, (0,))(x)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
def test_round():
    """
    Feature: Ops.
    Description: test op round.
    Expectation: expect correct result.
    """
    np_array = np.array([0.8, 1.5, 2.3, 2.5, -4.5]).astype(np.float32)
    x = ms.Tensor(np_array)
    out = round_forward_func(x)
    expect_out = np.round(np_array).astype(np.float32)
    assert (out.asnumpy() == expect_out).all()

    grad = round_backward_func(x)
    expect_grad = np.zeros(shape=(5,)).astype(np.float32)
    assert (grad.asnumpy() == expect_grad).all()


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
def test_round_vmap():
    """
    Feature: test vmap function.
    Description: test round op vmap.
    Expectation: expect correct result.
    """
    axes = -1
    x = ms.Tensor(np.random.uniform(low=-255, high=255, size=(4, 3, 2)).astype(np.float32))
    net_vmap = ops.vmap(ops.vmap(round_forward_func, in_axes=axes, out_axes=axes), in_axes=axes, out_axes=axes)
    out = net_vmap(x)
    expect_out = round_forward_func(x)
    assert (out.asnumpy() == expect_out.asnumpy()).all()
