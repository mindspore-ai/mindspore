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
def reduce_mean_forward_func(x):
    return ops.auto_generate.reduce_mean(x, axis=0, keep_dims=True)


@ms.jit
def reduce_mean_backward_func(x):
    return ops.grad(reduce_mean_forward_func, (0,))(x)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
def test_reduce_mean():
    """
    Feature: Ops.
    Description: test op reduce mean.
    Expectation: expect correct result.
    """
    x = ms.Tensor(np.array([[0.248653, 0.273924, 0.640271],
                            [0.746676, 0.004394, 0.437812]]).astype(np.float32))
    out = reduce_mean_forward_func(x)
    expect_out = np.array([[0.4976645, 0.139159, 0.5390415]]).astype(np.float32)
    assert np.allclose(out.asnumpy(), expect_out, rtol=1e-4, atol=1e-4)

    grads = reduce_mean_backward_func(x)
    expect_grad = np.array([[0.5, 0.5, 0.5],
                            [0.5, 0.5, 0.5]]).astype(np.float32)
    assert np.allclose(grads.asnumpy(), expect_grad, rtol=1e-4, atol=1e-4)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
def test_reduce_mean_vmap():
    """
    Feature: test vmap function.
    Description: test reduce_mean op vmap.
    Expectation: expect correct result.
    """
    in_axes = -1
    x = ms.Tensor(np.random.uniform(low=-1, high=1, size=(4, 3, 2, 2)).astype(np.float32))
    nest_vmap = ops.vmap(ops.vmap(reduce_mean_forward_func, in_axes=in_axes, out_axes=-1), in_axes=in_axes, out_axes=-1)
    out = nest_vmap(x)
    expect_out = reduce_mean_forward_func(x)
    assert np.allclose(out.asnumpy(), expect_out.asnumpy(), rtol=1e-4, atol=1e-4)
