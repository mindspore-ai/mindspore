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
def reduce_std_forward_func(x):
    return ops.auto_generate.reduce_std(x, axis=0, unbiased=True, keep_dims=True)


@test_utils.run_with_cell
def reduce_std_backward_func(x):
    return ops.grad(reduce_std_forward_func, (0,))(x)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
@test_utils.run_test_func
def test_reduce_std(mode):
    """
    Feature: Ops.
    Description: test op reduce std.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = ms.Tensor(np.array([[0.392192, 1.484581, 1.111067],
                            [1.614102, 0.676057, 3.279313]]).astype(np.float32))
    out = reduce_std_forward_func(x)
    expect_std = np.array([[0.864021, 0.571713, 1.533181]]).astype(np.float32)
    assert np.allclose(out[0].asnumpy(), expect_std, rtol=1e-4, atol=1e-4)
    expect_mean = np.array([[1.003147, 1.080319, 2.19519]]).astype(np.float32)
    assert np.allclose(out[1].asnumpy(), expect_mean, rtol=1e-4, atol=1e-4)

    grad = reduce_std_backward_func(x)
    expect_grad = np.array([[-0.207107, 1.207107, -0.207107],
                            [1.207107, -0.207107, 1.207107]]).astype(np.float32)
    assert np.allclose(grad.asnumpy(), expect_grad, rtol=1e-4, atol=1e-4)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
@test_utils.run_test_func
def test_reduce_std_vmap(mode):
    """
    Feature: test vmap function.
    Description: test reduce_std op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    in_axes = -1
    x = ms.Tensor(np.random.uniform(low=-1, high=1, size=(4, 3, 2, 2)).astype(np.float32))
    nest_vmap = ops.vmap(ops.vmap(reduce_std_forward_func, in_axes=in_axes, out_axes=-1), in_axes=in_axes, out_axes=-1)
    out = nest_vmap(x)
    expect_out = reduce_std_forward_func(x)
    assert np.allclose(out[0].asnumpy(), expect_out[0].asnumpy(), rtol=1e-4, atol=1e-4)
    assert np.allclose(out[1].asnumpy(), expect_out[1].asnumpy(), rtol=1e-4, atol=1e-4)
