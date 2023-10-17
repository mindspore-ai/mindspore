# Copyright 2020 Huawei Technologies Co., Ltd
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
def lin_space_forward_func(start, stop, num=5):
    return ops.auto_generate.lin_space_(start, stop, num)


@test_utils.run_with_cell
def lin_space_backward_func(start, stop, num=5):
    return ops.grad(lin_space_forward_func, (0,))(start, stop, num)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_lin_space_forward(mode):
    """
    Feature: Ops.
    Description: test op LinSpace.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    start, stop, num = 5, 25, 5
    output = lin_space_forward_func(ms.Tensor(start, ms.float32), ms.Tensor(stop, ms.float32), num)
    expect = np.linspace(start, stop, num, axis=-1)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_lin_space_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op LinSpace.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    start, stop, num = 5, 25, 5
    grads = lin_space_forward_func(ms.Tensor(start, ms.float32), ms.Tensor(stop, ms.float32), num)
    expect = np.array([5., 10., 15., 20., 25.]).astype(np.float32)
    assert np.allclose(grads.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_lin_space_vmap(mode):
    """
    Feature: test vmap function.
    Description: test LinSpace op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    np.random.seed(0)
    start_np = np.random.randn(5, 4)
    stop_np = np.random.randn(4, 5)
    num_np = 5
    start = ms.Tensor(start_np, dtype=ms.float32)
    stop = ms.Tensor(stop_np, dtype=ms.float32)
    result_ms = ops.vmap(ops.vmap(lin_space_forward_func, (0, 0)), (1, 0))(start, stop)
    start_np = np.moveaxis(start_np, 1, 0)
    result_np = np.linspace(start_np, stop_np, num_np, axis=-1)
    assert np.allclose(result_ms.asnumpy(), result_np)
