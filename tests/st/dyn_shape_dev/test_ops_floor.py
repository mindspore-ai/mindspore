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
def floor_forward_func(x):
    return ops.auto_generate.floor(x)


@test_utils.run_with_cell
def floor_backward_func(x):
    return ops.grad(floor_forward_func, (0,))(x)


def floor_dyn_shape_func(x):
    return ops.auto_generate.floor(x)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_func
def test_floor_forward(mode):
    """
    Feature: Ops.
    Description: test op floor.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(np.array([1.1, 2.5, -1.5]), ms.float32)
    output = floor_forward_func(x)
    expect = [1., 2., -2]
    assert np.allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_func
def test_floor_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op floor.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(np.array([1.1, -1.5]), ms.float32)
    output = floor_backward_func(x)
    expect = np.array([0., 0.])
    assert np.allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_func
def test_floor_vmap(mode):
    """
    Feature: test vmap function.
    Description: test floor op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(np.array([[[1.1, 2.5], [-1.5, 1.1]]]))
    nest_vmap = ops.vmap(ops.vmap(floor_forward_func, in_axes=0), in_axes=0)
    output = nest_vmap(x)
    expect = [[[1., 2.], [-2, 1.]]]
    assert np.allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.context.PYNATIVE_MODE, ms.context.GRAPH_MODE])
@test_utils.run_test_func
def test_floor_dynamic(mode):
    """
    Feature: test dynamic tensor of floor.
    Description: test dynamic tensor of floor.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(floor_dyn_shape_func)
    test_cell.set_inputs(x_dyn)
    np_x = np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4)
    x = ms.Tensor(np_x, ms.float32)
    output = test_cell(x)
    expect = np.floor(np_x)
    assert np.allclose(output.asnumpy(), expect)
    np_x1 = np.arange(1 * 2 * 3 * 4).reshape(4, 3, 2, 1)
    x1 = ms.Tensor(np_x1, ms.float32)
    output1 = test_cell(x1)
    expect1 = np.floor(np_x1)
    assert np.allclose(output1.asnumpy(), expect1)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.context.PYNATIVE_MODE, ms.context.GRAPH_MODE])
@test_utils.run_test_func
def test_floor_dynamic_rank(mode):
    """
    Feature: test dynamic rank tensor of floor.
    Description: test dynamic rank tensor of floor.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(floor_dyn_shape_func)
    test_cell.set_inputs(x_dyn)
    np_x = np.arange(1 * 3).reshape(1, 3)
    x = ms.Tensor(np_x, ms.float32)
    output = test_cell(x)
    expect = np.floor(np_x)
    assert np.allclose(output.asnumpy(), expect)
    np_x1 = np.arange(1 * 2 * 3 * 4).reshape(4, 3, 2, 1)
    x1 = ms.Tensor(np_x1, ms.float32)
    output1 = test_cell(x1)
    expect1 = np.floor(np_x1)
    assert np.allclose(output1.asnumpy(), expect1)
