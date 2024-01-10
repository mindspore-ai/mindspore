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
from mindspore import ops
import test_utils


@test_utils.run_with_cell
def silu_forward_func(x):
    return ops.auto_generate.silu(x)


@test_utils.run_with_cell
def silu_backward_func(x):
    return ops.grad(silu_forward_func, (0,))(x)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_silu_forward(mode):
    """
    Feature: Ops.
    Description: Test op SiLU forward.
    Expectation: Correct result.
    """
    ms.context.set_context(mode=mode)
    expect_out = ms.Tensor([[0, 0.7310586], [1.7615942, 2.8577223]], ms.float32)
    x = ms.Tensor([[0, 1], [2, 3]], ms.float32)
    out = silu_forward_func(x)
    assert np.allclose(out.numpy(), expect_out.numpy())

@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_silu_backward(mode):
    """
    Feature: Ops.
    Description: Test op SiLU backward.
    Expectation: Correct result.
    """
    ms.context.set_context(mode=mode)
    expect_out = ms.Tensor([[0.5, 0.92767054], [1.0907842, 1.0881041]], ms.float32)
    x = ms.Tensor([[0, 1], [2, 3]], ms.float32)
    grads = silu_backward_func(x)
    assert np.allclose(grads.numpy(), expect_out.numpy())

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_silu_vmap(mode):
    """
    Feature: Ops.
    Description: Test op SiLU vmap.
    Expectation: Correct result.
    """
    ms.context.set_context(mode=mode)
    expect_out = ms.Tensor([[0, 0.7310586], [1.7615942, 2.8577223]], ms.float32)
    x = ms.Tensor([[0, 1], [2, 3]], ms.float32)
    nest_vmap = ops.vmap(ops.vmap(silu_forward_func, in_axes=0), in_axes=0)
    out = nest_vmap(x)
    assert np.allclose(out.numpy(), expect_out.numpy())

@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_silu_dynamic(mode):
    """
    Feature: Ops.
    Description: Test op SiLU dynamic shape.
    Expectation: Correct result.
    """
    ms.context.set_context(mode=mode)
    test_cell = test_utils.to_cell_obj(ops.auto_generate.silu)
    x_dyn1 = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell.set_inputs(x_dyn1)
    expect_out1 = ms.Tensor([0], ms.float32)
    x1 = ms.Tensor([0], ms.float32)
    out1 = test_cell(x1)
    assert np.allclose(out1.numpy(), expect_out1.numpy())
    x_dyn2 = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell.set_inputs(x_dyn2)
    expect_out2 = ms.Tensor([[0, 0.7310586], [1.7615942, 2.8577223]], ms.float32)
    x2 = ms.Tensor([[0, 1], [2, 3]], ms.float32)
    out2 = test_cell(x2)
    assert np.allclose(out2.numpy(), expect_out2.numpy())
