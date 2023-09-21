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
def sigmoid_grad_forward_func(y, dy):
    return ops.auto_generate.sigmoid_grad(y, dy)


@test_utils.run_with_cell
def sigmoid_grad_backward_func(y, dy):
    return ops.grad(sigmoid_grad_forward_func, (0, 1))(y, dy)

def sigmoid_grad_dyn_shape_func(y, dy):
    return ops.auto_generate.sigmoid_grad(y, dy)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_sigmoid_grad_forward(mode):
    """
    Feature: Ops.
    Description: Test op SigmoidGrad forward.
    Expectation: Correct result.
    """
    ms.context.set_context(mode=mode)
    expect_out = ms.Tensor([[0, 0], [-2, -6]], ms.float32)
    y = ms.Tensor([[0, 1], [2, 3]], ms.float32)
    dy = ms.Tensor([[1, 1], [1, 1]], ms.float32)
    out = sigmoid_grad_forward_func(y, dy)
    assert np.allclose(out.numpy(), expect_out.numpy())


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_sigmoid_grad_backward(mode):
    """
    Feature: Ops.
    Description: Test op SigmoidGrad backward.
    Expectation: Correct result.
    """
    ms.context.set_context(mode=mode)
    expect_out = ms.Tensor([[1, -1], [-3, -5]], ms.float32)
    y = ms.Tensor([[0, 1], [2, 3]], ms.float32)
    dy = ms.Tensor([[1, 1], [1, 1]], ms.float32)
    ddy, _ = sigmoid_grad_backward_func(y, dy)
    assert np.allclose(ddy.numpy(), expect_out.numpy())

@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
def test_sigmoid_grad_vmap():
    """
    Feature: test vmap function.
    Description: test sigmoid_grad op vmap.
    Expectation: expect correct result.
    """
    y = ms.Tensor([[0, 1], [2, 3]], ms.float32)
    dy = ms.Tensor([[1, 1], [1, 1]], ms.float32)
    expect_out = ms.Tensor([[0, 0], [-2, -6]], ms.float32)
    nest_vmap = ops.vmap(ops.vmap(sigmoid_grad_forward_func, in_axes=(0, 0)), in_axes=(0, 0))
    out = nest_vmap(y, dy)
    assert np.allclose(out.numpy(), expect_out.numpy())

@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_sigmoid_grad_dynamic(mode):
    """
    Feature: test dynamic tensor and dynamic scalar of sigmoid grad.
    Description: test dynamic tensor and dynamic scalar of sigmoid grad.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    y_dyn1 = ms.Tensor(shape=None, dtype=ms.float32)
    dy_dyn1 = ms.Tensor(shape=None, dtype=ms.float32)
    expect_out1 = ms.Tensor([0], ms.float32)
    y1 = ms.Tensor([0], ms.float32)
    dy1 = ms.Tensor([1], ms.float32)
    test_cell = test_utils.to_cell_obj(sigmoid_grad_dyn_shape_func)
    test_cell.set_inputs(y_dyn1, dy_dyn1)
    out1 = test_cell(y1, dy1)
    assert np.allclose(out1.numpy(), expect_out1.numpy())
    y_dyn2 = ms.Tensor(shape=[None, None], dtype=ms.float32)
    dy_dyn2 = ms.Tensor(shape=[None, None], dtype=ms.float32)
    expect_out2 = ms.Tensor([[0, 0], [-2, -6]], ms.float32)
    y2 = ms.Tensor([[0, 1], [2, 3]], ms.float32)
    dy2 = ms.Tensor([[1, 1], [1, 1]], ms.float32)
    test_cell.set_inputs(y_dyn2, dy_dyn2)
    out2 = test_cell(y2, dy2)
    assert np.allclose(out2.numpy(), expect_out2.numpy())
