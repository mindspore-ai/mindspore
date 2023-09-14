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


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE])
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
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE])
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
