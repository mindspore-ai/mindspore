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

@ms.jit
def sigmoid_forward_func(x):
    return ops.auto_generate.sigmoid(x)

@ms.jit
def sigmoid_backward_func(x):
    return ops.grad(sigmoid_forward_func, (0,))(x)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
def test_sigmoid_forward():
    """
    Feature: Ops.
    Description: Test op Sigmoid forward.
    Expectation: Correct result.
    """
    ms.context.set_context(mode=ms.context.GRAPH_MODE)
    expect_out = ms.Tensor([[0.5, 0.7310586], [0.8807971, 0.95257413]], ms.float32)
    x = ms.Tensor([[0, 1], [2, 3]], ms.float32)
    out = sigmoid_forward_func(x)
    assert np.allclose(out.numpy(), expect_out.numpy())

@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
def test_sigmoid_backward():
    """
    Feature: Ops.
    Description: Test op Sigmoid backward.
    Expectation: Correct result.
    """
    ms.context.set_context(mode=ms.context.GRAPH_MODE)
    expect_out = ms.Tensor([[0.25, 0.19661193], [0.10499357, 0.04517666]], ms.float32)
    x = ms.Tensor([[0, 1], [2, 3]], ms.float32)
    grads = sigmoid_backward_func(x)
    assert np.allclose(grads.numpy(), expect_out.numpy())
    