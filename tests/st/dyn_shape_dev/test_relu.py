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
def relu_forward_func(x):
    return ops.auto_generate.relu(x)


@ms.jit
def relu_backward_func(x):
    return ops.grad(relu_forward_func, (0,))(x)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
def test_relu_forward():
    """
    Feature: Ops.
    Description: test op relu.
    Expectation: expect correct result.
    """
    x = ms.Tensor(np.array([[[[-1, 1, 10],
                              [1, -1, 1],
                              [10, 1, -1]]]]).astype(np.float32))
    out = relu_forward_func(x)
    expect_out = np.array([[[[0, 1, 10],
                             [1, 0, 1],
                             [10, 1, 0.]]]]).astype(np.float32)
    assert (out.asnumpy() == expect_out).all()


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
def test_relu_backward():
    """
    Feature: Auto grad.
    Description: test auto grad of op relu.
    Expectation: expect correct result.
    """
    x = ms.Tensor(np.array([[[[-1, 1, 10],
                              [1, -1, 1],
                              [10, 1, -1]]]]).astype(np.float32))
    grad = relu_backward_func(x)
    expect_grad = np.array([[[[0, 1, 1],
                              [1, 0, 1],
                              [1, 1, 0]]]]).astype(np.float32)
    assert (grad.asnumpy() == expect_grad).all()
