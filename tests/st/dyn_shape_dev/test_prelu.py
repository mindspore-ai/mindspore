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
from mindspore import Tensor, context
from mindspore import ops
import test_utils


@test_utils.run_with_cell
def prelu_forward_func(x, weight):
    return ops.auto_generate.prelu(x, weight)


@test_utils.run_with_cell
def prelu_backward_func(x, weight):
    return ops.grad(prelu_forward_func, (0, 1))(x, weight)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_prelu_forward(mode):
    """
    Feature: prelu ops.
    Description: test ops prelu.
    Expectation: output right results.
    """
    context.set_context(mode=mode)
    x = Tensor(np.arange(-6, 6).reshape((2, 3, 2)).astype(np.float32))
    weight = Tensor(np.array([0.1, 0.6, -0.3]).astype(np.float32))
    output = prelu_forward_func(x, weight)
    print("output:", output)
    expect_output = np.array([[[-0.6, -0.5],
                               [-2.4, -1.8],
                               [0.6, 0.3]],
                              [[0., 1.],
                               [2., 3.],
                               [4., 5.]]]).astype(np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expect_output, decimal=4)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_prelu_backward(mode):
    """
    Feature: prelu ops.
    Description: test auto grad of ops prelu.
    Expectation: output the right grad.
    """
    context.set_context(mode=mode)
    x = Tensor(np.arange(-6, 6).reshape((2, 3, 2)).astype(np.float32))
    weight = Tensor(np.array([0.1, 0.6, -0.3]).astype(np.float32))
    output = prelu_backward_func(x, weight)
    print("output:", output)
