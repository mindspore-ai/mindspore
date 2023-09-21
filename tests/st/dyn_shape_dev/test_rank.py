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


@ms.jit
def rank_forward_func(x):
    return ops.operations.manually_defined.rank(x)


@ms.jit
def rank_backward_func(x):
    return ops.grad(rank_forward_func, (0))(x)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_rank_forward(mode):
    """
    Feature: rank ops.
    Description: test ops rank.
    Expectation: output the right rank of a tensor.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[2, 2], [2, 2]]).astype(np.float32))
    output = rank_forward_func(x)
    expect_output = 2
    np.testing.assert_equal(output, expect_output)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_rank_backward(mode):
    """
    Feature: rank ops.
    Description: test auto grad of ops rank.
    Expectation: output the right grad.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[2, 2], [2, 2]]).astype(np.float32))
    output = rank_backward_func(x)
    expect_output = np.array([[0, 0], [0, 0]]).astype(np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect_output)
