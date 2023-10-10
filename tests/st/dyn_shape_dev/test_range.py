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
from mindspore import context
from mindspore import ops
import test_utils


@test_utils.run_with_cell
def range_forward_func(start, limit, delta):
    return ops.auto_generate.range(start, limit, delta, maxlen=10)


@test_utils.run_with_cell
def range_backward_func(start, limit, delta):
    return ops.grad(range_forward_func, (0, 1, 2))(start, limit, delta)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_range_forward_tensor_input(mode):
    """
    Feature: range ops.
    Description: test ops range for Tensor input.
    Expectation: output a sequence of numbers that begins at "start" and extlimits by increments of "delta" up to but
    not including "limit".
    """
    context.set_context(mode=mode)
    start = ms.Tensor([0])
    limit = ms.Tensor([10])
    delta = ms.Tensor([2])
    output = range_forward_func(start, limit, delta)
    print("output:", output)
    expect_output = np.array([0, 2, 4, 6, 8]).astype(np.int64)
    np.testing.assert_array_equal(output.asnumpy(), expect_output)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_range_forward(mode):
    """
    Feature: range ops.
    Description: test ops range.
    Expectation: output a sequence of numbers that begins at "start" and extlimits by increments of "delta" up to but
    not including "limit".
    """
    context.set_context(mode=mode)
    start = 0
    limit = 10
    delta = 2
    output = range_forward_func(start, limit, delta)
    print("output:", output)
    expect_output = np.array([0, 2, 4, 6, 8]).astype(np.int64)
    np.testing.assert_array_equal(output.asnumpy(), expect_output)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_range_backward(mode):
    """
    Feature: range ops.
    Description: test auto grad of ops range.
    Expectation: output the right grad.
    """
    context.set_context(mode=mode)
    start = 0
    limit = 10
    delta = 2
    output = range_backward_func(start, limit, delta)
    print("output:", output)
