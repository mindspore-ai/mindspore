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
# pylint: disable=unused-variable
import pytest
import numpy as np
import mindspore as ms
from mindspore import context
from mindspore import ops
from mindspore.common import dtype as mstype
import test_utils


@test_utils.run_with_cell
def randperm_v2_forward_func(n):
    return ops.auto_generate.randperm(n, seed=0, offset=0, dtype=mstype.float16)


@test_utils.run_with_cell
def randperm_v2_backward_func(n):
    return ops.grad(randperm_v2_forward_func, (0))(n)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
@test_utils.run_test_func
def test_randperm_v2_forward(mode):
    """
    Feature: randperm_v2 ops.
    Description: test ops randperm_v2.
    Expectation: generates random permutation of integers from 0 to n-1 without repeating.
    """
    context.set_context(mode=mode)
    output = randperm_v2_forward_func(4)
    np.testing.assert_equal(output.shape, (4,))
    np.testing.assert_equal(output.dtype, mstype.float16)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
@test_utils.run_test_func
def test_randperm_v2_backward(mode):
    """
    Feature: randperm_v2 ops.
    Description: test auto grad of ops randperm_v2.
    Expectation: output the right grad.
    """
    context.set_context(mode=mode)
    output = randperm_v2_backward_func(4)
