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
def real_forward_func(x):
    return ops.auto_generate.real(x)


@test_utils.run_with_cell
def real_backward_func(x):
    return ops.grad(real_forward_func, (0))(x)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
@test_utils.run_test_func
def test_real_forward(mode):
    """
    Feature: real ops.
    Description: test ops real.
    Expectation: output the real part of the input.
    """
    context.set_context(mode=mode)
    x = Tensor(np.asarray(np.complex(1.3 + 0.4j)).astype(np.complex64))
    output = real_forward_func(x)
    expect_output = np.asarray(np.complex(1.3)).astype(np.float32)
    np.testing.assert_equal(output.asnumpy(), expect_output)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
# @pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
@test_utils.run_test_func
def test_real_backward(mode):
    """
    Feature: real ops.
    Description: test auto grad of ops real.
    Expectation: output the right grad.
    """
    context.set_context(mode=mode)
    x = Tensor(np.asarray(np.complex(1.3 + 0.4j)).astype(np.complex64))
    output = real_backward_func(x)
    expect_output = np.asarray(np.complex(1. + 0j)).astype(np.complex64)
    np.testing.assert_equal(output.asnumpy(), expect_output)
