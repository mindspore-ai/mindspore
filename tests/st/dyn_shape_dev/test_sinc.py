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
import test_utils
import mindspore as ms
from mindspore import Tensor, context
from mindspore import ops


@test_utils.run_with_cell
def sinc_forward_func(x):
    return ops.auto_generate.sinc(x)


@test_utils.run_with_cell
def sinc_backward_func(x):
    return ops.grad(sinc_forward_func, (0))(x)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sinc_forward(mode):
    """
    Feature: sinc ops.
    Description: test ops sinc.
    Expectation: output the right result.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([0.62, 0.28, 0.43, 0.62]).astype(np.float32))
    output = sinc_forward_func(x)
    expect_output = np.asarray([0.47735003, 0.8759357, 0.7224278, 0.47735003]).astype(np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expect_output, decimal=4)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sinc_backward(mode):
    """
    Feature: sinc ops.
    Description: test auto grad of ops sinc.
    Expectation: output the right grad.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([0.62, 0.28, 0.43, 0.62]).astype(np.float32))
    output = sinc_backward_func(x)
    print("output:\n", output)
    expect_output = np.asarray([-1.3636689, -0.85182726, -1.1727549, -1.3636689]).astype(np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expect_output, decimal=4)
