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
def sinh_forward_func(x):
    return ops.auto_generate.sinh(x)


@test_utils.run_with_cell
def sinh_backward_func(x):
    return ops.grad(sinh_forward_func, (0))(x)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sinh_forward(mode):
    """
    Feature: sinh ops.
    Description: test ops sinh.
    Expectation: output the right result.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([0.62, 0.28, 0.43, 0.62]).astype(np.float32))
    output = sinh_forward_func(x)
    expect_output = np.asarray([0.6604918, 0.28367308, 0.44337422, 0.6604918]).astype(np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expect_output, decimal=4)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sinh_backward(mode):
    """
    Feature: sinh ops.
    Description: test auto grad of ops sinh.
    Expectation: output the right grad.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([0.62, 0.28, 0.43, 0.62]).astype(np.float32))
    output = sinh_backward_func(x)
    expect_output = np.asarray([1.1984363, 1.0394568, 1.0938833, 1.1984363]).astype(np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expect_output, decimal=4)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sinh_vmap(mode):
    """
    Feature: test vmap function.
    Description: test sinh ops vmap.
    Expectation: expect right result.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[[0.62, 0.28, 0.43, 0.62]]]).astype(np.float32))
    nest_vmap = ops.vmap(ops.vmap(sinh_forward_func))
    output = nest_vmap(x)
    print("output:", output)
    expect_out = sinh_forward_func(x)
    np.testing.assert_equal(output.asnumpy(), expect_out.asnumpy())
