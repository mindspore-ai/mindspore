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
def real_div_forward_func(x, y):
    return ops.auto_generate.real_div(x, y)


@test_utils.run_with_cell
def real_div_backward_func(x, y):
    return ops.grad(real_div_forward_func, (0, 1))(x, y)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_real_div_forward(mode):
    """
    Feature: real_div ops.
    Description: test ops real_div.
    Expectation: output the right result.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float32))
    y = Tensor(np.array([4.0, 5.0, 6.0]).astype(np.float32))
    output = real_div_forward_func(x, y)
    expect_output = np.asarray([0.25, 0.4, 0.5]).astype(np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect_output)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_real_div_backward(mode):
    """
    Feature: real_div ops.
    Description: test auto grad of ops real_div.
    Expectation: output the right grad.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float32))
    y = Tensor(np.array([4.0, 5.0, 6.0]).astype(np.float32))
    dx, dy = real_div_backward_func(x, y)
    except_dx = np.asarray([0.25, 0.2, 0.1666667]).astype(np.float32)
    except_dy = np.asarray([-0.0625, -0.08, -0.0833333]).astype(np.float32)
    np.testing.assert_array_almost_equal(dx.asnumpy(), except_dx, decimal=4)
    np.testing.assert_array_almost_equal(dy.asnumpy(), except_dy, decimal=4)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_real_div_vmap(mode):
    """
    Feature: test vmap function.
    Description: test real_div ops vmap.
    Expectation: expect right result.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[[1.0, 2.0, 3.0]]]).astype(np.float32))
    y = Tensor(np.array([[[4.0, 5.0, 6.0]]]).astype(np.float32))
    nest_vmap = ops.vmap(ops.vmap(real_div_forward_func, in_axes=(0, 0)), in_axes=(0, 0))
    output = nest_vmap(x, y)
    print("output:", output)
    expect_out = real_div_forward_func(x, y)
    np.testing.assert_equal(output.asnumpy(), expect_out.asnumpy())
