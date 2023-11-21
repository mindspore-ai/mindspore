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
def reciprocal_forward_func(x):
    return ops.auto_generate.reciprocal_(x)


@test_utils.run_with_cell
def reciprocal_backward_func(x):
    return ops.grad(reciprocal_forward_func, (0))(x)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_func
def test_reciprocal_forward(mode):
    """
    Feature: reciprocal ops.
    Description: test ops reciprocal.
    Expectation: output the right result.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([1.0, 2.0, 4.0]).astype(np.float32))
    output = reciprocal_forward_func(x)
    expect_output = np.asarray([1., 0.5, 0.25]).astype(np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expect_output, decimal=4)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_func
def test_reciprocal_backward(mode):
    """
    Feature: reciprocal ops.
    Description: test auto grad of ops reciprocal.
    Expectation: output the right grad.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([1.0, 2.0, 4.0]).astype(np.float32))
    output = reciprocal_backward_func(x)
    expect_output = np.asarray([-1., -0.25, -0.0625]).astype(np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expect_output, decimal=4)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_func
def test_reciprocal_vmap(mode):
    """
    Feature: test vmap function.
    Description: test reciprocal ops vmap.
    Expectation: expect right result.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[[1.0, 2.0, 4.0]]]).astype(np.float32))
    nest_vmap = ops.vmap(ops.vmap(reciprocal_forward_func))
    output = nest_vmap(x)
    expect_out = reciprocal_forward_func(x)
    np.testing.assert_equal(output.asnumpy(), expect_out.asnumpy())


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
# @pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_func
def test_reciprocal_dynamic(mode):
    """
    Feature: reciprocal ops.
    Description: test ops reciprocal dynamic tensor input.
    Expectation: output the right result.
    """
    context.set_context(mode=mode)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(ops.auto_generate.reciprocal_)
    test_cell.set_inputs(x_dyn)
    x1 = Tensor(np.array([1.0, 2.0, 4.0]).astype(np.float32))
    output1 = test_cell(x1)
    expect_output1 = np.asarray([1., 0.5, 0.25]).astype(np.float32)
    np.testing.assert_array_almost_equal(output1.asnumpy(), expect_output1, decimal=4)
    x2 = Tensor(np.array([[1.0, 2.0],
                          [4.0, 5.0]]).astype(np.float32))
    output2 = test_cell(x2)
    expect_output2 = np.asarray([[1, 0.5],
                                 [0.25, 0.2]]).astype(np.float32)
    np.testing.assert_array_almost_equal(output2.asnumpy(), expect_output2, decimal=4)
