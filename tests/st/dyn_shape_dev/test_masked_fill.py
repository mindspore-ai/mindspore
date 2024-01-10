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
def masked_fill_forward_func(input_x, mask, value):
    return ops.auto_generate.masked_fill(input_x, mask, value)


@test_utils.run_with_cell
def masked_fill_backward_func(input_x, mask, value):
    return ops.grad(masked_fill_forward_func, (0, 1))(input_x, mask, value)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_func
def test_masked_fill_forward(mode):
    """
    Feature: masked_fill ops.
    Description: test ops masked_fill.
    Expectation: output the right result.
    """
    context.set_context(mode=mode)
    input_x = Tensor(np.array([1., 2., 3., 4.]).astype(np.float32))
    mask = Tensor(np.array([True, True, False, True]).astype(np.bool_))
    output = masked_fill_forward_func(input_x, mask, 0.5)
    expect_output = np.asarray([0.5, 0.5, 3., 0.5]).astype(np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expect_output, decimal=4)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_func
def test_masked_fill_backward(mode):
    """
    Feature: masked_fill ops.
    Description: test auto grad of ops masked_fill.
    Expectation: output the right grad.
    """
    context.set_context(mode=mode)
    input_x = Tensor(np.array([1., 2., 3., 4.]).astype(np.float32))
    mask = Tensor(np.array([True, True, False, True]).astype(np.bool_))
    input_x_output, mask_output = masked_fill_backward_func(input_x, mask, 0.5)
    expect_input_x_output = np.asarray([0., 0., 1., 0.]).astype(np.float32)
    np.testing.assert_array_almost_equal(input_x_output.asnumpy(), expect_input_x_output, decimal=4)
    expect_mask_output = np.asarray([0., 0., 0., 0.]).astype(np.float32)
    np.testing.assert_array_almost_equal(mask_output.asnumpy(), expect_mask_output, decimal=4)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_func
def test_masked_fill_vmap(mode):
    """
    Feature: test vmap function.
    Description: test masked_fill ops vmap.
    Expectation: expect right result.
    """
    context.set_context(mode=mode)
    input_x = Tensor(np.array([[[1., 2., 3., 4.]]]).astype(np.float32))
    mask = Tensor(np.array([[[True, True, False, True]]]).astype(np.bool_))
    value = Tensor(np.array([[0.5]]).astype(np.float32))
    nest_vmap = ops.vmap(ops.vmap(masked_fill_forward_func, in_axes=(0, 0, 0)), in_axes=(0, 0, 0))
    output = nest_vmap(input_x, mask, value)
    expect_out = masked_fill_forward_func(input_x, mask, 0.5)
    np.testing.assert_equal(output.asnumpy(), expect_out.asnumpy())


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_func
def test_masked_fill_dynamic(mode):
    """
    Feature: masked_fill ops.
    Description: test ops masked_fill dynamic tensor input.
    Expectation: output the right result.
    """
    context.set_context(mode=mode)
    input_x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    mask_dyn = ms.Tensor(shape=None, dtype=ms.bool_)
    test_cell = test_utils.to_cell_obj(ops.auto_generate.masked_fill)
    test_cell.set_inputs(input_x_dyn, mask_dyn, 0.5)
    input_x1 = Tensor(np.array([1., 2., 3., 4.]).astype(np.float32))
    mask1 = Tensor(np.array([True, True, False, True]).astype(np.bool_))
    output1 = test_cell(input_x1, mask1, 0.5)
    expect_output1 = np.asarray([0.5, 0.5, 3., 0.5]).astype(np.float32)
    np.testing.assert_array_almost_equal(output1.asnumpy(), expect_output1, decimal=4)
    input_x2 = Tensor(np.array([[1, 2],
                                [3, 4]]).astype(np.float32))
    mask2 = Tensor(np.array([[True, True],
                             [False, True]]).astype(np.bool_))
    output2 = test_cell(input_x2, mask2, 0.5)
    expect_output2 = np.asarray([[0.5, 0.5],
                                 [3., 0.5]]).astype(np.float32)
    np.testing.assert_array_almost_equal(output2.asnumpy(), expect_output2, decimal=4)
