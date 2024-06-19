# Copyright 2024 Huawei Technologies Co., Ltd
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
import os
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops
from mindspore.ops import masked_fill

from tests.st.utils import test_utils


@test_utils.run_with_cell
def masked_fill_forward_func(input_x, mask, value):
    return masked_fill(input_x, mask, value)


@test_utils.run_with_cell
def masked_fill_backward_func(input_x, mask, value):
    return ops.grad(masked_fill_forward_func, (0, 1))(input_x, mask, value)


@test_utils.run_with_cell
def masked_fill_vmap_func(input_x, mask, value):
    return ops.vmap(masked_fill_forward_func)(input_x, mask, value)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_masked_fill_forward(context_mode):
    """
    Feature: pyboost function.
    Description: test function masked_fill forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    input_x = ms.Tensor(np.array([1., 2., 3., 4.]).astype(np.float32))
    mask = ms.Tensor(np.array([True, True, False, True]).astype(np.bool_))
    output = masked_fill_forward_func(input_x, mask, 0.5)
    expect_output = np.asarray([0.5, 0.5, 3., 0.5]).astype(np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect_output, rtol=1e-3)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_masked_fill_backward(context_mode):
    """
    Feature: pyboost function.
    Description: test function masked_fill backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    input_x = ms.Tensor(np.array([1., 2., 3., 4.]).astype(np.float32))
    mask = ms.Tensor(np.array([True, True, False, True]).astype(np.bool_))
    input_x_output, mask_output = masked_fill_backward_func(input_x, mask, 0.5)
    expect_input_x_output = np.asarray([0., 0., 1., 0.]).astype(np.float32)
    np.testing.assert_allclose(input_x_output.asnumpy(), expect_input_x_output, rtol=1e-3)
    expect_mask_output = np.asarray([0., 0., 0., 0.]).astype(np.float32)
    np.testing.assert_allclose(mask_output.asnumpy(), expect_mask_output, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_masked_fill_vmap(context_mode):
    """
    Feature: pyboost function.
    Description: test function masked_fill vmap feature.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    input_x = ms.Tensor(np.array([[[1., 2., 3., 4.]]]).astype(np.float32))
    mask = ms.Tensor(np.array([[[True, True, False, True]]]).astype(np.bool_))
    value = ms.Tensor(np.array([[0.5]]).astype(np.float32))
    nest_vmap = ops.vmap(ops.vmap(masked_fill_forward_func, in_axes=(0, 0, 0)), in_axes=(0, 0, 0))
    output = nest_vmap(input_x, mask, value)
    expect_out = masked_fill_forward_func(input_x, mask, 0.5)
    np.testing.assert_allclose(output.asnumpy(), expect_out.asnumpy(), rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_masked_fill_forward_dynamic_shape(context_mode):
    """
    Feature: pyboost function.
    Description: test function masked_fill forward with dynamic shape.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    input_x_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    mask_dyn = ms.Tensor(shape=[None, None], dtype=ms.bool_)
    test_cell = test_utils.to_cell_obj(masked_fill_forward_func)
    test_cell.set_inputs(input_x_dyn, mask_dyn, 0.5)
    input_x1 = ms.Tensor(np.array([[1.]]).astype(np.float32))
    mask1 = ms.Tensor(np.array([[True]]).astype(np.bool_))
    output1 = test_cell(input_x1, mask1, 0.5)
    expect_output1 = np.asarray([[0.5]]).astype(np.float32)
    np.testing.assert_allclose(output1.asnumpy(), expect_output1, rtol=1e-3)
    input_x2 = ms.Tensor(np.array([[1, 2],
                                   [3, 4]]).astype(np.float32))
    mask2 = ms.Tensor(np.array([[True, True],
                                [False, True]]).astype(np.bool_))
    output2 = test_cell(input_x2, mask2, 0.5)
    expect_output2 = np.asarray([[0.5, 0.5],
                                 [3., 0.5]]).astype(np.float32)
    np.testing.assert_allclose(output2.asnumpy(), expect_output2, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_masked_fill_forward_dynamic_rank(context_mode):
    """
    Feature: pyboost function.
    Description: test function masked_fill forward with dynamic rank.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    input_x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    mask_dyn = ms.Tensor(shape=None, dtype=ms.bool_)
    test_cell = test_utils.to_cell_obj(masked_fill_forward_func)
    test_cell.set_inputs(input_x_dyn, mask_dyn, 0.5)
    input_x1 = ms.Tensor(np.array([1., 2., 3., 4.]).astype(np.float32))
    mask1 = ms.Tensor(np.array([True, True, False, True]).astype(np.bool_))
    output1 = test_cell(input_x1, mask1, 0.5)
    expect_output1 = np.asarray([0.5, 0.5, 3., 0.5]).astype(np.float32)
    np.testing.assert_allclose(output1.asnumpy(), expect_output1, rtol=1e-3)
    input_x2 = ms.Tensor(np.array([[1, 2],
                                   [3, 4]]).astype(np.float32))
    mask2 = ms.Tensor(np.array([[True, True],
                                [False, True]]).astype(np.bool_))
    output2 = test_cell(input_x2, mask2, 0.5)
    expect_output2 = np.asarray([[0.5, 0.5],
                                 [3., 0.5]]).astype(np.float32)
    np.testing.assert_allclose(output2.asnumpy(), expect_output2, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_masked_fill_backward_dynamic_shape(context_mode):
    """
    Feature: pyboost function.
    Description: test function masked_fill backward with dynamic shape.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    if context_mode == ms.GRAPH_MODE and ms.context.get_context("device_target") == "Asecnd":
        os.environ["MS_DISABLE_KERNEL_BACKOFF"] = "0"
    input_x_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    mask_dyn = ms.Tensor(shape=[None, None], dtype=ms.bool_)
    test_cell = test_utils.to_cell_obj(masked_fill_backward_func)
    test_cell.set_inputs(input_x_dyn, mask_dyn, 0.5)

    input_x1 = ms.Tensor(np.array([[1.]]).astype(np.float32))
    mask1 = ms.Tensor(np.array([[False]]).astype(np.bool_))
    input_x_output1, mask_output1 = test_cell(input_x1, mask1, 0.5)
    expect_input_x_output1 = np.asarray([[1.]]).astype(np.float32)
    np.testing.assert_allclose(input_x_output1.asnumpy(), expect_input_x_output1, rtol=1e-3)
    expect_mask_output1 = np.asarray([[0.]]).astype(np.float32)
    np.testing.assert_allclose(mask_output1.asnumpy(), expect_mask_output1, rtol=1e-3)

    input_x2 = ms.Tensor(np.array([[1, 2],
                                   [3, 4]]).astype(np.float32))
    mask2 = ms.Tensor(np.array([[True, True],
                                [False, True]]).astype(np.bool_))
    input_x_output2, mask_output2 = test_cell(input_x2, mask2, 0.5)
    expect_input_x_output2 = np.asarray([[0., 0.], [1., 0.]]).astype(np.float32)
    np.testing.assert_allclose(input_x_output2.asnumpy(), expect_input_x_output2, rtol=1e-3)
    expect_mask_output2 = np.asarray([[0., 0.], [0., 0.]]).astype(np.float32)
    np.testing.assert_allclose(mask_output2.asnumpy(), expect_mask_output2, rtol=1e-3)
    if context_mode == ms.GRAPH_MODE and ms.context.get_context("device_target") == "Asecnd":
        del os.environ["MS_DISABLE_KERNEL_BACKOFF"]


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_masked_fill_backward_dynamic_rank(context_mode):
    """
    Feature: pyboost function.
    Description: test function masked_fill backward with dynamic rank.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    if context_mode == ms.GRAPH_MODE and ms.context.get_context("device_target") == "Asecnd":
        os.environ["MS_DISABLE_KERNEL_BACKOFF"] = "0"
    input_x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    mask_dyn = ms.Tensor(shape=None, dtype=ms.bool_)
    test_cell = test_utils.to_cell_obj(masked_fill_backward_func)
    test_cell.set_inputs(input_x_dyn, mask_dyn, 0.5)

    input_x1 = ms.Tensor(np.array([1., 2., 3., 4.]).astype(np.float32))
    mask1 = ms.Tensor(np.array([True, True, False, True]).astype(np.bool_))
    input_x_output1, mask_output1 = test_cell(input_x1, mask1, 0.5)
    expect_input_x_output1 = np.asarray([0., 0., 1., 0.]).astype(np.float32)
    np.testing.assert_allclose(input_x_output1.asnumpy(), expect_input_x_output1, rtol=1e-3)
    expect_mask_output1 = np.asarray([0., 0., 0., 0.]).astype(np.float32)
    np.testing.assert_allclose(mask_output1.asnumpy(), expect_mask_output1, rtol=1e-3)

    input_x2 = ms.Tensor(np.array([[1, 2],
                                   [3, 4]]).astype(np.float32))
    mask2 = ms.Tensor(np.array([[True, True],
                                [False, True]]).astype(np.bool_))
    input_x_output2, mask_output2 = test_cell(input_x2, mask2, 0.5)
    expect_input_x_output2 = np.asarray([[0., 0.], [1., 0.]]).astype(np.float32)
    np.testing.assert_allclose(input_x_output2.asnumpy(), expect_input_x_output2, rtol=1e-3)
    expect_mask_output2 = np.asarray([[0., 0.], [0., 0.]]).astype(np.float32)
    np.testing.assert_allclose(mask_output2.asnumpy(), expect_mask_output2, rtol=1e-3)
    if context_mode == ms.GRAPH_MODE and ms.context.get_context("device_target") == "Asecnd":
        del os.environ["MS_DISABLE_KERNEL_BACKOFF"]


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('param_jit_level', ["O2", "O0"])
def test_ops_masked_fill_forward_with_broadcast(context_mode, param_jit_level):
    """
    Feature: pyboost function.
    Description: test function masked_fill forward.
    Expectation: expect correct result.
    """
    ms.set_context(jit_level=param_jit_level)
    ms.context.set_context(mode=context_mode)
    input_x = ms.Tensor(np.array([[1., 2.]]).astype(np.float32))
    mask = ms.Tensor(np.array([[False], [True]]).astype(np.bool_))
    output = masked_fill_forward_func(input_x, mask, 0.5)
    expect_output = np.asarray([[1, 2], [0.5, 0.5]]).astype(np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect_output, rtol=1e-3)
