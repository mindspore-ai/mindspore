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
# pylint: disable=missing-module-docstring
import os
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops, context
from mindspore.ops import masked_fill

from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


def np_masked_fill_forward_func(input_x, mask, value):
    input_x = np.ma.array(input_x, mask=mask, fill_value=value)
    return input_x.filled()


@test_utils.run_with_cell
def masked_fill_forward_func(input_x, mask, value):
    return masked_fill(input_x, mask, value)


@test_utils.run_with_cell
def masked_fill_backward_func(input_x, mask, value):
    # pylint: disable=E1102
    return ops.grad(masked_fill_forward_func, 0)(input_x, mask, value)


@test_utils.run_with_cell
def masked_fill_backward_with_sens(input_x, mask, value, sens):
    return ms.grad(masked_fill_forward_func, (0,), sens_param=True)(input_x, mask, value, sens)


@test_utils.run_with_cell
def masked_fill_value_grad_with_sens(input_x, mask, value, sens):
    return ms.grad(masked_fill_forward_func, (2,), sens_param=True)(input_x, mask, value, sens)


@test_utils.run_with_cell
def masked_fill_vmap_func(input_x, mask, value):
    return ops.vmap(masked_fill_forward_func)(input_x, mask, value)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', mem_peak=1024.25)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_masked_fill_normal(context_mode):
    """
    Feature: pyboost function.
    Description: test function masked_fill forward and backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    input_shape = (77, 77)
    input_x = np.random.randn(*input_shape).astype(np.float32)
    mask_arry = np.random.choice([True, False], size=[77, 77])
    mask = ms.Tensor(mask_arry, dtype=ms.bool_)
    output = masked_fill_forward_func(ms.Tensor(input_x), mask, 0.0)
    expect_output = np_masked_fill_forward_func(input_x, mask_arry, 0.0)
    np.testing.assert_allclose(output.asnumpy(), expect_output, rtol=1e-3)

    input_x1 = ms.Tensor(np.array([1., 2., 3., 4.]).astype(np.float32))
    mask = ms.Tensor(np.array([True, True, False, True]).astype(np.bool_))
    input_x_output = masked_fill_backward_func(input_x1, mask, 0.5)
    expect_input_x_output = np.asarray([0., 0., 1., 0.]).astype(np.float32)
    np.testing.assert_allclose(input_x_output.asnumpy(), expect_input_x_output, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend910b', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1', card_mark='onecard', mem_peak=0.02)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_masked_fill_backward_nan_inf(context_mode):
    """
    Feature: pyboost function.
    Description: test masked_fill backward with nan/inf sens.
    Expectation: masked positions are zeroed without propagating nan/inf.
    """
    ms.context.set_context(mode=context_mode)
    input_x = ms.Tensor(np.array([1., 2., 3.]).astype(np.float32))
    mask = ms.Tensor(np.array([True, False, True]).astype(np.bool_))
    value = ms.Tensor(0.0, ms.float32)
    sens = ms.Tensor(np.array([np.nan, np.inf, -1.]).astype(np.float32))
    input_grad = masked_fill_backward_with_sens(input_x, mask, value, sens)
    expect_input_grad = np.asarray([0., np.inf, 0.]).astype(np.float32)
    np.testing.assert_allclose(input_grad.asnumpy(), expect_input_grad, rtol=1e-3, atol=1e-6)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', mem_peak=0.02)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_masked_fill_value_grad(context_mode):
    """
    Feature: pyboost function.
    Description: test masked_fill value grad with custom sens.
    Expectation: value grad equals sum of grad at masked positions.
    """
    ms.context.set_context(mode=context_mode)
    input_x = ms.Tensor(np.array([[1., 2.], [3., 4.]]).astype(np.float32))
    mask = ms.Tensor(np.array([[True, False], [False, True]]).astype(np.bool_))
    value = ms.Tensor(0.5, ms.float32)
    sens = ms.Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]).astype(np.float32))
    value_grad = masked_fill_value_grad_with_sens(input_x, mask, value, sens)
    expect_value_grad = np.asarray(1.0 + 4.0).astype(np.float32)
    np.testing.assert_allclose(value_grad.asnumpy(), expect_value_grad, rtol=1e-3, atol=1e-6)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', mem_peak=0.02)
def test_ops_masked_fill_value_grad_ge_o2():
    """
    Feature: pyboost function.
    Description: test masked_fill value grad under GE O2 mode.
    Expectation: run without SumExt adapter error.
    """
    context.set_context(mode=ms.GRAPH_MODE, jit_level='O2', device_target="Ascend")
    input_x = ms.Tensor(np.array([[1., 2.], [3., 4.]]).astype(np.float32))
    mask = ms.Tensor(np.array([[True, False], [False, True]]).astype(np.bool_))
    value = ms.Tensor(0.5, ms.float32)
    sens = ms.Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]).astype(np.float32))
    value_grad = masked_fill_value_grad_with_sens(input_x, mask, value, sens)
    expect_value_grad = np.asarray(1.0 + 4.0).astype(np.float32)
    np.testing.assert_allclose(value_grad.asnumpy(), expect_value_grad, rtol=1e-3, atol=1e-6)


@arg_mark(plat_marks=['platform_ascend910b', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1', card_mark='onecard', mem_peak=0.02)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_masked_fill_value_grad_nan_inf(context_mode):
    """
    Feature: pyboost function.
    Description: test masked_fill value grad with nan/inf sens.
    Expectation: nan/inf on masked-out positions are excluded.
    """
    ms.context.set_context(mode=context_mode)
    input_x = ms.Tensor(np.array([1., 2., 3., 4.]).astype(np.float32))
    mask = ms.Tensor(np.array([False, True, True, False]).astype(np.bool_))
    value = ms.Tensor(0.5, ms.float32)
    sens = ms.Tensor(np.array([np.nan, 2.0, np.inf, -3.0]).astype(np.float32))
    value_grad = masked_fill_value_grad_with_sens(input_x, mask, value, sens)
    assert np.isinf(value_grad.asnumpy())


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard')
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


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', mem_peak=0.02)
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


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', mem_peak=0.02)
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


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', mem_peak=2048.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_masked_fill_backward_dynamic_shape(context_mode):
    """
    Feature: pyboost function.
    Description: test function masked_fill backward with dynamic shape.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    context.set_context(jit_level='O0')
    if context_mode == ms.GRAPH_MODE and ms.context.get_context("device_target") == "Asecnd":
        os.environ["MS_DISABLE_KERNEL_BACKOFF"] = "0"
    input_x_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    mask_dyn = ms.Tensor(shape=[None, None], dtype=ms.bool_)
    test_cell = test_utils.to_cell_obj(masked_fill_backward_func)
    test_cell.set_inputs(input_x_dyn, mask_dyn, 0.5)

    input_x1 = ms.Tensor(np.array([[1.]]).astype(np.float32))
    mask1 = ms.Tensor(np.array([[False]]).astype(np.bool_))
    input_x_output1 = test_cell(input_x1, mask1, 0.5)
    expect_input_x_output1 = np.asarray([[1.]]).astype(np.float32)
    np.testing.assert_allclose(input_x_output1.asnumpy(), expect_input_x_output1, rtol=1e-3)

    input_x2 = ms.Tensor(np.array([[1, 2],
                                   [3, 4]]).astype(np.float32))
    mask2 = ms.Tensor(np.array([[True, True],
                                [False, True]]).astype(np.bool_))
    input_x_output2 = test_cell(input_x2, mask2, 0.5)
    expect_input_x_output2 = np.asarray([[0., 0.], [1., 0.]]).astype(np.float32)
    np.testing.assert_allclose(input_x_output2.asnumpy(), expect_input_x_output2, rtol=1e-3)
    if context_mode == ms.GRAPH_MODE and ms.context.get_context("device_target") == "Asecnd":
        del os.environ["MS_DISABLE_KERNEL_BACKOFF"]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', mem_peak=2048.00)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_masked_fill_backward_dynamic_rank(context_mode):
    """
    Feature: pyboost function.
    Description: test function masked_fill backward with dynamic rank.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    context.set_context(jit_level='O0')
    if context_mode == ms.GRAPH_MODE and ms.context.get_context("device_target") == "Asecnd":
        os.environ["MS_DISABLE_KERNEL_BACKOFF"] = "0"
    input_x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    mask_dyn = ms.Tensor(shape=None, dtype=ms.bool_)
    test_cell = test_utils.to_cell_obj(masked_fill_backward_func)
    test_cell.set_inputs(input_x_dyn, mask_dyn, 0.5)

    input_x1 = ms.Tensor(np.array([1., 2., 3., 4.]).astype(np.float32))
    mask1 = ms.Tensor(np.array([True, True, False, True]).astype(np.bool_))
    input_x_output1 = test_cell(input_x1, mask1, 0.5)
    expect_input_x_output1 = np.asarray([0., 0., 1., 0.]).astype(np.float32)
    np.testing.assert_allclose(input_x_output1.asnumpy(), expect_input_x_output1, rtol=1e-3)

    input_x2 = ms.Tensor(np.array([[1, 2],
                                   [3, 4]]).astype(np.float32))
    mask2 = ms.Tensor(np.array([[True, True],
                                [False, True]]).astype(np.bool_))
    input_x_output2 = test_cell(input_x2, mask2, 0.5)
    expect_input_x_output2 = np.asarray([[0., 0.], [1., 0.]]).astype(np.float32)
    np.testing.assert_allclose(input_x_output2.asnumpy(), expect_input_x_output2, rtol=1e-3)
    if context_mode == ms.GRAPH_MODE and ms.context.get_context("device_target") == "Asecnd":
        del os.environ["MS_DISABLE_KERNEL_BACKOFF"]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', mem_peak=1024.25)
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
