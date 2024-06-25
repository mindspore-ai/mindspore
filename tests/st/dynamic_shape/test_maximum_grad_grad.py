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

import numpy as np
import pytest
from tests.st.utils import test_utils

from mindspore import ops
import mindspore as ms
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def maximum_grad_grad_forward_func(x1, x2, dx, dy):
    return ops.auto_generate.MaximumGradGrad(grad_x=True, grad_y=True)(x1, x2, dx, dy)


@test_utils.run_with_cell
def maximum_grad_grad_vmap_func(x1, x2, dx, dy):
    return ops.vmap(maximum_grad_grad_forward_func, in_axes=0, out_axes=0)(x1, x2, dx, dy)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
def test_maximum_grad_grad_op_forward(context_mode, data_type):
    """
    Feature: Ops.
    Description: test op maximum_grad_grad forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = ms.Tensor(np.array([1, 2, 4]).astype(data_type))
    y = ms.Tensor(np.array([2, 4, 3]).astype(data_type))
    dx = ms.Tensor(np.array([0., 0., 3.]).astype(data_type))
    dy = ms.Tensor(np.array([1., 2., 0.]).astype(data_type))
    out = maximum_grad_grad_forward_func(x, y, dx, dy)
    expect_out = np.array([[0., 0., 0.], [0., 0., 0.], [1., 2., 3.]]).astype(np.float32)
    np.testing.assert_allclose(out[0].asnumpy(), expect_out[0], rtol=1e-3)
    np.testing.assert_allclose(out[1].asnumpy(), expect_out[1], rtol=1e-3)
    np.testing.assert_allclose(out[2].asnumpy(), expect_out[2], rtol=1e-3)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
def test_maximum_grad_grad_op_vmap(context_mode, data_type):
    """
    Feature: test vmap function.
    Description: test maximum_grad_grad op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = ms.Tensor(np.array([[3, 5, 11], [7, 5, 6]]).astype(data_type))
    y = ms.Tensor(np.array([[6, 6, 8], [3, 2, 7]]).astype(data_type))
    dx = ms.Tensor(np.array([[0., 0., 3.], [1., 2., 0.]]).astype(data_type))
    dy = ms.Tensor(np.array([[1., 2., 0.], [0., 0., 3.]]).astype(data_type))
    out = maximum_grad_grad_vmap_func(x, y, dx, dy)
    expect_out = np.array([[[0., 0., 0.], [0., 0., 0.]],
                           [[0., 0., 0.], [0., 0., 0.]],
                           [[1., 2., 3.], [1., 2., 3.]]]).astype(np.float32)
    np.testing.assert_allclose(out[0].asnumpy(), expect_out[0], rtol=1e-3)
    np.testing.assert_allclose(out[1].asnumpy(), expect_out[1], rtol=1e-3)
    np.testing.assert_allclose(out[2].asnumpy(), expect_out[2], rtol=1e-3)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_maximum_grad_grad_op_dynamic_shape(context_mode):
    """
    Feature: maximum_grad_grad ops.
    Description: test ops maximum_grad_grad dynamic tensor input.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x_dyn = ms.Tensor(shape=[None], dtype=ms.float32)
    y_dyn = ms.Tensor(shape=[None], dtype=ms.float32)
    dx_dyn = ms.Tensor(shape=[None], dtype=ms.float32)
    dy_dyn = ms.Tensor(shape=[None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(maximum_grad_grad_forward_func)
    test_cell.set_inputs(x_dyn, y_dyn, dx_dyn, dy_dyn)
    x = ms.Tensor(np.array([1, 2, 4]).astype(np.float32))
    y = ms.Tensor(np.array([2, 4, 3]).astype(np.float32))
    dx = ms.Tensor(np.array([0., 0., 3.]).astype(np.float32))
    dy = ms.Tensor(np.array([1., 2., 0.]).astype(np.float32))
    out = test_cell(x, y, dx, dy)
    expect_out = np.array([[0., 0., 0.], [0., 0., 0.], [1., 2., 3.]]).astype(np.float32)
    np.testing.assert_allclose(out[0].asnumpy(), expect_out[0], rtol=1e-3)
    np.testing.assert_allclose(out[1].asnumpy(), expect_out[1], rtol=1e-3)
    np.testing.assert_allclose(out[2].asnumpy(), expect_out[2], rtol=1e-3)

    x1 = ms.Tensor(np.array([3, 6, 2, 7]).astype(np.float32))
    y1 = ms.Tensor(np.array([7, 5, 9, 2]).astype(np.float32))
    dx1 = ms.Tensor(np.array([0., 5., 0., 2.]).astype(np.float32))
    dy1 = ms.Tensor(np.array([3., 0., 2., 0.]).astype(np.float32))
    out1 = test_cell(x1, y1, dx1, dy1)
    expect_out1 = np.array([[0., 0., 0., 0.], [0., 0., 0., 0.], [3., 5., 2., 2.]]).astype(np.float32)
    np.testing.assert_allclose(out1[0].asnumpy(), expect_out1[0], rtol=1e-3)
    np.testing.assert_allclose(out1[1].asnumpy(), expect_out1[1], rtol=1e-3)
    np.testing.assert_allclose(out1[2].asnumpy(), expect_out1[2], rtol=1e-3)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_maximum_grad_grad_op_dynamic_rank(context_mode):
    """
    Feature: maximum_grad_grad ops.
    Description: test ops maximum_grad_grad dynamic tensor input.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    y_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    dx_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    dy_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(maximum_grad_grad_forward_func)
    test_cell.set_inputs(x_dyn, y_dyn, dx_dyn, dy_dyn)
    x = ms.Tensor(np.array([1, 2, 4]).astype(np.float32))
    y = ms.Tensor(np.array([2, 4, 3]).astype(np.float32))
    dx = ms.Tensor(np.array([0., 0., 3.]).astype(np.float32))
    dy = ms.Tensor(np.array([1., 2., 0.]).astype(np.float32))
    out = test_cell(x, y, dx, dy)
    expect_out = np.array([[0., 0., 0.], [0., 0., 0.], [1., 2., 3.]]).astype(np.float32)
    np.testing.assert_allclose(out[0].asnumpy(), expect_out[0], rtol=1e-3)
    np.testing.assert_allclose(out[1].asnumpy(), expect_out[1], rtol=1e-3)
    np.testing.assert_allclose(out[2].asnumpy(), expect_out[2], rtol=1e-3)

    x1 = ms.Tensor(np.array([3, 6, 2, 7]).astype(np.float32))
    y1 = ms.Tensor(np.array([7, 5, 9, 2]).astype(np.float32))
    dx1 = ms.Tensor(np.array([0., 5., 0., 2.]).astype(np.float32))
    dy1 = ms.Tensor(np.array([3., 0., 2., 0.]).astype(np.float32))
    out1 = test_cell(x1, y1, dx1, dy1)
    expect_out1 = np.array([[0., 0., 0., 0.], [0., 0., 0., 0.], [3., 5., 2., 2.]]).astype(np.float32)
    np.testing.assert_allclose(out1[0].asnumpy(), expect_out1[0], rtol=1e-3)
    np.testing.assert_allclose(out1[1].asnumpy(), expect_out1[1], rtol=1e-3)
    np.testing.assert_allclose(out1[2].asnumpy(), expect_out1[2], rtol=1e-3)
