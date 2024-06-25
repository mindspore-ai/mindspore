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
from mindspore import Tensor
import mindspore as ms
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def cumsum_forward_func(x, axis, exclusive, reverse):
    return ops.CumSum(exclusive, reverse)(x, axis)


@test_utils.run_with_cell
def cumsum_backward_func(x, axis, exclusive, reverse):
    return ops.grad(cumsum_forward_func, (0,))(x, axis, exclusive, reverse)


@test_utils.run_with_cell
def cumsum_dyn_shape_func(x, axis, exclusive, reverse):
    return ops.CumSum(exclusive, reverse)(x, axis)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("dtype", [np.int8, np.int32, np.uint8,
                                   np.float64, np.float32, np.float16])
def test_cumsum_forward(context_mode, dtype):
    """
    Feature: Ops.
    Description: test op cumsum.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    np_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).astype(dtype)
    x = Tensor(np_array)
    axis = 1
    exclusive = False
    reverse = False
    out = cumsum_forward_func(x, axis, exclusive, reverse)
    expect = np.cumsum(np_array, axis)
    assert np.allclose(out.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("dtype", [np.int8, np.int32, np.uint8,
                                   np.float64, np.float32, np.float16])
def test_cumsum_backward(context_mode, dtype):
    """
    Feature: Auto grad.
    Description: test auto grad of op cumsum.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    np_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).astype(dtype)
    x = Tensor(np_array)
    axis = 1
    exclusive = False
    reverse = False
    out = cumsum_backward_func(x, axis, exclusive, reverse)
    expect = np.array([[4, 3, 2, 1],
                       [4, 3, 2, 1]]).astype(dtype)
    assert np.allclose(out.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_cumsum_dynamic(context_mode):
    """
    Feature: test dynamic shape feature of cumsum.
    Description: test dynamic shape feature of cumsum.
    Expectation: expect correct result.
    """

    ms.context.set_context(mode=context_mode)
    x_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    axis = 1
    exclusive = False
    reverse = False
    test_cell = test_utils.to_cell_obj(cumsum_dyn_shape_func)
    test_cell.set_inputs(x_dyn, axis, exclusive, reverse)

    x1 = ms.Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), ms.float32)
    out1 = test_cell(x1, axis, exclusive, reverse)
    expect = np.array([[1, 3, 6, 10], [5, 11, 18, 26]]).astype('float32')
    assert np.allclose(out1.asnumpy(), expect, rtol=1e-4, atol=1e-4)

    x2 = ms.Tensor(np.array([[2, 3, 4], [6, 7, 8]]), ms.float32)
    out2 = test_cell(x2, axis, exclusive, reverse)
    expect = np.array([[2, 5, 9], [6, 13, 21]]).astype('float32')
    assert np.allclose(out2.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_cumsum_dynamic_rank(context_mode):
    """
    Feature: test dynamic rank feature of cumsum.
    Description: test dynamic rank feature of cumsum.
    Expectation: expect correct result.
    """

    ms.context.set_context(mode=context_mode)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    axis = 1
    exclusive = False
    reverse = False
    test_cell = test_utils.to_cell_obj(cumsum_dyn_shape_func)
    test_cell.set_inputs(x_dyn, axis, exclusive, reverse)

    x1 = ms.Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), ms.float32)
    out1 = test_cell(x1, axis, exclusive, reverse)
    expect = np.array([[1, 3, 6, 10], [5, 11, 18, 26]]).astype('float32')
    assert np.allclose(out1.asnumpy(), expect, rtol=1e-4, atol=1e-4)

    x2 = ms.Tensor(np.array([[[2, 3, 4], [6, 7, 8]]]), ms.float32)
    out2 = test_cell(x2, axis, exclusive, reverse)
    expect = np.array([[[2, 3, 4], [8, 10, 12]]]).astype('float32')
    assert np.allclose(out2.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_cumsum_backward_dynamic(context_mode):
    """
    Feature: test barkward dynamic rank feature of cumsum.
    Description: test barkward dynamic rank feature of cumsum.
    Expectation: expect correct result.
    """

    ms.context.set_context(mode=context_mode)
    x_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    axis = 1
    exclusive = False
    reverse = False
    test_cell = test_utils.to_cell_obj(cumsum_backward_func)
    test_cell.set_inputs(x_dyn, axis, exclusive, reverse)

    x1 = ms.Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), ms.float32)
    out1 = test_cell(x1, axis, exclusive, reverse)
    expect = np.array([[4, 3, 2, 1], [4, 3, 2, 1]]).astype('float32')
    assert np.allclose(out1.asnumpy(), expect, rtol=1e-4, atol=1e-4)

    x2 = ms.Tensor(np.array([[2, 3, 4], [6, 7, 8]]), ms.float32)
    out2 = test_cell(x2, axis, exclusive, reverse)
    expect = np.array([[3, 2, 1], [3, 2, 1]]).astype('float32')
    assert np.allclose(out2.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_cumsum_backward_dynamic_rank(context_mode):
    """
    Feature: test barkward dynamic rank feature of cumsum.
    Description: test barkward dynamic rank feature of cumsum.
    Expectation: expect correct result.
    """

    ms.context.set_context(mode=context_mode)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    axis = 1
    exclusive = False
    reverse = False
    test_cell = test_utils.to_cell_obj(cumsum_backward_func)
    test_cell.set_inputs(x_dyn, axis, exclusive, reverse)

    x1 = ms.Tensor(np.array([[11, 1, 33, 4], [51, 16, 7, 98]]), ms.float32)
    out1 = test_cell(x1, axis, exclusive, reverse)
    expect = np.array([[4, 3, 2, 1], [4, 3, 2, 1]]).astype('float32')
    assert np.allclose(out1.asnumpy(), expect, rtol=1e-4, atol=1e-4)

    x2 = ms.Tensor(np.array([[[2, 13, 4], [6, 7, 128]]]), ms.float32)
    out2 = test_cell(x2, axis, exclusive, reverse)
    expect = np.array([[[2, 2, 2], [1, 1, 1]]]).astype('float32')
    assert np.allclose(out2.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@pytest.mark.skip(reason="Probabilistic failure on CI.")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("dtype", [np.int8, np.int32, np.uint8,
                                   np.float64, np.float32, np.float16])
def test_cumsum_vmap(context_mode, dtype):
    """
    Feature: Vmap.
    Description: test vmap of op cumsum.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    np_array = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]]]).astype(dtype)
    x = Tensor(np_array)
    axis = -1
    exclusive = False
    reverse = False
    nest_vmap = ops.vmap(ops.vmap(cumsum_forward_func, in_axes=(0, None, None, None)), in_axes=(0, None, None, None))
    out = nest_vmap(x, axis, exclusive, reverse)
    expect_out = np.array([[[1, 3, 6, 10], [5, 11, 18, 26]]]).astype(dtype)
    assert (out.asnumpy() == expect_out).all()
