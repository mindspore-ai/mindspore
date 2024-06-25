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
def cumprod_forward_func(x, axis, exclusive, reverse):
    return ops.CumProd(exclusive, reverse)(x, axis)


@test_utils.run_with_cell
def cumprod_backward_func(x, axis, exclusive, reverse):
    return ops.grad(cumprod_forward_func, (0,))(x, axis, exclusive, reverse)


@test_utils.run_with_cell
def cumprod_dyn_shape_func(x, axis, exclusive, reverse):
    return ops.CumProd(exclusive, reverse)(x, axis)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16,
                                   np.uint32, np.uint64, np.complex64, np.complex128,
                                   np.float64, np.float32, np.float16])
def test_cumprod_forward(context_mode, dtype):
    """
    Feature: Ops.
    Description: test op cumprod.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    np_array = np.array([[1, 2, 3, 4], [5, 1, 7, 2]]).astype(dtype)
    x = Tensor(np_array)
    axis = 1
    exclusive = False
    reverse = False
    out = cumprod_forward_func(x, axis, exclusive, reverse)
    expect = np.cumprod(np_array, axis)
    assert np.allclose(out.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.float16])
def test_cumprod_backward(context_mode, dtype):
    """
    Feature: Auto grad.
    Description: test auto grad of op cumprod.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    np_array = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).astype(dtype)
    x = Tensor(np_array)
    axis = 1
    exclusive = False
    reverse = False
    out = cumprod_backward_func(x, axis, exclusive, reverse)
    expect = np.array([[33, 16, 10, 6],
                       [385, 320, 270, 210]]).astype(dtype)
    assert np.allclose(out.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_cumprod_dynamic(context_mode):
    """
    Feature: test dynamic shape feature cumprod.
    Description: test dynamic shape feature cumprod.
    Expectation: expect correct result.
    """

    ms.context.set_context(mode=context_mode)
    x_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    axis = 1
    exclusive = False
    reverse = False
    test_cell = test_utils.to_cell_obj(cumprod_dyn_shape_func)
    test_cell.set_inputs(x_dyn, axis, exclusive, reverse)

    x1 = ms.Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), ms.float32)
    out1 = test_cell(x1, axis, exclusive, reverse)
    expect1 = np.array([[1, 2, 6, 24], [5, 30, 210, 1680]]).astype('float32')
    assert np.allclose(out1.asnumpy(), expect1, rtol=1e-4, atol=1e-4)

    x2 = ms.Tensor(np.array([[2, 3, 4], [6, 7, 8]]), ms.float32)
    out2 = test_cell(x2, axis, exclusive, reverse)
    expect2 = np.array([[2, 6, 24], [6, 42, 336]]).astype('float32')
    assert np.allclose(out2.asnumpy(), expect2, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_cumprod_dynamic_rank(context_mode):
    """
    Feature: test dynamic rank feature of cumprod.
    Description: test dynamic rank feature of cumprod.
    Expectation: expect correct result.
    """

    ms.context.set_context(mode=context_mode)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    axis = 1
    exclusive = False
    reverse = False
    test_cell = test_utils.to_cell_obj(cumprod_dyn_shape_func)
    test_cell.set_inputs(x_dyn, axis, exclusive, reverse)

    x1 = ms.Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), ms.float32)
    out1 = test_cell(x1, axis, exclusive, reverse)
    expect1 = np.array([[1, 2, 6, 24], [5, 30, 210, 1680]]).astype('float32')
    assert np.allclose(out1.asnumpy(), expect1, rtol=1e-4, atol=1e-4)

    x2 = ms.Tensor(np.array([[[2, 3, 4], [6, 7, 8]]]), ms.float32)
    out2 = test_cell(x2, axis, exclusive, reverse)
    expect2 = np.array([[[2, 3, 4], [12, 21, 32]]]).astype('float32')
    assert np.allclose(out2.asnumpy(), expect2, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_cumprod_backward_dynamic(context_mode):
    """
    Feature: test backward dynamic shape feature of cumprod.
    Description: test backward dynamic shape feature of cumprod.
    Expectation: expect correct result.
    """

    ms.context.set_context(mode=context_mode)
    x_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    axis = 1
    exclusive = False
    reverse = True
    test_cell = test_utils.to_cell_obj(cumprod_backward_func)
    test_cell.set_inputs(x_dyn, axis, exclusive, reverse)

    x1 = ms.Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), ms.float32)
    out1 = test_cell(x1, axis, exclusive, reverse)
    expect1 = np.array([[24, 24, 20, 16], [336, 336, 296, 260]]).astype('float32')
    assert np.allclose(out1.asnumpy(), expect1, rtol=1e-4, atol=1e-4)

    x2 = ms.Tensor(np.array([[2, 3, 4], [6, 7, 8]]), ms.float32)
    out2 = test_cell(x2, axis, exclusive, reverse)
    expect2 = np.array([[12, 12, 10], [56, 56, 50]]).astype('float32')
    assert np.allclose(out2.asnumpy(), expect2, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_cumprod_backward_dynamic_rank(context_mode):
    """
    Feature: test backward dynamic rank feature of cumprod.
    Description: test backward dynamic rank feature of cumprod.
    Expectation: expect correct result.
    """

    ms.context.set_context(mode=context_mode)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    axis = 1
    exclusive = False
    reverse = True
    test_cell = test_utils.to_cell_obj(cumprod_backward_func)
    test_cell.set_inputs(x_dyn, axis, exclusive, reverse)

    x1 = ms.Tensor(np.array([[111, 22, 3, 41], [5, 6, 77, 8]]), ms.float32)
    out1 = test_cell(x1, axis, exclusive, reverse)
    expect1 = np.array([[2706, 13776, 101065, 7396], [3696, 3696, 296, 2850]]).astype('float32')
    assert np.allclose(out1.asnumpy(), expect1, rtol=1e-4, atol=1e-4)

    x2 = ms.Tensor(np.array([[[22, 3, 4], [6, 76, 8]]]), ms.float32)
    out2 = test_cell(x2, axis, exclusive, reverse)
    expect2 = np.array([[[6, 76, 8], [23, 4, 5]]]).astype('float32')
    assert np.allclose(out2.asnumpy(), expect2, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int64, np.uint16,
                                   np.uint32, np.uint64, np.float64, np.float32, np.float16])
def test_cumprod_vmap(context_mode, dtype):
    """
    Feature: Vmap.
    Description: test vmap of op cumprod.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    np_array = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]]]).astype(dtype)
    x = Tensor(np_array)
    axis = -1
    exclusive = False
    reverse = False
    nest_vmap = ops.vmap(ops.vmap(cumprod_forward_func, in_axes=(0, None, None, None)), in_axes=(0, None, None, None))
    out = nest_vmap(x, axis, exclusive, reverse)
    expect_out = np.array([[[1, 2, 6, 24], [5, 30, 210, 1680]]]).astype(dtype)
    assert (out.asnumpy() == expect_out).all()
