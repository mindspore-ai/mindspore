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
def less_forward_func(x, y):
    return ops.less(x, y)


@test_utils.run_with_cell
def less_backward_func(x, y):
    return ops.grad(less_forward_func, (0, 1))(x, y)


def less_dyn_shape_func(x, y):
    return ops.less(x, y)


@test_utils.run_with_cell
def less_infervalue_func1():
    x = ms.Tensor(np.array([1, 2, 3]).astype(np.float32))
    y = ms.Tensor(np.array([1, 1, 4]).astype(np.float32))
    return ops.less(x, y)


@test_utils.run_with_cell
def less_infervalue_func2():
    x = ms.Tensor(np.array([1, 2, 4]).astype(np.float32))
    y = ms.Tensor(np.array([3, 2, 1]).astype(np.float32))
    return ops.less(x, y)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode',
                         [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_less_forward(mode):
    """
    Feature: Ops.
    Description: test op less.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = ms.Tensor(np.array([1, 2, 3]), ms.int32)
    y = ms.Tensor(np.array([1, 1, 4]), ms.int32)
    out = less_forward_func(x, y)
    expect = np.array([False, False, True], dtype=np.bool)
    assert np.allclose(out.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode',
                         [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_less_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op less.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = ms.Tensor(np.array([1, 2, 3]), ms.int32)
    y = ms.Tensor(np.array([1, 1, 4]), ms.int32)
    grad_x, grad_y = less_backward_func(x, y)
    expect_grad_x = np.array([0, 0, 0], np.int32)
    expect_grad_y = np.array([0, 0, 0], np.int32)
    assert np.allclose(grad_x.asnumpy(), expect_grad_x)
    assert np.allclose(grad_y.asnumpy(), expect_grad_y)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode',
                         [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_less_vmap(mode):
    """
    Feature: test vmap function.
    Description: test less op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = ms.Tensor(np.array([[1, 2, 3], [1, 2, 3]]), ms.int32)
    y = ms.Tensor(np.array([[1, 1, 4], [1, 1, 4]]), ms.int32)
    less_vmap = ops.vmap(less_forward_func)
    out = less_vmap(x, y)
    expect = np.array([[False, False, True], [False, False, True]],
                      dtype=np.bool)
    assert np.allclose(out.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode',
                         [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_less_dynamic_shape(mode):
    """
    Feature: test dynamic tensor of less.
    Description: test dynamic tensor of less.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x_dyn = ms.Tensor(shape=[None, 3], dtype=ms.int32)
    y_dyn = ms.Tensor(shape=[None, 3], dtype=ms.int32)
    x = ms.Tensor(np.array([[1, 2, 3], [1, 2, 3]]), ms.int32)
    y = ms.Tensor(np.array([[1, 1, 4], [1, 1, 4]]), ms.int32)
    test_cell = test_utils.to_cell_obj(less_dyn_shape_func)
    test_cell.set_inputs(x_dyn, y_dyn)
    out = test_cell(x, y)
    expect = np.array([[False, False, True], [False, False, True]],
                      dtype=np.bool)
    assert np.allclose(out.asnumpy(), expect, rtol=1e-4, atol=1e-4)

    x_2 = ms.Tensor(np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]), ms.int32)
    y_2 = ms.Tensor(np.array([[1, 1, 4], [1, 1, 4], [1, 1, 4]]), ms.int32)
    out_2 = test_cell(x_2, y_2)
    expect_2 = np.array(
        [[False, False, True], [False, False, True], [False, False, True]],
        dtype=np.bool)
    assert np.allclose(out_2.asnumpy(), expect_2, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode',
                         [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_less_dynamic_rank(mode):
    """
    Feature: test dynamic tensor of less.
    Description: test dynamic tensor of less.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x_dyn = ms.Tensor(shape=None, dtype=ms.int32)
    y_dyn = ms.Tensor(shape=None, dtype=ms.int32)
    x = ms.Tensor(np.array([[1, 2, 3], [1, 2, 3]]), ms.int32)
    y = ms.Tensor(np.array([[1, 1, 4], [1, 1, 4]]), ms.int32)
    test_cell = test_utils.to_cell_obj(less_dyn_shape_func)
    test_cell.set_inputs(x_dyn, y_dyn)
    out = test_cell(x, y)
    expect = np.array([[False, False, True], [False, False, True]],
                      dtype=np.bool)
    assert np.allclose(out.asnumpy(), expect, rtol=1e-4, atol=1e-4)

    x_2 = ms.Tensor(np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]]), ms.int32)
    y_2 = ms.Tensor(np.array([[1, 1, 4], [1, 1, 4], [1, 1, 4]]), ms.int32)
    out_2 = test_cell(x_2, y_2)
    expect_2 = np.array(
        [[False, False, True], [False, False, True], [False, False, True]],
        dtype=np.bool)
    assert np.allclose(out_2.asnumpy(), expect_2, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_less_op_infervalue(context_mode):
    """
    Feature: Ops.
    Description: test op less infervalue.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    out_1 = less_infervalue_func1()
    expect_out_1 = np.array([False, False, True], dtype=np.bool)
    assert np.allclose(out_1.asnumpy(), expect_out_1)
    out_2 = less_infervalue_func2()
    expect_out_2 = np.array([True, False, False], dtype=np.bool)
    assert np.allclose(out_2.asnumpy(), expect_out_2)
