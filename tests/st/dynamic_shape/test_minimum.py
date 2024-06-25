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
def minimum_forward_func(x, y):
    return ops.minimum(x, y)


@test_utils.run_with_cell
def minimum_backward_func(x, y):
    return ops.grad(minimum_forward_func, (0, 1))(x, y)


@test_utils.run_with_cell
def minimum_vmap_func(x, y):
    return ops.vmap(minimum_forward_func, in_axes=0, out_axes=0)(x, y)


@test_utils.run_with_cell
def minimum_infervalue_func1():
    x = ms.Tensor(np.array([1, 2, 4]).astype(np.float32))
    y = ms.Tensor(np.array([2, 4, 3]).astype(np.float32))
    return ops.minimum(x, y)


@test_utils.run_with_cell
def minimum_infervalue_func2():
    x = ms.Tensor(np.array([4, 5, 2]).astype(np.float32))
    y = ms.Tensor(np.array([3]).astype(np.float32))
    return ops.minimum(x, y)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
@test_utils.run_test_with_On
def test_minimum_op_forward(context_mode, data_type):
    """
    Feature: Ops.
    Description: test op minimum forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = ms.Tensor(np.array([1, 2, 4]).astype(data_type))
    y = ms.Tensor(np.array([2, 4, 3]).astype(data_type))
    out = minimum_forward_func(x, y)
    expect_out = np.array([1., 2., 3.]).astype(np.float32)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
@test_utils.run_test_with_On
def test_minimum_op_backward(context_mode, data_type):
    """
    Feature: Auto grad.
    Description: test auto grad of op minimum.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = ms.Tensor(np.array([1, 2, 4]).astype(data_type))
    y = ms.Tensor(np.array([2, 4, 3]).astype(data_type))
    grads = minimum_backward_func(x, y)
    expect_out = np.array([[1., 1., 0.], [0., 0., 1.]]).astype(np.float32)
    np.testing.assert_allclose(grads[0].asnumpy(), expect_out[0], rtol=1e-3)
    np.testing.assert_allclose(grads[1].asnumpy(), expect_out[1], rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
@test_utils.run_test_with_On
def test_minimum_op_vmap(context_mode, data_type):
    """
    Feature: test vmap function.
    Description: test minimum op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = ms.Tensor(np.array([[3, 5, 11], [7, 5, 6]]).astype(data_type))
    y = ms.Tensor(np.array([[6, 6, 8], [3, 2, 7]]).astype(data_type))
    out = minimum_vmap_func(x, y)
    expect_out = np.array([[3., 5., 8.], [3., 2., 6.]]).astype(np.float32)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_minimum_op_infervalue(context_mode):
    """
    Feature: Ops.
    Description: test op minimum infervalue.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    out_1 = minimum_infervalue_func1()
    expect_out_1 = np.array([1., 2., 3.]).astype(np.float32)
    np.testing.assert_array_equal(out_1.asnumpy(), expect_out_1)
    out_2 = minimum_infervalue_func2()
    expect_out_2 = np.array([3., 3., 2.]).astype(np.float32)
    np.testing.assert_array_equal(out_2.asnumpy(), expect_out_2)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_minimum_op_dynamic_shape(context_mode):
    """
    Feature: minimum ops.
    Description: test ops minimum dynamic shape tensor input.
    Expectation: output the right result.
    """
    ms.context.set_context(mode=context_mode)
    x_dyn = ms.Tensor(shape=[None], dtype=ms.float32)
    y_dyn = ms.Tensor(shape=[None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(ops.minimum)
    test_cell.set_inputs(x_dyn, y_dyn)
    x = ms.Tensor(np.array([6, 1, 2]).astype(np.float32))
    y = ms.Tensor(np.array([9, 2, 4]).astype(np.float32))
    out = test_cell(x, y)
    expect_out = np.array([6., 1., 2.]).astype(np.float32)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-3)
    x_2 = ms.Tensor(np.array([3, 8, 11]).astype(np.float32))
    y_2 = ms.Tensor(np.array([6, 5, 7]).astype(np.float32))
    out_2 = test_cell(x_2, y_2)
    expect_out_2 = np.array([3., 5., 7.]).astype(np.float32)
    np.testing.assert_allclose(out_2.asnumpy(), expect_out_2, rtol=1e-3)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_minimum_op_dynamic_rank(context_mode):
    """
    Feature: minimum ops.
    Description: test ops minimum dynamic rank tensor input.
    Expectation: output the right result.
    """
    ms.context.set_context(mode=context_mode)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    y_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(ops.minimum)
    test_cell.set_inputs(x_dyn, y_dyn)
    x = ms.Tensor(np.array([6, 1, 2, 4]).astype(np.float32))
    y = ms.Tensor(np.array([9, 2, 4, 3]).astype(np.float32))
    out = test_cell(x, y)
    expect_out = np.array([6., 1., 2., 3.]).astype(np.float32)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-3)
    x_2 = ms.Tensor(np.array([3, 8, 11]).astype(np.float32))
    y_2 = ms.Tensor(np.array([6, 5, 7]).astype(np.float32))
    out_2 = test_cell(x_2, y_2)
    expect_out_2 = np.array([3., 5., 7.]).astype(np.float32)
    np.testing.assert_allclose(out_2.asnumpy(), expect_out_2, rtol=1e-3)


# 反向动态shape有公共问题，待解决后再放开用例
@pytest.mark.skip
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_minimum_op_dynamic_backward_shape(context_mode):
    """
    Feature: minimum ops.
    Description: test ops minimum dynamic shape tensor input.
    Expectation: output the right result.
    """
    ms.context.set_context(mode=context_mode)
    x_dyn = ms.Tensor(shape=[None], dtype=ms.float32)
    y_dyn = ms.Tensor(shape=[None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(minimum_backward_func)
    test_cell.set_inputs(x_dyn, y_dyn)
    x = ms.Tensor(np.array([6, 1, 2]).astype(np.float32))
    y = ms.Tensor(np.array([9, 2, 4]).astype(np.float32))
    grads = test_cell(x, y)
    expect_out = np.array([[1., 1., 0.], [0., 0., 1.]]).astype(np.float32)
    np.testing.assert_allclose(grads[0].asnumpy(), expect_out[0], rtol=1e-3)
    np.testing.assert_allclose(grads[1].asnumpy(), expect_out[1], rtol=1e-3)

    x_2 = ms.Tensor(np.array([3, 8, 11]).astype(np.float32))
    y_2 = ms.Tensor(np.array([6, 5, 7]).astype(np.float32))
    grads_2 = test_cell(x_2, y_2)
    expect_out_2 = np.array([[1., 0., 0.], [0., 1., 1.]]).astype(np.float32)
    np.testing.assert_allclose(grads_2[0].asnumpy(), expect_out_2[0], rtol=1e-3)
    np.testing.assert_allclose(grads_2[1].asnumpy(), expect_out_2[1], rtol=1e-3)

# 反向动态shape有公共问题，待解决后再放开用例
@pytest.mark.skip
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_minimum_op_dynamic_backward_rank(context_mode):
    """
    Feature: minimum ops.
    Description: test ops minimum dynamic rank tensor input.
    Expectation: output the right result.
    """
    ms.context.set_context(mode=context_mode)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    y_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(minimum_backward_func)
    test_cell.set_inputs(x_dyn, y_dyn)
    x = ms.Tensor(np.array([6, 1, 2, 4]).astype(np.float32))
    y = ms.Tensor(np.array([9, 2, 4, 3]).astype(np.float32))
    grads = test_cell(x, y)
    expect_out = np.array([[1., 1., 1., 0.], [0., 0., 0., 1.]]).astype(np.float32)
    np.testing.assert_allclose(grads[0].asnumpy(), expect_out[0], rtol=1e-3)
    np.testing.assert_allclose(grads[1].asnumpy(), expect_out[1], rtol=1e-3)

    x_2 = ms.Tensor(np.array([3, 8, 11]).astype(np.float32))
    y_2 = ms.Tensor(np.array([6, 5, 7]).astype(np.float32))
    grads_2 = test_cell(x_2, y_2)
    expect_out_2 = np.array([[1., 0., 0.], [0., 1., 1.]]).astype(np.float32)
    np.testing.assert_allclose(grads_2[0].asnumpy(), expect_out_2[0], rtol=1e-3)
    np.testing.assert_allclose(grads_2[1].asnumpy(), expect_out_2[1], rtol=1e-3)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [np.complex64])
@test_utils.run_test_with_On
def test_minimum_op_check_dtype(context_mode, data_type):
    """
    Feature: Ops.
    Description: test op minimum forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = ms.Tensor(np.array([1, 2, 4]).astype(data_type))
    y = ms.Tensor(np.array([2, 4, 3]).astype(data_type))
    with pytest.raises(TypeError):
        _ = minimum_forward_func(x, y)
