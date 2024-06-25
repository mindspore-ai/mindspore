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
def cummin_forward_func(x, axis):
    return ops.Cummin(axis)(x)


@test_utils.run_with_cell
def cummin_vmap_func(x, axis):
    return ops.vmap(cummin_forward_func, in_axes=0, out_axes=0)(x, axis)

@test_utils.run_with_cell
def cummin_dyn_shape_func(x, axis):
    return ops.Cummin(axis)(x)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16,
                                   np.uint32, np.uint64, np.float64, np.float32, np.float16])
def test_cummin_forward(context_mode, dtype):
    """
    Feature: cummin ops.
    Description: test ops cummin forward.
    Expectation: output right results.
    """
    ms.context.set_context(mode=context_mode)
    x = Tensor(np.array([[3, 1, 4, 1], [1, 5, 9, 2]]).astype(dtype))
    axis = -2
    values, indices = cummin_forward_func(x, axis)
    expect_values = np.asarray([[3, 1, 4, 1], [1, 1, 4, 1]]).astype(dtype)
    expect_indices = np.asarray([[0, 0, 0, 0], [1, 0, 0, 0]]).astype(np.int64)
    assert np.allclose(values.asnumpy(), expect_values)
    assert (indices.asnumpy() == expect_indices).all()


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16,
                                   np.uint32, np.uint64, np.float64, np.float32, np.float16])
def test_cummin_vmap(context_mode, dtype):
    """
    Feature: Vmap.
    Description: test vmap of op cummin.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    np_array = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]]]).astype(dtype)
    x = Tensor(np_array)
    axis = -1
    nest_vmap = ops.vmap(ops.vmap(cummin_forward_func, in_axes=(0, None)), in_axes=(0, None))
    values, indices = nest_vmap(x, axis)
    expect_values = np.array([[[1, 1, 1, 1], [5, 5, 5, 5]]]).astype(dtype)
    expect_indices = np.array([[[0, 0, 0, 0], [0, 0, 0, 0]]]).astype(np.int64)
    assert (values.asnumpy() == expect_values).all()
    assert (indices.asnumpy() == expect_indices).all()


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_cummin_dynamic(context_mode):
    """
    Feature: test dynamic shape feature of cummin.
    Description: test dynamic shape feature of cummin.
    Expectation: expect correct result.
    """

    ms.context.set_context(mode=context_mode)
    x_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    axis = 1
    test_cell = test_utils.to_cell_obj(cummin_dyn_shape_func)
    test_cell.set_inputs(x_dyn, axis)

    x1 = ms.Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), ms.float32)
    values1, indices1 = test_cell(x1, axis)
    expect_values1 = np.array([[1, 1, 1, 1], [5, 5, 5, 5]]).astype(np.float32)
    expect_indices1 = np.array([[0, 0, 0, 0], [0, 0, 0, 0]]).astype(np.float32)
    assert np.allclose(values1.asnumpy(), expect_values1)
    assert np.allclose(indices1.asnumpy(), expect_indices1)

    x2 = ms.Tensor(np.array([[2, 3, 4], [6, 7, 8]]), ms.float32)
    values2, indices2 = test_cell(x2, axis)
    expect_values2 = np.array([[2, 2, 2], [6, 6, 6]]).astype(np.float32)
    expect_indices2 = np.array([[0, 0, 0], [0, 0, 0]]).astype(np.float32)
    assert np.allclose(values2.asnumpy(), expect_values2)
    assert np.allclose(indices2.asnumpy(), expect_indices2)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_cummin_dynamic_rank(context_mode):
    """
    Feature: test dynamic rank feature of cummin.
    Description: test dynamic rank feature of cummin.
    Expectation: expect correct result.
    """

    ms.context.set_context(mode=context_mode)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    axis = 1
    test_cell = test_utils.to_cell_obj(cummin_dyn_shape_func)
    test_cell.set_inputs(x_dyn, axis)

    x1 = ms.Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), ms.float32)
    values1, indices1 = test_cell(x1, axis)
    expect_values1 = np.array([[1, 1, 1, 1], [5, 5, 5, 5]]).astype(np.float32)
    expect_indices1 = np.array([[0, 0, 0, 0], [0, 0, 0, 0]]).astype(np.float32)
    assert np.allclose(values1.asnumpy(), expect_values1)
    assert np.allclose(indices1.asnumpy(), expect_indices1)

    x2 = ms.Tensor(np.array([[[23, 32, 13], [12, 43, 7]]]), ms.float32)
    values2, indices2 = test_cell(x2, axis)
    expect_values2 = np.array([[[23, 32, 13], [12, 32, 7]]]).astype(np.float32)
    expect_indices2 = np.array([[[0, 0, 0], [1, 0, 1]]]).astype(np.float32)
    assert np.allclose(values2.asnumpy(), expect_values2)
    assert np.allclose(indices2.asnumpy(), expect_indices2)
