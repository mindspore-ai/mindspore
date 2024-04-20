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
import numpy as np
import pytest
from tests.st.utils import test_utils

from mindspore import ops
from mindspore import Tensor
import mindspore as ms

@test_utils.run_with_cell
def cummax_forward_func(x, axis):
    return ops.Cummax(axis)(x)

@test_utils.run_with_cell
def cummax_backward_func(x, axis):
    return ops.grad(cummax_forward_func, (0))(x, axis)

@test_utils.run_with_cell
def cummax_vmap_func(x, axis):
    return ops.vmap(cummax_forward_func, in_axes=(0, None), out_axes=(0, None))(x, axis)

@test_utils.run_with_cell
def cummax_dyn_shape_func(x, axis):
    return ops.cummax(x, axis)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64, np.uint8,
                                   np.float64, np.float32, np.float16])
def test_cummax_forward(context_mode, dtype):
    """
    Feature: cummax ops.
    Description: test ops cummax forward.
    Expectation: output right results.
    """
    ms.context.set_context(mode=context_mode)
    x = Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).astype(dtype))
    axis = 0
    values, indices = cummax_forward_func(x, axis)
    expect_values = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]]).astype(dtype)
    expect_indices = np.asarray([[0, 0, 0, 0], [1, 1, 1, 1]]).astype(np.int64)
    assert np.allclose(values.asnumpy(), expect_values)
    assert (indices.asnumpy() == expect_indices).all()


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64, np.uint8,
                                   np.float64, np.float32, np.float16])
def test_cummax_vmap(context_mode, dtype):
    """
    Feature: Vmap.
    Description: test vmap of op cummax.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    np_array = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]]]).astype(dtype)
    x = Tensor(np_array)
    axis = 0
    nest_vmap = ops.vmap(ops.vmap(cummax_forward_func, in_axes=(0, None)), in_axes=(0, None))
    values, indices = nest_vmap(x, axis)
    expect_values = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]]]).astype(dtype)
    expect_indices = np.array([[[0, 1, 2, 3], [0, 1, 2, 3]]]).astype(np.int64)
    assert (values.asnumpy() == expect_values).all()
    assert (indices.asnumpy() == expect_indices).all()

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("dtype", [np.float32])
def test_cummax_backward(context_mode, dtype):
    """
    Feature: cummax ops.
    Description: test ops cummax forward.
    Expectation: output right results.
    """
    ms.context.set_context(mode=context_mode)
    x = Tensor(np.array([[3, 4, 6, 10], [1, 6, 7, 9], [4, 3, 8, 7], [1, 3, 7, 9]]).astype(dtype))
    expect_values = np.asarray([[2, 1, 1, 4], [0, 3, 1, 0], [2, 0, 2, 0], [0, 0, 0, 0]]).astype(dtype)
    output = cummax_backward_func(x, 0)
    np.testing.assert_allclose(output.asnumpy(), expect_values, rtol=1e-3)
