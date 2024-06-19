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


@test_utils.run_with_cell
def div_forward_func(x1, x2):
    return ops.Div()(x1, x2)


@test_utils.run_with_cell
def div_backward_func(x1, x2):
    return ops.grad(div_forward_func, (0, 1))(x1, x2)


@test_utils.run_with_cell
def div_dyn_shape_func(x1, x2):
    return ops.Div()(x1, x2)


@test_utils.run_with_cell
def div_infer_value():
    x = ms.Tensor(np.array([[2, 2], [3, 3]]).astype(np.float32))
    y = ms.Tensor(np.array([[1, 2], [3, 6]]).astype(np.float32))

    return div_forward_func(x, y)

@test_utils.run_with_cell
def div_infer_value1():
    x = ms.Tensor(np.array([[2, 2], [3, 3]]).astype(np.int64))
    y = ms.Tensor(np.array([3]).astype(np.int64))

    return div_forward_func(x, y)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_div_infer_value(mode):
    """
    Feature: Ops.
    Description: test op div infer value.
    Expectation: expect correct result.
    """
    out1 = div_infer_value()
    np_x1 = np.array([[2, 2], [3, 3]]).astype(np.float32)
    np_y1 = np.array([[1, 2], [3, 6]]).astype(np.float32)
    expect1 = np.divide(np_x1, np_y1)
    assert np.allclose(out1.asnumpy(), expect1)

    out2 = div_infer_value1()
    np_x2 = np.array([[2, 2], [3, 3]]).astype(np.int64)
    np_y2 = np.array([3]).astype(np.int64)
    expect2 = np_x2 / np_y2
    assert np.allclose(out2.asnumpy(), expect2)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
# @pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_div_forward(mode):
    """
    Feature: Ops.
    Description: test op div.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x1_array = np.array([7, 8, 9]).astype(np.float32)
    x2_array = np.array([14, 6, 12]).astype(np.float32)
    x1 = ms.Tensor(x1_array, ms.float32)
    x2 = ms.Tensor(x2_array, ms.float32)
    expect_out = np.divide(x1_array, x2_array)
    out = div_forward_func(x1, x2)
    assert np.allclose(out.asnumpy(), expect_out)

    x1_array = np.array([20]).astype(np.float32)
    x2_array = np.array([14, 6, 12]).astype(np.float32)
    x1 = ms.Tensor(x1_array, ms.float32)
    x2 = ms.Tensor(x2_array, ms.float32)
    expect_out = np.divide(x1_array, x2_array)
    out = div_forward_func(x1, x2)
    assert np.allclose(out.asnumpy(), expect_out)

    x1_array = 5
    x2_array = np.array([14, 6, 12]).astype(np.float32)
    x1 = ms.Tensor(x1_array, ms.float16)
    x2 = ms.Tensor(x2_array, ms.float32)
    expect_out = np.divide(x1_array, x2_array)
    out = div_forward_func(x1, x2)
    assert np.allclose(out.asnumpy(), expect_out)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
# @pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_div_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op div.
    Expectation: expect it raises RuntimeError.
    """
    ms.context.set_context(mode=mode)
    x1_array = np.array([7, 8, 9]).astype(np.float32)
    x2_array = np.array([14, 6, 12]).astype(np.float32)
    x1 = ms.Tensor(x1_array, ms.float32)
    x2 = ms.Tensor(x2_array, ms.float32)
    expect_x1_grad = [7.14285746e-02, 1.66666672e-01, 8.33333358e-02]
    expect_x2_grad = [-3.57142873e-02, -2.22222239e-01, -6.25000000e-02]
    out = div_backward_func(x1, x2)
    assert np.allclose(out[0].asnumpy(), expect_x1_grad)
    assert np.allclose(out[1].asnumpy(), expect_x2_grad)

    x1_array = np.array([20]).astype(np.float32)
    x2_array = np.array([14, 6, 12]).astype(np.float32)
    x1 = ms.Tensor(x1_array, ms.float32)
    x2 = ms.Tensor(x2_array, ms.float32)
    expect_x1_grad = [3.21428597e-01]
    expect_x2_grad = [-1.02040820e-01, -5.55555582e-01, -1.38888896e-01]
    out = div_backward_func(x1, x2)
    assert np.allclose(out[0].asnumpy(), expect_x1_grad)
    assert np.allclose(out[1].asnumpy(), expect_x2_grad)

    x1_array = 5.0
    x2_array = np.array([14, 6, 12]).astype(np.float32)
    x1 = ms.Tensor(x1_array, ms.float16)
    x2 = ms.Tensor(x2_array, ms.float32)
    expect_out = [-2.55102050e-02, -1.38888896e-01, -3.47222239e-02]
    out = div_backward_func(x1, x2)
    assert np.allclose(out[1].asnumpy(), expect_out)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
# @pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_div_vmap(mode):
    """
    Feature: test vmap function.
    Description: test div op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    in_axes = -1
    x1 = ms.Tensor(np.array([7, 8, 9]), ms.int32)
    x2 = ms.Tensor(np.array([14, 6, 12]), ms.int32)
    nest_vmap = ops.vmap(div_forward_func, in_axes=in_axes, out_axes=0)
    out = nest_vmap(x1, x2)
    expect = div_forward_func(x1, x2)
    assert np.allclose(out.asnumpy(), expect.asnumpy())


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
# @pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_div_dynamic(context_mode):
    """
    Feature: test dynamic tensor and dynamic scalar of div.
    Description: test dynamic tensor and dynamic scalar of div.
    Expectation: expect correct result.
    """

    ms.context.set_context(mode=context_mode)
    x1_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    x2_dyn = ms.Tensor(shape=[None], dtype=ms.float32)

    test_cell = test_utils.to_cell_obj(div_dyn_shape_func)
    test_cell.set_inputs(x1_dyn, x2_dyn)

    x1 = ms.Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), ms.float32)
    x2 = ms.Tensor(np.array([1, 2, 3, 4]), ms.float32)
    out1 = test_cell(x1, x2)
    expect = np.array([[1, 1, 1, 1], [5, 3, 2.33333, 2]]).astype('float32')
    assert np.allclose(out1.asnumpy(), expect, rtol=1e-4, atol=1e-4)

    x1_1 = ms.Tensor(np.array([[1, 2, 3], [5, 6, 7]]), ms.float32)
    x2_1 = ms.Tensor(np.array([1, 2, 3]), ms.float32)
    out1_1 = test_cell(x1_1, x2_1)
    expect_1 = np.array([[1, 1, 1], [5, 3, 2.33333]]).astype('float32')
    assert np.allclose(out1_1.asnumpy(), expect_1, rtol=1e-4, atol=1e-4)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
# @pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_div_dynamic_rank(context_mode):
    """
    Feature: test dynamic tensor and dynamic scalar of div.
    Description: test dynamic tensor and dynamic scalar of div.
    Expectation: expect correct result.
    """

    ms.context.set_context(mode=context_mode)
    x1_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    x2_dyn = ms.Tensor(shape=None, dtype=ms.float32)

    test_cell = test_utils.to_cell_obj(div_dyn_shape_func)
    test_cell.set_inputs(x1_dyn, x2_dyn)

    x1 = ms.Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), ms.float32)
    x2 = ms.Tensor(np.array([1, 2, 3, 4]), ms.float32)
    out1 = test_cell(x1, x2)
    expect = np.array([[1, 1, 1, 1], [5, 3, 2.33333, 2]]).astype('float32')
    assert np.allclose(out1.asnumpy(), expect, rtol=1e-4, atol=1e-4)

    x1_1 = ms.Tensor(np.array([[[1, 2, 3], [5, 6, 7]]]), ms.float32)
    x2_1 = ms.Tensor(np.array([1, 2, 3]), ms.float32)
    out1_1 = test_cell(x1_1, x2_1)
    expect_1 = np.array([[[1, 1, 1], [5, 3, 2.33333]]]).astype('float32')
    assert np.allclose(out1_1.asnumpy(), expect_1, rtol=1e-4, atol=1e-4)
