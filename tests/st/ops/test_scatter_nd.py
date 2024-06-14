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
import tests.st.utils.test_utils as test_utils

from mindspore import ops
import mindspore as ms


@test_utils.run_with_cell
def scatter_nd_forward_func(indices, updates, shape):
    return ops.scatter_nd(indices, updates, shape)


@test_utils.run_with_cell
def scatter_nd_backward_func(indices, updates, shape):
    return ops.grad(scatter_nd_forward_func, (0, 1))(indices, updates, shape)


@test_utils.run_with_cell
def scatter_nd_vmap_func(indices, updates, shape):
    return ops.vmap(scatter_nd_forward_func, in_axes=(0, 0, None), out_axes=0)(indices, updates, shape)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [np.float32, np.float16, np.int64, np.int32, np.int16, np.int8])
@pytest.mark.parametrize("indices_type", [np.int64, np.int32])
@test_utils.run_test_with_On
def test_scatter_nd_op_forward_1(context_mode, data_type, indices_type):
    """
    Feature: Ops.
    Description: test op scatter_nd forward 1.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    indices = ms.Tensor(np.array([[1, 0], [1, 1], [1, 0], [1, 0], [1, 0]]).astype(indices_type))
    updates = ms.Tensor(np.array([-13.4, -3.1, 5.1, -12.1, -1.0]).astype(data_type))
    shape = (2, 2)
    out = scatter_nd_forward_func(indices, updates, shape)
    expect_out = np.array([[0., 0.], [-21.4, -3.1]]).astype(data_type)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-6)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [np.float32, np.float16, np.int64, np.int32, np.int16, np.int8])
@pytest.mark.parametrize("indices_type", [np.int64, np.int32])
@test_utils.run_test_with_On
def test_scatter_nd_op_forward_2(context_mode, data_type, indices_type):
    """
    Feature: Ops.
    Description: test op scatter_nd forward 2.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    indices = ms.Tensor(np.array([[0, 1], [1, 1], [0, 1], [0, 1], [0, 1]]).astype(indices_type))
    updates = ms.Tensor(np.array([3.2, 1.1, 5.3, -2.2, -1.0]).astype(data_type))
    shape = (2, 2)
    out = scatter_nd_forward_func(indices, updates, shape)
    expect_out = np.array([[0., 5.3], [0., 1.1]]).astype(data_type)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [np.float32, np.float16, np.int64, np.int32, np.int16, np.int8])
@pytest.mark.parametrize("indices_type", [np.int64, np.int32])
@test_utils.run_test_with_On
def test_scatter_nd_op_backward_1(context_mode, data_type, indices_type):
    """
    Feature: Ops.
    Description: test op scatter_nd backward 1.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    indices = ms.Tensor(np.array([[1, 0], [1, 1], [1, 0], [1, 0], [1, 0]]).astype(indices_type))
    updates = ms.Tensor(np.array([-13.4, -3.1, 5.1, -12.1, -1.0]).astype(data_type))
    shape = (2, 2)
    grad_out_0, grad_out_1 = scatter_nd_backward_func(indices, updates, shape)
    expect_out_0 = np.array([[0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.]]).astype(data_type)
    expect_out_1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0]).astype(data_type)
    np.testing.assert_allclose(grad_out_0.asnumpy(), expect_out_0, rtol=1e-6)
    np.testing.assert_allclose(grad_out_1.asnumpy(), expect_out_1, rtol=1e-6)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [np.float32, np.float16, np.int64, np.int32, np.int16, np.int8])
@pytest.mark.parametrize("indices_type", [np.int64, np.int32])
@test_utils.run_test_with_On
def test_scatter_nd_op_backward_2(context_mode, data_type, indices_type):
    """
    Feature: Ops.
    Description: test op scatter_nd backward 2.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    indices = ms.Tensor(np.array([[0, 1], [1, 1], [0, 1], [0, 1], [0, 1]]).astype(indices_type))
    updates = ms.Tensor(np.array([3.2, 1.1, 5.3, -2.2, -1.0]).astype(data_type))
    shape = (2, 2)
    grad_out_0, grad_out_1 = scatter_nd_backward_func(indices, updates, shape)
    expect_out_0 = np.array([[0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.]]).astype(data_type)
    expect_out_1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0]).astype(data_type)
    np.testing.assert_allclose(grad_out_0.asnumpy(), expect_out_0, rtol=1e-6)
    np.testing.assert_allclose(grad_out_1.asnumpy(), expect_out_1, rtol=1e-6)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [np.float32, np.int32])
@pytest.mark.parametrize("indices_type", [np.int32])
@test_utils.run_test_with_On
def test_scatter_nd_op_vmap(context_mode, data_type, indices_type):
    """
    Feature: test vmap function.
    Description: test scatter_nd op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    indices = ms.Tensor(np.array([[[1, 0], [1, 1], [1, 0], [1, 0], [1, 0]],
                                  [[0, 1], [1, 1], [0, 1], [0, 1], [0, 1]]]).astype(indices_type))
    updates = ms.Tensor(np.array([[-13.4, -3.1, 5.1, -12.1, -1.0],
                                  [3.2, 1.1, 5.3, -2.2, -1.0]]).astype(data_type))
    shape = (2, 2)
    out = scatter_nd_vmap_func(indices, updates, shape)
    expect_out = np.array([[[0., 0.], [-21.4, -3.1]],
                           [[0., 5.3], [0., 1.1]]]).astype(data_type)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-6)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [ms.float32, ms.int64])
@pytest.mark.parametrize("indices_type", [ms.int64, ms.int32])
@test_utils.run_test_with_On
def test_scatter_nd_op_dynamic_shape(context_mode, data_type, indices_type):
    """
    Feature: Ops.
    Description: test op scatter_nd dynamic_shape.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    np_type = np.float32 if data_type == ms.float32 else np.int64
    indices_dyn = ms.Tensor(shape=[None, None], dtype=indices_type)
    updates_dyn = ms.Tensor(shape=[None], dtype=data_type)
    test_cell = test_utils.to_cell_obj(ops.scatter_nd)
    shape = (2, 2)
    test_cell.set_inputs(indices_dyn, updates_dyn, shape)

    indices = ms.Tensor(np.array([[1, 0], [1, 1], [1, 0], [1, 0], [1, 0]])).astype(indices_type)
    updates = ms.Tensor(np.array([-13.4, -3.1, 5.1, -12.1, -1.0])).astype(data_type)
    out = test_cell(indices, updates, shape)
    expect_out = np.array([[0., 0.], [-21.4, -3.1]]).astype(np_type)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-6)

    indices = ms.Tensor(np.array([[1, 0], [1, 1], [1, 0], [1, 0]])).astype(indices_type)
    updates = ms.Tensor(np.array([-13.4, -3.1, 5.1, -12.1])).astype(data_type)
    out = test_cell(indices, updates, shape)
    expect_out = np.array([[0., 0.], [-20.4, -3.1]]).astype(np_type)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-6)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [ms.float32, ms.int64])
@pytest.mark.parametrize("indices_type", [ms.int64, ms.int32])
@test_utils.run_test_with_On
def test_scatter_nd_op_dynamic_rank(context_mode, data_type, indices_type):
    """
    Feature: Ops.
    Description: test op scatter_nd dynamic_rank.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    np_type = np.float32 if data_type == ms.float32 else np.int64
    indices_dyn = ms.Tensor(shape=None, dtype=indices_type)
    updates_dyn = ms.Tensor(shape=None, dtype=data_type)
    test_cell = test_utils.to_cell_obj(ops.scatter_nd)
    shape = (2, 2)
    test_cell.set_inputs(indices_dyn, updates_dyn, shape)

    indices = ms.Tensor(np.array([[1, 0], [1, 1], [1, 0], [1, 0], [1, 0]])).astype(indices_type)
    updates = ms.Tensor(np.array([-13.4, -3.1, 5.1, -12.1, -1.0])).astype(data_type)
    out = test_cell(indices, updates, shape)
    expect_out = np.array([[0., 0.], [-21.4, -3.1]]).astype(np_type)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-6)

    indices = ms.Tensor(np.array([[0], [1]])).astype(indices_type)
    updates = ms.Tensor(np.array([[1.0, 2.0], [3.0, 4.0]])).astype(data_type)
    out = test_cell(indices, updates, shape)
    expect_out = np.array([[1.0, 2.0], [3.0, 4.0]]).astype(np_type)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-6)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_scatter_nd_op_1d_shape(context_mode):
    """
    Feature: Ops.
    Description: test op scatter_nd forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    indices = ms.Tensor(np.array([[1], [3], [5], [7], [9]]).astype(np.int64))
    updates = ms.Tensor(np.array([-13.4, -3.1, 5.1, -12.1, -1.0]).astype(np.float32))
    shape = (10,)
    out = scatter_nd_forward_func(indices, updates, shape)
    expect_out = np.array([0, -13.4, 0, -3.1, 0, 5.1, 0, -12.1, 0, -1.0]).astype(np.float32)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_scatter_nd_exception(context_mode):
    """
    Feature: test exception case.
    Description: test scatter_nd op  exception case.
    Expectation: expect catching the error.
    """
    np.random.seed(98094768)
    indices = ms.Tensor(np.random.uniform(-10, 10, size=()).astype(np.int64))
    updates = ms.Tensor(np.random.uniform(-10, 10, size=[2]).astype(np.float32))
    shape = (-23, 9, 36, -4)
    ms.context.set_context(pynative_synchronize=True)
    with pytest.raises(RuntimeError) as info:
        _ = scatter_nd_forward_func(indices, updates, shape)
    assert "a scalar" in str(info)
