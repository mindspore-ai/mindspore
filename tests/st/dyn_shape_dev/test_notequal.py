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
import test_utils

from mindspore import ops
import mindspore as ms


@test_utils.run_with_cell
def notequal_forward_func(x, y):
    return ops.auto_generate.not_equal(x, y)


@test_utils.run_with_cell
def notequal_infervalue_func1():
    x = ms.Tensor(np.array([1, 2, 4]).astype(np.float32))
    y = ms.Tensor(np.array([1, 2, 3]).astype(np.float32))
    return ops.auto_generate.not_equal(x, y)


@test_utils.run_with_cell
def notequal_infervalue_func2():
    x = ms.Tensor(np.array([1, 2, 4]).astype(np.float32))
    y = ms.Tensor(np.array([3]).astype(np.float32))
    return ops.auto_generate.not_equal(x, y)


@test_utils.run_with_cell
def notequal_backward_func(x, y):
    return ops.grad(notequal_forward_func, (0, 1))(x, y)


@test_utils.run_with_cell
def notequal_vmap_func(x, y):
    return ops.vmap(notequal_forward_func, in_axes=0, out_axes=0)(x, y)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
def test_notequal_op_forward(context_mode, data_type):
    """
    Feature: Ops.
    Description: test op notequal forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = ms.Tensor(np.array([1, 2, 4]).astype(data_type))
    y = ms.Tensor(np.array([2, 4, 3]).astype(data_type))
    out = notequal_forward_func(x, y)
    expect_out = np.array([True, True, True]).astype(np.bool)
    np.testing.assert_array_equal(out.asnumpy(), expect_out)
    print("out:", out)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_notequal_op_infervalue(context_mode):
    """
    Feature: Ops.
    Description: test op notequal infervalue.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    out = notequal_infervalue_func1()
    expect_out = np.array([False, False, True]).astype(np.bool)
    np.testing.assert_array_equal(out.asnumpy(), expect_out)
    print("out1:", out)
    out = notequal_infervalue_func2()
    expect_out = np.array([True, True, True]).astype(np.bool)
    np.testing.assert_array_equal(out.asnumpy(), expect_out)
    print("out2:", out)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
def test_notequal_op_backward(context_mode, data_type):
    """
    Feature: Auto grad.
    Description: test auto grad of op notequal.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = ms.Tensor(np.array([1, 2, 4]).astype(data_type))
    y = ms.Tensor(np.array([2, 4, 3]).astype(data_type))
    grads = notequal_backward_func(x, y)
    expect_out = np.array([0., 0., 0.]).astype(np.float32)
    np.testing.assert_allclose(grads[0].asnumpy(), expect_out, rtol=1e-3)
    print("grads:", grads)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
def test_notequal_op_vmap(context_mode, data_type):
    """
    Feature: test vmap function.
    Description: test notequal op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = ms.Tensor(np.array([[1, 2, 3], [3, 2, 1]]).astype(data_type))
    y = ms.Tensor(np.array([[1, 2, 3], [3, 2, 1]]).astype(data_type))
    out = notequal_vmap_func(x, y)
    expect_out = np.array([[False, False, False], [False, False, False]]).astype(np.bool)
    np.testing.assert_array_equal(out.asnumpy(), expect_out)
    print("vmap:", out)
