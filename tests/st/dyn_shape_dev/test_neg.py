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
def neg_forward_func(x):
    return ops.auto_generate.neg(x)


@test_utils.run_with_cell
def neg_backward_func(x):
    return ops.grad(neg_forward_func, 0)(x)


@test_utils.run_with_cell
def neg_vmap_func(x):
    return ops.vmap(neg_forward_func, in_axes=0, out_axes=0)(x)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
@test_utils.run_test_func
def test_neg_op_forward(context_mode, data_type):
    """
    Feature: Ops.
    Description: test op neg forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = ms.Tensor(np.array([1, 2, -1, 2, 0, -3.5]).astype(data_type))
    out = neg_forward_func(x)
    expect_out = np.array([-1., -2., 1., -2., 0., 3.5]).astype(np.float32)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
@test_utils.run_test_func
def test_neg_op_backward(context_mode, data_type):
    """
    Feature: Auto grad.
    Description: test auto grad of op neg.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = ms.Tensor(np.array([1, 2, -1, 2, 0, -3.5]).astype(data_type))
    grads = neg_backward_func(x)
    expect_out = np.array([-1., -1., -1., -1., -1., -1.]).astype(np.float32)
    np.testing.assert_allclose(grads.asnumpy(), expect_out, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
@test_utils.run_test_func
def test_neg_op_vmap(context_mode, data_type):
    """
    Feature: test vmap function.
    Description: test neg op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = ms.Tensor(np.array([1, 2, -1, 2, 0, -3.5]).astype(data_type))
    out = neg_vmap_func(x)
    expect_out = np.array([-1., -2., 1., -2., 0., 3.5]).astype(np.float32)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-3)
