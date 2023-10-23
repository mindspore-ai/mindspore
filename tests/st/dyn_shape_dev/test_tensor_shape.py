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
from mindspore import ops
import mindspore as ms
import test_utils


@test_utils.run_with_cell
def tensor_shape_forward_func(x):
    return ops.auto_generate.tensor_shape(x)


@test_utils.run_with_cell
def tensor_shape_backward_func(x):
    return ops.grad(tensor_shape_forward_func, (0,))(x)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_func
def test_tensor_shape_forward(mode):
    """
    Feature: Ops.
    Description: test op tensor_shape.
    Expectation: expect correct result.
    """
    ms.set_context(mode=mode)
    x = ms.Tensor(np.ones([3, 2, 1]).astype(np.float32))
    out = tensor_shape_forward_func(x)
    expect = np.array([3, 2, 1]).astype(np.int64)
    assert (out.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_func
def test_tensor_shape_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op tensor_shape.
    Expectation: expect correct result.
    """
    ms.set_context(mode=mode)
    x = ms.Tensor(np.ones([3, 2, 1]).astype(np.float32))
    grad = tensor_shape_backward_func(x)
    expect_grad = np.zeros((3, 2, 1)).astype(np.float32)
    assert (grad.asnumpy() == expect_grad).all()


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_func
def test_tensor_shape_vmap(mode):
    """
    Feature: test vmap function.
    Description: test tensor_shape op vmap.
    Expectation: expect correct result.
    """
    ms.set_context(mode=mode)
    in_axes = -1
    x = ms.Tensor(np.ones([3, 2, 1, 2, 2]).astype(np.float32))
    nest_vmap = ops.vmap(ops.vmap(tensor_shape_forward_func, in_axes=in_axes, out_axes=0), in_axes=in_axes, out_axes=0)
    out = nest_vmap(x)
    expect = np.array([[[3., 2., 1.], [3., 2., 1.]],
                       [[3., 2., 1.], [3., 2., 1.]]]).astype(np.float32)
    assert (out.asnumpy() == expect).all()
