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


@ms.jit
def onehot_forward_func(indices, depth, on_value, off_value, axis):
    return ops.auto_generate.one_hot_(indices, depth, on_value, off_value, axis)


@ms.jit
def onehot_backward_func(indices, depth, on_value, off_value, axis):
    return ops.grad(onehot_forward_func, (0, 2, 3))(indices, depth, on_value, off_value, axis)


@ms.jit
def onehot_vmap_func(indices, depth, on_value, off_value, axis):
    in_axis = (-1, None, None, None, None)
    return ops.vmap(onehot_forward_func, in_axes=in_axis, out_axes=0)(indices, depth, on_value, off_value, axis)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE])
@pytest.mark.parametrize("data_type", [np.int32])
def test_onehot_op_forward(context_mode, data_type):
    """
    Feature: Ops.
    Description: test op onehot forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    indices = ms.Tensor(np.array([0, 1, 2]).astype(data_type))
    depth = 3
    on_value = ms.Tensor(1.0, ms.float32)
    off_value = ms.Tensor(0.0, ms.float32)
    out = onehot_forward_func(indices, depth, on_value, off_value, axis=-1)
    expect_out = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]).astype(np.float32)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-3)
    print("out:", out)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE])
@pytest.mark.parametrize("data_type", [np.int32])
def test_onehot_op_backward(context_mode, data_type):
    """
    Feature: Auto grad.
    Description: test auto grad of op onehot.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    indices = ms.Tensor(np.array([0, 1, 2]).astype(data_type))
    depth = 3
    on_value = ms.Tensor(1.0, ms.float32)
    off_value = ms.Tensor(0.0, ms.float32)
    grads = onehot_backward_func(indices, depth, on_value, off_value, axis=-1)
    expect_out = np.array([0., 0., 0.]).astype(np.float32)
    np.testing.assert_allclose(grads[0].asnumpy(), expect_out, rtol=1e-3)
    print("grads:", grads)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE])
@pytest.mark.parametrize("data_type", [np.int32])
def test_onehot_op_vmap(context_mode, data_type):
    """
    Feature: test vmap function.
    Description: test onehot op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    indices = ms.Tensor(np.array([0, 1, 2]).astype(data_type))
    depth = 3
    on_value = ms.Tensor(1.0, ms.float32)
    off_value = ms.Tensor(0.0, ms.float32)
    out = onehot_vmap_func(indices, depth, on_value, off_value, axis=-1)
    expect_out = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]).astype(np.float32)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-3)
    print("vmap:", out)
