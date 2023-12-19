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
import pytest
import numpy as np
import test_utils

from mindspore import Tensor, context
from mindspore import ops
from mindspore import dtype as mstype


@test_utils.run_with_cell
def softmax_forward_func(x):
    return ops.auto_generate.softmax_(x, axis=0)


@test_utils.run_with_cell
def softmax_backward_func(x):
    return ops.grad(softmax_forward_func, (0,))(x)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_softmax_op(mode):
    """
    Feature: Ops
    Description: test op softmax
    Expectation: expect correct result.
    """
    x = Tensor(np.random.rand(10, 36, 12, 12).astype(np.float32))
    context.set_context(mode=mode)
    out = softmax_forward_func(x)
    expect_shape = (10, 36, 12, 12)
    assert out.asnumpy().shape == expect_shape
    logits = Tensor(np.array([1, 2, 3, 4, 5]), mstype.float32)
    output = softmax_forward_func(logits)
    expect_out = np.array([0.01165623, 0.03168492, 0.08612854, 0.23412167, 0.6364086]).astype(np.float32)
    assert np.allclose(output.asnumpy(), expect_out, 1e-04, 1e-04)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_softmax_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op softmax pool.
    Expectation: expect correct result.
    """
    x = Tensor(np.random.rand(10, 36, 12, 12).astype(np.float32))
    context.set_context(mode=mode)
    grads = softmax_backward_func(x)
    expect_shape = (10, 36, 12, 12)
    assert grads.asnumpy().shape == expect_shape


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_sofmax_vmap(mode):
    """
    Feature: test vmap function.
    Description: test softmax op vmap.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)
    in_axes = -1
    x = Tensor(np.random.randn(1, 6, 6, 3, 6).astype(np.float32))
    nest_vmap = ops.vmap(ops.vmap(softmax_forward_func,
                                  in_axes=in_axes,
                                  out_axes=0),
                         in_axes=in_axes,
                         out_axes=0)
    out = nest_vmap(x)
    expect_shape = (6, 3, 1, 6, 6)
    assert out.asnumpy().shape == expect_shape


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_softmax_dynamic_shape(mode):
    """
    Feature: Ops
    Description: test op softmax dynamic shape
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)
    x_dyn = Tensor(shape=[None, None, None, None], dtype=mstype.float32)
    test_cell = test_utils.to_cell_obj(ops.auto_generate.softmax_)
    test_cell.set_inputs(x_dyn)
    x_1 = Tensor(np.random.rand(10, 36, 12, 12).astype(np.float32))
    out_1 = test_cell(x_1)
    expect_shape_1 = (10, 36, 12, 12)
    assert out_1.asnumpy().shape == expect_shape_1
    x_2 = Tensor(np.random.rand(6, 20, 10, 10).astype(np.float32))
    out_2 = test_cell(x_2)
    expect_shape_2 = (6, 20, 10, 10)
    assert out_2.asnumpy().shape == expect_shape_2


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
# @pytest.mark.platform_arm_ascend_training 动态rank ge存在缺陷
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_softmax_dynamic_rank(mode):
    """
    Feature: Ops
    Description: test op softmax dynamic rank
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)
    x_dyn = Tensor(shape=None, dtype=mstype.float32)
    test_cell = test_utils.to_cell_obj(ops.auto_generate.softmax_)
    test_cell.set_inputs(x_dyn)
    x_1 = Tensor(np.random.rand(10, 36, 12).astype(np.float32))
    out_1 = test_cell(x_1)
    expect_shape_1 = (10, 36, 12)
    assert out_1.asnumpy().shape == expect_shape_1
    x_2 = Tensor(np.random.rand(6, 20, 10, 10).astype(np.float32))
    out_2 = test_cell(x_2)
    expect_shape_2 = (6, 20, 10, 10)
    assert out_2.asnumpy().shape == expect_shape_2


# 反向动态shape有公共问题，待解决后再放开用例
@pytest.mark.skip
@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_softmax_dynamic_backward_shape(mode):
    """
    Feature: Ops
    Description: test op softmax backward dynamic shape
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)
    x_dyn = Tensor(shape=[None, None, None, None], dtype=mstype.float32)
    test_cell = test_utils.to_cell_obj(softmax_backward_func)
    test_cell.set_inputs(x_dyn)
    x_1 = Tensor(np.random.rand(10, 36, 12, 12).astype(np.float32))
    out_1 = test_cell(x_1)
    expect_shape_1 = (10, 36, 12, 12)
    assert out_1.asnumpy().shape == expect_shape_1
    x_2 = Tensor(np.random.rand(6, 20, 10, 10).astype(np.float32))
    out_2 = test_cell(x_2)
    expect_shape_2 = (6, 20, 10, 10)
    assert out_2.asnumpy().shape == expect_shape_2


# 反向动态shape有公共问题，待解决后再放开用例
@pytest.mark.skip
@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_softmax_backward_dynamic_rank(mode):
    """
    Feature: Ops
    Description: test op softmax backward dynamic rank
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)
    x_dyn = Tensor(shape=None, dtype=mstype.float32)
    test_cell = test_utils.to_cell_obj(softmax_backward_func)
    test_cell.set_inputs(x_dyn)
    x_1 = Tensor(np.random.rand(10, 36, 12).astype(np.float32))
    out_1 = test_cell(x_1)
    expect_shape_1 = (10, 36, 12)
    assert out_1.asnumpy().shape == expect_shape_1
    x_2 = Tensor(np.random.rand(6, 20, 10, 10).astype(np.float32))
    out_2 = test_cell(x_2)
    expect_shape_2 = (6, 20, 10, 10)
    assert out_2.asnumpy().shape == expect_shape_2
