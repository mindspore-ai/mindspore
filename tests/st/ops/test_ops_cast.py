# Copyright 2023 Huawei Technocasties Co., Ltd
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
from mindspore import ops
from mindspore.ops import cast
import mindspore.common.dtype as mstype
import mindspore as ms
import tests.st.utils.test_utils as test_utils


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


@test_utils.run_with_cell
def cast_forward_func(x, dtype):
    return cast(x, dtype)


@test_utils.run_with_cell
def cast_backward_func(x, dtype):
    return ops.grad(cast_forward_func, (0, 1))(x, dtype)


@test_utils.run_with_cell
def cast_vmap_func(x, dtype):
    return ops.vmap(cast_forward_func, in_axes=(0, None), out_axes=0)(x, dtype)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_cast_normal(context_mode):
    """
    Feature: pyboost function.
    Description: test function cast forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((1280, 1280), np.float16)
    dtype = mstype.float32
    output = cast_forward_func(ms.Tensor(x), dtype)
    assert output.asnumpy().dtype == 'float32'
    assert output.asnumpy().shape == (1280, 1280)

    x = generate_random_input((1280,), np.float32)
    dtype = mstype.float16
    output = cast_backward_func(ms.Tensor(x), dtype)
    assert output.asnumpy().dtype == 'float32'
    assert output.asnumpy().shape == (1280,)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_cast_bf16(context_mode):
    """
    Feature: pyboost function.
    Description: test function cast backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x1 = generate_random_input((3, 32, 1), np.float32)
    x2 = generate_random_input((3, 32, 1), np.float32)
    x = tuple((x1, x2))
    dtype = mstype.bfloat16
    output = cast_backward_func(ms.Tensor(x), dtype)
    assert output.asnumpy().dtype == 'float32'
    assert output.asnumpy().shape == (2, 3, 32, 1)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_cast_vmap(context_mode):
    """
    Feature: pyboost function.
    Description: test function cast vmap feature.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((640, 640), np.float32)
    dtype = mstype.float16
    output = cast_vmap_func(ms.Tensor(x), dtype)
    assert output.asnumpy().dtype == 'float16'
    assert output.asnumpy().shape == (640, 640)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_cast_forward_dynamic_shape(context_mode):
    """
    Feature: pyboost function.
    Description: test function cast forward with dynamic shape.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    dtype = mstype.float16
    test_cell = test_utils.to_cell_obj(cast_forward_func)
    test_cell.set_inputs(x_dyn, dtype)
    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(ms.Tensor(x1), dtype)
    assert output.asnumpy().dtype == 'float16'
    assert output.asnumpy().shape == (2, 3, 4, 5)
    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = test_cell(ms.Tensor(x2), dtype)
    assert output.asnumpy().dtype == 'float16'
    assert output.asnumpy().shape == (3, 4, 5, 6)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_cast_forward_dynamic_rank(context_mode):
    """
    Feature: pyboost function.
    Description: test function cast forward with dynamic rank.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float64)
    test_cell = test_utils.to_cell_obj(cast_forward_func)
    dtype = mstype.float32
    test_cell.set_inputs(x_dyn, dtype)
    x1 = generate_random_input((1280, 1280, 3, 3), np.float64)
    output = test_cell(ms.Tensor(x1), dtype)
    assert output.asnumpy().dtype == 'float32'
    assert output.asnumpy().shape == (1280, 1280, 3, 3)
    x2 = generate_random_input((320, 320), np.float64)
    output = test_cell(ms.Tensor(x2), dtype)
    assert output.asnumpy().dtype == 'float32'
    assert output.asnumpy().shape == (320, 320)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_cast_backward_dynamic_shape(context_mode):
    """
    Feature: pyboost function.
    Description: test function cast backward with dynamic shape.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    dtype = mstype.float16
    test_cell = test_utils.to_cell_obj(cast_backward_func)
    test_cell.set_inputs(x_dyn, dtype)
    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(ms.Tensor(x1), dtype)
    assert output.asnumpy().dtype == 'float32'
    assert output.asnumpy().shape == (2, 3, 4, 5)
    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = test_cell(ms.Tensor(x2), dtype)
    assert output.asnumpy().dtype == 'float32'
    assert output.asnumpy().shape == (3, 4, 5, 6)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_cast_backward_dynamic_rank(context_mode):
    """
    Feature: pyboost function.
    Description: test function cast backward with dynamic rank.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    dtype = mstype.float16
    test_cell = test_utils.to_cell_obj(cast_backward_func)
    test_cell.set_inputs(x_dyn, dtype)
    x1 = generate_random_input((1280, 1280, 3, 3), np.float32)
    output = test_cell(ms.Tensor(x1), dtype)
    assert output.asnumpy().dtype == 'float32'
    assert output.asnumpy().shape == (1280, 1280, 3, 3)
    x2 = generate_random_input((1280, 1280), np.float32)
    output = test_cell(ms.Tensor(x2), dtype)
    assert output.asnumpy().dtype == 'float32'
    assert output.asnumpy().shape == (1280, 1280)
