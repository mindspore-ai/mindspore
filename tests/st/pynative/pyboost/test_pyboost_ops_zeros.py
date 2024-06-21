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
import mindspore as ms
from mindspore import ops
from mindspore import Tensor, context
from mindspore.ops.auto_generate.gen_ops_def import zeros
import test_utils


@test_utils.run_with_cell
def zeros_forward_func(size, dtype):
    return zeros(size, dtype)


@test_utils.run_with_cell
def zeros_backward_func(size, dtype):
    return ops.grad(zeros_forward_func, (0,))(size, dtype)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_zeros_tensor_api(mode):
    """
    Feature: test zeros operator
    Description: test zeros forward tensor api
    Expectation: success
    """
    context.set_context(mode=mode)
    x = Tensor([1], ms.int32)
    size = (2, 3)
    zeros_output = x.new_zeros(size, ms.float32)
    expect_output = np.zeros(size, np.float32)
    assert np.allclose(zeros_output.asnumpy(), expect_output)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_zeros_forward(mode):
    """
    Feature: test zeros operator
    Description: test zeros forward with pyboost
    Expectation: success
    """
    context.set_context(mode=mode)
    size = (2, 3)
    zeros_output = zeros_forward_func(size, ms.float32)
    expect_output = np.zeros(size, np.float32)
    assert np.allclose(zeros_output.asnumpy(), expect_output)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_zeros_backward(mode):
    """
    Feature: test zeros operator
    Description: test zeros backward
    Expectation: success
    """
    context.set_context(mode=mode)
    size = Tensor(np.array([2, 3]).astype(np.int64))
    zeros_output = zeros_backward_func(size, ms.float64)
    expect_output = [0, 0]
    assert np.allclose(zeros_output.asnumpy(), expect_output)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_func
def test_zeros_forward_dynamic_shape(context_mode):
    """
    Feature: zeros ops.
    Description: test ops zeros with dynamic shape tensor input.
    Expectation: output the right result.
    """
    context.set_context(mode=context_mode)
    size_dyn = Tensor(shape=[None], dtype=ms.int64)
    test_cell = test_utils.to_cell_obj(zeros)
    test_cell.set_inputs(size_dyn, ms.int32)
    size = Tensor(np.array([2, 3]).astype(np.int64))
    out = test_cell(size, ms.int32)
    expect_output = np.zeros(size, np.int32)
    assert np.allclose(out.asnumpy(), expect_output)

    size = Tensor(np.array([3, 4]).astype(np.int64))
    out = test_cell(size, ms.int32)
    expect_output = np.zeros(size, np.int32)
    assert np.allclose(out.asnumpy(), expect_output)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_func
def test_zeros_forward_dynamic_rank(context_mode):
    """
    Feature: zeros ops.
    Description: test ops zeros with dynamic shape tensor input.
    Expectation: output the right result.
    """
    context.set_context(mode=context_mode)
    size_dyn = Tensor(shape=None, dtype=ms.int64)
    test_cell = test_utils.to_cell_obj(zeros)
    test_cell.set_inputs(size_dyn, ms.int32)
    size = Tensor(np.array([2, 3]).astype(np.int64))
    out = test_cell(size, ms.int32)
    expect_output = np.zeros(size, np.int32)
    assert np.allclose(out.asnumpy(), expect_output)

    with pytest.raises((TypeError, ValueError)):
        size = Tensor(np.array([[2, 3], [4, 5]]).astype(np.int64))
        _ = test_cell(size, ms.int32)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_func
def test_zeros_backward_dynamic_shape(context_mode):
    """
    Feature: zeros ops.
    Description: test ops zeros with dynamic shape tensor input.
    Expectation: output the right result.
    """
    context.set_context(mode=context_mode)
    size_dyn = Tensor(shape=[None], dtype=ms.int64)
    test_cell = test_utils.to_cell_obj(ops.grad(zeros, (0,)))
    test_cell.set_inputs(size_dyn, ms.int32)
    size = Tensor(np.array([2, 3]).astype(np.int64))
    out = test_cell(size, ms.int32)
    expect_output = [0, 0]
    assert np.allclose(out.asnumpy(), expect_output)

    size = Tensor(np.array([2, 3, 4]).astype(np.int64))
    out = test_cell(size, ms.int32)
    expect_output = [0, 0, 0]
    assert np.allclose(out.asnumpy(), expect_output)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_func
def test_zeros_backward_dynamic_rank(context_mode):
    """
    Feature: zeros ops.
    Description: test ops zeros with dynamic shape tensor input.
    Expectation: output the right result.
    """
    context.set_context(mode=context_mode)
    size_dyn = Tensor(shape=None, dtype=ms.int64)
    test_cell = test_utils.to_cell_obj(ops.grad(zeros, (0,)))
    test_cell.set_inputs(size_dyn, ms.int32)
    size = Tensor(np.array([2, 3]).astype(np.int64))
    out = test_cell(size, ms.int32)
    expect_output = [0, 0]
    assert np.allclose(out.asnumpy(), expect_output)

    with pytest.raises((TypeError, ValueError)):
        size = Tensor(np.array([[2, 3], [4, 5]]).astype(np.int64))
        _ = test_cell(size, ms.int32)
