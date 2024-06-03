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

import mindspore as ms
from mindspore import ops
from mindspore.ops.auto_generate import index_select_ext
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.random(shape).astype(dtype)


@test_utils.run_with_cell
def index_select_ext_forward_func(input_x, axis, index):
    return index_select_ext(input_x, axis, index)


@test_utils.run_with_cell
def index_select_ext_backward_func(input_x, axis, index):
    return ops.grad(index_select_ext_forward_func, (0))(input_x, axis, index)


def generate_expect_forward_output(input_np, axis, index_np):
    if axis < 0:
        axis += input_np.ndim
    return input_np[(slice(None),) * axis + (index_np,)]


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_func_index_select_ext_normal(context_mode):
    """
    Feature: pyboost function.
    Description: test function index_select_ext normal.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    if context_mode == ms.GRAPH_MODE:
        ms.set_context(jit_level='O0')

    input_np = generate_random_input((2, 3, 3, 4), np.float32)
    axis = -1
    index_np = np.array([0, 1, 2, 3, 2, 1]).astype(np.int64)
    output = index_select_ext_forward_func(ms.Tensor(input_np), axis, ms.Tensor(index_np))
    expect = generate_expect_forward_output(input_np, axis, index_np)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    input_np1 = np.arange(2 * 2 * 3).reshape(2, 2, 3).astype(np.float32)
    axis1 = -1
    index_np1 = np.array([0, 1, 2, 2, 1]).astype(np.int64)
    grad = index_select_ext_backward_func(ms.Tensor(input_np1), axis1, ms.Tensor(index_np1))
    expect1 = np.array([[[1, 2, 2], [1, 2, 2]], [[1, 2, 2], [1, 2, 2]]]).astype(np.float32)
    np.testing.assert_allclose(grad.asnumpy(), expect1, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_arm_ascend910b_training
def test_func_index_select_ext_dynamic():
    """
    Feature: pyboost function.
    Description: test function dropout dynamic.
    Expectation: expect correct result.
    """
    input1 = ms.Tensor(generate_random_input((2, 2, 3, 4), np.float32))
    axis1 = 2
    index1 = ms.Tensor(np.array([0, 1, 2, 1, 2]).astype(np.int64))
    input2 = ms.Tensor(generate_random_input((3, 4, 5), np.float32))
    axis2 = 2
    index2 = ms.Tensor(np.array([0, 1, 2, 1, 2, 3]).astype(np.int64))

    TEST_OP(index_select_ext, [[input1, axis1, index1], [input2, axis2, index2]], 'index_select',
            disable_input_check=True, disable_mode=['GRAPH_MODE'])
