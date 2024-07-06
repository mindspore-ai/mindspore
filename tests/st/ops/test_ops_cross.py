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
import pytest
import numpy as np
import mindspore as ms
from mindspore.mint import cross
from mindspore import Tensor, ops
import tests.st.utils.test_utils as test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def cross_forward_func(input1, other, dim=None):
    return cross(input1, other, dim)


def cross_backward_func(input1, other, dim=None):
    return ops.grad(cross_forward_func, (0, 1))(input1, other, dim)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', ['pynative', 'KBK', "graph"])
@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64, np.uint8, np.float64,
                                   np.float32, np.float16, np.complex128])
@pytest.mark.parametrize("dim_value", [-1, 1])
def test_ops_cross(mode, dtype, dim_value):
    """
    Feature: pyboost function.
    Description: test function cross forward.
    Expectation: expect correct result.
    """
    input1 = Tensor(np.array([[1, 2, 3], [3, 4, 5], [6, 7, 8], [3, 6, 9]]).astype(dtype))
    other = Tensor(np.array([[9, 7, 5], [3, 2, 1], [4, 6, 8], [2, 7, 5]]).astype(dtype))
    expect_forward = np.asarray([[-11, 22, -11], [-6, 12, -6], [8, -16, 8], [-33, 3, 9]]).astype(dtype)
    expect_grad_input = np.asarray([[2, -4, 2], [1, -2, 1], [-2, 4, -2], [2, 3, -5]]).astype(dtype)
    expect_grad_other = np.asarray([[1, -2, 1], [1, -2, 1], [1, -2, 1], [3, -6, 3]]).astype(dtype)
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = cross_forward_func(input1, other, dim_value)
        output_grad = cross_backward_func(input1, other, dim_value)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE)
        op_froward = ms.jit(cross_forward_func, jit_config=ms.JitConfig(jit_level="O0"))
        output = op_froward(input1, other, dim_value)
        op_backward = ms.jit(cross_backward_func, jit_config=ms.JitConfig(jit_level="O0"))
        output_grad = op_backward(input1, other, dim_value)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE)
        op_froward = ms.jit(cross_forward_func, jit_config=ms.JitConfig(jit_level="O2"))
        output = op_froward(input1, other, dim_value)
        op_backward = ms.jit(cross_backward_func, jit_config=ms.JitConfig(jit_level="O2"))
        output_grad = op_backward(input1, other, dim_value)
    np.testing.assert_allclose(output.asnumpy(), expect_forward, rtol=1e-3)
    np.testing.assert_allclose(output_grad[0].asnumpy(), expect_grad_input, rtol=1e-3)
    np.testing.assert_allclose(output_grad[1].asnumpy(), expect_grad_other, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
def test_ops_cross_forward_dynamic():
    """
    Feature: pyboost function.
    Description: test function cross forward with dynamic shape.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    test_cell = test_utils.to_cell_obj(cross_forward_func)
    input1 = Tensor(generate_random_input((3, 4, 5), np.float32))
    other1 = Tensor(generate_random_input((3, 4, 5), np.float32))

    input2 = Tensor(generate_random_input((2, 3, 5, 4), np.float32))
    other2 = Tensor(generate_random_input((2, 3, 5, 4), np.float32))

    TEST_OP(test_cell, [[input1, other1, 0], [input2, other2, -3]], "cross", disable_mode=["GRAPH_MODE"])
