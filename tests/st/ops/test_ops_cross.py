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
from mindspore import Tensor, ops, jit, JitConfig
import tests.st.utils.test_utils as test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


def cross_forward_func(input1, other, dim=None):
    return cross(input1, other, dim)


def cross_backward_func(input1, other, dim=None):
    return ops.grad(cross_forward_func, (0, 1))(input1, other, dim)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64, np.uint8, np.float64,
                                   np.float32, np.float16, np.complex64, np.complex128])
def test_ops_cross_forward(mode, dtype):
    """
    Feature: pyboost function.
    Description: test function cross forward.
    Expectation: expect correct result.
    """
    input1 = Tensor(np.array([[1, 2, 3], [1, 2, 3]]).astype(dtype))
    other = Tensor(np.array([[4, 5, 6], [4, 5, 6]]).astype(dtype))
    if mode == 'pynative':
        output = cross_forward_func(input1, other)
    elif mode == 'KBK':
        output = (jit(cross_forward_func, jit_config=JitConfig(jit_level="O0")))(input1, other)
    expect_values = np.asarray([[-3, 6, -3], [-3, 6, -3]]).astype(dtype)
    np.testing.assert_allclose(output.asnumpy(), expect_values, rtol=1e-3)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64, np.float64,
                                   np.float32, np.float16, np.complex128])
@pytest.mark.parametrize("dim_value", [-1, 1])
def test_ops_cross_backward(mode, dtype, dim_value):
    """
    Feature: pyboost function.
    Description: test function cross backward.
    Expectation: expect correct result.
    """
    input1 = Tensor(np.array([[1, 2, 3], [3, 4, 5], [6, 7, 8], [3, 6, 9]]).astype(dtype))
    other = Tensor(np.array([[9, 7, 5], [3, 2, 1], [4, 6, 8], [2, 7, 5]]).astype(dtype))
    if mode == 'pynative':
        output = cross_backward_func(input1, other, dim=1)
    elif mode == 'KBK':
        output = (jit(cross_backward_func, jit_config=JitConfig(jit_level="O0")))(input1, other, dim=dim_value)
    expect_input_grad = np.asarray([[2, -4, 2], [1, -2, 1], [-2, 4, -2], [2, 3, -5]]).astype(dtype)
    expect_other_grad = np.asarray([[1, -2, 1], [1, -2, 1], [1, -2, 1], [3, -6, 3]]).astype(dtype)
    np.testing.assert_allclose(output[0].asnumpy(), expect_input_grad, rtol=1e-3)
    np.testing.assert_allclose(output[1].asnumpy(), expect_other_grad, rtol=1e-3)

@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_ops_cross_backward_uint8(mode):
    """
    Feature: pyboost function.
    Description: test function cross backward.
    Expectation: expect correct result.
    """
    input1 = Tensor(np.array([[1, 2, 3], [3, 4, 5], [6, 7, 8], [3, 6, 9]]).astype(np.uint8))
    other = Tensor(np.array([[9, 7, 5], [3, 2, 1], [4, 6, 8], [2, 7, 5]]).astype(np.uint8))
    if mode == 'pynative':
        output = cross_backward_func(input1, other, dim=1)
    elif mode == 'KBK':
        output = (jit(cross_backward_func, jit_config=JitConfig(jit_level="O0")))(input1, other, dim=1)
    expect_input_grad = np.asarray([[2, 252, 2], [1, 254, 1], [254, 4, 254], [2, 3, 251]]).astype(np.uint8)
    expect_other_grad = np.asarray([[1, 254, 1], [1, 254, 1], [1, 254, 1], [3, 250, 3]]).astype(np.uint8)
    np.testing.assert_allclose(output[0].asnumpy(), expect_input_grad, rtol=1e-3)
    np.testing.assert_allclose(output[1].asnumpy(), expect_other_grad, rtol=1e-3)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.PYNATIVE_MODE])
def test_ops_cross_forward_dynamic(context_mode):
    """
    Feature: pyboost function.
    Description: test function cross forward with dynamic shape.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    test_cell = test_utils.to_cell_obj(cross_forward_func)
    input1 = Tensor(np.array([[-0.3956, 1.1455, 1.6895],
                              [-0.5849, 1.3672, 0.3599],
                              [-1.1626, 0.7180, -0.0521],
                              [-0.1339, 0.9902, -2.0225]]).astype(np.float32))
    other = Tensor(np.array([[-0.0257, -1.4725, -1.2251],
                             [-1.1479, -0.7005, -1.9757],
                             [-1.3904, 0.3726, -1.1836],
                             [-0.9688, -0.7153, 0.2159]]).astype(np.float32))
    TEST_OP(test_cell, [[input1, other], [other, input1]], grad=True, jit_level="O0")
