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
from mindspore.mint import masked_select
from mindspore import Tensor, ops, jit, JitConfig
import tests.st.utils.test_utils as test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


def masked_select_forward_func(input1, mask):
    return masked_select(input1, mask)


def masked_select_backward_func(input1, mask):
    return ops.grad(masked_select_forward_func, (0, 1))(input1, mask)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64,
                                   np.uint8, np.float64, np.float32, np.float16])
def test_ops_masked_select_forward(mode, dtype):
    """
    Feature: pyboost function.
    Description: test function masked_select forward.
    Expectation: expect correct result.
    """
    input1 = Tensor(np.array([[1, 2, 3], [1, 2, 3]]).astype(dtype))
    mask = Tensor(np.array([[True, False, True], [False, True, False]]).astype(np.bool_))
    if mode == 'pynative':
        output = masked_select_forward_func(input1, mask)
    elif mode == 'KBK':
        output = (jit(masked_select_forward_func, jit_config=JitConfig(jit_level="O0")))(input1, mask)
    expect_values = np.asarray([1, 3, 2]).astype(dtype)
    np.testing.assert_allclose(output.asnumpy(), expect_values, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64,
                                   np.uint8, np.float64, np.float32, np.float16])
def test_ops_masked_select_backward(mode, dtype):
    """
    Feature: pyboost function.
    Description: test function masked_select backward.
    Expectation: expect correct result.
    """
    input1 = Tensor(np.array([[1, 2, 3], [3, 4, 5], [6, 7, 8], [3, 6, 9]]).astype(dtype))
    mask = Tensor(np.array([[True], [False], [True], [False]]).astype(np.bool_))
    if mode == 'pynative':
        output = masked_select_backward_func(input1, mask)
    elif mode == 'KBK':
        output = (jit(masked_select_backward_func, jit_config=JitConfig(jit_level="O0")))(input1, mask)
    expect_input_grad = np.asarray([[1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0]]).astype(dtype)
    expect_mask_grad = np.asarray([[False], [False], [False], [False]]).astype(np.bool_)
    np.testing.assert_allclose(output[0].asnumpy(), expect_input_grad, rtol=1e-3)
    np.testing.assert_allclose(output[1].asnumpy(), expect_mask_grad, rtol=1e-3)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('context_mode', [ms.PYNATIVE_MODE])
def test_ops_masked_select_forward_dynamic(context_mode):
    """
    Feature: pyboost function.
    Description: test function masked_select forward with dynamic shape.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    test_cell = test_utils.to_cell_obj(masked_select_forward_func)
    input1 = Tensor(np.array([[-0.3956, 1.1455, 1.6895],
                              [-0.5849, 1.3672, 0.3599],
                              [-1.1626, 0.7180, -0.0521],
                              [-0.1339, 0.9902, -2.0225]]).astype(np.float32))
    mask1 = Tensor(np.array([[False, True, True],
                             [True, False, True],
                             [True, False, False],
                             [False, True, False]]).astype(np.bool_))
    input2 = Tensor(np.array([[-0.3956, 1.1455, 1.6895],
                              [-0.5849, 1.3672, 0.3599],
                              [-1.1626, 0.7180, -0.0521],
                              [-0.1339, 0.9902, -2.0225]]).astype(np.float32))
    mask2 = Tensor(np.array([[False], [False], [True], [True]]).astype(np.bool_))
    TEST_OP(test_cell, [[input1, mask1], [input2, mask2]], grad=True, jit_level="O0")
