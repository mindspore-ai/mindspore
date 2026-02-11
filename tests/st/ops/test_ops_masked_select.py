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
from mindspore import Tensor, ops, jit
import tests.st.utils.test_utils as test_utils
from tests.st.ops.test_tools.test_op import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def masked_select_forward_func(input1, mask):
    return masked_select(input1, mask)


def masked_select_backward_func(input1, mask):
    return ops.grad(masked_select_forward_func, (0, 1))(input1, mask)  # pylint: disable=not-callable


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64,
                                   np.uint8, np.float64, np.float32, np.float16])
def test_ops_masked_select(mode, dtype):
    """
    Feature: pyboost function.
    Description: test function masked_select backward.
    Expectation: expect correct result.
    """
    input1 = Tensor(np.array([[1, 2, 3], [3, 4, 5], [6, 7, 8], [3, 6, 9]]).astype(dtype))
    mask = Tensor(np.array([[True], [False], [True], [False]]).astype(np.bool_))
    expect_value = np.asarray([1, 2, 3, 6, 7, 8]).astype(dtype)
    expect_input_grad = np.asarray([[1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0]]).astype(dtype)
    expect_mask_grad = np.asarray([[False], [False], [False], [False]]).astype(np.bool_)
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = masked_select_forward_func(input1, mask)
        output_grad = masked_select_backward_func(input1, mask)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE)
        output = (jit(masked_select_forward_func, backend="ms_backend", jit_level="O0"))(input1, mask)
        output_grad = (jit(masked_select_backward_func, backend="ms_backend", jit_level="O0"))(input1, mask)
    np.testing.assert_allclose(output.asnumpy(), expect_value, rtol=1e-3)
    np.testing.assert_allclose(output_grad[0].asnumpy(), expect_input_grad, rtol=1e-3)
    np.testing.assert_allclose(output_grad[1].asnumpy(), expect_mask_grad, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
def test_ops_masked_select_forward_dynamic():
    """
    Feature: pyboost function.
    Description: test function masked_select forward with dynamic shape.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)

    input1 = ms.Tensor(generate_random_input((7, 8, 9), np.float32))
    mask1 = ms.Tensor(generate_random_input((7, 8, 9), np.bool_))

    input2 = ms.Tensor(generate_random_input((3, 4, 5), np.float32))
    mask2 = ms.Tensor(generate_random_input((4, 5), np.bool_))

    test_cell = test_utils.to_cell_obj(masked_select_forward_func)
    TEST_OP(test_cell, [[input1, mask1], [input2, mask2]],
            case_config={'disable_input_check': True,
                         'all_dim_zero': True})
