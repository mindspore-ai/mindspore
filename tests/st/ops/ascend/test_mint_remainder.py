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
from functools import partial
import random
import pytest

import mindspore as ms
from mindspore.mint import remainder
from mindspore import ops, set_context, Tensor, jit, JitConfig
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


def build_context_func(fn, mode):
    if mode == 'Pynative':
        return fn
    if mode == 'KBK':
        return jit(fn, jit_config=JitConfig(jit_level="O0"))
    # Graph Mode with GE
    return jit(fn, jit_config=JitConfig(jit_level="O2"))


def remainder_forward(x, y):
    return remainder(x, y)


def remainder_backward(x, y):
    return ops.grad(remainder, grad_position=(0, 1))(x, y)


def generate_random_array(shape, dtype, scale=10):
    return np.random.rand(*shape).astype(dtype) * scale


def generate_random_number(scale=10, offset=0.01):
    return np.random.random() * scale + offset


cases = {
    "TensorTensor": [
        {
            'input': np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
            'other': np.array([[1], [2]]),
            'forward_out': np.array([[0, 0, 0, 0], [1, 0, 1, 0]]),
            'backward_out': np.array([[-10], [-12]])
        },
        {
            'input': np.array([-3, -2, -1, 1, 2, 3]),
            'other': np.array([[1], [2]]),
            'forward_out': np.array([[0, 0, 0, 0, 0, 0], [1, 0, 1, 1, 0, 1]]),
            'backward_out': np.array([[0], [2]])
        },
        {
            'input': 5,
            'other': np.array([1, 2, 3, 4, 5]),
            'forward_out': np.array([0, 1, 2, 1, 0]),
            'backward_out': np.array([-5, -2, -1, -1, -1])
        }
    ],
    "TensorScalar": [
        {
            'input': np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
            'other': 3,
            'forward_out': np.array([[1, 2, 0, 1], [2, 0, 1, 2]]),
            'backward_out': np.array([[1, 1, 1, 1], [1, 1, 1, 1]])
        },
        {
            'input': np.array([-3, -2, -1, 1, 2, 3]),
            'other': 2,
            'forward_out': np.array([1, 0, 1, 1, 0, 1]),
            'backward_out': np.array([1, 1, 1, 1, 1, 1])
        },
        {
            'input': np.array([1, 2, 3, 4, 5]),
            'other': -1.5,
            'forward_out': np.array([-0.5, -1, 0, -0.5, -1]),
            'backward_out': np.array([1, 1, 1, 1, 1])
        }
    ],
    "ScalarTensor": [
        {
            'input': 3,
            'other': np.array([[1], [2]]),
            'forward_out': np.array([[0], [1]])
        },
        {
            'input': 10,
            'other': np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
            'forward_out': np.array([[0, 0, 1, 2], [0, 4, 3, 2]])
        },
        {
            'input': 5,
            'other': np.array([1, 2, 3, 4, 5]),
            'forward_out': np.array([0, 1, 2, 1, 0])
        }
    ]
}

dtypes = (ms.int32, ms.int64, ms.float16, ms.float32, ms.float64)  # ms.bfloat16


def check_forward(forward_func, x, y, expected):
    out = forward_func(x, y)
    assert np.allclose(out.asnumpy(), expected)


def check_backward(backward_func, x, y, expected, is_tensor_tensor=False):
    dout = backward_func(x, y)
    out = dout[1] if is_tensor_tensor else dout
    assert np.allclose(out.asnumpy(), expected)


def check_backward_tensor_tensor(backward_func, x, y, expected):
    check_backward(backward_func, x, y, expected, is_tensor_tensor=True)


def run_cases(op_type, fn, expect_out_key):
    for case in cases[op_type]:
        expected = case[expect_out_key]
        x, y = case['input'], case['other']
        if op_type == 'TensorTensor':
            x = Tensor(x, random.choice(dtypes))
            y = Tensor(y, random.choice(dtypes))
            fn(x, y, expected)
        elif op_type == 'TensorScalar':
            x = Tensor(x, random.choice(dtypes))
            fn(x, y, expected)
            fn(x, float(y), expected)
        elif op_type == 'ScalarTensor':
            y = Tensor(y, random.choice(dtypes))
            fn(x, y, expected)
            fn(float(x), y, expected)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.parametrize("context_mode", [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_remainder_normal(context_mode):
    """
    Feature: mint.remainder static shape.
    Description: test mint.remainder in Pynative / Graph KBK mode with valid inputs.
    Expectation: expect correct results.
    """
    set_context(mode=context_mode)
    if context_mode == ms.GRAPH_MODE:
        op_forward = build_context_func(remainder_forward, mode="KBK")
        op_backward = build_context_func(remainder_backward, mode="KBK")
    else:
        op_forward = build_context_func(remainder_forward, mode="Pynative")
        op_backward = build_context_func(remainder_backward, mode="Pynative")
    # test forward
    check_func_forward = partial(check_forward, op_forward)
    run_cases('TensorTensor', fn=check_func_forward, expect_out_key='forward_out')
    run_cases('TensorScalar', fn=check_func_forward, expect_out_key='forward_out')
    run_cases('ScalarTensor', fn=check_func_forward, expect_out_key='forward_out')
    # test backward
    run_cases('TensorTensor', fn=partial(check_backward_tensor_tensor, op_backward), expect_out_key='backward_out')
    run_cases('TensorScalar', fn=partial(check_backward, op_backward), expect_out_key='backward_out')


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.parametrize("context_mode", [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_remainder_dynamic(context_mode):
    """
    Feature: mint.remainder dynamic shape.
    Description: test mint.remainder with dynamic shape and dynamic rank in Pynative / Graph KBK mode.
    Expectation: test passed.
    """
    input_tensor_1 = Tensor(generate_random_array((2, 3, 4), np.float32))
    other_tensor_1 = Tensor(generate_random_array((2, 1, 4), np.float32))
    input_tensor_2 = Tensor(generate_random_array((3, 5), np.float32))
    other_tensor_2 = Tensor(generate_random_array((5,), np.float32))

    input_scalar_1 = generate_random_number(scale=10, offset=-5)
    other_scalar_1 = generate_random_number(scale=10, offset=0.01)
    input_scalar_2 = generate_random_number(scale=20, offset=-10)
    other_scalar_2 = generate_random_number(scale=20, offset=0.01)

    inputs_seq_tensor_tensor = [[input_tensor_1, other_tensor_1], [input_tensor_2, other_tensor_2]]
    inputs_seq_tensor_scalar = [[input_tensor_1, other_scalar_1], [input_tensor_2, other_scalar_2]]
    inputs_seq_scalar_tensor = [[input_scalar_1, other_tensor_1], [input_scalar_2, other_tensor_2]]

    TEST_OP(remainder, inputs_seq_tensor_tensor, 'remainder_tensor_tensor', disable_mode=['GRAPH_MODE'])
    TEST_OP(remainder, inputs_seq_tensor_scalar, 'remainder_tensor_scalar', disable_mode=['GRAPH_MODE'])
    TEST_OP(remainder, inputs_seq_scalar_tensor, 'remainder_scalar_tensor', disable_mode=['GRAPH_MODE'],
            disable_grad=True)
