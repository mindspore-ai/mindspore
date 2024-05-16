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
def test_remainder_forward(context_mode):
    """
    Feature: remainder_ext ops forward.
    Description: test remainder_ext forward in Pynative mode and Graph KBK mode with valid inputs.
    Expectation: return correct results.
    """
    set_context(mode=context_mode)
    if context_mode == ms.GRAPH_MODE:
        op_func = build_context_func(remainder_forward, mode="KBK")
    else:
        op_func = build_context_func(remainder_forward, mode="Pynative")
    check_func = partial(check_forward, op_func)
    run_cases('TensorTensor', fn=check_func, expect_out_key='forward_out')
    run_cases('TensorScalar', fn=check_func, expect_out_key='forward_out')
    run_cases('ScalarTensor', fn=check_func, expect_out_key='forward_out')


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.parametrize("context_mode", [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_remainder_backward(context_mode):
    """
    Feature: remainder_ext ops backward.
    Description: test remainder_ext backward in Pynative mode and Graph KBK mode with valid inputs.
    Expectation: output right results.
    """
    if context_mode == ms.GRAPH_MODE:
        op_func = build_context_func(remainder_backward, mode="KBK")
    else:
        op_func = build_context_func(remainder_backward, mode="Pynative")
    run_cases('TensorTensor', fn=partial(check_backward_tensor_tensor, op_func), expect_out_key='backward_out')
    run_cases('TensorScalar', fn=partial(check_backward, op_func), expect_out_key='backward_out')
