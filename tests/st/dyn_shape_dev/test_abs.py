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
from mindspore import Tensor, context
from mindspore import ops

from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP

def get_inputs_np(shapes, dtypes):
    np.random.seed(10)
    inputs_np = []
    for shape, dtype in zip(shapes, dtypes):
        inputs_np.append(np.random.randn(*shape).astype(dtype))
    return inputs_np


def get_inputs_tensor(inputs_np):
    inputs = []
    for input_np in inputs_np:
        inputs.append(Tensor(input_np))
    return inputs


def compare(output, expect):
    if isinstance(output, (tuple, list)):
        for o_ele, e_ele in zip(output, expect):
            compare(o_ele, e_ele)
    else:
        if expect.dtype == np.float32:
            rtol, atol = 1e-4, 1e-4
        else:
            rtol, atol = 1e-3, 1e-3
        assert np.allclose(output.asnumpy(), expect, rtol, atol)

def eltwise_case(prim_func, expect_func, expect_grad_func, mode, inputs_np=None):
    if inputs_np is None:
        input_shape = (2, 4, 8, 16)
        input_dtype = np.float32
        inputs_np = get_inputs_np([input_shape], [input_dtype])
    context.set_context(mode=mode)
    # inputs data
    input_args = get_inputs_tensor(inputs_np)
    # forward:
    print("start forward test:", flush=True)
    output = prim_func(*input_args)
    expect_output = expect_func(*inputs_np)
    compare(output, expect_output)
    # backward:
    print("start backward test:", flush=True)
    output_grad = ops.grad(prim_func)(*input_args)
    expect_grad = expect_grad_func(inputs_np, expect_output)
    compare(output_grad, expect_grad)


def eltwise_case_vmap(prim_func, mode, inputs_np=None):
    if inputs_np is None:
        input_shape = (2, 4, 8, 16)
        input_dtype = np.float32
        inputs_np = get_inputs_np([input_shape], [input_dtype])
    context.set_context(mode=mode)
    x_batched = inputs_np[0]
    input_args_batched = get_inputs_tensor([x_batched])
    output_vmap = ops.vmap(prim_func, in_axes=0)(*input_args_batched)
    output_batched = []
    for x in x_batched:
        input_args = get_inputs_tensor([x])
        output_batched.append(prim_func(*input_args).asnumpy())
    expect = np.stack(output_batched)
    assert np.allclose(output_vmap.asnumpy(), expect)


def abs_func(x):
    return ops.auto_generate.abs(x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_abs(mode):
    """
    Feature: Test abs op.
    Description: Test abs.
    Expectation: the result match with expected result.
    """

    def abs_grad(inputs_np, _):
        input_data = inputs_np[0]
        expect_grads = np.zeros_like(input_data)
        expect_grads[input_data > 0] = 1
        expect_grads[input_data < 0] = -1
        return expect_grads

    prim_func = abs_func
    expect_func = np.abs
    expect_grad_func = abs_grad
    eltwise_case(prim_func, expect_func, expect_grad_func, mode)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_abs_vmap(mode):
    """
    Feature: Test abs op.
    Description: Test abs vmap.
    Expectation: the result match with expected result.
    """
    eltwise_case_vmap(abs_func, mode)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_abs_dyn(mode):
    """
    Feature: Test abs op.
    Description: Test abs dynamic shape.
    Expectation: the result match with expected result.
    """
    input_case1 = get_inputs_tensor(get_inputs_np([(2, 4, 8)], [np.float32]))
    input_case2 = get_inputs_tensor(get_inputs_np([(2, 4, 8, 16)], [np.float32]))
    TEST_OP(abs_func, [input_case1, input_case2], mode=mode)
