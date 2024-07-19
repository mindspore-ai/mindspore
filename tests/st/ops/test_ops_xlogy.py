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


"""test xlogy"""
import numpy as np
import pytest
import mindspore.common.dtype as mstype
import mindspore as ms
from mindspore.mint import xlogy
from mindspore import ops, Tensor
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.st.utils import test_utils


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x, y):
    return x * np.log(y)

def reshape_grad(x, grad_x):
    if x.shape == grad_x.shape:
        return grad_x
    axis = ()
    lenx = len(grad_x.shape) - len(x.shape)
    for dim in range(0, lenx):
        axis = axis + (dim,)
    for dim in range(lenx, len(grad_x.shape)):
        if x.shape[dim - lenx] == 1:
            axis = axis + (dim,)
    res = np.sum(grad_x, axis, keepdims=True)
    res.reshape(x.shape)
    return res

def generate_expect_backward_output(x, y):
    if isinstance(x, np.ndarray):
        a = np.not_equal(x, 0.0)
        a = a.astype(x.dtype)
        a = a * np.log(y)
        a = reshape_grad(x, a)
    else:
        a = 0
    if isinstance(y, np.ndarray):
        b = np.divide(x, y)
        b = b.reshape(b.shape)
        b = reshape_grad(y, b)
    else:
        b = 0
    return a, b


@test_utils.run_with_cell
def xlogy_forward_func(x, y):
    return xlogy(x, y)


@test_utils.run_with_cell
def xlogy_backward_func(x, y):
    return ops.grad(xlogy_forward_func, (0, 1))(x, y)


@test_utils.run_with_cell
def xlogy_vmap_func(x, y, in_axes=0):
    return ops.vmap(xlogy_forward_func, in_axes, out_axes=0)(x, y)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_ops_xlogy(mode):
    """
    Feature: pyboost function.
    Description: test function xlogy.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 3, 4), np.float32)
    y = np.abs(generate_random_input((2, 3, 1), np.float32)) + 0.01  # y > 0

    expect_out = generate_expect_forward_output(x, y)
    expect_out2 = generate_expect_forward_output(x, 2)
    expect_out3 = generate_expect_forward_output(2, y)

    expect_doutx, expect_douty = generate_expect_backward_output(x, y)
    expect_doutx2, _ = generate_expect_backward_output(x, 2)
    _, expect_douty3 = generate_expect_backward_output(2, y)

    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        out = xlogy_forward_func(Tensor(x), Tensor(y))
        out2 = xlogy_forward_func(Tensor(x), 2)
        out3 = xlogy_forward_func(2, Tensor(y))
        doutx, douty = xlogy_backward_func(Tensor(x), Tensor(y))
        doutx2 = xlogy_backward_func(Tensor(x), 2)
        douty3 = xlogy_backward_func(2, Tensor(y))
    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE)
        op = ms.jit(xlogy_forward_func, jit_config=ms.JitConfig(jit_level="O0"))
        out = op(Tensor(x), Tensor(y))
        out2 = op(Tensor(x), 2)
        out3 = op(2, Tensor(y))
        op = ms.jit(xlogy_backward_func, jit_config=ms.JitConfig(jit_level="O0"))
        doutx, douty = op(Tensor(x), Tensor(y))
        doutx2 = op(Tensor(x), 2)
        douty3 = op(2, Tensor(y))
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE)
        out = xlogy_forward_func(Tensor(x), Tensor(y))
        out2 = xlogy_forward_func(Tensor(x), 2)
        out3 = xlogy_forward_func(2, Tensor(y))
        doutx, douty = xlogy_backward_func(Tensor(x), Tensor(y))
        doutx2 = xlogy_backward_func(Tensor(x), 2)
        douty3 = xlogy_backward_func(2, Tensor(y))

    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-3)
    np.testing.assert_allclose(out2.asnumpy(), expect_out2, rtol=1e-3)
    np.testing.assert_allclose(out3.asnumpy(), expect_out3, rtol=1e-3)
    np.testing.assert_allclose(doutx.asnumpy(), expect_doutx, rtol=1e-3)
    np.testing.assert_allclose(douty.asnumpy(), expect_douty, rtol=1e-3)
    np.testing.assert_allclose(doutx2.asnumpy(), expect_doutx2, rtol=1e-3)
    np.testing.assert_allclose(douty3.asnumpy(), expect_douty3, rtol=1e-3)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", ["pynative", "KBK"])
def test_ops_xlogy_forward_910b_nan(mode):
    """
    Feature: pyboost function.
    Description: test function xlogy forward.
    Expectation: expect correct result.
    """
    x = Tensor([[[-1.1, 7.4], [2.2, 0], [3.3, -2]], [[0, -8.3], [-2.3, 2], [4.5, 0]]], mstype.float32)

    y = Tensor(np.array([[[np.nan], [0], [3.2]], [[np.inf], [-2.0], [3]]]), mstype.float32)

    expect_out = np.array([[[np.nan, np.nan], [-np.inf, 0.0], [3.8384, -2.3263]],
                           [[0.0, -np.inf], [np.nan, np.nan], [4.9438, 0.0]]])


    if mode == "pynative":
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        out = xlogy_forward_func(x, y)

    elif mode == "KBK":
        ms.context.set_context(mode=ms.GRAPH_MODE)
        op = ms.jit(xlogy_forward_func, jit_config=ms.JitConfig(jit_level="O0"))
        out = op(x, y)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE)
        out = xlogy_forward_func(x, y)

    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_xlogy_dynamic_shape():
    """
    Feature: pyboost function.
    Description: test function xlogy forward with dynamic shape.
    Expectation: expect correct result.
    """
    x1 = Tensor(generate_random_input((2, 3, 4), np.float32))
    y1 = Tensor(np.abs(generate_random_input((2, 3, 1), np.float32)) + 0.01)

    x2 = Tensor(generate_random_input((5, 2, 3, 4), np.float32))
    y2 = Tensor(np.abs(generate_random_input((5, 2, 3, 1), np.float32)) + 0.01)

    test_cell = test_utils.to_cell_obj(xlogy_forward_func)
    TEST_OP(test_cell, [[x1, y1], [x2, y2]], '', disable_yaml_check=True, disable_mode=['GRAPH_MODE'])
