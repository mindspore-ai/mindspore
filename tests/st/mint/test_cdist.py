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
from mindspore import mint, Tensor, context
from mindspore.ops.functional import vmap
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.ops_binary_cases import ops_binary_cases, OpsBinaryCase
from tests.st.utils.test_utils import run_with_cell

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def _count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_me) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, \
        "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}". \
            format(data_expected[greater], data_me[greater], error[greater])


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if np.any(np.isnan(data_expected)):
        assert np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan)
    elif not np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan):
        _count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert True


@run_with_cell
def cdist_forward_func(x1, x2, p):
    return mint.cdist(x1, x2, p)


@run_with_cell
def cdist_forward_vmap_func(x1, x2):
    return mint.cdist(x1, x2)


@run_with_cell
def cdist_backward_func(x1, x2, p):
    grad_fn = ms.grad(cdist_forward_func, grad_position=(0, 1))
    return grad_fn(x1, x2, p)


def mint_cdist_binary_compare(input_binary_data, output_binary_data, loss, mode):
    if mode == "KBK":
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    elif mode == "GRAPH":
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O2")
    x1 = Tensor(input_binary_data[0])
    x2 = Tensor(input_binary_data[1])

    output = cdist_forward_func(x1, x2, 2.0)
    allclose_nparray(output.asnumpy(), output_binary_data[0], loss, loss)

    grads = cdist_backward_func(x1, x2, 2.0)
    allclose_nparray(grads[0].asnumpy(), output_binary_data[1], loss, loss)
    allclose_nparray(grads[1].asnumpy(), output_binary_data[2], loss, loss)


@ops_binary_cases(
    OpsBinaryCase(
        input_info=[
            ((10, 10, 10), np.float32),
            ((10, 10, 10), np.float32),
        ],
        output_info=[
            ((10, 10, 10), np.float32),
            ((10, 10, 10), np.float32),
            ((10, 10, 10), np.float32),
        ],
        extra_info="auto_drive",
    )
)
def mint_cdist_binary_case1(input_binary_data=None, output_binary_data=None, loss=1e-04, mode="pynative"):
    mint_cdist_binary_compare(input_binary_data, output_binary_data, loss, mode)


@arg_mark(
    plat_marks=[
        "platform_ascend",
        "cpu_linux",
        "cpu_windows",
        "cpu_macos",
        "platform_gpu",
    ],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
    mem_peak=20.00
)
@pytest.mark.parametrize("mode", ["KBK", "GRAPH"])
def test_mint_cdist_binary_cases(mode):
    """
    Feature: mint.cdist
    Description: Verify the result of cdist
    Expectation: success
    """
    mint_cdist_binary_case1(loss=1e-04, mode=mode)


@arg_mark(
    plat_marks=[
        "cpu_linux",
        "cpu_windows",
        "cpu_macos",
    ],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
@pytest.mark.parametrize("mode", ["KBK", "GRAPH"])
def test_mint_cdist_cpu_normal(mode):
    """
    Feature: mint.cdist
    Description: test the cdist on cpu.
    Expectation: success
    """
    if mode == "KBK":
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    elif mode == "GRAPH":
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O2")
    x1 = Tensor(np.array([[[1.0, 1.0], [2.0, 2.0]]]).astype(np.float32))
    x2 = Tensor(np.array([[[3.0, 3.0], [3.0, 3.0]]]).astype(np.float32))
    output = cdist_forward_func(x1, x2, 1.)
    grads = cdist_backward_func(x1, x2, 1.)
    expect = np.array([[[4.0, 4.0], [2.0, 2.0]]]).astype(np.float32)
    except_x1_grad = np.array([[[-2.0, -2.0], [-2.0, -2.0]]]).astype(np.float32)
    except_x2_grad = np.array([[[2.0, 2.0], [2.0, 2.0]]]).astype(np.float32)
    allclose_nparray(output.asnumpy(), expect, 1e-04, 1e-04)
    allclose_nparray(grads[0].asnumpy(), except_x1_grad, 1e-04, 1e-04)
    allclose_nparray(grads[1].asnumpy(), except_x2_grad, 1e-04, 1e-04)


@arg_mark(
    plat_marks=[
        "platform_ascend",
        "platform_ascend910b",
        "cpu_linux",
        "cpu_windows",
        "cpu_macos",
        "platform_gpu",
    ],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=4.01
)
@pytest.mark.parametrize("mode", ["KBK", "GRAPH"])
def test_mint_cdist_vmap(mode):
    """
    Feature: mint.cdist
    Description: Verify the result of cdist
    Expectation: success
    """
    if mode == "KBK":
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    elif mode == "GRAPH":
        context.set_context(mode=ms.GRAPH_MODE, jit_level="O2")
    x1 = Tensor(np.array([[[1.0, 1.0], [2.0, 2.0]], [[1.0, 1.0], [2.0, 2.0]],
                          [[1.0, 1.0], [2.0, 2.0]], [[1.0, 1.0], [2.0, 2.0]],
                          [[1.0, 1.0], [2.0, 2.0]], [[1.0, 1.0], [2.0, 2.0]],
                          [[1.0, 1.0], [2.0, 2.0]], [[1.0, 1.0], [2.0, 2.0]]]).astype(np.float32))
    x2 = Tensor(np.array([[[3.0, 3.0], [3.0, 3.0]], [[3.0, 3.0], [3.0, 3.0]],
                          [[3.0, 3.0], [3.0, 3.0]], [[3.0, 3.0], [3.0, 3.0]],
                          [[3.0, 3.0], [3.0, 3.0]], [[3.0, 3.0], [3.0, 3.0]],
                          [[3.0, 3.0], [3.0, 3.0]], [[3.0, 3.0], [3.0, 3.0]]]).astype(np.float32))

    expect = np.array([[[2.828427, 2.828427], [1.4142135, 1.4142135]],
                       [[2.828427, 2.828427], [1.4142135, 1.4142135]],
                       [[2.828427, 2.828427], [1.4142135, 1.4142135]],
                       [[2.828427, 2.828427], [1.4142135, 1.4142135]],
                       [[2.828427, 2.828427], [1.4142135, 1.4142135]],
                       [[2.828427, 2.828427], [1.4142135, 1.4142135]],
                       [[2.828427, 2.828427], [1.4142135, 1.4142135]],
                       [[2.828427, 2.828427], [1.4142135, 1.4142135]]]).astype(np.float32)
    out = vmap(cdist_forward_vmap_func, in_axes=(0), out_axes=0)(x1, x2)
    allclose_nparray(out.asnumpy(), expect, 1e-04, 1e-04)
