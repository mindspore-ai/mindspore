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
from mindspore import Tensor, context
from mindspore.mint.linalg import qr
from tests.mark_utils import arg_mark
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
import scipy

context.set_context(jit_level='O0')


def syminvadj_np(mat):
    ret = mat + mat.T
    np.fill_diagonal(ret, np.diag(ret).real * 0.5)
    return ret


def trilImInvAdjSkew_np(mat):
    ret = np.tril(mat - mat.T)
    return ret


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def expect_forward_func(a, mode='reduced'):
    # mode: {'reduced', 'complete', 'r'}.
    q, r = np.linalg.qr(a, mode)
    return q, r


# To simplify the test, only input a two-dimensional tensor for verify.
def expect_backward_func(a, mode='reduced'):
    q, r = expect_forward_func(a, mode)

    m_, n_ = q.shape[-2], r.shape[-1]
    # In backward, mode only can choice 'reduced' and 'complete', not 'r'.
    if mode not in ['reduced', 'complete']:
        raise ValueError(f"Not supported mode: '{mode}'.")

    if mode == 'complete' and m_ > n_:
        raise ValueError("The QR decomposition is not differentiable.")

    dq = np.ones_like(q)
    dr = np.ones_like(r)

    # Init dA.
    da = np.dot(dr, r.T) - np.dot(q.T, dq)
    if m_ >= n_:
        da = np.dot(q, syminvadj_np(np.triu(da)))
        da = da + dq
        scipy_triangular = scipy.linalg.solve_triangular(r, da.T, lower=False, trans='N')
        da = scipy_triangular.T
    else:
        da = np.dot(q, trilImInvAdjSkew_np(-da))
        r_narrow = r[:, :m_]
        da = scipy.linalg.solve_triangular(r_narrow, da.T, lower=False, trans='N').T
        zero_tensor = np.zeros((m_, n_ - m_), dtype=r.dtype)
        da = np.hstack([da, zero_tensor])
        da += np.dot(q, dr)

    return da


@test_utils.run_with_cell
def forward_func(a, mode='reduced'):
    q, r = qr(a, mode)
    return q, r


@test_utils.run_with_cell
def backward_func(a, mode='reduced'):
    return ms.grad(forward_func, (0))(a, mode)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="essential", mem_peak=1024.01
)
@pytest.mark.parametrize("context_mode", [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
@test_utils.run_test_with_On
def test_ops_qr_forward(context_mode):
    """
    Feature: pyboost function.
    Description: test function mint.linalg.qr forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    if context_mode == ms.GRAPH_MODE:
        ms.set_context(jit_level='O0')

    x_np = generate_random_input((3, 2), np.float32)
    out_q, out_r = forward_func(Tensor(x_np))
    expect_q, expect_r = expect_forward_func(x_np)

    np.testing.assert_allclose(out_q.asnumpy(), expect_q, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(out_r.asnumpy(), expect_r, atol=1e-3, rtol=1e-3)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="essential", mem_peak=1.00
)
@pytest.mark.parametrize("context_mode", [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
@test_utils.run_test_with_On
def test_ops_qr_backward(context_mode):
    """
    Feature: pyboost function.
    Description: test function mint.linalg.qr backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    if context_mode == ms.GRAPH_MODE:
        ms.set_context(jit_level='O0')

    # Input A is a thin matrix.
    x_np = generate_random_input((5, 3), np.float32)
    dx_expect = expect_backward_func(x_np)
    dx = backward_func(Tensor(x_np))
    np.testing.assert_allclose(dx.asnumpy(), dx_expect, atol=1e-3, rtol=1e-3)

    # Input A is a wide matrix.
    x_np = generate_random_input((3, 5), np.float32)
    dx_expect = expect_backward_func(x_np)
    dx = backward_func(Tensor(x_np))
    np.testing.assert_allclose(dx.asnumpy(), dx_expect, atol=1e-3, rtol=1e-3)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="essential", mem_peak=2048.01
)
@pytest.mark.parametrize("context_mode", [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
@test_utils.run_test_with_On
def test_ops_qr_dtype(context_mode):
    """
    Feature: pyboost function.
    Description: test function mint.mv ops dtype support.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    if context_mode == ms.GRAPH_MODE:
        ms.set_context(jit_level='O0')

    # At present, mint.linalg.qr only supports FLOAT32.
    types = [np.float32]
    for dtype in types:
        x_np = generate_random_input((2, 3), dtype)
        out_q, out_r = forward_func(Tensor(x_np))
        expect_q, expect_r = expect_forward_func(x_np)

        np.testing.assert_allclose(out_q.asnumpy(), expect_q, atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(out_r.asnumpy(), expect_r, atol=1e-3, rtol=1e-3)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=2048.01
)
def test_linalg_qr_dynamic_shape():
    """
    Feature: Test dynamic shape.
    Description: test function expand_as dynamic feature.
    Expectation: expect correct result.
    """
    input1 = generate_random_input((3, 5), np.float32)
    input2 = generate_random_input((3, 4, 3, 5), np.float32)

    TEST_OP(
        forward_func,
        [[Tensor(input1), "reduced"], [Tensor(input2), "reduced"]],
        disable_mode=["GRAPH_MODE_GE"],
        disable_case=['ScalarTensor'],
        case_config={'disable_input_check': True}
    )
