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
from mindspore import Tensor
from mindspore.mint import mv
from tests.mark_utils import arg_mark
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def mv_expect_forward_func(x_np, vec_np):
    return x_np @ vec_np


def mv_expect_backward_func(x_np, vec_np):
    # Construct a hypothetical gradient to simulate.
    dout = np.ones_like(mv_expect_forward_func(x_np, vec_np))
    dx = np.outer(dout, vec_np)
    dvec = x_np.T @ dout
    return dx, dvec


@test_utils.run_with_cell
def mv_forward_func(x, vec):
    return mv(x, vec)


@test_utils.run_with_cell
def mv_backward_func(x, vec):
    return ms.grad(mv_forward_func, (0, 1))(x, vec)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential", mem_peak=1024.01
)
@pytest.mark.parametrize("context_mode", [ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_mv_forward(context_mode):
    """
    Feature: pyboost function.
    Description: test function mint.mv forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x_np = generate_random_input((2, 3), np.float32)
    vec_np = generate_random_input((3,), np.float32)
    out_mint = mv_forward_func(Tensor(x_np), Tensor(vec_np))
    out_expect = mv_expect_forward_func(x_np, vec_np)

    np.testing.assert_allclose(out_mint.asnumpy(), out_expect, atol=1e-3, rtol=1e-3)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential", mem_peak=1024.01
)
@pytest.mark.parametrize("context_mode", [ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_mv_backward(context_mode):
    """
    Feature: pyboost function.
    Description: test function mint.mv backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x_np = generate_random_input((2, 3), np.float32)
    vec_np = generate_random_input((3,), np.float32)
    dx, dvec = mv_backward_func(Tensor(x_np), Tensor(vec_np))
    dx_expect, dvec_expect = mv_expect_backward_func(x_np, vec_np)

    np.testing.assert_allclose(dx.asnumpy(), dx_expect, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(dvec.asnumpy(), dvec_expect, atol=1e-3, rtol=1e-3)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential", mem_peak=1024.01
)
@pytest.mark.parametrize("context_mode", [ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_mv_dtype(context_mode):
    """
    Feature: pyboost function.
    Description: test function mint.mv ops dtype support.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x_shape = (2, 3)
    vec_shape = (3,)

    # At present, alcnn supports FLOAT16 and FLOAT32.
    types = [np.float16, np.float32]
    for dtype in types:
        x = generate_random_input(x_shape, dtype)
        vec = generate_random_input(vec_shape, dtype)
        out = mv_forward_func(Tensor(x), Tensor(vec))
        out_expect = mv_expect_forward_func(x, vec)
        np.testing.assert_allclose(out.asnumpy(), out_expect, atol=1e-3, rtol=1e-3)

        dx, dvec = mv_backward_func(Tensor(x), Tensor(vec))
        dx_expect, dvec_expect = mv_expect_backward_func(x, vec)
        np.testing.assert_allclose(dx.asnumpy(), dx_expect, atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(dvec.asnumpy(), dvec_expect, atol=1e-3, rtol=1e-3)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=1.00
)
def test_mv_dynamic_shape():
    """
    Feature: Test dynamic shape.
    Description: test function mv dynamic feature.
    Expectation: expect correct result.
    """
    input1 = generate_random_input((2, 3), np.float32)
    vec1 = generate_random_input((3,), np.float32)
    input2 = generate_random_input((3, 4), np.float32)
    vec2 = generate_random_input((4,), np.float32)

    TEST_OP(
        mv_forward_func,
        [[Tensor(input1), Tensor(vec1)], [Tensor(input2), Tensor(vec2)]],
        disable_mode=['GRAPH_MODE_GE'],
        disable_case=['ScalarTensor'],
        case_config={'disable_input_check': True,
                     'all_dim_zero': True}
    )
