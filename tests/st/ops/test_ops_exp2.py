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
from mindspore.mint import exp2
from mindspore.mint.special import exp2 as special_exp2
from tests.mark_utils import arg_mark
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def exp2_expect_forward_func(x):
    return np.exp2(x)


def exp2_expect_backward_func(x):
    LOG_2 = 0.693147
    return np.exp2(x) * LOG_2


@test_utils.run_with_cell
def exp2_forward_func(x):
    return exp2(x)


@test_utils.run_with_cell
def exp2_special_forward_func(x):
    return special_exp2(x)


@test_utils.run_with_cell
def exp2_backward_func(x):
    return ms.grad(exp2_forward_func, (0))(x)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="essential", mem_peak=1024.01
)
@pytest.mark.parametrize("context_mode", [ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_exp2_normal(context_mode):
    """
    Feature: pyboost function.
    Description: test function exp2 forward and backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    # forward
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output_f = exp2_forward_func(ms.Tensor(x))
    output_f_special = exp2_special_forward_func(ms.Tensor(x))
    expect_f = exp2_expect_forward_func(x)
    np.testing.assert_allclose(output_f.asnumpy(), expect_f, rtol=1e-3)
    np.testing.assert_allclose(output_f_special.asnumpy(), expect_f, rtol=1e-3)

    # backward
    x = generate_random_input((1, 2, 3, 4), np.float32)
    output_b = exp2_backward_func(ms.Tensor(x))
    expect_b = exp2_expect_backward_func(x)
    np.testing.assert_allclose(output_b.asnumpy(), expect_b, rtol=1e-3)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="essential", mem_peak=1024.01
)
@pytest.mark.parametrize("context_mode", [ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_exp2_dtype(context_mode):
    """
    Feature: pyboost function.
    Description: test function exp2 forward and backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    shape = (2, 3, 4, 5)
    # Warning: the type conversion of the uint8 PTA is different from the NumPy.
    # After the conversion, the type of the PTA is FLOAT32, and the output type of the NumPy is FLOAT16.
    # types = [np.uint8]
    types = [np.float16, np.float32, np.int8, np.int16, np.int32, np.int64]
    for dtype in types:
        x = generate_random_input(shape, dtype)
        x_ms = ms.Tensor(x)
        out = exp2_forward_func(x_ms)
        out_f_special = exp2_special_forward_func(x_ms)
        expect = exp2_expect_forward_func(x)
        np.testing.assert_allclose(out.asnumpy(), expect, rtol=1e-3)
        np.testing.assert_allclose(out_f_special.asnumpy(), expect, rtol=1e-3)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=2048.01
)
@pytest.mark.parametrize("mode", [ms.PYNATIVE_MODE])
def test_exp2_dynamic_shape(mode):
    """
    Feature: Test dynamic shape.
    Description: test function exp2 dynamic feature.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    ms_data1 = generate_random_input((2, 3, 4, 5), np.float32)
    ms_data2 = generate_random_input((3, 4, 5, 6, 7), np.float32)
    TEST_OP(
        exp2_forward_func,
        [[ms.Tensor(ms_data1)], [ms.Tensor(ms_data2)]],
        disable_mode=["GRAPH_MODE_GE"],
    )
