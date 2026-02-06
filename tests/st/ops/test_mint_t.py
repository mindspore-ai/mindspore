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
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x):
    return np.transpose(x)


def generate_expect_backward_output(x):
    return np.ones_like(x)


@test_utils.run_with_cell
def t_forward_func(x):
    return ms.mint.t(x)


@test_utils.run_with_cell
def t_backward_func(x):
    return ms.grad(t_forward_func, (0))(x)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.0)
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_t_normal0(mode):
    """
    Feature: pyboost function.
    Description: test function t forward and backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    if mode == ms.GRAPH_MODE:
        ms.context.set_context(jit_level='O0')
    x = generate_random_input((3, 4), np.float32)
    output = t_forward_func(ms.Tensor(x))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    output = t_backward_func(ms.Tensor(x))
    expect = generate_expect_backward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=2048.00)
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_t_normal1(mode):
    """
    Feature: pyboost function.
    Description: test function t forward and backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    if mode == ms.GRAPH_MODE:
        ms.context.set_context(jit_level='O0')
    x = generate_random_input((6,), np.float32)
    output = t_forward_func(ms.Tensor(x))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    output = t_backward_func(ms.Tensor(x))
    expect = generate_expect_backward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=2048.00)
def test_t_dynamic_shape0():
    """
    Feature: pyboost function.
    Description: test function t with dynamic shape and dynamic rank.
    Expectation: return the correct value.
    """
    x = generate_random_input((2,), np.float32)
    y = generate_random_input((5, 6), np.float32)

    TEST_OP(t_forward_func, [[ms.Tensor(x)], [ms.Tensor(y)]], disable_mode=["GRAPH_MODE_GE"])


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=2048.0)
def test_t_dynamic_shape1():
    """
    Feature: pyboost function.
    Description: test function t with dynamic shape and dynamic rank.
    Expectation: return the correct value.
    """
    x = generate_random_input((2, 3), np.float32)
    y = generate_random_input((8,), np.float32)

    TEST_OP(t_forward_func, [[ms.Tensor(x)], [ms.Tensor(y)]], disable_mode=["GRAPH_MODE_GE"])
