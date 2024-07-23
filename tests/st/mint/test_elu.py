# Copyright 2024 Huawei Technoeluies Co., Ltd
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
from mindspore import mint, context, Tensor
from tests.mark_utils import arg_mark
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def generate_expect_forward_output(x, alpha, dtype):
    return np.where(x > 0, x * 1, alpha * 1 * (np.exp(x * 1) - 1)).astype(dtype)

def generate_expect_backward_output(x, alpha, dtype):
    return  np.where(x > 0, 1, alpha * 1 * 1 * np.exp(x * 1)).astype(dtype)


@test_utils.run_with_cell
def elu_forward_func(x, alpha):
    return mint.nn.functional.elu(x, alpha)

@test_utils.run_with_cell
def elu_backward_func(x, alpha):
    return ms.ops.grad(elu_forward_func, (0))(x, alpha)

@test_utils.run_with_cell
def elu_vmap_func(x, alpha):
    return ms.ops.vmap(elu_forward_func, in_axes=(0, None), out_axes=0)(x, alpha)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("mode", [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_elu_normal(mode):
    """
    Feature: test elu operator
    Description: test elu run by pyboost
    Expectation: success
    """
    context.set_context(mode=mode)

    alpha = np.random.uniform(0.5, 2)
    x_np = generate_random_input((2, 3, 4), np.float32)
    x_tensor = Tensor(x_np, ms.float32)
    output = elu_forward_func(x_tensor, alpha)
    expect = generate_expect_forward_output(x_np, alpha, np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, 1e-4, 1e-4)

    output = elu_backward_func(x_tensor, alpha)
    expect = generate_expect_backward_output(x_np, alpha, np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("mode", [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_elu_backward_alpha_neg(mode):
    """
    Feature: test elu operator
    Description: test elu run by pyboost
    Expectation: success
    """
    context.set_context(mode=mode)

    alpha = np.random.uniform(-0.5, -2)
    x_np = generate_random_input((2, 3, 4), np.float32)
    x_tensor = Tensor(x_np, ms.float32)
    output = elu_backward_func(x_tensor, alpha)
    expect = generate_expect_backward_output(x_np, alpha, np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("mode", [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_elu_vmap(mode):
    """
    Feature: pyboost function.
    Description: test function elu vmap feature.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)

    alpha = np.random.uniform(0.5, 2)
    x_np = generate_random_input((2, 3, 4, 5), np.float32)
    x_tensor = Tensor(x_np, ms.float32)
    output = elu_vmap_func(x_tensor, alpha)
    expect = generate_expect_forward_output(x_np, alpha, np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_elu_dynamic_shape_testop():
    """
    Feature: Test elu with dynamic shape in graph mode using TEST_OP.
    Description: call ops.elu with valid input and index.
    Expectation: return the correct value.
    """

    alpha1 = np.random.uniform(0.5, 2)
    alpha2 = np.random.uniform(0.5, 2)
    x1 = generate_random_input((3, 4, 5), np.float32)
    x2 = generate_random_input((3, 7, 8, 3), np.float32)
    TEST_OP(mint.nn.functional.elu, [[ms.Tensor(x1), alpha1], [ms.Tensor(x2), alpha2]], 'elu')
