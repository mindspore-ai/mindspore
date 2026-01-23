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
from mindspore import ops, jit
from mindspore.mint.nn import SELU
from mindspore.mint.nn.functional import selu
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x):
    alpha = 1.67326324
    scale = 1.05070098
    return np.where(x >= 0, scale * x, scale * alpha * (np.exp(x) - 1))


def generate_expect_backward_output(x):
    alpha = 1.67326324
    scale = 1.05070098
    return np.where(x >= 0, scale, scale * alpha * np.exp(x))


def selu_forward_nn(x):
    op = SELU()
    return op(x)


@test_utils.run_with_cell
def selu_forward_func(x):
    return selu(x)


@test_utils.run_with_cell
def selu_backward_func(x):
    return ms.grad(selu_forward_func, (0))(x)


@jit(backend="ms_backend")
@test_utils.run_with_cell
def selu_vmap_func(x):
    return ops.vmap(selu_forward_func)(x)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=2048.01)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_selu_normal(context_mode):
    """
    Feature: pyboost function.
    Description: test function selu forward and backward.
    Expectation: expect correct result.
    """
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=context_mode)
    # forward
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output_f1 = selu_forward_func(ms.Tensor(x))
    output_f2 = selu_forward_nn(ms.Tensor(x))
    expect_f = generate_expect_forward_output(x)
    np.testing.assert_allclose(output_f1.asnumpy(), expect_f, rtol=1e-3)
    np.testing.assert_allclose(output_f2.asnumpy(), expect_f, rtol=1e-3)

    # backward
    output_b = selu_backward_func(ms.Tensor(x))
    expect_b = generate_expect_backward_output(x)
    np.testing.assert_allclose(output_b.asnumpy(), expect_b, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'],
          level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=2048.00)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_selu_vmap(context_mode):
    """
    Feature: pyboost function.
    Description: test function selu vmap feature.
    Expectation: expect correct result.
    """
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = selu_vmap_func(ms.Tensor(x))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'],
          level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=2048.00)
def test_ops_selu_dynamic_shape():
    """
    Feature: pyboost function.
    Description: test function selu with dynamic shape and dynamic rank.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    x2 = generate_random_input((4, 5), np.float32)
    TEST_OP(selu_forward_func, [[ms.Tensor(x1)], [ms.Tensor(x2)]], disable_mode=['GRAPH_MODE_GE'])
