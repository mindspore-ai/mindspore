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
from mindspore import ops
from mindspore.mint import expm1
from mindspore.mint.special import expm1 as special_expm1
from tests.mark_utils import arg_mark
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.common.random_generator import generate_numpy_ndarray_by_randn


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def expm1_expect_forward_func(x):
    return np.expm1(x)

@test_utils.run_with_cell
def expm1_forward_func(x):
    return expm1(x)

@test_utils.run_with_cell
def expm1_special_forward_func(x):
    return special_expm1(x)

@test_utils.run_with_cell
def expm1_backward_func(x):
    return ms.grad(expm1_forward_func, (0))(x)


@test_utils.run_with_cell
def expm1_vmap_func(x):
    return ops.vmap(expm1_forward_func)(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.75)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_expm1_normal(context_mode):
    """
    Feature: pyboost function.
    Description: test function expm1 forward and backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    # forward
    x = generate_numpy_ndarray_by_randn((2, 3, 4, 5), np.float32, 'x')
    output_f = expm1_forward_func(ms.Tensor(x))
    output_f_special = expm1_special_forward_func(ms.Tensor(x))
    expect_f = expm1_expect_forward_func(x)
    np.testing.assert_allclose(output_f.asnumpy(), expect_f, rtol=1e-3)
    np.testing.assert_allclose(output_f_special.asnumpy(), expect_f, rtol=1e-3)

    # backward
    x2 = np.random.randn(2, 3, 4, 5).astype(np.float32)
    output_b = expm1_backward_func(ms.Tensor(x2))
    expect_b = np.expm1(x2) + 1
    np.testing.assert_allclose(output_b.asnumpy(), expect_b, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_expm1_bf16(context_mode):
    """
    Feature: pyboost function.
    Description: test function expm1 bfloat16 forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    input_bf16 = ms.Tensor([-4.2, 2.4, -0.3, -1.2, 5.1], dtype=ms.bfloat16)
    output_f = expm1_forward_func(input_bf16)
    expect_f = np.array([-0.984375, 10.0625, -0.259765, -0.699218, 162.0])
    np.testing.assert_allclose(output_f.float().asnumpy(), expect_f, rtol=4e-3, atol=4e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b', 'platform_gpu', 'cpu_linux', 'cpu_windows',
    'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=2048.01)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_expm1_vmap(context_mode):
    """
    Feature: pyboost function.
    Description: test function expm1 vmap feature.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = expm1_vmap_func(ms.Tensor(x))
    expect = expm1_expect_forward_func(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b', 'platform_gpu', 'cpu_linux', 'cpu_windows',
    'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=4.0)
def test_expm1_dynamic_shape():
    """
    Feature: Test dynamic shape.
    Description: test function expm1 dynamic feature.
    Expectation: expect correct result.
    """
    ms_data1 = generate_random_input((2, 3, 4, 5), np.float32)
    ms_data2 = generate_random_input((3, 4, 5, 6, 7), np.float32)
    TEST_OP(expm1_forward_func, [[ms.Tensor(ms_data1)], [ms.Tensor(ms_data2)]])
