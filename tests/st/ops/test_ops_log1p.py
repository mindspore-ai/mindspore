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
from mindspore.mint import log1p
from mindspore.mint.special import log1p as special_log1p
from tests.mark_utils import arg_mark
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP


def generate_random_input(shape, dtype):
    # value must be greater than -1
    return np.random.rand(*shape).astype(dtype)


def log1p_expect_forward_func(x):
    return np.log1p(x)


@test_utils.run_with_cell
def log1p_forward_func(x):
    return log1p(x)


@test_utils.run_with_cell
def log1p_special_forward_func(x):
    return special_log1p(x)

@test_utils.run_with_cell
def log1p_backward_func(x):
    return ms.grad(log1p_forward_func, (0))(x)


@test_utils.run_with_cell
def log1p_vmap_func(x):
    return ops.vmap(log1p_forward_func)(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows',
                      'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=2048.02)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_log1p_normal(context_mode):
    """
    Feature: pyboost function.
    Description: test function log1p forward and backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    # forward
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output_f = log1p_forward_func(ms.Tensor(x))
    output_f_special = log1p_special_forward_func(ms.Tensor(x))
    expect_f = log1p_expect_forward_func(x)
    np.testing.assert_allclose(output_f.asnumpy(), expect_f, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(output_f_special.asnumpy(), expect_f, rtol=1e-3, atol=1e-3)

    # backward
    x2 = np.random.rand(2, 3, 4, 5).astype(np.float32)
    output_b = log1p_backward_func(ms.Tensor(x2))
    expect_b = np.reciprocal(x2 + 1)
    np.testing.assert_allclose(output_b.asnumpy(), expect_b, rtol=1e-3, atol=1e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_log1p_bf16(context_mode):
    """
    Feature: pyboost function.
    Description: test function log1p forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    input_bf16 = ms.Tensor([4.2, 2.4, -0.3, -1.2, 5.1], dtype=ms.bfloat16)
    output_f = log1p_forward_func(input_bf16)
    expect_f = np.array([1.648438, 1.2265625, -0.357421875, np.nan, 1.8046875])
    np.testing.assert_allclose(output_f.float().asnumpy(), expect_f, rtol=4e-3, atol=4e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b', 'platform_gpu', 'cpu_linux', 'cpu_windows',
    'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_log1p_vmap(context_mode):
    """
    Feature: pyboost function.
    Description: test function log1p vmap feature.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = log1p_vmap_func(ms.Tensor(x))
    expect = log1p_expect_forward_func(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b', 'platform_gpu', 'cpu_linux', 'cpu_windows',
    'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=4.0)
def test_log1p_dynamic_shape():
    """
    Feature: Test dynamic shape.
    Description: test function log1p  dynamic feature.
    Expectation: expect correct result.
    """
    ms_data1 = generate_random_input((2, 3, 4, 5), np.float32)
    ms_data2 = generate_random_input((3, 4, 5, 6, 7), np.float32)
    TEST_OP(log1p_forward_func, [[ms.Tensor(ms_data1)], [ms.Tensor(ms_data2)]])
