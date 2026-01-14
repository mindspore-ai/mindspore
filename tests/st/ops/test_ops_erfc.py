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
from scipy import special
import mindspore as ms
from mindspore import ops
from mindspore.mint import erfc
from tests.mark_utils import arg_mark
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def erfc_expect_forward_func(x):
    return special.erfc(x)


@test_utils.run_with_cell
def erfc_forward_func(x):
    return erfc(x)


@test_utils.run_with_cell
def erfc_backward_func(x):
    return ms.grad(erfc_forward_func, (0))(x)


@test_utils.run_with_cell
def erfc_vmap_func(x):
    return ops.vmap(erfc_forward_func)(x)

@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=2048.01)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_erfc_normal(context_mode):
    """
    Feature: pyboost function.
    Description: test function erfc forward and backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    # forward
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output_f = erfc_forward_func(ms.Tensor(x))
    expect_f = erfc_expect_forward_func(x)
    diff = output_f.asnumpy() - expect_f
    error = np.ones(shape=expect_f.shape) * 1e-4
    assert np.all(np.abs(diff) < error)

    # backward
    x2 = np.array([0.1, 0.2, 0.3, 1, 2]).astype('float32')
    output_b = erfc_backward_func(ms.Tensor(x2))
    expect_b = np.array([-1.1171516, -1.0841347, -1.0312609, -0.4151074, -0.02066698]).astype('float32')
    np.testing.assert_allclose(output_b.asnumpy(), expect_b, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_erfc_bf16(context_mode):
    """
    Feature: pyboost function.
    Description: test function erfc forward(bf16).
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x_tensor = ms.Tensor([0.3, -1.2, 10.7], dtype=ms.bfloat16)
    output = erfc_forward_func(x_tensor)
    expect = np.array([0.6719, 1.9141, 0.000])
    np.testing.assert_allclose(output.float().asnumpy(), expect, rtol=4e-3, atol=4e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b', 'platform_gpu', 'cpu_linux', 'cpu_windows',
                      'cpu_macos'], level_mark='level1', card_mark='onecard',
                      essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_erfc_vmap(context_mode):
    """
    Feature: pyboost function.
    Description: test function erfc vmap feature.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = erfc_vmap_func(ms.Tensor(x))
    expect = erfc_expect_forward_func(x)

    diff = output.asnumpy() - expect
    error = np.ones(shape=expect.shape) * 1e-4
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b', 'platform_gpu', 'cpu_linux', 'cpu_windows',
                      'cpu_macos'], level_mark='level1', card_mark='onecard',
                      essential_mark='unessential', mem_peak=4.0)
def test_erfc_dynamic_shape():
    """
    Feature: Test dynamic shape.
    Description: test function erfc  dynamic feature.
    Expectation: expect correct result.
    """
    ms_data1 = generate_random_input((2, 3, 4), np.float32)
    ms_data2 = generate_random_input((3, 4, 5, 6), np.float32)
    TEST_OP(erfc_forward_func, [[ms.Tensor(ms_data1)], [ms.Tensor(ms_data2)]])
