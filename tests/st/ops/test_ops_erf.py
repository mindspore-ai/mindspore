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
from tests.mark_utils import arg_mark
import pytest
import numpy as np
from scipy import special
import mindspore as ms
from mindspore import ops
from mindspore.mint import erf
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x):
    return special.erf(x)


@test_utils.run_with_cell
def erf_forward_func(x):
    return erf(x)


@test_utils.run_with_cell
def erf_backward_func(x):
    return ops.grad(erf_forward_func, (0))(x)


@test_utils.run_with_cell
def erf_vmap_func(x):
    return ops.vmap(erf_forward_func)(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype, tol', [(np.float16, 1.0e-3), (np.float32, 1.0e-4)])
@pytest.mark.parametrize('shape', [(2, 3, 4, 5), (1, 256, 2048), (1, 256, 5120)])
@test_utils.run_test_with_On
def test_ops_erf_forward(context_mode, shape, dtype, tol):
    """
    Feature: pyboost function.
    Description: test function erf forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input(shape, dtype)
    output = erf_forward_func(ms.Tensor(x))
    expect = generate_expect_forward_output(x)
    diff = output.asnumpy() - expect
    error = np.ones(shape=expect.shape) * tol
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_erf_bf16(context_mode):
    """
    Feature: pyboost function.
    Description: test function erf forward(bf16).
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x_tensor = ms.Tensor([0, -1., 10.], dtype=ms.bfloat16)
    output = erf_forward_func(x_tensor)
    expect = np.array([0.000, -0.8427, 1.0000])
    np.testing.assert_allclose(output.float().asnumpy(), expect, rtol=5e-3, atol=5e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype, tol', [(np.float16, 1.0e-3), (np.float32, 1.0e-4)])
@test_utils.run_test_with_On
def test_ops_erf_backward(context_mode, dtype, tol):
    """
    Feature: pyboost function.
    Description: test function erf backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = np.array([0.1, 0.2, 0.3, 1, 2]).astype(dtype)
    output = erf_backward_func(ms.Tensor(x))
    expect = np.array([1.1171516, 1.0841347, 1.0312609, 0.4151074, 0.02066698]).astype(dtype)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=tol)



@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype, tol', [(np.float16, 1.0e-3), (np.float32, 1.0e-4)])
@pytest.mark.parametrize('shape', [(2, 3, 4, 5), (1, 256, 2048), (1, 256, 5120)])
@test_utils.run_test_with_On
def test_ops_erf_vmap(context_mode, shape, dtype, tol):
    """
    Feature: pyboost function.
    Description: test function erf vmap feature.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input(shape, dtype)
    output = erf_vmap_func(ms.Tensor(x))
    expect = generate_expect_forward_output(x)
    diff = output.asnumpy() - expect
    error = np.ones(shape=expect.shape) * tol
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_erf_dynamic_shape():
    """
    Feature: Test dynamic shape.
    Description: test function erf  dynamic feature.
    Expectation: expect correct result.
    """
    ms_data1 = generate_random_input((2, 3, 4), np.float32)
    ms_data2 = generate_random_input((3, 4, 5, 6), np.float32)
    TEST_OP(erf_forward_func
            , [[ms.Tensor(ms_data1)], [ms.Tensor(ms_data2)]], 'erf')
