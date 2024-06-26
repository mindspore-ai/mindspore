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

import numpy as np
import pytest
from mindspore import ops
import mindspore as ms
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark


def generate_std_input():
    return np.array([0.0370, 0.2970, 1.5420, 0.9105])


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x):
    return 1 / np.sqrt(x)


def generate_expect_backward_output(x):
    return -1 / (2 * np.sqrt(np.power(x, 3)))


@test_utils.run_with_cell
def rsqrt_forward_func(x):
    return ops.rsqrt(x)


@test_utils.run_with_cell
def rsqrt_backward_func(x):
    return ops.grad(rsqrt_forward_func, (0,))(x)


@test_utils.run_with_cell
def rsqrt_vmap_func(x):
    return ops.vmap(rsqrt_forward_func, in_axes=0, out_axes=0)(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_rsqrt_forward(mode):
    """
    Feature: pyboost function.
    Description: test function rsqrt forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = generate_std_input()
    output = rsqrt_forward_func(ms.Tensor(x))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_rsqrt_bf16(mode):
    """
    Feature: pyboost function.
    Description: test function rsqrt forward(bf16).
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = ms.Tensor([1.0, 2.0, 3.0, 4.0], dtype=ms.bfloat16)
    output = rsqrt_forward_func(x)
    expect = np.array([1.0000, 0.7071, 0.5774, 0.5000])
    np.testing.assert_allclose(output.float().asnumpy(), expect, rtol=5e-3, atol=5e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_rsqrt_backward(mode):
    """
    Feature: pyboost function.
    Description: test function rsqrt backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = generate_std_input()
    output = rsqrt_backward_func(ms.Tensor(x))
    expect = generate_expect_backward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_rsqrt_vmap(mode):
    """
    Feature: pyboost function.
    Description: test function rsqrt vmap feature.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = generate_std_input()
    output = rsqrt_vmap_func(ms.Tensor(x))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
def test_rsqrt_dynamic_shape():
    """
    Feature: Test dynamic shape.
    Description: test function rsqrt dynamic feature.
    Expectation: expect correct result.
    """
    ms_data1 = ms.Tensor(np.abs(np.random.randn(4, 5, 6).astype(np.float32)))
    ms_data2 = ms.Tensor(np.abs(np.random.randn(3, 2, 5, 1).astype(np.float32)))
    TEST_OP(rsqrt_forward_func, [[ms_data1], [ms_data2]], 'rsqrt')
