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
from mindspore.mint.nn import Mish
from mindspore.mint.nn.functional import mish
from tests.mark_utils import arg_mark
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def mish_expect_forward_func(x):
    result = np.zeros_like(x, dtype=x.dtype)
    for index, _ in np.ndenumerate(x):
        result[index] = x[index] * np.tanh(np.log(np.exp(x[index]) + 1))
    return result


def mish_forward_nn(x):
    op = Mish()
    return op(x)


@test_utils.run_with_cell
def mish_forward_func(x):
    return mish(x)


@test_utils.run_with_cell
def mish_backward_func(x):
    return ms.grad(mish_forward_func, (0))(x)


@jit(backend="ms_backend")
@test_utils.run_with_cell
def mish_vmap_func(x):
    return ops.vmap(mish_forward_func)(x)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=2048.01)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_mish_normal(context_mode):
    """
    Feature: pyboost function.
    Description: test function mish forward and backward.
    Expectation: expect correct result.
    """
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=context_mode)
    # forward
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output_f1 = mish_forward_func(ms.Tensor(x))
    output_f2 = mish_forward_nn(ms.Tensor(x))
    expect_f = mish_expect_forward_func(x)
    np.testing.assert_allclose(output_f1.asnumpy(), expect_f, rtol=1e-3)
    np.testing.assert_allclose(output_f2.asnumpy(), expect_f, rtol=1e-3)

    # backward
    x2 = np.array([11.2, 1.3, 20.3, -0.5, -8.1]).astype('float32')
    output_b = mish_backward_func(ms.Tensor(x2))
    expect_b = np.array([1.0, 1.08363771, 1.0, 0.28951066, -0.00215442])
    np.testing.assert_allclose(output_b.asnumpy(), expect_b, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'],
          level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=2048.00)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_mish_vmap(context_mode):
    """
    Feature: pyboost function.
    Description: test function mish vmap feature.
    Expectation: expect correct result.
    """
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = mish_vmap_func(ms.Tensor(x))
    expect = mish_expect_forward_func(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'],
          level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=4.0)
def test_mish_dynamic_shape():
    """
    Feature: Test dynamic shape.
    Description: test function mish dynamic feature.
    Expectation: expect correct result.
    """
    ms_data1 = generate_random_input((2, 3, 4, 5), np.float32)
    ms_data2 = generate_random_input((3, 4, 5, 6, 7), np.float32)
    TEST_OP(mish_forward_func, [[ms.Tensor(ms_data1)], [ms.Tensor(ms_data2)]],
            disable_mode=['GRAPH_MODE_GE'])
