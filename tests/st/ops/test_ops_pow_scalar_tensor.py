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

"""
Test cases for pow with scalar base and tensor exponent.
"""

import numpy as np
import pytest
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark
from mindspore import mint, ops
import mindspore as ms


def generate_random_input(shape, dtype):
    return np.random.uniform(0.9, 1.0, size=shape).astype(dtype), np.random.uniform(0.9, 1.0, size=shape).astype(dtype)


@test_utils.run_with_cell
def pow_forward_func(x, y):
    return mint.pow(x, y)


@test_utils.run_with_cell
def pow_backward_func(x, y):
    return ops.grad(pow_forward_func, (0, 1))(x, y) # pylint: disable=not-callable


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
          essential_mark='essential', mem_peak=1024.01)
@test_utils.run_test_with_On
def test_pow_scalar_tensor_fallback():
    """
    Feature: Ops.
    Description: test pow_scalar_tensor fallback.
    Expectation: expect correct result.
    """
    # mint.pow it self has no need to support graph mode O2, but backward of some existing
    # operations may use PowScalarTensor. PowScalarTensor will fallback to Pow in this case.
    ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O2')
    x_np = 2
    _, y_np = generate_random_input((2, 3, 4), dtype=np.int32)
    x = x_np
    y = ms.Tensor(y_np)
    out = pow_forward_func(x, y)
    expect_out = np.power(x_np, y_np)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-3)
    assert out.dtype == ms.int32


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
          essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
@test_utils.run_test_with_On
def test_pow_op_normal(context_mode, data_type):
    """
    Feature: Ops.
    Description: test op pow forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    if context_mode == ms.GRAPH_MODE:
        ms.set_context(jit_config={'jit_level': 'O0'})
    x_np = 0
    _, y_np = generate_random_input((2, 3, 4, 5), dtype=data_type)

    x = x_np
    y = ms.Tensor(y_np)
    out = pow_forward_func(x, y)
    expect_out = np.power(x_np, y_np)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-3)

    x_np = 2.3
    _, y_np = generate_random_input((3, 4, 5), dtype=data_type)

    x = x_np
    y = ms.Tensor(y_np)
    out = pow_forward_func(x, y)
    expect_out = np.power(x_np, y_np)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-3)

    ## auto grad
    ms.context.set_context(mode=context_mode)
    if context_mode == ms.GRAPH_MODE:
        ms.set_context(jit_config={'jit_level': 'O0'})
    _, y_np = generate_random_input((3, 4, 5), dtype=data_type)
    x = x_np = 2
    y = ms.Tensor(y_np)
    grads = pow_backward_func(x, y)
    expect_out = np.power(x_np, y_np) * np.log(x_np)
    np.testing.assert_allclose(grads.asnumpy(), expect_out, rtol=1e-3)

    x = x_np = 0
    _, y_np = generate_random_input((3, 4, 5), dtype=data_type)
    y = ms.Tensor(y_np)
    grads = pow_backward_func(x, y)
    expect_out = np.zeros_like(y_np)
    np.testing.assert_allclose(grads.asnumpy(), expect_out, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=4.0)
def test_pow_dynamic_shape():
    """
    Feature: Test pow with dynamic shape in graph mode.
    Description: call ops.pow with valid input and index.
    Expectation: return the correct value.
    """
    _, exp1 = generate_random_input((2, 3, 4), np.float32)
    _, exp2 = generate_random_input((3, 4, 5, 6), np.float32)

    TEST_OP(pow_forward_func, [[2.3, ms.Tensor(exp1)], [0, ms.Tensor(exp2)]],
            disable_mode=['GRAPH_MODE_GE'],
            case_config={'disable_input_check': True,
                         'disable_resize': True})
