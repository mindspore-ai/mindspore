# Copyright 2025 Huawei Technologies Co., Ltd
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
# pylint: disable=unused-variable
import pytest
import numpy as np
import mindspore as ms
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.uniform(-1, 1, shape).astype(dtype)


@test_utils.run_with_cell
def inplace_sign_forward_func(x):
    x = x * 1
    return x.sign_()


@test_utils.run_with_cell
def inplace_sign_backward_func(x):
    return ms.grad(inplace_sign_forward_func, (0))(x)



@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=4.00)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_inplace_sign(context_mode):
    """
    Feature: standard forward, backward features.
    Description: test function sign_.
    Expectation: expect correct result.
    """
    if context_mode == ms.GRAPH_MODE:
        ms.set_context(jit_level='O0')
    ms.context.set_context(mode=context_mode, device_target="Ascend")
    x = generate_random_input((2, 3, 4), np.float32)
    expect_out = np.sign(x)
    expect_grad = np.zeros_like(x, dtype=np.float32)
    output = inplace_sign_forward_func(ms.Tensor(x))
    output_grad = inplace_sign_backward_func(ms.Tensor(x))
    assert np.allclose(output.asnumpy(), expect_out, rtol=1e-5, equal_nan=True)
    assert np.allclose(output_grad.asnumpy(), expect_grad, rtol=1e-5, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=4.0)
def test_sign_dynamic_shape():
    """
    Feature: dynamic shape forward, backward features.
    Description: test sign forward with dynamic shape.
    Expectation: expect correct result.
    """
    tensor_x1 = ms.Tensor(generate_random_input((2, 3, 4, 5), np.float32))
    tensor_x2 = ms.Tensor(generate_random_input((2, 5, 4), np.float32))
    TEST_OP(inplace_sign_forward_func, [[tensor_x1], [tensor_x2]], disable_mode=['GRAPH_MODE_GE'])


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sign_float16(context_mode):
    """
    Feature: test sign_ functional API.
    Description: testcase for sign_ functional API.
    Expectation: the result match with expected result.
    """
    if context_mode == ms.GRAPH_MODE:
        ms.set_context(jit_level='O0')
    ms.context.set_context(mode=context_mode, device_target="Ascend")
    x = generate_random_input((2, 3, 4), np.float32)
    expect_out = np.sign(x)
    expect_grad = np.zeros_like(x, dtype=np.float32)

    output = inplace_sign_forward_func(ms.Tensor(x, dtype=ms.float16))
    output_grad = inplace_sign_backward_func(ms.Tensor(x, dtype=ms.float16))
    assert np.allclose(output.asnumpy(), expect_out, rtol=1e-4, equal_nan=True)
    assert  np.allclose(output_grad.asnumpy(), expect_grad, rtol=1e-4, equal_nan=True)
