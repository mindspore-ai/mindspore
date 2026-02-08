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
# pylint: disable=unused-variable
import pytest
import numpy as np
import mindspore as ms
from mindspore import mint, jit
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


def generate_random_input(shape):
    return np.random.randn(*shape).astype(np.float32)


def generate_expect_forward_output(x, min_val, max_val):
    return np.minimum(np.maximum(min_val, x), max_val)


@test_utils.run_with_cell
def hardtanh_forward_func(x, min_val, max_val):
    mint.nn.functional.hardtanh_(x, min_val, max_val)
    return x


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.0)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_hardtanh_std(mode):
    """
    Feature: standard forward, backward features.
    Description: test function hardtanh_
    Expectation: expect correct result.
    """
    x_np = generate_random_input((2, 3))
    x = ms.Tensor(x_np, dtype=ms.float32)
    min_val = -1.0
    max_val = 4.0
    expect = generate_expect_forward_output(x_np, min_val, max_val)

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = hardtanh_forward_func(x, min_val, max_val)
        inplace_x = hardtanh_forward_func(ms.Tensor(x_np, dtype=ms.float32), min_val, max_val)
    else:
        output = (jit(hardtanh_forward_func, backend="ms_backend", jit_level="O0"))(x, min_val, max_val)
        inplace_x = (jit(hardtanh_forward_func, backend="ms_backend", jit_level="O0"))(
            ms.Tensor(x_np, dtype=ms.float32), min_val, max_val)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(inplace_x.asnumpy(), expect, rtol=1e-5, atol=1e-5)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=2048.01)
def test_hardtanh_dynamic_shape():
    """
    Feature: dynamic shape forward, backward features.
    Description: test hardtanh_ forward with dynamic shape.
    Expectation: expect correct result.
    """
    x1_np = generate_random_input((3, 4, 5))
    min_val_1 = -2
    max_val_1 = 4
    x2_np = generate_random_input((3, 7, 8, 3))
    min_val_2 = -3
    max_val_2 = 3
    TEST_OP(hardtanh_forward_func,
            [[ms.Tensor(x1_np), min_val_1, max_val_1], [ms.Tensor(x2_np), min_val_2, max_val_2]],
            disable_mode=['GRAPH_MODE_GE'],
            case_config={'disable_grad': True,
                         'all_dim_zero': True})


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_hardtanh_bfloat16(mode):
    """
    Feature: test hardtanh_ functional API.
    Description: testcase for hardtanh_ functional API.
    Expectation: the result match with expected result.
    """
    x_np = generate_random_input((2, 3, 4))
    x = ms.Tensor(x_np, dtype=ms.bfloat16)
    min_val = -2
    max_val = 4
    expect = generate_expect_forward_output(x_np, min_val, max_val)

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = hardtanh_forward_func(x, min_val, max_val)
        inplace_x = hardtanh_forward_func(ms.Tensor(x_np, dtype=ms.bfloat16), min_val, max_val)
    else:
        output = (jit(hardtanh_forward_func, jit_level="O0"))(x, min_val, max_val)
        inplace_x = (jit(hardtanh_forward_func, jit_level="O0"))(
            ms.Tensor(x_np, dtype=ms.bfloat16), min_val, max_val)
    np.allclose(output.float().asnumpy(), expect, 4e-3, 4e-3, equal_nan=True)
    np.testing.assert_allclose(inplace_x.asnumpy(), expect, rtol=4e-3, atol=4e-3)
