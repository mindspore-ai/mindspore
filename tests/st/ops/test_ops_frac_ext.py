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
Tests for frac ext
"""
import pytest
import numpy as np
import mindspore as ms
from mindspore import mint
from tests.mark_utils import arg_mark
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


@test_utils.run_with_cell
def frac_forward_func(x):
    return mint.frac(x)


@test_utils.run_with_cell
def frac_backward_func(x):
    return ms.grad(frac_forward_func, (0))(x)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.01)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_mint_frac_normal(mode):
    """
    Feature: pyboost function.
    Description: test function frac forward.
    Expectation: expect correct result.
    """
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=mode)
    x = generate_random_input((2, 3), np.float32)
    expect_output = np.fmod(x, 1)
    output = frac_forward_func(ms.Tensor(x))
    np.testing.assert_allclose(output.asnumpy(), expect_output, rtol=1e-4)

    x1 = np.array([-1.5, 1.8, 2.3]).astype(np.float32)
    expect_output = np.array([1., 1., 1.]).astype(np.float32)
    output = frac_backward_func(ms.Tensor(x1))
    np.testing.assert_allclose(output.asnumpy(), expect_output, rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=2048.01)
def test_mint_std_dynamic_shape():
    """
    Feature: Test operator Frac by TEST_OP.
    Description: Test operator Frac with dynamic input.
    Expectation: return the correct value.
    """
    x1 = ms.Tensor(generate_random_input((2, 3), np.float32))
    x2 = ms.Tensor(generate_random_input((2, 3, 4), np.float32))
    TEST_OP(frac_forward_func, [[x1], [x2]], disable_mode=["GRAPH_MODE_GE"])

#mint.frac support int_dtype but torch does not, so int type cases maintained here
@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=2048.01)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_mint_frac_dtype_int(mode):
    """
    Feature: Test operator Frac by TEST_OP.
    Description: Test operator Frac with dynamic input.
    Expectation: return the correct value.
    """
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=mode)
    np_arr = ms.Tensor(generate_random_input((2, 3), np.int32))
    x1 = mint.frac(ms.Tensor(np_arr, dtype = ms.uint8))
    x2 = mint.frac(ms.Tensor(np_arr, dtype = ms.int8))
    x3 = mint.frac(ms.Tensor(np_arr, dtype = ms.int16))
    x4 = mint.frac(ms.Tensor(np_arr, dtype=ms.int32))
    x5 = mint.frac(ms.Tensor(np_arr, dtype=ms.int64))
    assert np.allclose(x1.asnumpy(), np.zeros((2,3)))
    assert np.allclose(x2.asnumpy(), np.zeros((2, 3)))
    assert np.allclose(x3.asnumpy(), np.zeros((2, 3)))
    assert np.allclose(x4.asnumpy(), np.zeros((2, 3)))
    assert np.allclose(x5.asnumpy(), np.zeros((2, 3)))
