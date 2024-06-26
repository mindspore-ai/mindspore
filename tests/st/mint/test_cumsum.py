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
from mindspore import ops, jit, JitConfig
from mindspore.mint import cumsum
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark

def generate_random_input(shape, dim):
    x = np.random.randn(*shape)
    return x, np.cumsum(x, dim)


def cumsum_func(x, dim):
    return cumsum(x, dim)


@test_utils.run_with_cell
def cumsum_forward_func(x, dim):
    return cumsum_func(x, dim)


def cumsum_bwd_func(x, dim):
    return ops.grad(cumsum_func, (0,))(x, dim)


@test_utils.run_with_cell
def cumsum_backward_func(x, dim):
    return cumsum_bwd_func(x, dim)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_cumsum_forward(mode):
    """
    Feature: Ops.
    Description: test op cumsum.
    Expectation: expect correct result.
    """
    test_shape = (2, 3, 4, 5)
    dim1 = 2
    x, expect = generate_random_input(test_shape, dim1)
    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
        output = cumsum_forward_func(ms.Tensor(x), dim1)
    else:
        output = (jit(cumsum_forward_func, jit_config=JitConfig(jit_level="O0")))(ms.Tensor(x), dim1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_cumsum_bfloat16(mode):
    """
    Feature: test ne functional API.
    Description: testcase for ne functional API.
    Expectation: the result match with expected result.
    """
    test_shape = (2, 3, 4)
    dim1 = 1
    x, expect = generate_random_input(test_shape, dim1)
    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
        output = cumsum_forward_func(ms.Tensor(x), dim1)
    else:
        output = (jit(cumsum_forward_func, jit_config=JitConfig(jit_level="O0")))(ms.Tensor(x), dim1)
    np.testing.assert_allclose(output.float().asnumpy(), expect, rtol=5e-3, atol=5e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_cumsum_backward(mode):
    """
    Feature: Ops.
    Description: test op cumsum.
    Expectation: expect correct result.
    """
    test_shape = (2, 3, 4, 5)
    dim1 = 0
    x, _ = generate_random_input(test_shape, dim1)
    expect = np.flip(np.cumsum(np.flip(np.ones(test_shape), dim1), dim1), dim1)
    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
        output = cumsum_backward_func(ms.Tensor(x), dim1)
    else:
        output = (jit(cumsum_backward_func, jit_config=JitConfig(jit_level="O0")))(ms.Tensor(x), dim1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4)



@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_cumsum_dynamic_shape():
    """
    Feature: Test dynamic shape.
    Description: test function div dynamic feature.
    Expectation: expect correct result.
    """
    dim1 = 0
    ms_data1, _ = generate_random_input((2, 3, 4), dim1)
    dim2 = 1
    ms_data2, _ = generate_random_input((3, 4, 5, 6), dim2)
    TEST_OP(cumsum_forward_func, [[ms.Tensor(ms_data1), dim1], [ms.Tensor(ms_data2), dim2]],
            '', disable_yaml_check=True, disable_mode=['GRAPH_MODE'])
