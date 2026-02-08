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
from mindspore import mint
from tests.mark_utils import arg_mark
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x):
    return np.bitwise_not(x)


@test_utils.run_with_cell
def bitwise_not_forward_func(x):
    return mint.bitwise_not(x)



@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=2048.01)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_mint_bitwise_not_normal(mode):
    """
    Feature: pyboost function.
    Description: test function bitwise_not forward.
    Expectation: expect correct result.
    """
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=mode)
    x = generate_random_input((2, 3), np.int8)
    expect_output = generate_expect_forward_output(x)
    output = bitwise_not_forward_func(ms.Tensor(x))
    np.testing.assert_allclose(output.asnumpy(), expect_output, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=2048.0)
def test_mint_bitwise_not_dynamic_shape():
    """
    Feature: Test operator bitwise_not by TEST_OP.
    Description: Test operator bitwise_not with dynamic input.
    Expectation: return the correct value.
    """
    x1 = ms.Tensor(generate_random_input((2, 3), np.int32))
    x2 = ms.Tensor(generate_random_input((2, 3, 4), np.int32))
    TEST_OP(bitwise_not_forward_func, [[x1], [x2]], disable_mode=["GRAPH_MODE_GE"])
