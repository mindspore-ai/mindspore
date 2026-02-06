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
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark

def generate_random_input(shape, dtype):
    res = np.random.randn(*shape).astype(dtype)
    threshold = 0
    res[res > threshold] = np.nan
    return res

def generate_expect_forward_output(x):
    return np.isnan(x)

@test_utils.run_with_cell
def isnan_forward_func(x):
    return mint.isnan(x)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=4.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_isnan_ascend(context_mode):
    """
    Feature: pyboost function.
    Description: test function round forward and backward on Ascend.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3), np.float32)
    output = isnan_forward_func(ms.Tensor(x))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    x = generate_random_input((4, 6, 8, 3), np.float32)
    output = isnan_forward_func(ms.Tensor(x))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.0)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_isnan_dynamic_shape_ascend(context_mode):
    """
    Feature: pyboost function.
    Description: test function round with dynamic shape on Ascend.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    x2 = generate_random_input((6, 7, 8), np.float32)
    TEST_OP(isnan_forward_func, [[ms.Tensor(x1)], [ms.Tensor(x2)]], disable_mode=['GRAPH_MODE_GE'])
