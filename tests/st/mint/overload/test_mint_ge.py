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
from mindspore import Tensor, mint

import tests.st.utils.test_utils as test_utils
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype), np.random.randn(*shape).astype(dtype)


@test_utils.run_with_cell
def ge_forward_func(x, other):
    return mint.ge(x, other)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=2048.01)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_ge_normal(context_mode):
    """
    Feature: pyboost function.
    Description: test function ge forward.
    Expectation: expect correct result.
    """
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=context_mode)
    x = Tensor(np.array([1, 2, 3]), ms.int32)
    y = Tensor(np.array([1, 1, 4]), ms.int32)
    output = ge_forward_func(x, y)
    expect_output = Tensor([True, True, False])
    assert np.allclose(output.asnumpy(), expect_output.asnumpy())
    y2 = 2
    output2 = ge_forward_func(x, y2)
    expect_output2 = Tensor([False, True, True])
    assert np.allclose(output2.asnumpy(), expect_output2.asnumpy())


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=4.0)
def test_ops_ge_forward_dynamic_shape():
    """
    Feature: pyboost function.
    Description: test function ge forward with dynamic shape.
    Expectation: expect correct result.
    """
    x, y = generate_random_input((3, 4, 5, 6), np.float16)
    x = ms.Tensor(x, dtype=ms.float16)
    y = ms.Tensor(y, dtype=ms.float16)
    x2, y2 = generate_random_input((3, 4), np.float16)
    x2 = ms.Tensor(x2, dtype=ms.float16)
    y2 = ms.Tensor(y2, dtype=ms.float16)
    TEST_OP(ge_forward_func, [[x, y], [x2, y2]],
            disable_mode=['GRAPH_MODE_GE'],
            case_config={'all_dim_zero': True})
    y3 = 1
    y4 = 2
    TEST_OP(ge_forward_func, [[x, y3], [x2, y4]],
            disable_mode=['GRAPH_MODE_GE'],
            case_config={'all_dim_zero': True})
