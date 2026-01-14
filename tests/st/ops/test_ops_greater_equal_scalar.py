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

import mindspore as ms
from mindspore import mint, Tensor
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.ops.test_tools.ops_binary_cases import ops_binary_cases, OpsBinaryCase
from tests.st.utils import test_utils


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


@test_utils.run_with_cell
def greater_equal_scalar_forward_func(input1, other1):
    return mint.greater_equal(input1, other1)


@test_utils.run_with_cell
def greater_equal_scalar_forward_dyn_func(input2, other2):
    return mint.greater_equal(input2, other2)


@test_utils.run_with_cell
def greater_equal_scalar_backward_func(input3, other3):
    return ms.grad(greater_equal_scalar_forward_func, (0))(input3, other3)


@ops_binary_cases(OpsBinaryCase(input_info=[((8, 24, 48, 160), np.float32)],
                                output_info=[((8, 24, 48, 160), np.bool_)],
                                extra_info='SD5B'))
def ops_greater_equal_scalar_binary_case1(input_binary_data=None, output_binary_data=None):
    output = greater_equal_scalar_forward_func(Tensor(input_binary_data[0]), 9.7)
    assert np.allclose(output.asnumpy(), output_binary_data[0], 1e-04, 1e-04)


@ops_binary_cases(OpsBinaryCase(input_info=[((144, 96, 24), np.float32)],
                                output_info=[((144, 96, 24), np.bool_)],
                                extra_info='SD5B'))
def ops_greater_equal_scalar_binary_case2(input_binary_data=None, output_binary_data=None):
    output = greater_equal_scalar_forward_func(Tensor(input_binary_data[0]), 1)
    assert np.allclose(output.asnumpy(), output_binary_data[0], 1e-04, 1e-04)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_greater_equal_scalar_binary_cases(context_mode):
    """
    Feature: Ops
    Description: test op greater_equal_scalar pynative
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode, jit_config={"jit_level": "O0"})

    ops_greater_equal_scalar_binary_case1()
    ops_greater_equal_scalar_binary_case2()


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=4.0)
def test_ops_greater_equal_scalar_dynamic_shape():
    """
    Feature: Ops
    Description: test op greater_equal_scalar dynamic shape
    Expectation: expect correct result.
    """
    input1 = generate_random_input((6, 4, 8, 9, 4), np.float32)
    other1 = 1.0

    input2 = generate_random_input((3, 7, 8, 5), np.float32)
    other2 = 2.4
    TEST_OP(greater_equal_scalar_forward_dyn_func, [[Tensor(input1), other1], [Tensor(input2), other2]],
            disable_mode=['GRAPH_MODE_GE'])
