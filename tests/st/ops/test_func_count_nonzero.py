# Copyright 2024 Huawei Technocasties Co., Ltd
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
from mindspore.mint import count_nonzero
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark
import tests.st.utils.test_utils as test_utils


def generate_random_input(shape, dtype):
    return np.random.randint(0, 2, shape).astype(dtype)


@test_utils.run_with_cell
def count_nonzero_forward_func(x, dim=None):
    return count_nonzero(x, dim=dim)


@test_utils.run_with_cell
def count_nonzero_backward_func(x, dim=None):
    return ms.grad(count_nonzero_forward_func, (0))(x, dim=dim)


def set_mode(mode):
    if mode == "GRAPH_MODE":
        ms.context.set_context(mode=ms.GRAPH_MODE,
                               jit_config={"jit_level": "O0"})
    else:
        ms.context.set_context(mode=ms.PYNATIVE_MODE)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="essential", mem_peak=2048.01
)
@pytest.mark.parametrize("context_mode", ["PYNATIVE_MODE", "GRAPH_MODE"])
def test_mint_count_nonzero_normal(context_mode):
    """
    Feature: pyboost function.
    Description: test function count_nonzero forward.
    Expectation: expect correct result.
    """
    set_mode(context_mode)
    x = generate_random_input((2, 3, 4), np.bool_)
    output = count_nonzero_forward_func(ms.Tensor(x), dim=(1))
    backward_output = count_nonzero_backward_func(ms.Tensor(x), dim=(1))
    assert output.asnumpy().dtype == "int64"
    assert output.asnumpy().shape == (2, 4)
    assert backward_output.asnumpy().dtype == "bool"
    assert backward_output.asnumpy().shape == (2, 3, 4)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=1.00
)
def test_mint_count_nonzero_forward_dynamic_shape():
    """
    Feature: pyboost function.
    Description: test function count_nonzero forward with dynamic shape.
    Expectation: expect correct result.
    """
    input1 = ms.Tensor(generate_random_input((2, 3, 4), np.float32))
    axis1 = (0, -1)
    input2 = ms.Tensor(generate_random_input((3, 3, 4, 4), np.float32))
    axis2 = (0, 1, -1)
    TEST_OP(count_nonzero_forward_func, [[input1, axis1], [input2, axis2]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['ScalarTensor'])

    input3 = ms.Tensor(generate_random_input((2, 3, 4), np.float32))
    input4 = ms.Tensor(generate_random_input((2, 3), np.float32))
    TEST_OP(count_nonzero_forward_func, [[input3], [input4]],
            disable_mode=['GRAPH_MODE_GE'])
