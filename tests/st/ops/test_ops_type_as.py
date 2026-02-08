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
from mindspore.ops import type_as
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark
import tests.st.utils.test_utils as test_utils


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


@test_utils.run_with_cell
def type_as_forward_func(x, y):
    return type_as(x, y)


@test_utils.run_with_cell
def type_as_backward_func(x, y):
    return ms.grad(type_as_forward_func, (0))(x, y)


def set_mode(mode):
    if mode == "GRAPH_MODE":
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})
    else:
        ms.context.set_context(mode=ms.PYNATIVE_MODE)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="essential", mem_peak=2048.01
)
@pytest.mark.parametrize("context_mode", ["PYNATIVE_MODE", "GRAPH_MODE"])
def test_ops_type_as_normal(context_mode):
    """
    Feature: pyboost function.
    Description: test function type_as forward.
    Expectation: expect correct result.
    """
    set_mode(context_mode)
    x = generate_random_input((1280, 1280), np.float16)
    y = generate_random_input((1280, 1280), np.float32)
    output = type_as_forward_func(ms.Tensor(x), ms.Tensor(y))
    backward_output = type_as_backward_func(ms.Tensor(x), ms.Tensor(y))
    assert output.asnumpy().dtype == "float32"
    assert output.asnumpy().shape == (1280, 1280)
    assert backward_output.asnumpy().dtype == "float16"
    assert backward_output.asnumpy().shape == (1280, 1280)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=2048.01
)
def test_ops_type_as_forward_dynamic_shape():
    """
    Feature: pyboost function.
    Description: test function type_as forward with dynamic shape.
    Expectation: expect correct result.
    """
    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    y1 = generate_random_input((1, 2, 3), np.float16)
    x2 = generate_random_input((1, 2, 3), np.float32)
    y2 = generate_random_input((2, 3, 4, 5), np.float16)
    TEST_OP(
        type_as_forward_func,
        [[ms.Tensor(x1), ms.Tensor(y1)], [ms.Tensor(x2), ms.Tensor(y2)]],
        disable_mode=["GRAPH_MODE_GE"],
    )
