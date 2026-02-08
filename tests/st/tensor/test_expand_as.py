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
import mindspore.context as context
from mindspore.common.tensor import Tensor
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP


class Net(ms.nn.Cell):
    def construct(self, x, other):
        return x.expand_as(other)


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def forward_func(x, other):
    return Net()(x, other)


def expect_forward_func(x_np, other_np):
    return np.broadcast_to(x_np, other_np.shape)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="essential", mem_peak=1024.01
)
@pytest.mark.parametrize("mode", [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_expand_as_forward(mode):
    """
    Feature: pyboost function.
    Description: test function expand_as forward.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)
    if mode == ms.GRAPH_MODE:
        ms.set_context(jit_level="O0")

    input_np = generate_random_input((128, 1, 1, 1), np.float32)
    other_np = generate_random_input((128, 1, 77, 77), np.float32)
    out = forward_func(Tensor(input_np), Tensor(other_np))
    expect = expect_forward_func(input_np, other_np)
    assert np.allclose(out.asnumpy(), expect, atol=1e-4, rtol=1e-4)

    input_np = generate_random_input((4, 5, 1), np.float32)
    other_np = generate_random_input((3, 5, 7, 4, 5, 6), np.float32)
    out = forward_func(Tensor(input_np), Tensor(other_np))
    expect = expect_forward_func(input_np, other_np)
    assert np.allclose(out.asnumpy(), expect, atol=1e-4, rtol=1e-4)

    input_np = generate_random_input((2, 3, 1, 5, 1), np.float32)
    other_np = generate_random_input((4, 5, 2, 3, 4, 5, 6), np.float32)
    out = forward_func(Tensor(input_np), Tensor(other_np))
    expect = expect_forward_func(input_np, other_np)
    assert np.allclose(out.asnumpy(), expect, atol=1e-4, rtol=1e-4)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="essential", mem_peak=1.00
)
@pytest.mark.parametrize("mode", [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_expand_as_dtype(mode):
    """
    Feature: Test supported data types of Tensor.expand_as.
    Description: all data types
    Expectation: success.
    """
    context.set_context(mode=mode)
    if mode == ms.GRAPH_MODE:
        ms.set_context(jit_level="O0")

    types = [np.float16, np.float32, np.uint8, np.int8, np.int32, np.int64, np.bool_]
    for dtype in types:
        input_np = generate_random_input((10, 1), dtype)
        other_np = generate_random_input((10, 10), dtype)
        out = forward_func(Tensor(input_np), Tensor(other_np))
        expect = expect_forward_func(input_np, other_np)
        assert np.allclose(out.asnumpy(), expect, atol=1e-3, rtol=1e-3)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=2048.01
)
def test_expand_as_dynamic_shape():
    """
    Feature: Test dynamic shape.
    Description: test function expand_as dynamic feature.
    Expectation: expect correct result.
    """
    input1 = generate_random_input((1, 1), np.float32)
    other1 = generate_random_input((10, 10), np.float32)
    input2 = generate_random_input((1, 1, 1, 1), np.float32)
    other2 = generate_random_input((10, 10, 10, 10), np.float32)

    TEST_OP(
        forward_func,
        [[Tensor(input1), Tensor(other1)], [Tensor(input2), Tensor(other2)]],
        disable_mode=["GRAPH_MODE_GE"],
        case_config={'disable_input_check': True,
                     'all_dim_zero': True}
    )
