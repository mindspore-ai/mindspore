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
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark
import torch


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x, y):
    return x.view_as(y)


def generate_expect_backward_output(x):
    return np.ones_like(x)


@test_utils.run_with_cell
def view_as_forward_func(x, y):
    return ms.ops.view_as(x, y)


@test_utils.run_with_cell
def view_as_backward_func(x, y):
    return ms.grad(view_as_forward_func, (0))(x, y)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.01)
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_view_as_normal0(mode):
    """
    Feature: test ops.
    Description: test function view_as forward and backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    if mode == ms.GRAPH_MODE:
        ms.context.set_context(jit_level='O0')
    x = generate_random_input((3, 4, 5), np.float32)
    y = generate_random_input((5, 2, 6), np.float32)

    output = view_as_forward_func(ms.Tensor(x), ms.Tensor(y))
    expect = generate_expect_forward_output(torch.Tensor(x), torch.Tensor(y))
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    output = view_as_backward_func(ms.Tensor(x), ms.Tensor(y))
    expect = generate_expect_backward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=2048.00)
def test_ops_view_as_dynamic_shape0():
    """
    Feature: test ops.
    Description: test function view_as with dynamic shape and dynamic rank.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((2, 3, 4), np.float32)
    y1 = generate_random_input((3, 8), np.float32)
    x2 = generate_random_input((5, 6), np.float32)
    y2 = generate_random_input((2, 3, 5), np.float32)

    TEST_OP(view_as_forward_func,
            [[ms.Tensor(x1), ms.Tensor(y1)], [ms.Tensor(x2), ms.Tensor(y2)]],
            disable_mode=["GRAPH_MODE_GE"])
