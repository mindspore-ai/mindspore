# Copyright 2023 Huawei Technologies Co., Ltd
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
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark

import mindspore as ms
from mindspore import Tensor, context
from mindspore import mint


@test_utils.run_with_cell
def gelu_none_forward_func(x):
    return mint.nn.functional.gelu(x, approximate="none")


@test_utils.run_with_cell
def gelu_none_backward_func(x):
    return ms.grad(gelu_none_forward_func, (0,))(x)


@test_utils.run_with_cell
def gelu_tanh_forward_func(x):
    return mint.nn.functional.gelu(x, approximate="tanh")


@test_utils.run_with_cell
def gelu_tanh_backward_func(x):
    return ms.grad(gelu_tanh_forward_func, (0,))(x)


def set_mode(mode):
    """
    set mode
    """
    if mode == "KBK":
        context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    else:
        context.set_context(mode=context.PYNATIVE_MODE)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard",
        essential_mark="essential", mem_peak=2.00)
@pytest.mark.parametrize("mode", ["KBK", "PYBOOST"])
def test_gelu_static(mode):
    """
    Feature: GeluExt
    Description: test interface of mint.nn.functional.gelu
    Expectation: expect correct result.
    """
    set_mode(mode)
    x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), ms.float32)
    loss = 1e-4

    # mode "none"
    expect_out = np.array([[-1.5866e-01, 3.9999e+00, -0.0000e+00],
                           [1.9545e+00, -1.4901e-06, 9.0000e+00]]).astype(np.float32)
    expect_dx = np.array([[-8.3315e-02, 1.0005e+00, -4.0418e-14],
                          [1.0852e+00, -7.1356e-06, 1.0000e+00]]).astype(np.float32)
    out = gelu_none_forward_func(x)
    dx = gelu_none_backward_func(x)
    assert np.allclose(out.asnumpy(), expect_out, loss, loss)
    assert np.allclose(dx.asnumpy(), expect_dx, loss, loss)

    # mode "tanh"
    expect_out = np.array([[-1.5881e-01, 3.9999e+00, -0.0000e+00],
                           [1.9546e+00, -2.9802e-07, 9.0000e+00]]).astype(np.float32)
    expect_dx = np.array([[-8.2964e-02, 1.0003e+00, 0.0000e+00],
                          [1.0861e+00, -2.0109e-06, 1.0000e+00]]).astype(np.float32)
    out = gelu_tanh_forward_func(x)
    dx = gelu_tanh_backward_func(x)
    assert np.allclose(out.asnumpy(), expect_out, loss, loss)
    assert np.allclose(dx.asnumpy(), expect_dx, loss, loss)


def generate_testcases():
    input_case1 = [Tensor(np.random.randn(2, 10, 20).astype(np.float32)),]
    input_case2 = [Tensor(np.random.randn(2, 3, 5, 7, 8, 4, 10, 20).astype(np.float32)),]
    return [input_case1, input_case2]


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
    mem_peak=38.01
)
def test_gelu_dynamic():
    """
    Feature: Ops
    Description: test op GeluExt dynamic shape
    Expectation: expect correct result.
    """
    context.set_context(
        runtime_num_threads=1
    )  # multi-threads have none-initialized bug now.

    TEST_OP(
        gelu_none_forward_func,
        generate_testcases(),
        disable_mode=["GRAPH_MODE_GE"]
    )

    TEST_OP(
        gelu_tanh_forward_func,
        generate_testcases(),
        disable_mode=["GRAPH_MODE_GE"]
    )
