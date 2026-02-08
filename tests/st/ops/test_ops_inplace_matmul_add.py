# Copyright 2025 Huawei Technologies Co., Ltd
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
from mindspore import ops, mint


def set_mode(mode):
    if mode == "KBK":
        context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    else:
        context.set_context(mode=context.PYNATIVE_MODE)


@test_utils.run_with_cell
def matmul_add_forward_func(x, weight, c):
    return ops.auto_generate.matmul_add_(x, weight, c)


@test_utils.run_with_cell
def matmul_add_golden_forward_func(x, weight, c):
    xt = mint.transpose(x, -1, -2)
    out = mint.matmul(xt, weight)
    return c + out


def generate_inputs(m, k, n, batch=None, dtype=ms.float16):
    if batch is not None:
        x_shape = (batch, k, m)
        w_shape = (batch, k, n)
        c_shape = (batch, m, n)
    else:
        x_shape = (k, m)
        w_shape = (k, n)
        c_shape = (m, n)
    x = Tensor(np.random.randn(*x_shape), dtype=dtype)
    w = Tensor(np.random.randn(*w_shape), dtype=dtype)
    c = Tensor(np.random.randn(*c_shape), dtype=ms.float32)
    return x, w, c


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
@pytest.mark.parametrize('mode', ['KBK'])
def test_inplace_matmul_add_static(mode):
    """
    Feature: Ops
    Description: test op InplaceMatmulAdd
    Expectation: expect correct result.
    """
    set_mode(mode)

    x, weight, c = generate_inputs(10, 20, 8)

    expect_forward_out = matmul_add_golden_forward_func(x, weight, c)
    expect_forward_out = expect_forward_out.asnumpy()

    output_forward = matmul_add_forward_func(x, weight, c)

    loss = 1e-3
    assert np.allclose(output_forward.asnumpy(), expect_forward_out, loss, loss)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=4.01
)
def test_inplace_matmul_add_dyn_shape():
    """
    Feature: Ops
    Description: test op InplaceMatmulAdd
    Expectation: expect correct result.
    """
    context.set_context(runtime_num_threads=1)  # multi-threads have none-initialized bug now.

    TEST_OP(
        matmul_add_forward_func,
        [
            list(generate_inputs(20, 30, 40, dtype=ms.bfloat16)),
            list(generate_inputs(20, 30, 40, batch=4, dtype=ms.bfloat16)),
        ],
        disable_mode=['GRAPH_MODE_GE', 'PYNATIVE_MODE'],
        disable_case=['ViewTensor', 'EmptyTensor', 'ScalarTensor'],
        case_config={'disable_grad': True},
        inplace_update=True
    )
