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
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark
from mindspore import Tensor, context
from mindspore import ops, mint


def set_mode(mode):
    if mode == "KBK":
        context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    else:
        context.set_context(mode=context.PYNATIVE_MODE)


@test_utils.run_with_cell
def gmm_add_forward_func(x, grad, group_list, out):
    return ops.auto_generate.grouped_matmul_add_(x, grad, group_list, out)


def gmm_k(x, weight, split_sizes):
    x_list = mint.split(x, split_sizes, 1)
    w_list = mint.split(weight, split_sizes, 0)
    output = []
    for i in range(len(x_list)):
        output_i = mint.matmul(x_list[i], w_list[i])
        output.append(output_i)
    out = mint.stack(output, dim=0)
    return out


@test_utils.run_with_cell
def gmm_add_golden_forward_func(grad, x, split_sizes, out):
    xt = mint.transpose(x, -1, -2)
    dw = gmm_k(xt, grad, split_sizes)
    out = dw.view(out.shape) + out
    return out


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
@pytest.mark.parametrize('mode', ['KBK'])
def test_gmm_add_static_shape(mode):
    """
    Feature: Ops
    Description: test op gmm_add
    Expectation: expect correct result.
    """
    set_mode(mode)

    m = 10
    k = 20
    n = 8
    split_sizes = [2, 4, 2, 2]
    x = Tensor(np.random.randn(m, k).astype(np.float16))
    grad = Tensor(np.random.randn(m, n).astype(np.float16))
    out = Tensor(np.random.randn(len(split_sizes), k, n).astype(np.float32))
    expect_forward_out = gmm_add_golden_forward_func(grad, x, split_sizes, out)
    expect_forward_out = expect_forward_out.asnumpy()

    group_list = []
    for i in range(len(split_sizes)):
        if i == 0:
            group_list.append(split_sizes[i])
            continue
        group_list.append(group_list[i-1] + split_sizes[i])
    group_list = Tensor(group_list)
    output_forward = gmm_add_forward_func(x, grad, group_list, out)

    loss = 1e-3
    assert np.allclose(output_forward.asnumpy(), expect_forward_out, loss, loss)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=4.01
)
def test_gmm_dyn_shape():
    """
    Feature: Ops
    Description: test op gmm and gmm_backward
    Expectation: expect correct result.
    """
    context.set_context(runtime_num_threads=1)  # multi-threads have none-initialized bug now.

    m = 10
    k = 20
    n = 8
    group_list = Tensor([2, 6, 8, 10])
    x = Tensor(np.random.randn(m, k).astype(np.float16))
    grad = Tensor(np.random.randn(m, n).astype(np.float16))
    out = Tensor(np.random.randn(group_list.shape[0] * k, n).astype(np.float32))
    inputs_0 = [x, grad, group_list, out]

    m = 20
    k = 30
    n = 8
    group_list = Tensor([2, 6, 8, 10, 14, 17, 20])
    x = Tensor(np.random.randn(m, k).astype(np.float16))
    grad = Tensor(np.random.randn(m, n).astype(np.float16))
    out = Tensor(np.random.randn(group_list.shape[0], k, n).astype(np.float32))
    inputs_1 = [x, grad, group_list, out]

    TEST_OP(
        gmm_add_forward_func,
        [
            inputs_0,
            inputs_1,
        ],
        disable_mode=['GRAPH_MODE_GE', 'PYNATIVE_MODE'],
        disable_case=['ViewTensor', 'EmptyTensor', 'ScalarTensor'],
        case_config={'disable_input_check': True,
                     'disable_grad': True,
                     'deterministic_use_origin_inputs': True},
        inplace_update=True
    )
