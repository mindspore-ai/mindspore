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
import mindspore as ms
from mindspore import Tensor, context
from mindspore import ops, mint


def set_mode(mode):
    if mode == "KBK":
        context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    else:
        context.set_context(mode=context.PYNATIVE_MODE)


@test_utils.run_with_cell
def gmm_forward_func(x, weight, group_list):
    out = ops.function.math_func.gmm([x,], [weight,], bias=None, group_list=group_list, group_type=0)
    return out[0]


@test_utils.run_with_cell
def gmm_backward_func(x, weight, group_list):
    return ms.grad(gmm_forward_func, (0, 1))(x, weight, group_list)


@test_utils.run_with_cell
def gmm_backward_frontend_func(grad, x, weight, group_list):
    dx, dw, _ = ops.function.math_func.gmm_backward([grad,], [x,], [weight], group_list=group_list)
    return dx[0], dw[0]


@test_utils.run_with_cell
def gmm_backward_fusion_frontend_func(grad, weight, group_list):
    dx, _, _ = ops.function.math_func.gmm_backward_fusion([grad,], [weight], group_list=group_list)
    return dx[0]


@test_utils.run_with_cell
def gmm_golden_forward_func(x, weight, split_sizes):
    x_list = mint.split(x, split_sizes, 0)
    w_list = mint.split(weight, 1, 0)
    output = []
    for i in range(len(x_list)):
        output_i = mint.matmul(x_list[i], w_list[i].squeeze(0))
        output.append(output_i)
    return mint.cat(output, dim=-2)


@test_utils.run_with_cell
def gmm_golden_backward_func(x, weight, split_sizes):
    return ms.grad(gmm_golden_forward_func, (0, 1))(x, weight, split_sizes)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=20.0)
@pytest.mark.parametrize('mode', ['PYBOOST', 'KBK'])
def test_gmm_static_shape(mode):
    """
    Feature: Ops
    Description: test func gmm and gmm_backward
    Expectation: expect correct result.
    """
    set_mode(mode)
    m = 10
    k = 20
    n = 8
    split_sizes = [2, 4, 2, 2]
    x = Tensor(np.random.randn(m, k).astype(np.float32))
    w = Tensor(np.random.randn(len(split_sizes), k, n).astype(np.float32))
    expect_forward_out = gmm_golden_forward_func(x, w, split_sizes)
    expect_backward_out = gmm_golden_backward_func(x, w, split_sizes)
    expect_forward_out = expect_forward_out.asnumpy()
    expect_backward_out = expect_backward_out[0].asnumpy(), expect_backward_out[1].asnumpy()

    loss = 1e-3
    group_list = []
    for i in range(len(split_sizes)):
        if i == 0:
            group_list.append(split_sizes[i])
            continue
        group_list.append(group_list[i-1] + split_sizes[i])
    output_forward = gmm_forward_func(x, w, group_list)
    assert np.allclose(output_forward.asnumpy(), expect_forward_out, loss, loss)

    output_backward1 = gmm_backward_func(x, w, group_list)
    assert np.allclose(output_backward1[0].asnumpy(), expect_backward_out[0], loss, loss)
    assert np.allclose(output_backward1[1].asnumpy(), expect_backward_out[1], loss, loss)

    grad = Tensor(np.ones(output_forward.shape).astype(np.float32))
    output_backward2 = gmm_backward_frontend_func(grad, x, w, group_list)
    assert np.allclose(output_backward2[0].asnumpy(), expect_backward_out[0], loss, loss)
    assert np.allclose(output_backward2[1].asnumpy(), expect_backward_out[1], loss, loss)

    output_backward_fusion = gmm_backward_fusion_frontend_func(grad, w, group_list)
    assert np.allclose(output_backward_fusion.asnumpy(), expect_backward_out[0], 1e-04, 1e-04)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=20.01
)
def test_gmm_dyn_shape():
    """
    Feature: Ops
    Description: test func gmm and it's backward
    Expectation: expect correct result.
    """
    context.set_context(runtime_num_threads=1)  # multi-threads have none-initialized bug now.

    m = 10
    k = 20
    n = 14
    group_list = [2, 6, 8, 10]
    x = Tensor(np.random.randn(m, k).astype(np.float32))
    w = Tensor(np.random.randn(len(group_list), k, n).astype(np.float32))
    inputs_0 = [x, w, group_list]

    m = 20
    k = 30
    n = 8
    group_list = [2, 6, 8, 10, 14, 17, 20]
    x = Tensor(np.random.randn(m, k).astype(np.float32))
    w = Tensor(np.random.randn(len(group_list), k, n).astype(np.float32))
    inputs_1 = [x, w, group_list]

    TEST_OP(
        gmm_forward_func,
        [
            inputs_0,
            inputs_1,
        ],
        disable_case=['DiscontiguousInput', 'EmptyTensor', 'ScalarTensor'],
        disable_mode=['GRAPH_MODE_GE'],
        case_config={'disable_input_check': True,
                     'deterministic_use_origin_inputs': True}
    )
