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


@test_utils.run_with_cell
def unpermuted_forward_func(permuted_tokens, sorted_indices, probs, padded_mode=False, restore_shape=None):
    return ops.moe_token_unpermute(permuted_tokens, sorted_indices, probs, padded_mode, restore_shape)


@test_utils.run_with_cell
def unpermuted_backward_func(permuted_tokens, sorted_indices, probs, padded_mode=False, restore_shape=None):
    return ms.grad(unpermuted_forward_func, (0, 2))(
        permuted_tokens, sorted_indices, probs, padded_mode, restore_shape
    )

def set_mode(mode):
    """
    set mode
    """
    if mode == "KBK":
        context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    else:
        context.set_context(mode=context.PYNATIVE_MODE)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="essential", mem_peak=4.01
)
@pytest.mark.parametrize("mode", ["KBK", "PYBOOST"])
def test_moe_token_unpermute_empty(mode):
    """
    Feature: moe_token_unpermute
    Description: test ops.moe_token_unpermute for empty tensor
    Expectation: expect correct result.
    """
    set_mode(mode)

    permuted_tokens = Tensor(np.random.randn(9, 20), dtype=ms.bfloat16)
    permuted_tokens = ops.slice(permuted_tokens, (0, 0), (0, 20))
    sorted_indices = Tensor(np.random.randn(18,), dtype=ms.int32)
    sorted_indices = ops.slice(sorted_indices, (0,), (0,))
    out = unpermuted_forward_func(permuted_tokens, sorted_indices, None)
    out_grad = ms.grad(unpermuted_forward_func, (0,))(permuted_tokens, sorted_indices, None, False, None)
    assert out.shape == (0, 20)
    assert out_grad.shape == (0, 20)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="essential", mem_peak=20.01
)
@pytest.mark.parametrize("mode", ["KBK", "PYBOOST"])
def test_moe_token_unpermute(mode):
    """
    Feature: moe_token_unpermute
    Description: test ops.moe_token_unpermute
    Expectation: expect correct result.
    """
    set_mode(mode)

    permuted_tokens = Tensor([
        [1, 1, 1], [0, 0, 0], [0, 0, 0], [3, 3, 3], [2, 2, 2], [1, 1, 1], [2, 2, 2], [3, 3, 3]
    ], dtype=ms.bfloat16)
    sorted_indices = Tensor([0, 6, 7, 5, 3, 1, 2, 4], dtype=ms.int32)
    probs = Tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=ms.bfloat16)

    out = unpermuted_forward_func(permuted_tokens, sorted_indices, probs)
    dx, dprob = unpermuted_backward_func(permuted_tokens, sorted_indices, probs)

    expect_out = np.array([[1.5, 1.5, 1.5], [2.0, 2.0, 2.0], [1.5, 1.5, 1.5], [1.0, 1.0, 1.0]]).astype(np.float32)
    expect_dx = (np.ones((8, 3)) * 0.5).astype(np.float32)
    expect_dprob = np.array([[3.0, 6.0], [9.0, 3.0], [9.0, 0.], [0., 6.0]]).astype(np.float32)

    np.allclose(out.float().asnumpy(), expect_out, 0.004, 0.004)
    np.allclose(dx.float().asnumpy(), expect_dx, 0.004, 0.004)
    np.allclose(dprob.float().asnumpy(), expect_dprob, 0.004, 0.004)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
    mem_peak=20.01
)
def test_unpermute_dynamic():
    """
    Feature: Ops
    Description: test op moe_token_unpermute dynamic shape
    Expectation: expect correct result.
    """
    num_tokens = 6
    hidden_size = 50
    topk = 2
    num_experts = 4
    permuted_token = Tensor(np.random.randn(num_tokens * topk, hidden_size), dtype=ms.bfloat16)
    indices = Tensor(np.random.randint(0, num_experts, (num_tokens, topk))).to(ms.int32)
    indices = mint.argsort(indices.view(-1)).to(dtype=ms.int32)
    probs = (mint.ones((num_tokens, topk)) / topk).to(ms.bfloat16)
    input_case1 = [permuted_token, indices, probs, False, None]

    # test case 2
    num_tokens = 2
    hidden_size = 20
    topk = 3
    num_experts = 6
    permuted_token = Tensor(np.random.randn(num_tokens * topk, hidden_size), dtype=ms.bfloat16)
    indices = Tensor(np.random.randint(0, num_experts, (num_tokens, topk))).to(ms.int32)
    indices = mint.argsort(indices.view(-1)).to(dtype=ms.int32)
    probs = (mint.ones((num_tokens, topk)) / topk).to(ms.bfloat16)
    input_case2 = [permuted_token, indices, probs, False, None]
    TEST_OP(
        unpermuted_forward_func,
        [input_case1, input_case2],
        disable_mode=["GRAPH_MODE_GE"],
        disable_case=['EmptyTensor', 'ScalarTensor'],
        case_config={'disable_input_check': True,
                     'deterministic_use_origin_inputs': True}
    )
