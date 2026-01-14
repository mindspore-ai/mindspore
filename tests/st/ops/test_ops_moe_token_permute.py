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
from mindspore import ops


@test_utils.run_with_cell
def permuted_forward_func(tokens, indices, num_out_tokens=0, padded_mode=False):
    return ops.moe_token_permute(tokens, indices, num_out_tokens, padded_mode)


@test_utils.run_with_cell
def permuted_backward_func(tokens, indices, num_out_tokens=0, padded_mode=False):
    return ms.grad(permuted_forward_func, (0, 1))(
        tokens, indices, num_out_tokens, padded_mode
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
    essential_mark="essential",
    mem_peak=20.01
)
@pytest.mark.parametrize("mode", ["KBK", "PYBOOST"])
@pytest.mark.parametrize("dtype", [ms.float16, ms.float32, ms.bfloat16])
def test_moe_token_permute(mode, dtype):
    """
    Feature: moe_token_permute
    Description: test ops.moe_token_permute
    Expectation: expect correct result.
    """
    set_mode(mode)
    loss = {ms.float16: 1e-3, ms.float32: 1e-4, ms.bfloat16: 4e-3}[dtype]

    tokens = Tensor([
        [1.41924731, -1.1341249, 11.987412], [0, 0, 0], [0, 0, 0], [3.1049871, -1233.91741, 15.13424], [2, 2, 2],
        [1.12413, 1.1974134, -1.91741234], [2, 2, 2], [-3.14210, 33.24124, 3.917513]
    ], dtype=dtype)
    indices = Tensor([0, 6, 7, 5, 3, 1, 2, 4], dtype=ms.int32)
    num_out_tokens = 0
    permuted_tokens, sorted_indices = permuted_forward_func(tokens, indices, num_out_tokens)
    expect_permuted_tokens = np.array([[1.41924731, -1.1341249, 11.987412],
                                       [1.12413, 1.1974134, -1.91741234],
                                       [2.000000, 2.000000, 2.000000],
                                       [2.000000, 2.000000, 2.000000],
                                       [-3.14210, 33.24124, 3.917513],
                                       [3.1049871, -1233.91741, 15.13424],
                                       [0.000000, 0.000000, 0.000000],
                                       [0.000000, 0.000000, 0.000000]]).astype(np.float32)
    expect_sorted_indices = np.array([0, 6, 7, 5, 3, 1, 2, 4]).astype(np.float32)

    assert np.allclose(permuted_tokens.float().asnumpy() if dtype == ms.bfloat16 else permuted_tokens.asnumpy(),
                       expect_permuted_tokens, loss, loss)
    assert np.allclose(sorted_indices.float().asnumpy() if dtype == ms.bfloat16 else sorted_indices.asnumpy(),
                       expect_sorted_indices, loss, loss)

    permuted_grad = permuted_backward_func(tokens, indices, num_out_tokens)
    expect_permuted_grad = (np.ones((8, 3))).astype(np.float32)
    assert np.allclose(permuted_grad[0].float().asnumpy() if dtype == ms.bfloat16 else permuted_grad[0].asnumpy(),
                       expect_permuted_grad, loss, loss)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=20.01
)
def test_permute_dynamic():
    """
    Feature: Ops
    Description: test op moe_token_permute dynamic shape
    Expectation: expect correct result.
    """
    num_tokens = 7
    hidden_size = 50
    topk = 3
    num_experts = 4
    tokens = Tensor(np.random.randn(num_tokens, hidden_size), dtype=ms.bfloat16)
    indices = Tensor(np.random.randint(0, num_experts, (num_tokens, topk))).to(ms.int32)
    num_out_tokens = 1
    input_case1 = [tokens, indices, num_out_tokens, False]

    # test case 2
    num_tokens = 2
    hidden_size = 20
    topk = 3
    num_experts = 6
    tokens = Tensor(np.random.randn(num_tokens, hidden_size), dtype=ms.bfloat16)
    indices = Tensor(np.random.randint(0, num_experts, (num_tokens, topk))).to(ms.int32)
    num_out_tokens = 2
    input_case2 = [tokens, indices, num_out_tokens, False]
    TEST_OP(
        permuted_forward_func,
        [input_case1, input_case2],
        disable_mode=["GRAPH_MODE_GE"],
        disable_case=['EmptyTensor', 'ScalarTensor'],
        case_config={'disable_input_check': True},
    )
