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

import numpy as np
import pytest
import mindspore as ms
from mindspore import context
from mindspore import jit
from mindspore.ops.auto_generate import MoeInitRouting
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark

moe_init_routing_op = MoeInitRouting()

# MoeInitRouting has 4 inputs and 3 outputs
# x:                       2D Tensor (num_row, h)
# rowIdx:                  2D Tensor (num_row, k)
# expertIdx:               2D Tensor (num_row, k)
# activeNum:               int64
# -------------------------------------------------
# expandedX:               2D Tensor (min(num_row, activeNum) * k, h)
# expandedRowIdx           1D Tensor (num_rows * k,)
# expandedExpertIdx        2D Tensor (num_row, k)

def moe_init_routing_exec(x, row_idx, expert_idx, active_num):
    num_rows = x.shape[0]
    k = expert_idx.shape[-1]
    sort_expert_for_source_row = np.argsort(
        expert_idx.reshape((-1,)), axis=-1, kind="stable")
    expanded_expert_idx = np.sort(
        expert_idx.reshape((-1,)), axis=-1)

    expanded_dst_to_src_row = np.take_along_axis(
        row_idx.reshape((-1,)), sort_expert_for_source_row, axis=-1)
    expanded_row_idx = np.zeros(expanded_dst_to_src_row.shape).astype(np.int32)
    expanded_row_idx[expanded_dst_to_src_row] = np.arange(
        expanded_dst_to_src_row.shape[-1])
    active_num = min(active_num, num_rows) * k
    expanded_x = x[expanded_dst_to_src_row[:active_num] % num_rows, :]
    return expanded_x, expanded_row_idx, expanded_expert_idx

@test_utils.run_with_cell
def moe_init_routing_forward_func(x, row_idx, expert_idx, active_num):
    return moe_init_routing_op(x, row_idx, expert_idx, active_num)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=20.0)
@pytest.mark.parametrize('mode', ['pynative', 'KBK', 'GE'])
def test_moe_init_routing_case0(mode):
    """
    Feature: Test the moe_init_routing calculate
    Description: Test the moe_init_routing ops in Ascend backend
    Expectation: The result match to the expect value.
    """
    n = 10
    col = 200
    k = 2
    activeNum = n

    # numpy input
    x = np.random.uniform(-1, 1, size=(n, col)).astype(np.float16)
    rowIdx = np.arange(n * k).reshape([k, n]).transpose(1, 0).astype(np.int32)
    expertIdx = np.random.randint(0, 100, size=(n, k)).astype(np.int32)

    expanded_x, expanded_row_idx, expanded_expert_idx = moe_init_routing_exec(x, rowIdx, expertIdx, activeNum)

    # tensor input
    x = ms.Tensor(x, ms.float16)
    rowIdx = ms.Tensor(rowIdx, ms.int32)
    expertIdx = ms.Tensor(expertIdx, ms.int32)

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        expanded_x_ms, expanded_row_idx_ms, expanded_expert_idx_ms = \
            moe_init_routing_forward_func(x, rowIdx, expertIdx, activeNum)
    elif mode == 'KBK':
        context.set_context(mode=ms.GRAPH_MODE)
        expanded_x_ms, expanded_row_idx_ms, expanded_expert_idx_ms = \
            (jit(moe_init_routing_forward_func, jit_level="O0"))(x, rowIdx, expertIdx, activeNum)
    else:
        context.set_context(mode=ms.GRAPH_MODE)
        expanded_x_ms, expanded_row_idx_ms, expanded_expert_idx_ms = \
            (jit(moe_init_routing_forward_func, backend="GE"))(x, rowIdx, expertIdx, activeNum)

    np.testing.assert_allclose(expanded_x_ms.asnumpy(), expanded_x, rtol=1e-3)
    np.testing.assert_allclose(expanded_row_idx_ms.asnumpy(), expanded_row_idx, rtol=1e-3)
    np.testing.assert_allclose(expanded_expert_idx_ms.asnumpy(), expanded_expert_idx, rtol=1e-3)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=20.0)
def test_ops_dynamic():
    """
    Feature: test moe_init_routing
    Description: dynamic shape and rank
    Expectation: success
    """
    n = 10
    col = 200
    k = 2
    activeNum = n

    # tensor input
    x_1 = ms.Tensor(np.random.uniform(-1, 1, size=(n, col)).astype(np.float16), ms.float16)
    rowIdx_1 = ms.Tensor(np.arange(n * k).reshape([k, n]).transpose(1, 0).astype(np.int32), ms.int32)
    expertIdx_1 = ms.Tensor(np.random.randint(0, 100, size=(n, k)).astype(np.int32), ms.int32)

    x_2 = ms.Tensor(np.random.uniform(-1, 1, size=(n, col)).astype(np.float16), ms.float16)
    rowIdx_2 = ms.Tensor(np.arange(n * k).reshape([k, n]).transpose(1, 0).astype(np.int32), ms.int32)
    expertIdx_2 = ms.Tensor(np.random.randint(0, 100, size=(n, k)).astype(np.int32), ms.int32)

    TEST_OP(moe_init_routing_op,
            [[x_1, rowIdx_1, expertIdx_1, activeNum], [x_2, rowIdx_2, expertIdx_2, activeNum]],
            disable_case=['ScalarTensor'],
            case_config={'disable_input_check': True,
                         'disable_grad': True})
