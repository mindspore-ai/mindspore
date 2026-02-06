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
from copy import deepcopy

import mindspore as ms
from mindspore import context
from mindspore.nn import Cell
from mindspore.ops.auto_generate import MoeFinalizeRouting

from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark


# MoeFinalizeRouting has 7 inputs and 1 outputs (token_num is  bs*seq)
# expanded_x:            2D Tensor (token_num * top_k, hidden)
# skip1:                 2D Tensor (token_num, hidden)
# skip2(optional):       2D Tensor (token_num, hidden)
# bias:                  2D Tensor (expert_num, hidden)
# scales:                2D Tensor (token_num, top_k)
# expanded_row_idx:      1D Tensor (token_num * top_k)
# expanded_expert_idx:   2D Tensor (token_num, top_k)
# ------------------------------
# y:                     2D Tensor (token_num, hidden)

def moe_finalize_routing(expanded_permuted_rows: np.ndarray,
                         skip1: np.ndarray,
                         skip2_optional: np.ndarray,
                         bias: np.ndarray,
                         scales: np.ndarray,
                         expanded_src_to_dst_row: np.ndarray,
                         expert_for_source_row: np.ndarray) -> np.ndarray:
    out = deepcopy(skip1[:])
    if skip2_optional is not None:
        out += skip2_optional
    token_num = skip1.shape[0]
    top_k = expanded_src_to_dst_row.shape[0] // token_num  # topk k

    for i in range(token_num):
        for k in range(top_k):
            dst_row = expanded_permuted_rows[expanded_src_to_dst_row[k * token_num + i], :]
            expert_id = expert_for_source_row[i, k]
            out[i, :] += scales[i, k] * (dst_row + bias[expert_id, :])
    return out


class MoeFinalizeRoutingNet(Cell):
    def __init__(self):
        super().__init__()
        self.moefzr = MoeFinalizeRouting()

    def construct(self, x, skip1, skip2, bias, scale, expanded_src_to_dst, expert_for_source_row):
        out = self.moefzr(x, skip1, skip2, bias, scale, expanded_src_to_dst, expert_for_source_row)
        return out

@test_utils.run_with_cell
def moe_finalize_routing_forward_func(x, skip1, skip2, bias, scale, expanded_src_to_dst, expert_for_source_row):
    net = MoeFinalizeRoutingNet()
    return net(x, skip1, skip2, bias, scale, expanded_src_to_dst, expert_for_source_row)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=36.0)
@pytest.mark.parametrize('mode', ['GE', 'KBK', 'pynative'])
def test_moe_finalize_routing_case0(mode):
    """
    Feature: Test the moe_finalize_routing calculate
    Description: Test the moe_finalize_routing ops in Ascend backend
    Expectation: The result match to the expect value.
    """
    context.set_context(device_target="Ascend")
    if mode == 'KBK':
        ms.set_context(mode=ms.GRAPH_MODE)
        ms.set_context(jit_level='O0')
    elif mode == 'GE':
        ms.context.set_context(mode=ms.GRAPH_MODE)
        ms.set_context(jit_level='O2')
    elif mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
    moefzr_net = MoeFinalizeRoutingNet()

    top_k = 2
    token_num = 6
    hidden = 4
    expert_num = 8

    # numpy input
    x = np.random.random((token_num * top_k, hidden)).astype(np.float16)
    skip1 = np.random.random((token_num, hidden)).astype(np.float16)
    skip2 = None
    bias = np.random.random((expert_num, hidden)).astype(np.float16)
    scale = np.random.random((token_num, top_k)).astype(np.float16)
    expanded_src_to_dst = np.arange(token_num * top_k).astype(np.int32)
    np.random.shuffle(expanded_src_to_dst)
    expert_for_source_row = np.random.randint(low=0, high=expert_num, size=(token_num, top_k)).astype(np.int32)

    expect0 = moe_finalize_routing(x, skip1, skip2, bias, scale, expanded_src_to_dst, expert_for_source_row)

    # tensor input
    x = ms.Tensor(x, ms.float16)
    skip1 = ms.Tensor(skip1, ms.float16)
    skip2 = None
    bias = ms.Tensor(bias, ms.float16)
    scale = ms.Tensor(scale, ms.float16)
    expanded_src_to_dst = ms.Tensor(expanded_src_to_dst, ms.int32)
    expert_for_source_row = ms.Tensor(expert_for_source_row, ms.int32)

    res = moefzr_net(x, skip1, skip2, bias, scale, expanded_src_to_dst, expert_for_source_row)
    resnpy = res.asnumpy()

    np.testing.assert_allclose(resnpy, expect0, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=52.0)
@pytest.mark.parametrize('mode', ['GE', 'KBK', 'pynative'])
def test_moe_finalize_routing_case1(mode):
    """
    Feature: Test the moe_finalize_routing calculate
    Description: Test the moe_finalize_routing ops in Ascend backend
    Expectation: The result match to the expect value.
    """
    context.set_context(device_target="Ascend")
    if mode == 'KBK':
        ms.set_context(mode=ms.GRAPH_MODE)
        ms.set_context(jit_level='O0')
    elif mode == 'GE':
        ms.context.set_context(mode=ms.GRAPH_MODE)
        ms.set_context(jit_level='O2')
    elif mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
    moefzr_net = MoeFinalizeRoutingNet()

    top_k = 2
    token_num = 512
    hidden = 4096
    expert_num = 8

    # numpy input
    x = np.random.random((token_num * top_k, hidden)).astype(np.float16)
    skip1 = np.random.random((token_num, hidden)).astype(np.float16)
    skip2 = None
    bias = np.random.random((expert_num, hidden)).astype(np.float16)
    scale = np.random.random((token_num, top_k)).astype(np.float16)
    expanded_src_to_dst = np.arange(token_num * top_k).astype(np.int32)
    np.random.shuffle(expanded_src_to_dst)
    expert_for_source_row = np.random.randint(low=0, high=expert_num, size=(token_num, top_k)).astype(np.int32)

    expect0 = moe_finalize_routing(x, skip1, skip2, bias, scale, expanded_src_to_dst, expert_for_source_row)

    # tensor input
    x = ms.Tensor(x, ms.float16)
    skip1 = ms.Tensor(skip1, ms.float16)
    skip2 = None
    bias = ms.Tensor(bias, ms.float16)
    scale = ms.Tensor(scale, ms.float16)
    expanded_src_to_dst = ms.Tensor(expanded_src_to_dst, ms.int32)
    expert_for_source_row = ms.Tensor(expert_for_source_row, ms.int32)

    res = moefzr_net(x, skip1, skip2, bias, scale, expanded_src_to_dst, expert_for_source_row)
    resnpy = res.asnumpy()

    np.testing.assert_allclose(resnpy, expect0, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=18.0)
def test_ops_moe_finalize_routing_dyn():
    """
    Feature: pyboost function.
    Description: test MoeFinalizeRouting forward with dynamic rank/shape.
    Expectation: success.
    """
    # inputs1
    x1 = ms.Tensor(np.random.random((6 * 2, 4)).astype(np.float16))
    skip1_1 = ms.Tensor(np.random.random((6, 4)).astype(np.float16))
    skip2_1 = ms.Tensor(np.random.random((6, 4)).astype(np.float16))
    bias1 = ms.Tensor(np.random.random((8, 4)).astype(np.float16))
    scale1 = ms.Tensor(np.random.random((6, 2)).astype(np.float16))
    expanded_src_to_dst1 = np.arange(6 * 2).astype(np.int32)
    np.random.shuffle(expanded_src_to_dst1)
    expanded_src_to_dst1 = ms.Tensor(expanded_src_to_dst1)
    expert_for_source_row1 = ms.Tensor(np.random.randint(low=0, high=8, size=(6, 2)).astype(np.int32))

    # inputs2
    x2 = ms.Tensor(np.random.random((12 * 4, 8)).astype(np.float16))
    skip1_2 = ms.Tensor(np.random.random((12, 8)).astype(np.float16))
    skip2_2 = ms.Tensor(np.random.random((12, 8)).astype(np.float16))
    bias2 = ms.Tensor(np.random.random((16, 8)).astype(np.float16))
    scale2 = ms.Tensor(np.random.random((12, 4)).astype(np.float16))
    expanded_src_to_dst2 = np.arange(12 * 4).astype(np.int32)
    np.random.shuffle(expanded_src_to_dst2)
    expanded_src_to_dst2 = ms.Tensor(expanded_src_to_dst2)
    expert_for_source_row2 = ms.Tensor(np.random.randint(low=0, high=16, size=(12, 4)).astype(np.int32))

    TEST_OP(moe_finalize_routing_forward_func,
            [[x1, skip1_1, skip2_1, bias1, scale1, expanded_src_to_dst1, expert_for_source_row1],
             [x2, skip1_2, skip2_2, bias2, scale2, expanded_src_to_dst2, expert_for_source_row2]],
            disable_case=['ScalarTensor'],
            case_config={'disable_input_check': True,
                         'disable_grad': True,
                         'all_dim_zero': True,
                         'deterministic_use_origin_inputs': True})
