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
from tests.mark_utils import arg_mark
import numpy as np
import pytest
from copy import deepcopy

import mindspore as ms
from mindspore import context
from mindspore.nn import Cell
from mindspore.ops.auto_generate import MoeFinalizeRouting

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
    top_k = expanded_src_to_dst_row.shape[0] // token_num # topk k

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


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_moe_finalize_routing_case0(mode):
    """
    Feature: Test the moe_finalize_routing calculate
    Description: Test the moe_finalize_routing ops in Ascend backend
    Expectation: The result match to the expect value.
    """
    context.set_context(device_target="Ascend", mode=mode)

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

    moefzr_net = MoeFinalizeRoutingNet()
    res = moefzr_net(x, skip1, skip2, bias, scale, expanded_src_to_dst, expert_for_source_row)
    resnpy = res.asnumpy()

    np.testing.assert_allclose(resnpy, expect0, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_moe_finalize_routing_case1(mode):
    """
    Feature: Test the moe_finalize_routing calculate
    Description: Test the moe_finalize_routing ops in Ascend backend
    Expectation: The result match to the expect value.
    """
    context.set_context(device_target="Ascend", mode=mode)

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

    moefzr_net = MoeFinalizeRoutingNet()
    res = moefzr_net(x, skip1, skip2, bias, scale, expanded_src_to_dst, expert_for_source_row)
    resnpy = res.asnumpy()

    np.testing.assert_allclose(resnpy, expect0, rtol=1e-3)
