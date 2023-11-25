# Copyright 2022 Huawei Technologies Co., Ltd
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
from mindspore import Tensor
import mindspore.context as context
import mindspore.nn as nn
from mindspore.ops.composite import GradOperation
from mindspore.ops.operations.nn_ops import FlashAttentionScore


class FlashAttentionScoreCell(nn.Cell):
    def __init__(self, head_num, input_layout):
        super(FlashAttentionScoreCell, self).__init__()
        self.fa_op = FlashAttentionScore(head_num=head_num, input_layout=input_layout)

    def construct(self, *inputs):
        return self.fa_op(*inputs)


class Grad(nn.Cell):
    def __init__(self, network):
        super().__init__()
        self.grad = GradOperation(get_all=True, sens_param=False)
        self.network = network

    def construct(self, *inputs):
        gout = self.grad(self.network)(*inputs)
        return gout


def generate_inputs(B, N, S, D, input_layout, use_mqa=False):
    N_Q = N
    N_KV = 1 if use_mqa else N
    if input_layout == "BSH":
        H_Q = N_Q * D
        H_KV = N_KV * D
        query = Tensor(np.ones((B, S, H_Q), dtype=np.float16))
        key = Tensor(np.ones((B, S, H_KV), dtype=np.float16))
        value = Tensor(np.ones((B, S, H_KV), dtype=np.float16))
    elif input_layout == "BNSD":
        query = Tensor(np.ones((B, N_Q, S, D), dtype=np.float16))
        key = Tensor(np.ones((B, N_KV, S, D), dtype=np.float16))
        value = Tensor(np.ones((B, N_KV, S, D), dtype=np.float16))
    else:
        raise ValueError(f"input_layout is invalid.")
    attn_mask = Tensor(np.ones((B, 1, S, S), dtype=np.uint8))
    return query, key, value, attn_mask


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('input_layout', ["BSH", "BNSD"])
def test_flash_attention_score_fwd_bwd(input_layout):
    """
    Feature: test FlashAttentionScore forward and backward in Graph modes.
    Description: test case for FlashAttentionScore and FlashAttentionScoreGrad.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    B = 1
    N = 4
    S = 1024
    D = 128
    query, key, value, attn_mask = generate_inputs(B, N, S, D, input_layout)
    drop_mask = None
    real_shift = None
    padding_mask = None
    prefix = None
    net_with_grad = Grad(FlashAttentionScoreCell(N, input_layout))

    dq, dk, dv, _ = net_with_grad(query, key, value, attn_mask, drop_mask, real_shift, padding_mask, prefix)

    assert dq.shape == query.shape
    assert dk.shape == key.shape
    assert dv.shape == value.shape
