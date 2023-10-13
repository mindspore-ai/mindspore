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
    def __init__(self, head_num):
        super(FlashAttentionScoreCell, self).__init__()
        self.fa_op = FlashAttentionScore(head_num=head_num)

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


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_flash_attention_score_fwd():
    """
    Feature: test FlashAttentionScore forward in Graph modes.
    Description: test case for FlashAttentionScore.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    B = 1
    S = 8192
    H = 1280
    N = 10
    query = Tensor(np.ones((B, S, H), dtype=np.float16))
    key = Tensor(np.ones((B, S, H), dtype=np.float16))
    value = Tensor(np.ones((B, S, H), dtype=np.float16))
    attn_mask = Tensor(np.ones((B, 1, S, S), dtype=np.uint8))
    drop_mask = None
    real_shift = None
    padding_mask = None

    net = FlashAttentionScoreCell(N)
    attention_out, softmax_max, softmax_sum = net(
        query, key, value, attn_mask, drop_mask, real_shift, padding_mask)
    assert attention_out.shape == (B, S, H)
    assert softmax_max.shape == (B, N, S, 8)
    assert softmax_sum.shape == (B, N, S, 8)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_flash_attention_score_fwd_bwd():
    """
    Feature: test FlashAttentionScore forward and backward in Graph modes.
    Description: test case for FlashAttentionScore and FlashAttentionScoreGrad.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    B = 1
    S = 8192
    H = 1280
    N = 10
    query = Tensor(np.ones((B, S, H), dtype=np.float16))
    key = Tensor(np.ones((B, S, H), dtype=np.float16))
    value = Tensor(np.ones((B, S, H), dtype=np.float16))
    attn_mask = Tensor(np.ones((B, 1, S, S), dtype=np.uint8))
    drop_mask = None
    real_shift = None
    padding_mask = None
    net_with_grad = Grad(FlashAttentionScoreCell(N))

    dq, dk, dv, _ = net_with_grad(query, key, value, attn_mask, drop_mask, real_shift, padding_mask)
    assert dq.shape == (B, S, H)
    assert dk.shape == (B, S, H)
    assert dv.shape == (B, S, H)
