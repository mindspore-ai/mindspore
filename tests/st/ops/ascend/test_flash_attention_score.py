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
import mindspore.common.dtype as mstype
import mindspore.context as context
import mindspore.nn as nn
from mindspore.ops.composite import GradOperation
from mindspore.ops.operations.nn_ops import FlashAttentionScore


class FlashAttentionScoreCell(nn.Cell):
    def __init__(self, head_num, input_layout):
        super(FlashAttentionScoreCell, self).__init__()
        self.fa_op = FlashAttentionScore(head_num=head_num, input_layout=input_layout, keep_prob=0.9)

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


def generate_inputs(B, N1, N2, S1, S2, D, input_layout, dtype):
    if input_layout == "BSH":
        query = Tensor(np.ones((B, S1, N1 * D)), dtype=dtype)
        key = Tensor(np.ones((B, S2, N2 * D)), dtype=dtype)
        value = Tensor(np.ones((B, S2, N2 * D)), dtype=dtype)
    elif input_layout == "BNSD":
        query = Tensor(np.ones((B, N1, S1, D)), dtype=dtype)
        key = Tensor(np.ones((B, N2, S2, D)), dtype=dtype)
        value = Tensor(np.ones((B, N2, S2, D)), dtype=dtype)
    else:
        raise ValueError(f"input_layout is invalid.")
    real_shift = Tensor(np.ones((B, N1, 1, S2)), dtype=dtype)
    drop_mask = Tensor(np.ones((B, N1, S1, S2 // 8)), dtype=mstype.uint8)
    attn_mask = Tensor(np.ones((B, 1, S1, S2)), dtype=mstype.uint8)
    prefix = Tensor(np.ones((B,)), dtype=mstype.int64)
    return query, key, value, real_shift, drop_mask, attn_mask, prefix


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
    N1 = 4
    N2 = 4
    S1 = 1024
    S2 = 1024
    D = 128
    dtype = mstype.float16
    query, key, value, real_shift, drop_mask, attn_mask, prefix = \
        generate_inputs(B, N1, N2, S1, S2, D, input_layout, dtype)
    padding_mask = None
    net_with_grad = Grad(FlashAttentionScoreCell(N1, input_layout))

    dq, dk, dv, _, _, _, _ = net_with_grad(query, key, value, real_shift, drop_mask, padding_mask, attn_mask, prefix)

    assert dq.shape == query.shape
    assert dk.shape == key.shape
    assert dv.shape == value.shape
