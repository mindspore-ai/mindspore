# Copyright 2023 Huawei Technologies Co., Ltd
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
import mindspore.ops.operations.nn_ops as P
import mindspore.nn as nn


class PromptFlashAttention(nn.Cell):
    def __init__(self, num_heads, scale_value=1.0, pre_tokens=2147483547, next_tokens=0, input_layout='BSH',
                 num_key_value_heads=0):
        super(PromptFlashAttention, self).__init__()
        self.fa_op = P.PromptFlashAttention(num_heads, scale_value, pre_tokens, next_tokens, input_layout,
                                            num_key_value_heads)

    def construct(self, query, key, value, attn_mask):
        return self.fa_op(query, key, value, attn_mask, None, None)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_prompt_flash_attention__bsh_fwd():
    """
    Feature: test PromptFlashAttention forward in Graph modes.
    Description: test case for PromptFlashAttention.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    B = 2
    S = 256
    H = 256
    N = 16
    query = Tensor(np.ones((B, S, H), dtype=np.float16))
    key = Tensor(np.ones((B, S, H), dtype=np.float16))
    value = Tensor(np.ones((B, S, H), dtype=np.float16))
    attn_mask = Tensor(np.ones((B, 1, S, S), dtype=np.float16))
    net = PromptFlashAttention(N)
    attention_out = net(query, key, value, attn_mask)
    assert attention_out[0].shape == (B, S, H)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_prompt_flash_attention__bnsd_fwd():
    """
    Feature: test PromptFlashAttention forward in Graph modes.
    Description: test case for PromptFlashAttention.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    B = 1
    N = 16
    S = 256
    D = 16

    query = Tensor(np.ones((B, N, S, D), dtype=np.float16))
    key = Tensor(np.ones((B, N, S, D), dtype=np.float16))
    value = Tensor(np.ones((B, N, S, D), dtype=np.float16))
    attn_mask = Tensor(np.ones((B, 1, S, S), dtype=np.float16))
    net = PromptFlashAttention(N, input_layout='BNSD')
    attention_out = net(query, key, value, attn_mask)
    assert attention_out[0].shape == (B, N, S, D)
