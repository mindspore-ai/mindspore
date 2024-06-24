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
from tests.mark_utils import arg_mark
import pytest
import numpy as np
import mindspore
from mindspore import Tensor
import mindspore.context as context
from mindspore.ops.function.nn_func import prompt_flash_attention
import mindspore.nn as nn


class PromptFlashAttention(nn.Cell):
    def __init__(self):
        super(PromptFlashAttention, self).__init__()
        self.pfa = prompt_flash_attention

    def construct(self, query, key, value, attn_mask, actual_seq_lengths, actual_seq_lengths_kv, pse_shift,
                  deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2, num_heads, scale_value=1.0,
                  pre_tokens=2147483547, next_tokens=0, input_layout='BSH', num_key_value_heads=0, sparse_mode=0):
        return self.pfa(query, key, value, attn_mask, actual_seq_lengths, actual_seq_lengths_kv, pse_shift,
                        deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2, num_heads=num_heads,
                        scale_value=scale_value, pre_tokens=pre_tokens, next_tokens=next_tokens,
                        input_layout=input_layout, num_key_value_heads=num_key_value_heads, sparse_mode=sparse_mode)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_prompt_flash_attention_bsh_fwd():
    """
    Feature: test PromptFlashAttention forward in Graph modes.
    Description: test case for PromptFlashAttention.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    B = 1
    S = 256
    Q_H = 256
    KV_H = 128
    N = 16
    KV_N = 8
    query = Tensor(np.ones((B, S, Q_H), dtype=np.float16))
    key = Tensor(np.ones((B, S, KV_H), dtype=np.float16))
    value = Tensor(np.ones((B, S, KV_H), dtype=np.float16))
    attn_mask = Tensor(np.ones((B, 1, S, S), dtype=np.float16))
    net = PromptFlashAttention()
    attention_out = net(query, key, value, attn_mask, None, None, None, None, None, None, None, None, N,
                        num_key_value_heads=KV_N)
    assert attention_out.shape == (B, S, Q_H)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_prompt_flash_attention_bnsd_fwd():
    """
    Feature: test PromptFlashAttention forward in Graph modes.
    Description: test case for PromptFlashAttention.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    B = 1
    Q_N = 10
    N = 5
    S = 1024
    D = 32

    query = Tensor(np.ones((B, Q_N, S, D)), dtype=mindspore.bfloat16)
    key = Tensor(np.ones((B, N, S, D)), dtype=mindspore.bfloat16)
    value = Tensor(np.ones((B, N, S, D)), dtype=mindspore.bfloat16)
    net = PromptFlashAttention()
    attention_out = net(query, key, value, None, None, None, None, None, None, None, None, None, num_heads=Q_N,
                        input_layout='BNSD', num_key_value_heads=N)
    assert attention_out.shape == (B, Q_N, S, D)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_prompt_flash_attention_bnsd_mod2_fwd():
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
    attn_mask = Tensor(np.ones((2048, 2048), dtype=np.bool))
    net = PromptFlashAttention()
    attention_out = net(query, key, value, attn_mask, None, None, None, None, None, None, None, None, N,
                        input_layout='BNSD', sparse_mode=2)
    assert attention_out.shape == (B, N, S, D)
