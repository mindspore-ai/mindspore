#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
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
import os
import numpy as np
import pytest
import math
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import mutable
from mindspore.common.api import jit
from mindspore.common import dtype as mstype
from mindspore.ops.function.nn_func import fused_infer_attention_score
from mindspore.ops.function.nn_func import prompt_flash_attention

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

class FusedInferAttentionScoreFunc(nn.Cell):
    def __init__(self, num_heads, input_layout='BSH', scale_value=1.0, num_key_value_heads=0,
                 pre_tokens=2147483547, next_tokens=0, sparse_mode=0, inner_precise=0):
        super().__init__()
        self.num_heads = num_heads
        self.input_layout = input_layout
        self.scale_value = scale_value
        self.num_key_value_heads = num_key_value_heads
        self.pre_tokens = pre_tokens
        self.next_tokens = next_tokens
        self.sparse_mode = sparse_mode
        self.inner_precise = inner_precise
        self.fias = fused_infer_attention_score

    @jit
    def construct(self, query, key, value, pse_shift, attn_mask, actual_seq_lengths, actual_seq_lengths_kv,
                  dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2, antiquant_scale,
                  antiquant_offset, block_table, query_padding_size, kv_padding_size):
        out = self.fias(query, key, value, pse_shift, attn_mask, actual_seq_lengths, actual_seq_lengths_kv,
                        dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2,
                        antiquant_scale, antiquant_offset, block_table, query_padding_size, kv_padding_size,
                        num_heads=self.num_heads, scale_value=self.scale_value,
                        input_layout=self.input_layout, num_key_value_heads=self.num_key_value_heads,
                        pre_tokens=self.pre_tokens, next_tokens=self.next_tokens, sparse_mode=self.sparse_mode,
                        inner_precise=self.inner_precise)
        return out

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_prompt_flash_attention_bsh_fwd(context_mode):
    """
    Feature: test PromptFlashAttention forward in Graph modes.
    Description: test case for PromptFlashAttention.
    Expectation: the result match with expected result.
    """
    ms.context.set_context(mode=context_mode)
    os.environ['GRAPH_OP_RUN'] = "1"
    B = 1
    S = 256
    Q_H = 256
    KV_H = 128
    N = 16
    KV_N = 8
    query = Tensor(np.ones([B, S, Q_H]), dtype=mstype.float16)
    key = Tensor(np.ones([B, S, KV_H]), dtype=mstype.float16)
    value = Tensor(np.ones([B, S, KV_H]), dtype=mstype.float16)
    attn_mask = Tensor(np.ones([B, S, S]), dtype=mstype.bool_)

    net = PromptFlashAttention()
    pfa_attention_out = net(query, key, value, attn_mask, None, None, None, None, None, None, None, None, N,
                            num_key_value_heads=KV_N)
    key_mut = mutable((key,))
    value_mut = mutable((value,))
    net_fias = FusedInferAttentionScoreFunc(N, num_key_value_heads=KV_N)
    fias_result = net_fias(query, key_mut, value_mut, None, attn_mask,
                           None, None, None, None, None, None,
                           None, None, None, None, None, None)
    fias_result_att = fias_result[0]
    assert fias_result_att.shape == pfa_attention_out.shape
    np.testing.assert_allclose(fias_result_att.asnumpy(), pfa_attention_out.asnumpy(), rtol=5e-3, atol=5e-3)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_prompt_flash_attention_bnsd_fwd(context_mode):
    """
    Feature: test PromptFlashAttention forward in Graph modes.
    Description: test case for PromptFlashAttention.
    Expectation: the result match with expected result.
    """
    ms.context.set_context(mode=context_mode)
    os.environ['GRAPH_OP_RUN'] = "1"
    B = 1
    Q_N = 10
    N = 5
    S = 1024
    D = 32

    query = Tensor(np.ones((B, Q_N, S, D)), dtype=ms.bfloat16)
    key = Tensor(np.ones((B, N, S, D)), dtype=ms.bfloat16)
    value = Tensor(np.ones((B, N, S, D)), dtype=ms.bfloat16)
    net = PromptFlashAttention()
    attention_out = net(query, key, value, None, None, None, None, None, None, None, None, None, num_heads=Q_N,
                        input_layout='BNSD', num_key_value_heads=N)

    key_mut = mutable((key,))
    value_mut = mutable((value,))
    net_fias = FusedInferAttentionScoreFunc(Q_N, input_layout='BNSD', num_key_value_heads=N)
    fias_result = net_fias(query, key_mut, value_mut, None, None,
                           None, None, None, None, None, None,
                           None, None, None, None, None, None)
    fias_result_att = fias_result[0]
    assert fias_result_att.shape == attention_out.shape
    np.testing.assert_allclose(fias_result_att.float().asnumpy(), attention_out.float().asnumpy(), rtol=5e-3, atol=5e-3)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_fused_infer_attention_score_bsh_fwd(context_mode):
    """
    Feature: Test functional ifa operator.
    Description: bsh mode for ifa test.
    Expectation: Assert result compare with expect value.
    """
    ms.context.set_context(mode=context_mode)
    os.environ['GRAPH_OP_RUN'] = "1"
    B, N, S, D = 1, 5, 32, 32
    H = N * D
    query = Tensor(np.ones([B, S, H]), dtype=mstype.float16)
    key = mutable((Tensor(np.ones([B, S, H]), dtype=mstype.float16),))
    value = mutable((Tensor(np.ones([B, S, H]), dtype=mstype.float16),))
    pse_shift = Tensor(np.ones([B, N, S, S]), dtype=mstype.float16)
    attn_mask = Tensor(np.ones([B, S, S]), dtype=mstype.bool_)
    scale_value = 1 / math.sqrt(D)
    input_layout = "BSH"
    num_key_value_heads = N

    net = FusedInferAttentionScoreFunc(N, input_layout, scale_value, num_key_value_heads,
                                       2147483647, 2147483647, 0, 1)
    fias_result = net(query, key, value, pse_shift, attn_mask,
                      None, None, None, None, None, None,
                      None, None, None, None, None, None)
    assert fias_result[0].shape == (B, S, H)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_fused_infer_attention_score_bnsd_fwd(context_mode):
    """
    Feature: Test functional ifa operator.
    Description: bnsd mode for ifa test.
    Expectation: Assert result compare with expect value.
    """
    ms.context.set_context(mode=context_mode)
    os.environ['GRAPH_OP_RUN'] = "1"
    B, N, S, D = 4, 6, 32, 32
    query = Tensor(np.ones([B, N, 1, D]), dtype=mstype.float16)
    key = mutable((Tensor(np.ones([B, N, S, D]), dtype=mstype.float16),))
    value = mutable((Tensor(np.ones([B, N, S, D]), dtype=mstype.float16),))
    attn_mask = Tensor(np.ones([B, S, S]), dtype=mstype.bool_)
    scale_value = 1 / math.sqrt(D)
    input_layout = "BNSD"
    num_key_value_heads = N

    net = FusedInferAttentionScoreFunc(N, input_layout, scale_value, num_key_value_heads,
                                       2147483647, 2147483647, 0, 1)

    fias_result = net(query, key, value, None, attn_mask, None, None, None, None, None, None,
                      None, None, None, None, None, None)
    assert fias_result[0].shape == (B, N, 1, D)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_fused_infer_attention_score_prompt_fwd(context_mode):
    """
    Feature: Test functional ifa operator.
    Description: bnsd mode for ifa test.
    Expectation: Assert result compare with expect value.
    """
    ms.context.set_context(mode=context_mode)
    os.environ['GRAPH_OP_RUN'] = "1"
    B, N, S, D = 1, 4, 2142, 256
    query = Tensor(np.ones([B, N, S, D]), dtype=mstype.float16)
    key = mutable((Tensor(np.ones([B, N, S, D]), dtype=mstype.float16),))
    value = mutable((Tensor(np.ones([B, N, S, D]), dtype=mstype.float16),))
    attn_mask = Tensor(np.ones([B, S, S]), dtype=mstype.bool_)
    num_heads = N
    scale_value = 1 / math.sqrt(D)
    pre_tokens = 65535
    next_tokens = 0
    input_layout = "BNSD"
    num_key_value_heads = N
    sparse_mode = 0
    inner_precise = 1
    net = FusedInferAttentionScoreFunc(num_heads, input_layout, scale_value, num_key_value_heads,
                                       pre_tokens, next_tokens, sparse_mode, inner_precise)

    fias_result = net(query, key, value, None, attn_mask, None, None, None, None, None, None,
                      None, None, None, None, None, None)
    assert fias_result[0].shape == (B, N, S, D)
    assert fias_result[0].dtype == mstype.float16
