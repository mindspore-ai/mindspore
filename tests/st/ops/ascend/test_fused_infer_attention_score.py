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
import numpy as np
import pytest
import math
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import auto_generate as P
from mindspore.ops.function.nn_func import prompt_flash_attention, incre_flash_attention

class PromptFlashAttention(nn.Cell):
    def __init__(self):
        super(PromptFlashAttention, self).__init__()
        self.pfa = prompt_flash_attention

    def construct(self, query, key, value, attn_mask, actual_seq_lengths, actual_seq_lengths_kv, pse_shift,
                  deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2, num_heads, scale_value=1.0,
                  pre_tokens=2147483547, next_tokens=0, input_layout='BSH', num_key_value_heads=0, sparse_mode=0,
                  inner_precise=0):
        out = self.pfa(query, key, value, attn_mask, actual_seq_lengths, actual_seq_lengths_kv, pse_shift,
                       deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2, num_heads=num_heads,
                       scale_value=scale_value, pre_tokens=pre_tokens, next_tokens=next_tokens,
                       input_layout=input_layout, num_key_value_heads=num_key_value_heads, sparse_mode=sparse_mode,
                       inner_precise=inner_precise)
        return out

class IncreFlashAttentionFunc(nn.Cell):

    def __init__(self, num_heads, input_layout, scale_value, num_key_value_heads):
        super().__init__()
        self.num_heads = num_heads
        self.input_layout = input_layout
        self.scale_value = scale_value
        self.num_key_value_heads = num_key_value_heads
        self.ifa = incre_flash_attention

    def construct(self, query, key_i, value_i, attn_mask, actual_seq_lengths, pse_shift,
                  dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2,
                  antiquant_scale, antiquant_offset, block_table=None, block_size=0, inner_precise=0):
        out = self.ifa(query, key_i, value_i, attn_mask, actual_seq_lengths,
                       pse_shift, dequant_scale1, quant_scale1, dequant_scale2,
                       quant_scale2, quant_offset2, antiquant_scale, antiquant_offset,
                       block_table, self.num_heads, self.input_layout, self.scale_value,
                       self.num_key_value_heads, inner_precise=inner_precise)
        return out

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
        self.fias = P.FusedInferAttentionScore(num_heads=self.num_heads, scale_value=self.scale_value,
                                               pre_tokens=self.pre_tokens, next_tokens=self.next_tokens,
                                               input_layout=self.input_layout,
                                               num_key_value_heads=self.num_key_value_heads,
                                               sparse_mode=self.sparse_mode, inner_precise=self.inner_precise)

    def construct(self, query, key, value, pse_shift, attn_mask, actual_seq_lengths, actual_seq_lengths_kv,
                  dequant_scale1, quant_scale1, dequant_scale2, quant_scale2, quant_offset2, antiquant_scale,
                  antiquant_offset, block_table, query_padding_size, kv_padding_size):
        out = self.fias(query, key, value, pse_shift, attn_mask, actual_seq_lengths,
                        actual_seq_lengths_kv, dequant_scale1, quant_scale1, dequant_scale2,
                        quant_scale2, quant_offset2, antiquant_scale, antiquant_offset,
                        block_table, query_padding_size, kv_padding_size)
        return out

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_fused_infer_attention_score_pfa_bsh_fwd(context_mode):
    """
    Feature: test FusedInferAttentionScore.
    Description: test case for FusedInferAttentionScore, input_layout :'BSH'
    Expectation: the result match with PromptFlashAttention result.
    """
    ms.context.set_context(mode=context_mode)
    ms.context.set_context(jit_level='O2')
    np.random.seed(968941859)
    B = 1
    S = 256
    Q_H = 256
    KV_H = 128
    N = 16
    KV_N = 8
    scale_value = 1 / math.sqrt(256.0)
    q = np.random.rand(B, S, Q_H).astype(np.float16)
    k = np.random.rand(B, S, KV_H).astype(np.float16)
    v = np.random.rand(B, S, KV_H).astype(np.float16)
    mask_shape = [B, S, S]
    att = np.zeros(mask_shape).astype(np.bool)
    query = Tensor(q, dtype=mstype.float16)
    key = Tensor(k, dtype=mstype.float16)
    value = Tensor(v, dtype=mstype.float16)
    attn_mask = Tensor(att, dtype=mstype.bool_)

    net = PromptFlashAttention()
    pfa_attention_out = net(query, key, value, attn_mask, None, None,
                            None, None, None, None, None, None, N,
                            num_key_value_heads=KV_N, scale_value=scale_value, inner_precise=0)
    key_mut = [key]
    value_mut = [value]
    net_fias = FusedInferAttentionScoreFunc(num_heads=N, num_key_value_heads=KV_N, inner_precise=0,
                                            scale_value=scale_value)
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
def test_fused_infer_attention_score_pfa_bnsd_fwd(context_mode):
    """
    Feature: test FusedInferAttentionScore.
    Description: test case for FusedInferAttentionScore, input_layout :'BNSD'
    Expectation: the result match with PromptFlashAttention result.
    """
    ms.context.set_context(mode=context_mode)
    ms.context.set_context(jit_level='O2')
    np.random.seed(968941859)
    B = 1
    Q_N = 10
    N = 5
    S = 1024
    D = 32
    scale_value = 1 / math.sqrt(D)
    q = np.random.rand(B, Q_N, S, D).astype(np.float16)
    k = np.random.rand(B, N, S, D).astype(np.float16)
    v = np.random.rand(B, N, S, D).astype(np.float16)
    query = Tensor(q, dtype=mstype.float16)
    key = Tensor(k, dtype=mstype.float16)
    value = Tensor(v, dtype=mstype.float16)

    net = PromptFlashAttention()
    attention_out = net(query, key, value, None, None, None, None, None,
                        None, None, None, None, num_heads=Q_N,
                        input_layout='BNSD', num_key_value_heads=N,
                        scale_value=scale_value, inner_precise=0)
    key_mut = [key]
    value_mut = [value]
    net_fias = FusedInferAttentionScoreFunc(num_heads=Q_N, input_layout='BNSD', num_key_value_heads=N,
                                            scale_value=scale_value, inner_precise=0)
    fias_result = net_fias(query, key_mut, value_mut, None, None,
                           None, None, None, None, None, None,
                           None, None, None, None, None, None)
    fias_result_att = fias_result[0]

    assert fias_result_att.shape == attention_out.shape
    np.testing.assert_allclose(attention_out.asnumpy(), fias_result_att.asnumpy(), rtol=5e-3, atol=5e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_fused_infer_attention_score_bnsd_incre(context_mode):
    """
    Feature: test FusedInferAttentionScore.
    Description: test case for FusedInferAttentionScore, input_layout :'BNSD'
    Expectation: the result match with increFlashAttention result.
    """
    ms.context.set_context(mode=context_mode)
    ms.context.set_context(jit_level='O2')
    np.random.seed(968941859)
    B = 1
    Q_N = 5
    N = 5
    S = 4096
    D = 128
    scale_value = 1 / math.sqrt(D)
    q = np.random.rand(B, Q_N, 1, D).astype(np.float16)
    k = np.random.rand(B, N, S, D).astype(np.float16)
    v = np.random.rand(B, N, S, D).astype(np.float16)
    query = Tensor(q, dtype=ms.float16)
    key = Tensor(k, dtype=ms.float16)
    value = Tensor(v, dtype=ms.float16)

    key_mut = [key]
    value_mut = [value]

    net = IncreFlashAttentionFunc(Q_N, input_layout='BNSD', scale_value=scale_value,
                                  num_key_value_heads=N)
    ifa_out = net(query, key_mut, value_mut, None, None, None,
                  None, None, None, None, None, None, None)

    net_fias = FusedInferAttentionScoreFunc(num_heads=Q_N, input_layout='BNSD', scale_value=scale_value,
                                            num_key_value_heads=N)
    fias_result = net_fias(query, key_mut, value_mut, None, None,
                           None, None, None, None, None, None, None,
                           None, None, None, None, None)
    fias_result_att = fias_result[0]
    assert fias_result_att.shape == ifa_out.shape
    np.testing.assert_allclose(fias_result_att.asnumpy(), ifa_out.asnumpy(), rtol=5e-3, atol=5e-3)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_fused_infer_attention_score_bnsd_incre_antiquant(context_mode):
    """
    Feature: test FusedInferAttentionScore.
    Description: test case for FusedInferAttentionScore, with antiquant
    Expectation: the result match with increFlashAttention result.
    """
    ms.context.set_context(mode=context_mode)
    ms.context.set_context(jit_level='O2')
    np.random.seed(968941859)
    B = 1
    Q_N = 5
    N = 5
    S = 4096
    D = 128
    scale_value = 1 / math.sqrt(D)
    q = np.random.rand(B, Q_N, 1, D).astype(np.float16)
    k = np.random.rand(B, N, S, D).astype(np.int8)
    v = np.random.rand(B, N, S, D).astype(np.int8)
    query = Tensor(q, dtype=ms.float16)
    key = Tensor(k, dtype=ms.int8)
    value = Tensor(v, dtype=ms.int8)
    antiquant_s = np.random.rand(2, N, 1, D).astype(np.float16)
    antiquant_o = np.random.rand(2, N, 1, D).astype(np.float16)
    antiquant_scale = Tensor(antiquant_s, dtype=ms.float16)
    antiquant_offset = Tensor(antiquant_o, dtype=ms.float16)
    key_mut = [key]
    value_mut = [value]

    net = IncreFlashAttentionFunc(Q_N, input_layout='BNSD', scale_value=scale_value,
                                  num_key_value_heads=N)
    ifa_out = net(query, key_mut, value_mut, None, None,
                  None, None, None, None, None, None,
                  antiquant_scale, antiquant_offset)

    net_fias = FusedInferAttentionScoreFunc(num_heads=Q_N, input_layout='BNSD', scale_value=scale_value,
                                            num_key_value_heads=N)
    fias_result = net_fias(query, key_mut, value_mut, None, None,
                           None, None, None, None, None, None, None,
                           antiquant_scale, antiquant_offset, None, None, None)
    fias_result_att = fias_result[0]
    assert fias_result_att.shape == ifa_out.shape
    np.testing.assert_allclose(fias_result_att.asnumpy(), ifa_out.asnumpy(), rtol=5e-3, atol=5e-3)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_fused_infer_attention_score_bsh_incre(context_mode):
    """
    Feature: test FusedInferAttentionScore.
    Description: test case for FusedInferAttentionScore, input_layout :'BSH'
    Expectation: the result match with increFlashAttention result.
    """
    ms.context.set_context(mode=context_mode)
    ms.context.set_context(jit_level='O2')
    np.random.seed(968941859)
    B = 1
    Q_N = 5
    S = 4096
    D = 128
    scale_value = 1 / math.sqrt(D)
    q = np.random.rand(B, 1, Q_N * D).astype(np.float16)
    k = np.random.rand(B, S, D).astype(np.float16)
    v = np.random.rand(B, S, D).astype(np.float16)
    query = Tensor(q, dtype=ms.float16)
    key = Tensor(k, dtype=ms.float16)
    value = Tensor(v, dtype=ms.float16)

    key_mut = [key]
    value_mut = [value]

    net = IncreFlashAttentionFunc(Q_N, input_layout='BSH', scale_value=scale_value, num_key_value_heads=1)
    ifa_out = net(query, key_mut, value_mut, None, None, None, None, None, None, None, None, None, None)

    net_fias = FusedInferAttentionScoreFunc(num_heads=Q_N, input_layout='BSH', scale_value=scale_value,
                                            num_key_value_heads=1)
    fias_result = net_fias(query, key_mut, value_mut, None, None,
                           None, None, None, None, None, None, None,
                           None, None, None, None, None)
    fias_result_att = fias_result[0]
    assert fias_result_att.shape == ifa_out.shape
    np.testing.assert_allclose(fias_result_att.asnumpy(), ifa_out.asnumpy(), rtol=5e-3, atol=5e-3)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_fused_infer_attention_score_bsh_incre_antiquant(context_mode):
    """
    Feature: test FusedInferAttentionScore.
    Description: test case for FusedInferAttentionScore, with antiquant
    Expectation: the result match with increFlashAttention result.
    """
    ms.context.set_context(mode=context_mode)
    ms.context.set_context(jit_level='O2')
    np.random.seed(968941859)
    B = 1
    Q_N = 5
    S = 4096
    D = 128
    scale_value = 1 / math.sqrt(D)
    q = np.random.rand(B, 1, Q_N * D).astype(np.float16)
    k = np.random.rand(B, S, D).astype(np.int8)
    v = np.random.rand(B, S, D).astype(np.int8)
    query = Tensor(q, dtype=ms.float16)
    key = Tensor(k, dtype=ms.int8)
    value = Tensor(v, dtype=ms.int8)
    antiquant_s = np.random.rand(2, D).astype(np.float16)
    antiquant_o = np.random.rand(2, D).astype(np.float16)
    antiquant_scale = Tensor(antiquant_s, dtype=ms.float16)
    antiquant_offset = Tensor(antiquant_o, dtype=ms.float16)

    key_mut = [key]
    value_mut = [value]

    net = IncreFlashAttentionFunc(Q_N, input_layout='BSH', scale_value=scale_value,
                                  num_key_value_heads=1)
    ifa_out = net(query, key_mut, value_mut, None, None, None,
                  None, None, None, None, None, antiquant_scale, antiquant_offset)

    net_fias = FusedInferAttentionScoreFunc(num_heads=Q_N, input_layout='BSH', scale_value=scale_value,
                                            num_key_value_heads=1)
    fias_result = net_fias(query, key_mut, value_mut, None, None,
                           None, None, None, None, None, None, None,
                           antiquant_scale, antiquant_offset, None, None, None)
    fias_result_att = fias_result[0]
    assert fias_result_att.shape == ifa_out.shape
    np.testing.assert_allclose(fias_result_att.asnumpy(), ifa_out.asnumpy(), rtol=5e-3, atol=5e-3)

@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_fused_infer_attention_score_pfa_bnsd_fwd_dynamic(context_mode):
    """
    Feature: test FusedInferAttentionScore.
    Description: test case for FusedInferAttentionScore, input_layout :'BNSD'
    Expectation: the result match with PromptFlashAttention result.
    """
    ms.context.set_context(mode=context_mode)
    ms.context.set_context(jit_level='O2')
    np.random.seed(968941859)
    B = 1
    Q_N = 10
    N = 5
    S = 1024
    D = 32
    query = Tensor(shape=[None, Q_N, None, None], dtype=mstype.float16)
    key = Tensor(shape=[None, N, None, None], dtype=mstype.float16)
    value = Tensor(shape=[None, N, None, None], dtype=mstype.float16)

    key_mut = [key]
    value_mut = [value]
    net_fias = FusedInferAttentionScoreFunc(num_heads=Q_N, input_layout='BNSD', num_key_value_heads=N)
    net_fias.set_inputs(query, key_mut, value_mut, None, None,
                        None, None, None, None, None, None,
                        None, None, None, None, None, None)

    q = np.random.rand(B, Q_N, S, D).astype(np.float16)
    k = np.random.rand(B, N, S, D).astype(np.float16)
    v = np.random.rand(B, N, S, D).astype(np.float16)
    query_v = Tensor(q, dtype=mstype.float16)
    key_v = Tensor(k, dtype=mstype.float16)
    value_v = Tensor(v, dtype=mstype.float16)
    key_mut_v = [key_v]
    value_mut_v = [value_v]
    fias_result = net_fias(query_v, key_mut_v, value_mut_v, None, None,
                           None, None, None, None, None, None,
                           None, None, None, None, None, None)
    fias_result_att = fias_result[0]

    net = PromptFlashAttention()
    pfa_attention_out = net(query_v, key_v, value_v, None, None, None, None, None,
                            None, None, None, None, num_heads=Q_N,
                            input_layout='BNSD', num_key_value_heads=N)
    assert fias_result_att.shape == pfa_attention_out.shape
    np.testing.assert_allclose(pfa_attention_out.asnumpy(), fias_result_att.asnumpy(), rtol=5e-3, atol=5e-3)
