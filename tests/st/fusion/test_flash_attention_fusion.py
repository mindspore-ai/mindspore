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
import mindspore.common.dtype as mstype
import mindspore.context as context
import mindspore.nn as nn
from mindspore import ops, Tensor
from mindspore import jit
import mindspore as ms
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark

context.set_context(device_target="Ascend")


@jit
def attention_compuation(query, key, value, attention_mask, nums_heads_per_partition, head_dim):
    attn_weights = ops.matmul(query, key.permute(0, 1, 3, 2))
    attn_weights = attn_weights / (value.shape[-1] ** 0.5)

    attention_mask = (1 - attention_mask) * (-20000)
    attn_weights = attn_weights + attention_mask.to(attn_weights.dtype)

    attn_weights = ops.softmax(attn_weights, axis=-1)
    attn_output = ops.matmul(attn_weights, value)

    return attn_output


class PFA_FusionV1_Net(nn.Cell):
    def __init__(self):
        super(PFA_FusionV1_Net, self).__init__()
        self.attention_compuation = attention_compuation

    def construct(self, query, key, value, attention_mask, nums_heads_per_partition, head_dim):
        query, key, value, attention_mask = query.to(mstype.float16), key.to(mstype.float16),\
                                            value.to(mstype.float16), attention_mask.to(mstype.float16)
        attn_output = self.attention_compuation(query, key, value, attention_mask, nums_heads_per_partition, head_dim)
        return attn_output

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE, context.GRAPH_MODE])
def test_prompt_flash_attention_fusion_v1(mode):
    """
    Feature: test_prompt_flash_attention_fusion_v1
    Description: test PromptFlashAttention fusion v1 function.
    Expectation: compare the result with exception value.
    """
    context.set_context(mode=mode)
    key = ops.rand(4, 12, 32, 64)
    query = ops.rand(4, 12, 32, 64)
    value = ops.rand(4, 12, 32, 64)
    atten_mask = ops.rand(4, 12, 32, 32)
    nums_heads_per_partition = 12
    head_dim = 64

    net = PFA_FusionV1_Net()
    atten_output_pass = net(query, key, value, atten_mask, nums_heads_per_partition, head_dim)
    assert atten_output_pass.shape == (4, 12, 32, 64)


@jit
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    L, S = query.shape[-2], key.shape[-2]
    query_last_dim = query.shape[-1]
    key_dtype = key.dtype
    scale_factor = 1 / ops.sqrt(Tensor(query_last_dim, dtype=key_dtype)) if scale is None else scale
    attn_bias = ops.zeros((L, S), dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = ops.ones((L, S), dtype=ms.bool_)
        temp_mask = ops.tril(temp_mask, 0)
        attn_bias = attn_bias.masked_fill(ops.logical_not(temp_mask), ops.cast(float("-inf"), attn_bias.dtype))
        attn_bias.astype(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == ms.bool_:
            attn_bias = attn_bias.masked_fill(ops.logical_not(attn_mask), ops.cast(float("-inf"), attn_bias.dtype))
        else:
            attn_bias += attn_mask

    key_transpose = ops.swapaxes(key, -2, -1)
    attn_weight = query @ key_transpose * scale_factor
    attn_weight += attn_bias
    attn_weight = ops.softmax(attn_weight, -1)
    if dropout_p != 0.:
        attn_weight = ops.dropout(attn_weight, p=dropout_p)
    output = attn_weight @ value
    return output


class PFA_FusionV2_Net(nn.Cell):
    def __init__(self):
        super(PFA_FusionV2_Net, self).__init__()
        self.scaled_dot_product_attention = scaled_dot_product_attention

    def construct(self, query, key, value, attention_mask):
        query, key, value = query.half(), key.half(), value.half()
        attn_output = self.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask,
                                                        dropout_p=0.0, is_causal=False)
        return attn_output


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_prompt_flash_attention_fusion_v2(mode):
    """
    Feature: test_prompt_flash_attention_fusion_v2
    Description: test PromptFlashAttention fusion v2 function.
    Expectation: compare the result with exception value.
    """
    context.set_context(mode=mode)
    key = ops.rand(4, 12, 32, 64)
    query = ops.rand(4, 12, 32, 64)
    value = ops.rand(4, 12, 32, 64)
    atten_mask = None

    net = PFA_FusionV2_Net()
    atten_output_pass = net(query, key, value, atten_mask)
    assert atten_output_pass.shape == (4, 12, 32, 64)
