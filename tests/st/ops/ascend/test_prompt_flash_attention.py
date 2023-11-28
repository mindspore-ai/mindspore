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
from mindspore import ops
from mindspore.ops.operations.nn_ops import PromptFlashAttention

context.set_context(device_target="Ascend")


class Net(nn.Cell):
    def __init__(self, num_heads, scale_value=1.0, pre_tokens=2147483547, next_tokens=0,
                 input_layout='BSH', num_key_value_heads=0):
        super(Net, self).__init__()
        self.pfa_op = PromptFlashAttention(num_heads, scale_value, pre_tokens, next_tokens,
                                           input_layout, num_key_value_heads)

    def construct(self, query, key, value, attn_mask):
        return self.pfa_op(query, key, value, attn_mask, attn_mask, None)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE, context.GRAPH_MODE])
def test_prompt_flash_attention_atten_mask_none(mode):
    """
    Feature: test_prompt_flash_attention_atten_mask_none
    Description: test PromptFlashAttention attention_mask is none case.
    Expectation: compare the result with exception value.
    """
    context.set_context(mode=mode)
    B = 1
    N = 16
    S = 256
    D = 16

    query = ops.randn((B, N, S, D), dtype=mstype.float16)
    key = ops.randn((B, N, S, D), dtype=mstype.float16)
    value = ops.randn((B, N, S, D), dtype=mstype.float16)
    attn_mask = None
    net = Net(N, input_layout='BNSD')
    output = net(query, key, value, attn_mask)
    assert output[0].shape == (B, N, S, D)
