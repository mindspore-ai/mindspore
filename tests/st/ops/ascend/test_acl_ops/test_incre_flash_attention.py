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
# ============================================================================
import numpy as np
import pytest
import math
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import mutable
from mindspore.common.api import jit
from mindspore.common import dtype as mstype
from mindspore.ops.function.nn_func import incre_flash_attention

class IncreFlashAttentionFunc(nn.Cell):
    def __init__(self, num_heads, input_layout, scale_value, num_key_value_heads):
        super().__init__()
        self.num_heads = num_heads
        self.input_layout = input_layout
        self.scale_value = scale_value
        self.num_key_value_heads = num_key_value_heads
        self.ifa = incre_flash_attention

    @jit
    def construct(self, query, key, value, attn_mask, actual_seq_lengths, padding_mask, dequant_scale1, quant_scale1,
                  dequant_scale2, quant_scale2, quant_offset2):
        out = self.ifa(query, key, value, attn_mask, actual_seq_lengths, padding_mask, dequant_scale1, quant_scale1,
                       dequant_scale2, quant_scale2, quant_offset2, None, None, None,
                       self.num_heads, self.input_layout, self.scale_value, self.num_key_value_heads, 0, 1)
        return out


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_incre_flash_attention_bsh_fwd():
    """
    Feature: Test functional ifa operator.
    Description: bsh mode for ifa test.
    Expectation: Assert result compare with expect value.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_level='O0')
    B, N, S, D = 1, 1, 32, 32
    H = N * D

    query = Tensor(np.ones([B, 1, H]), dtype=mstype.float16)
    key = mutable((Tensor(np.ones([B, S, H]), dtype=mstype.float16),))
    value = mutable((Tensor(np.ones([B, S, H]), dtype=mstype.float16),))
    padding_mask = Tensor(np.ones([B, 1, 1, S]), dtype=mstype.float16)
    attn_mask = Tensor(np.ones([B, 1, 1, S]), dtype=mstype.float16)
    actual_seq_lengths = [S] * B
    asl_tensor = Tensor(actual_seq_lengths, dtype=mstype.int64)

    scale_value = 1 / math.sqrt(D)
    input_layout = "BSH"
    num_key_value_heads = N

    net = IncreFlashAttentionFunc(N, input_layout, scale_value, num_key_value_heads)

    ifa_result = net(query, key, value, attn_mask, asl_tensor, padding_mask, None, None, None, None, None)
    assert ifa_result.shape == (B, 1, H)
