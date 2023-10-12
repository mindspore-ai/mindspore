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
import numpy as np
import math
import pytest

import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Tensor, context
from mindspore.nn.layer.flash_attention import FlashAttention
from mindspore.ops import operations as P
from mindspore.ops import functional as F


class FlashAttentionNet(nn.Cell):
    def __init__(self, num_heads, head_dim, dropout_rate=0.0, prev_tockens=65536, next_tockens=65536):
        super(FlashAttentionNet, self).__init__()
        self.flash_attention = FlashAttention(head_dim, num_heads, dropout_rate, prev_tockens, next_tockens,
                                              high_precision=True)
        self.transpose_key = P.Transpose()

    def construct(self, query, key, value, attention_mask):
        key = self.transpose_key(key, (0, 1, 3, 2))
        attention_mask = 1.0 - attention_mask
        attention = self.flash_attention(query, key, value, attention_mask)
        return attention


class AttnNet(nn.Cell):
    """
    Get the weighted score along the seq_length

    Inputs:
        query: the query matrix
        key: the key matrix
        value: the value matrix
        attention_mask: the attention mask matrix with shape (batch_size,
        1, seq_length, seq_length)
    Outputs:
        weighted_values: Tensor, the weighted sum scores
    """
    def __init__(self, head_dim, dropout_rate=0., softmax_compute_type=mstype.float16):
        super(AttnNet, self).__init__()
        self.softmax_compute_type = softmax_compute_type
        self._is_ascend = context.get_context('device_target') in ["Ascend"]
        self.scale_factor = Tensor(math.sqrt(math.sqrt(head_dim)))
        self.real_div = P.RealDiv()
        self.sub = P.Sub()
        self.mul = P.Mul()
        self.add = P.Add()
        self.multiply_data = Tensor([-10000.0,], dtype=softmax_compute_type)
        self.softmax = nn.Softmax().to_float(softmax_compute_type)
        self.softmax_3d = nn.Softmax().to_float(softmax_compute_type)
        self.dropout = nn.Dropout(keep_prob=1 - dropout_rate)
        self.prob_dropout = nn.Dropout(keep_prob=1 - dropout_rate)
        self.batch_matmul = P.BatchMatMul()

    def construct(self, query, key, value, attention_mask):
        """
        Get the weighted score along the seq_length

        Inputs:
            query: the query matrix
            key: the key matrix
            value: the value matrix
            attention_mask: the attention mask matrix with shape (batch_size,
            1, seq_length, seq_length)
        Outputs:
            weighted_values: Tensor, the weighted sum scores
        """
        # Normalize query and key before MatMul, default off
        # Attention score [bs, num_heads, seq_length, seq_length]
        factor = P.Cast()(self.scale_factor, P.DType()(query))
        query = self.real_div(query, factor)
        key = self.real_div(key, factor)
        score = self.batch_matmul(query, key)

        ori_dtype = P.DType()(score)
        attention_scores = P.Cast()(score, self.softmax_compute_type)

        # for input size of (bs, 1) namely the second graph,
        # the shape of attention_mask matrix should be (bs, 1, 1, seq_length)
        if attention_mask is not None:
            # Minus 10000 for the position where masked to exclude them from softmax
            multiple_out = self.sub(
                P.Cast()(F.tuple_to_array((1.0,)), P.DType()(attention_scores)),
                P.Cast()(attention_mask, P.DType()(attention_scores)))

            adder = self.mul(multiple_out, self.multiply_data)
            attention_scores = self.add(adder, attention_scores)

        # attention probs
        attention_probs = self._softmax(attention_scores)
        attention_probs = P.Cast()(attention_probs, ori_dtype)

        attention_probs = self.prob_dropout(attention_probs)
        # Weighted sum output [bs, num_heads, seq_length, size_per_head]
        weighted_values = self.batch_matmul(attention_probs, value)
        return weighted_values

    def _softmax(self, attention_scores):
        """
        For the consideration of the performance, do softmax according to different situations
        :param attention_scores: a 3d tensor before softmax
        :return: the attention scores.
        """
        if self._is_ascend and self.softmax_compute_type == mstype.float16 or not self._is_ascend:
            attention_probs = self.softmax(attention_scores)
        else:
            shape = F.shape(attention_scores)
            # attention probs
            attention_probs = self.softmax_3d(
                F.reshape(attention_scores,
                          (shape[0], -1, shape[-1])))
            attention_probs = F.reshape(attention_probs, shape)
        return attention_probs


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [mstype.float16])
def test_nn_flash_attention_fwd(dtype):
    """
    Feature: nn.FlashAttention
    Description: Verify the result of FlashAttention forward
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE)
    B = 1
    S = 8192
    H = 1280
    N = 10
    D = H // N

    # Generate inputs
    np.random.seed(1234)
    qv_tensor = Tensor(np.random.uniform(-3, 3, (B, N, S, D)), dtype=dtype)
    k_tensor = Tensor(np.random.uniform(-3, 3, (B, N, D, S)), dtype=dtype)
    attention_mask = Tensor(np.repeat(np.expand_dims(np.tril(np.ones(shape=(S, S))), 0), B, axis=0), dtype=dtype)

    sa_net = AttnNet(head_dim=D)
    sa_out = sa_net(qv_tensor, k_tensor, qv_tensor, attention_mask).asnumpy()
    fa_net = FlashAttentionNet(num_heads=N, head_dim=D)
    fa_out = fa_net(qv_tensor, k_tensor, qv_tensor, attention_mask).asnumpy()
    assert sa_out.shape == fa_out.shape
