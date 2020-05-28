# Copyright 2020 Huawei Technologies Co., Ltd
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
"""Self-Attention block."""
import mindspore.common.dtype as mstype
from mindspore import nn

from .multi_head_attention import MultiHeadAttention
from .residual_conn import ResidualConnection
from .components import LayerNorm


class SelfAttention(nn.Cell):
    """
    Self-Attention.

    Layer norm -> Multi-Head Self-Attention -> Add & Dropout.

    Args:
        attn_embed_dim (int): Dimensions of attention weight, e.g. Q, K, V.
        num_attn_heads (int): Attention heads number. Default: 1.
        attn_dropout_prob (float): Dropout rate in attention. Default: 0.1.
        initializer_range (float): Initial range.
        dropout_prob (float): Dropout rate.
        has_attention_mask (bool): Whether has attention mask.
        compute_type (mstype): Mindspore data type. Default: mstype.float32.

    Returns:
        Tensor, shape (N, T, D).
    """

    def __init__(self,
                 attn_embed_dim,
                 num_attn_heads,
                 attn_dropout_prob=0.1,
                 initializer_range=0.02,
                 dropout_prob=0.1,
                 has_attention_mask=True,
                 compute_type=mstype.float32):
        super(SelfAttention, self).__init__()
        self.multi_head_self_attention = MultiHeadAttention(
            src_dim=attn_embed_dim,
            tgt_dim=attn_embed_dim,
            attn_embed_dim=attn_embed_dim,
            num_attn_heads=num_attn_heads,
            attention_dropout_prob=attn_dropout_prob,
            initializer_range=initializer_range,
            has_attention_mask=has_attention_mask,
            do_return_2d_tensor=False,
            compute_type=compute_type)

        self.layer_norm = LayerNorm(in_channels=attn_embed_dim)
        self.residual = ResidualConnection(dropout_prob=dropout_prob)

    def construct(self, queries, keys, values, attention_mask):
        """
        Construct self-attention block.

        Layer norm -> Multi-Head Self-Attention -> Add & Dropout.

        Args:
            queries (Tensor): Shape (N, T, D).
            keys (Tensor): Shape (N, T', D).
            values (Tensor): Shape (N, T', D).
            attention_mask (Tensor): Shape (N, T, T').

        Returns:
            Tensor, shape (N, T, D).
        """
        q = self.layer_norm(queries)  # (N, T, D)
        attention_output = self.multi_head_self_attention(
            q, keys, values, attention_mask
        )  # (N, T, D)
        q = self.residual(attention_output, queries)
        return q
