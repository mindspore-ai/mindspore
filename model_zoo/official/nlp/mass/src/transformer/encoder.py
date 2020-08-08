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
"""Encoder of Transformer."""
import mindspore.common.dtype as mstype
from mindspore import nn

from .feed_forward_network import FeedForwardNet
from .self_attention import SelfAttention
from .components import LayerNorm


class EncoderCell(nn.Cell):
    """
    Single Encoder layer.

    Layer structure is as below:
        -> pre_LayerNorm
        -> Multi-head Self-Attention
        -> Dropout & Add
        -> pre_LayerNorm
        -> Fc1
        -> Activation Function
        -> Dropout
        -> Fc2
        -> Dropout & Add

    Args:
        attn_embed_dim (int): Dimensions of attention weights.
        num_attn_heads (int): Heads number.
        intermediate_size (int): Hidden size in FFN.
        attention_dropout_prob (float): Dropout rate in attention layer.
        initializer_range (float): Initial range.
        hidden_dropout_prob (float): Dropout rate in FFN.
        hidden_act (str): Activation function in FFN.
        compute_type (mstype): Mindspore data type.

    Returns:
        Tensor, shape of (N, T, D).
    """

    def __init__(self,
                 attn_embed_dim=768,
                 num_attn_heads=12,
                 intermediate_size=3072,
                 attention_dropout_prob=0.02,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 hidden_act="relu",
                 compute_type=mstype.float32):
        super(EncoderCell, self).__init__()
        self.attention = SelfAttention(
            attn_embed_dim=attn_embed_dim,
            num_attn_heads=num_attn_heads,
            attn_dropout_prob=attention_dropout_prob,
            initializer_range=initializer_range,
            dropout_prob=hidden_dropout_prob,
            compute_type=compute_type)
        self.feed_forward_net = FeedForwardNet(
            in_channels=attn_embed_dim,
            hidden_size=intermediate_size,
            out_channels=attn_embed_dim,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            hidden_dropout_prob=hidden_dropout_prob,
            dropout=hidden_dropout_prob,
            compute_type=compute_type)

    def construct(self, queries, attention_mask):
        """
        Construct network.

        Args:
            queries (Tensor): Shape (N, T, D).
            attention_mask (Tensor): Shape (N, T, T').

        Returns:
            Tensor, shape (N, T, D).
        """
        attention_output = self.attention(queries, queries, queries,
                                          attention_mask)  # (N, T, D)
        output = self.feed_forward_net(attention_output)  # (N, T, D)
        return output


class TransformerEncoder(nn.Cell):
    """
    Implements of Transformer encoder.

    According to Google Tensor2Tensor lib experience, they found that
    put layer norm behind the multi-head self-attention and ffn would
    make model more robust.

    Thus, we take the same action.

    Encoder layer structure is as below:
        -> pre_LayerNorm
        -> Multi-head Self-Attention
        -> Dropout & Add
        -> pre_LayerNorm
        -> Fc1
        -> Activation Function
        -> Dropout
        -> Fc2
        -> Dropout & Add

    Args:
        attn_embed_dim (int): Dimensions of attention weights.
        encoder_layers (int): Encoder layers.
        num_attn_heads (int): Heads number.
        intermediate_size (int): Hidden size in FFN.
        attention_dropout_prob (float): Dropout rate in attention.
        initializer_range (float): Initial range.
        hidden_dropout_prob (float): Dropout rate in FFN.
        hidden_act (str): Activation function.
        compute_type (mstype): Mindspore data type.

    Returns:
        Tensor, shape of (N, T, D).
    """

    def __init__(self,
                 attn_embed_dim,
                 encoder_layers,
                 num_attn_heads=12,
                 intermediate_size=3072,
                 attention_dropout_prob=0.1,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 hidden_act="relu",
                 compute_type=mstype.float32):
        super(TransformerEncoder, self).__init__()
        self.num_layers = encoder_layers

        layers = []
        for _ in range(encoder_layers):
            layer = EncoderCell(
                attn_embed_dim=attn_embed_dim,
                num_attn_heads=num_attn_heads,
                intermediate_size=intermediate_size,
                attention_dropout_prob=attention_dropout_prob,
                initializer_range=initializer_range,
                hidden_dropout_prob=hidden_dropout_prob,
                hidden_act=hidden_act,
                compute_type=compute_type
            )
            layers.append(layer)

        self.layers = nn.CellList(layers)
        self.layer_norm = LayerNorm(in_channels=attn_embed_dim)

    def construct(self, input_tensor, attention_mask):
        """
        Construct network.

        Args:
            input_tensor (Tensor): Shape (N, T, D).
            attention_mask (Tensor): Shape (N, T, T).

        Returns:
            Tensor, shape (N, T, D).
        """
        prev_output = input_tensor
        for layer_module in self.layers:
            prev_output = layer_module(prev_output,
                                       attention_mask)  # (N, T, D)
        prev_output = self.layer_norm(prev_output)  # (N, T, D)
        return prev_output
