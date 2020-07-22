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
"""Decoder of Transformer."""
import mindspore.common.dtype as mstype
from mindspore import nn

from .feed_forward_network import FeedForwardNet
from .self_attention import SelfAttention
from .components import LayerNorm


class DecoderCell(nn.Cell):
    """
    Decoder cells used in Transformer.

    Args:
        attn_embed_dim (int): Dimensions of attention weight, e.g. Q, K, V.
        num_attn_heads (int): Attention heads number.
        intermediate_size (int): Hidden size in FFN.
        attn_dropout_prob (float): Dropout rate in attention layer. Default: 0.1.
        initializer_range (float): Initial range. Default: 0.02.
        dropout_prob (float): Dropout rate between layers. Default: 0.1.
        hidden_act (str): Activation function in FFN. Default: "relu".
        compute_type (mstype): Mindspore data type. Default: mstype.float32.

    Returns:
        Tensor, output with shape (N, T', D).
    """

    def __init__(self,
                 attn_embed_dim=768,
                 num_attn_heads=12,
                 intermediate_size=3072,
                 attn_dropout_prob=0.02,
                 initializer_range=0.02,
                 dropout_prob=0.1,
                 hidden_act="relu",
                 compute_type=mstype.float32):
        super(DecoderCell, self).__init__()
        self.masked_attn = SelfAttention(
            attn_embed_dim=attn_embed_dim,
            num_attn_heads=num_attn_heads,
            attn_dropout_prob=attn_dropout_prob,
            initializer_range=initializer_range,
            dropout_prob=dropout_prob,
            compute_type=compute_type)
        self.enc_dec_attn = SelfAttention(
            attn_embed_dim=attn_embed_dim,
            num_attn_heads=num_attn_heads,
            attn_dropout_prob=attn_dropout_prob,
            initializer_range=initializer_range,
            dropout_prob=dropout_prob,
            compute_type=compute_type)
        self.feed_forward_net = FeedForwardNet(
            in_channels=attn_embed_dim,
            hidden_size=intermediate_size,
            out_channels=attn_embed_dim,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            hidden_dropout_prob=dropout_prob,
            compute_type=compute_type)

    def construct(self, queries, attention_mask, encoder_out, enc_attention_mask):
        """
        Construct network.

        Args:
            queries (Tensor): With shape (N, T', D).
            attention_mask (Tensor): With shape (N, T', T').
            encoder_out (Tensor): With shape (N, T, D).
            enc_attention_mask (Tensor): With shape (N, T, T).

        Returns:
            Tensor, output.
        """
        attention_output = self.masked_attn(
            queries, queries, queries,
            attention_mask
        )
        attention_output = self.enc_dec_attn(
            attention_output,  # (N, T', D)
            encoder_out, encoder_out,  # (N, T, D)
            enc_attention_mask  # (N, T, T)
        )
        output = self.feed_forward_net(attention_output)
        return output


class TransformerDecoder(nn.Cell):
    """
    Implements of Transformer decoder.

    Args:
        attn_embed_dim (int): Dimensions of attention layer.
        decoder_layers (int): Decoder layers.
        num_attn_heads (int): Attention heads number.
        intermediate_size (int): Hidden size of FFN.
        attn_dropout_prob (float): Dropout rate in attention. Default: 0.1.
        initializer_range (float): Initial range. Default: 0.02.
        dropout_prob (float): Dropout rate between layers. Default: 0.1.
        hidden_act (str): Non-linear activation function in FFN. Default: "relu".
        compute_type (mstype): Mindspore data type. Default: mstype.float32.

    Returns:
        Tensor, shape of (N, T', D).
    """

    def __init__(self,
                 attn_embed_dim,
                 decoder_layers,
                 num_attn_heads,
                 intermediate_size,
                 attn_dropout_prob=0.1,
                 initializer_range=0.02,
                 dropout_prob=0.1,
                 hidden_act="relu",
                 compute_type=mstype.float32):
        super(TransformerDecoder, self).__init__()
        self.num_layers = decoder_layers
        self.attn_embed_dim = attn_embed_dim

        self.layer0 = DecoderCell(
            attn_embed_dim=attn_embed_dim,
            num_attn_heads=num_attn_heads,
            intermediate_size=intermediate_size,
            attn_dropout_prob=attn_dropout_prob,
            initializer_range=initializer_range,
            dropout_prob=dropout_prob,
            hidden_act=hidden_act,
            compute_type=compute_type
        )
        self.layer1 = DecoderCell(
            attn_embed_dim=attn_embed_dim,
            num_attn_heads=num_attn_heads,
            intermediate_size=intermediate_size,
            attn_dropout_prob=attn_dropout_prob,
            initializer_range=initializer_range,
            dropout_prob=dropout_prob,
            hidden_act=hidden_act,
            compute_type=compute_type
        )
        self.layer2 = DecoderCell(
            attn_embed_dim=attn_embed_dim,
            num_attn_heads=num_attn_heads,
            intermediate_size=intermediate_size,
            attn_dropout_prob=attn_dropout_prob,
            initializer_range=initializer_range,
            dropout_prob=dropout_prob,
            hidden_act=hidden_act,
            compute_type=compute_type
        )
        self.layer3 = DecoderCell(
            attn_embed_dim=attn_embed_dim,
            num_attn_heads=num_attn_heads,
            intermediate_size=intermediate_size,
            attn_dropout_prob=attn_dropout_prob,
            initializer_range=initializer_range,
            dropout_prob=dropout_prob,
            hidden_act=hidden_act,
            compute_type=compute_type
        )
        self.layer4 = DecoderCell(
            attn_embed_dim=attn_embed_dim,
            num_attn_heads=num_attn_heads,
            intermediate_size=intermediate_size,
            attn_dropout_prob=attn_dropout_prob,
            initializer_range=initializer_range,
            dropout_prob=dropout_prob,
            hidden_act=hidden_act,
            compute_type=compute_type
        )
        self.layer5 = DecoderCell(
            attn_embed_dim=attn_embed_dim,
            num_attn_heads=num_attn_heads,
            intermediate_size=intermediate_size,
            attn_dropout_prob=attn_dropout_prob,
            initializer_range=initializer_range,
            dropout_prob=dropout_prob,
            hidden_act=hidden_act,
            compute_type=compute_type
        )

        self.layer_preprocess = LayerNorm(in_channels=attn_embed_dim,
                                          return_2d=False)

    def construct(self, input_tensor, attention_mask, encoder_out, enc_attention_mask):
        """
        Construct network.

        Args:
            input_tensor (Tensor): With shape of (N, T', D).
            attention_mask (Tensor): With shape of (N, T', T').
            encoder_out (Tensor): With shape of (N, T, D).
            enc_attention_mask (Tensor): With shape of (N, T, T).

        Returns:
            Tensor, shape of (N, T', D).
        """
        prev_output = input_tensor
        prev_output = self.layer0(prev_output, attention_mask, encoder_out, enc_attention_mask)
        prev_output = self.layer1(prev_output, attention_mask, encoder_out, enc_attention_mask)
        prev_output = self.layer2(prev_output, attention_mask, encoder_out, enc_attention_mask)
        prev_output = self.layer3(prev_output, attention_mask, encoder_out, enc_attention_mask)
        prev_output = self.layer4(prev_output, attention_mask, encoder_out, enc_attention_mask)
        prev_output = self.layer5(prev_output, attention_mask, encoder_out, enc_attention_mask)

        # Add layer norm, and full connection layer.
        prev_output = self.layer_preprocess(prev_output)
        return prev_output
