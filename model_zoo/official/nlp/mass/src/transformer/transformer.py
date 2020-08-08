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
"""Transformer model addressed by Vaswani et al., 2017."""
import copy
import math

from mindspore import nn, Tensor
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype

from config.config import TransformerConfig

from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
from .create_attn_mask import CreateAttentionMaskFromInputMask
from .embedding import EmbeddingLookup
from .positional_embedding import PositionalEmbedding
from .components import SaturateCast


class Transformer(nn.Cell):
    """
    Transformer with encoder and decoder.

    In Transformer, we define T = src_max_len, T' = tgt_max_len.

    Args:
        config (TransformerConfig): Model config.
        is_training (bool): Whether is training.
        use_one_hot_embeddings (bool): Whether use one-hot embedding.

    Returns:
        Tuple[Tensor], network outputs.
    """

    def __init__(self,
                 config: TransformerConfig,
                 is_training: bool,
                 use_one_hot_embeddings: bool = False,
                 use_positional_embedding: bool = True):
        super(Transformer, self).__init__()

        self.use_positional_embedding = use_positional_embedding
        config = copy.deepcopy(config)
        self.is_training = is_training
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_dropout_prob = 0.0

        self.input_mask_from_dataset = config.input_mask_from_dataset
        self.batch_size = config.batch_size
        self.max_positions = config.seq_length
        self.attn_embed_dim = config.hidden_size
        self.num_layers = config.num_hidden_layers
        self.word_embed_dim = config.hidden_size

        self.last_idx = self.num_layers - 1

        self.embedding_lookup = EmbeddingLookup(
            vocab_size=config.vocab_size,
            embed_dim=self.word_embed_dim,
            use_one_hot_embeddings=use_one_hot_embeddings)

        if self.use_positional_embedding:
            self.positional_embedding = PositionalEmbedding(
                embedding_size=self.word_embed_dim,
                max_position_embeddings=config.max_position_embeddings)

        self.encoder = TransformerEncoder(
            attn_embed_dim=self.attn_embed_dim,
            encoder_layers=self.num_layers,
            num_attn_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            attention_dropout_prob=config.attention_dropout_prob,
            initializer_range=config.initializer_range,
            hidden_dropout_prob=config.hidden_dropout_prob,
            hidden_act=config.hidden_act,
            compute_type=config.compute_type)

        self.decoder = TransformerDecoder(
            attn_embed_dim=self.attn_embed_dim,
            decoder_layers=self.num_layers,
            num_attn_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            attn_dropout_prob=config.attention_dropout_prob,
            initializer_range=config.initializer_range,
            dropout_prob=config.hidden_dropout_prob,
            hidden_act=config.hidden_act,
            compute_type=config.compute_type)

        self.cast = P.Cast()
        self.dtype = config.dtype
        self.cast_compute_type = SaturateCast(dst_type=config.compute_type)
        self.slice = P.StridedSlice()
        self.dropout = nn.Dropout(keep_prob=1 - config.hidden_dropout_prob)

        self._create_attention_mask_from_input_mask = CreateAttentionMaskFromInputMask(config)

        self.scale = Tensor([math.sqrt(float(self.word_embed_dim))],
                            dtype=mstype.float32)
        self.multiply = P.Mul()

    def construct(self, source_ids, source_mask, target_ids, target_mask):
        """
        Construct network.

        In this method, T = src_max_len, T' = tgt_max_len.

        Args:
            source_ids (Tensor): Source sentences with shape (N, T).
            source_mask (Tensor): Source sentences padding mask with shape (N, T),
                where 0 indicates padding position.
            target_ids (Tensor): Target sentences with shape (N, T').
            target_mask (Tensor): Target sentences padding mask with shape (N, T'),
                where 0 indicates padding position.

        Returns:
            Tuple[Tensor], network outputs.
        """
        # Process source sentences.
        src_embeddings, embedding_tables = self.embedding_lookup(source_ids)
        src_embeddings = self.multiply(src_embeddings, self.scale)
        if self.use_positional_embedding:
            src_embeddings = self.positional_embedding(src_embeddings)
        src_embeddings = self.dropout(src_embeddings)

        # Attention mask with shape (N, T, T).
        enc_attention_mask = self._create_attention_mask_from_input_mask(source_mask)
        # Transformer encoder.
        encoder_output = self.encoder(
            self.cast_compute_type(src_embeddings),  # (N, T, D).
            self.cast_compute_type(enc_attention_mask)  # (N, T, T).
        )

        # Process target sentences.
        tgt_embeddings, _ = self.embedding_lookup(target_ids)
        tgt_embeddings = self.multiply(tgt_embeddings, self.scale)
        if self.use_positional_embedding:
            tgt_embeddings = self.positional_embedding(tgt_embeddings)
        tgt_embeddings = self.dropout(tgt_embeddings)

        # Attention mask with shape (N, T', T').
        tgt_attention_mask = self._create_attention_mask_from_input_mask(
            target_mask, True
        )
        # Transformer decoder.
        decoder_output = self.decoder(
            self.cast_compute_type(tgt_embeddings),  # (N, T', D)
            self.cast_compute_type(tgt_attention_mask),  # (N, T', T')
            encoder_output,  # (N, T, D)
            enc_attention_mask  # (N, T, T)
        )

        return encoder_output, decoder_output, embedding_tables
