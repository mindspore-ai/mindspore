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
"""Transformer for infer."""
import math
import copy
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor

from .beam_search import BeamSearchDecoder, TileBeam
from .embedding import EmbeddingLookup
from .positional_embedding import PositionalEmbedding
from .components import SaturateCast
from .create_attn_mask import CreateAttentionMaskFromInputMask
from .decoder import TransformerDecoder
from .encoder import TransformerEncoder


class PredLogProbs(nn.Cell):
    """
    Get log probs.

    Args:
        batch_size (int): Batch size of input dataset.
        seq_length (int): The length of sequences.
        width (int): Number of parameters of a layer
        compute_type (int): Type of input type.
        dtype (int): Type of MindSpore output type.
    """

    def __init__(self,
                 batch_size,
                 seq_length,
                 width,
                 compute_type=mstype.float32,
                 dtype=mstype.float32):
        super(PredLogProbs, self).__init__()
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.width = width
        self.compute_type = compute_type
        self.dtype = dtype

        self.reshape = P.Reshape()
        self.matmul = P.MatMul(transpose_b=True)
        self.log_softmax = nn.LogSoftmax(axis=-1)
        self.shape_flat_sequence_tensor = (self.batch_size * self.seq_length, self.width)
        self.cast = P.Cast()

    def construct(self, input_tensor, output_weights):
        """
        Calculate the log_softmax.

        Inputs:
            input_tensor (Tensor): A batch of sentences with shape (N, T).
            output_weights (Tensor): A batch of masks with shape (N, T).

        Returns:
            Tensor, the prediction probability with shape (N, T').
        """
        input_tensor = self.reshape(input_tensor, self.shape_flat_sequence_tensor)
        input_tensor = self.cast(input_tensor, self.compute_type)
        output_weights = self.cast(output_weights, self.compute_type)

        logits = self.matmul(input_tensor, output_weights)
        logits = self.cast(logits, self.dtype)

        log_probs = self.log_softmax(logits)
        return log_probs


class TransformerDecoderStep(nn.Cell):
    """
    Multi-layer transformer decoder step.

    Args:
        config (TransformerConfig): The config of Transformer.
        num_hidden_layers (int): The numbers of hidden layers.
        attn_embed_dim (int): Dimensions of attention weights.
        num_attn_heads=12 (int): Heads number.
        seq_length (int): The length of a sequence.
        intermediate_size: Hidden size in FFN.
        attn_dropout_prob (float): Dropout rate in attention. Default: 0.1.
        initializer_range (float): Initial range.
        hidden_dropout_prob (float): Dropout rate in FFN.
        hidden_act (str): Activation function in FFN.
        compute_type (mstype): Mindspore data type. Default: mstype.float32.
        embedding_lookup (function): Embeddings lookup operation. Default: None.
        positional_embedding (function): Position Embedding operation. Default: None.
        projection (function): Function to get log probs. Default: None.
    """

    def __init__(self,
                 config,
                 num_hidden_layers,
                 attn_embed_dim,
                 num_attn_heads=12,
                 seq_length=64,
                 intermediate_size=3072,
                 attn_dropout_prob=0.1,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 hidden_act="relu",
                 compute_type=mstype.float32,
                 embedding_lookup=None,
                 positional_embedding=None,
                 projection=None):
        super(TransformerDecoderStep, self).__init__(auto_prefix=False)
        self.embedding_lookup = embedding_lookup
        self.positional_embedding = positional_embedding
        self.projection = projection
        self.seq_length = seq_length
        self.decoder = TransformerDecoder(
            attn_embed_dim=attn_embed_dim,
            num_attn_heads=num_attn_heads,
            decoder_layers=num_hidden_layers,
            intermediate_size=intermediate_size,
            attn_dropout_prob=attn_dropout_prob,
            initializer_range=initializer_range,
            dropout_prob=hidden_dropout_prob,
            hidden_act=hidden_act,
            compute_type=compute_type)

        self.ones_like = P.OnesLike()
        self.shape = P.Shape()

        self._create_attention_mask_from_input_mask = CreateAttentionMaskFromInputMask(config)
        self.expand = P.ExpandDims()
        self.multiply = P.Mul()

        ones = np.ones(shape=(seq_length, seq_length))
        self.future_mask = Tensor(np.tril(ones), dtype=mstype.float32)

        self.cast_compute_type = SaturateCast(dst_type=compute_type)
        self.scale = Tensor([math.sqrt(float(attn_embed_dim))], dtype=mstype.float32)

    def construct(self, input_ids, enc_states, enc_attention_mask):
        """
        Get log probs.

        Args:
            input_ids: [batch_size * beam_width, m]
            enc_states: [batch_size * beam_width, T, D]
            enc_attention_mask: [batch_size * beam_width, T, D]

        Returns:
            Tensor, the log_probs. [batch_size * beam_width, 1, Vocabulary_Dimension]
        """

        # process embedding. input_embedding: [batch_size * beam_width, m, D], embedding_tables: [V, D]
        input_embedding, embedding_tables = self.embedding_lookup(input_ids)
        input_embedding = self.multiply(input_embedding, self.scale)
        input_embedding = self.positional_embedding(input_embedding)
        input_embedding = self.cast_compute_type(input_embedding)

        input_shape = self.shape(input_ids)
        input_len = input_shape[1]
        # [m,m]
        future_mask = self.future_mask[0:input_len:1, 0:input_len:1]
        # [batch_size * beam_width, m]
        input_mask = self.ones_like(input_ids)
        # [batch_size * beam_width, m, m]
        input_mask = self._create_attention_mask_from_input_mask(input_mask)
        # [batch_size * beam_width, m, m]
        input_mask = self.multiply(input_mask, self.expand(future_mask, 0))
        input_mask = self.cast_compute_type(input_mask)

        # [batch_size * beam_width, m, D]
        enc_attention_mask = enc_attention_mask[::, 0:input_len:1, ::]

        # call TransformerDecoder:  [batch_size * beam_width, m, D]
        decoder_output = self.decoder(input_embedding, input_mask, enc_states, enc_attention_mask)

        # take the last step, [batch_size * beam_width, 1, D]
        decoder_output = decoder_output[::, input_len - 1:input_len:1, ::]

        # projection and log_prob
        log_probs = self.projection(decoder_output, embedding_tables)

        # [batch_size * beam_width, 1, vocabulary_size]
        return log_probs


class TransformerInferModel(nn.Cell):
    """
    Transformer Infer.

    Args:
        config (TransformerConfig): The config of Transformer.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
    """

    def __init__(self,
                 config,
                 use_one_hot_embeddings=False):
        super(TransformerInferModel, self).__init__()
        config = copy.deepcopy(config)
        config.hidden_dropout_prob = 0.0
        config.attention_dropout_prob = 0.0

        self.input_mask_from_dataset = config.input_mask_from_dataset
        self.batch_size = config.batch_size
        self.seq_length = config.seq_length
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embedding_size = config.hidden_size
        self.attn_embed_dim = config.hidden_size
        self.num_layers = config.num_hidden_layers
        self.last_idx = self.num_hidden_layers - 1

        self.embedding_lookup = EmbeddingLookup(
            vocab_size=config.vocab_size,
            embed_dim=self.embedding_size,
            use_one_hot_embeddings=use_one_hot_embeddings)

        self.positional_embedding = PositionalEmbedding(
            embedding_size=self.embedding_size,
            max_position_embeddings=config.max_position_embeddings)
        # use for infer
        self.projection = PredLogProbs(
            batch_size=config.batch_size * config.beam_width,
            seq_length=1,
            width=self.hidden_size,
            compute_type=config.compute_type)

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

        decoder_cell = TransformerDecoderStep(
            config=config,
            num_hidden_layers=config.num_hidden_layers,
            attn_embed_dim=self.attn_embed_dim,
            seq_length=config.seq_length,
            num_attn_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            hidden_dropout_prob=config.hidden_dropout_prob,
            compute_type=config.compute_type,
            initializer_range=config.initializer_range,
            hidden_act="relu",
            embedding_lookup=self.embedding_lookup,
            positional_embedding=self.positional_embedding,
            attn_dropout_prob=config.attention_dropout_prob,
            projection=self.projection
        )

        # link beam_search after decoder
        self.decoder = BeamSearchDecoder(
            batch_size=config.batch_size,
            seq_length=config.seq_length,
            vocab_size=config.vocab_size,
            decoder=decoder_cell,
            beam_width=config.beam_width,
            length_penalty_weight=config.length_penalty_weight,
            max_decode_length=config.max_decode_length)

        self.decoder.add_flags(loop_can_unroll=True)

        self.cast = P.Cast()
        self.dtype = config.dtype
        self.cast_compute_type = SaturateCast(dst_type=config.compute_type)
        self.expand = P.ExpandDims()
        self.multiply = P.Mul()

        self._create_attention_mask_from_input_mask = CreateAttentionMaskFromInputMask(config)

        # use for infer
        self.tile_beam = TileBeam(beam_width=config.beam_width)
        ones = np.ones(shape=(config.batch_size, config.max_decode_length))
        self.encode_mask = Tensor(ones, dtype=mstype.float32)

        self.scale = Tensor([math.sqrt(float(self.embedding_size))],
                            dtype=mstype.float32)
        self.reshape = P.Reshape()

    def construct(self, source_ids, source_mask, target_ids=None, target_mask=None):
        """
        Process source sentence

        Inputs:
            source_ids (Tensor): Source sentences with shape (N, T).
            source_mask (Tensor): Source sentences padding mask with shape (N, T),
                where 0 indicates padding position.

        Returns:
            Tensor, Predictions with shape (N, T').
        """
        # word_embeddings
        src_embeddings, _ = self.embedding_lookup(source_ids)
        src_embeddings = self.multiply(src_embeddings, self.scale)
        # position_embeddings
        src_embeddings = self.positional_embedding(src_embeddings)
        # attention mask, [batch_size, seq_length, seq_length]
        enc_attention_mask = self._create_attention_mask_from_input_mask(source_mask)
        # encode
        encoder_output = self.encoder(self.cast_compute_type(src_embeddings),
                                      self.cast_compute_type(enc_attention_mask))

        # bean search for encoder output
        beam_encoder_output = self.tile_beam(encoder_output)
        # [batch_size, T, D]
        enc_attention_mask = self.multiply(
            enc_attention_mask[::, 0:1:1, ::],
            self.expand(self.encode_mask, -1))
        # [N*batch_size, T, D]
        beam_enc_attention_mask = self.tile_beam(enc_attention_mask)
        beam_enc_attention_mask = self.cast_compute_type(beam_enc_attention_mask)
        predicted_ids, predicted_probs = self.decoder(beam_encoder_output, beam_enc_attention_mask)
        predicted_ids = self.reshape(predicted_ids, (self.batch_size, -1))
        return predicted_ids, predicted_probs
