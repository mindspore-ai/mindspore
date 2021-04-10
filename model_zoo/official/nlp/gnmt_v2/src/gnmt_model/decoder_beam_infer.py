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
"""Decoder for beam_search of GNMT."""
import numpy as np

from mindspore import nn, Tensor
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype

from .embedding import EmbeddingLookup
from .decoder import GNMTDecoder
from .create_attn_padding import CreateAttentionPaddingsFromInputPaddings
from .components import SaturateCast


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
        self.log_softmax = nn.LogSoftmax(axis=-1)
        self.cast = P.Cast()

    def construct(self, logits):
        """
        Calculate the log_softmax.

        Inputs:
            input_tensor (Tensor): A batch of sentences with shape (N, T).
            output_weights (Tensor): A batch of masks with shape (N, T).

        Returns:
            Tensor, the prediction probability with shape (N, T').
        """
        log_probs = self.log_softmax(logits)
        return log_probs


class BeamDecoderStep(nn.Cell):
    """
    Multi-layer transformer decoder step.

    Args:
        config (GNMTConfig): The config of Transformer.
    """

    def __init__(self,
                 config,
                 use_one_hot_embeddings,
                 compute_type=mstype.float32):
        super(BeamDecoderStep, self).__init__(auto_prefix=True)

        self.vocab_size = config.vocab_size
        self.word_embed_dim = config.hidden_size
        self.embedding_lookup = EmbeddingLookup(
            is_training=False,
            vocab_size=config.vocab_size,
            embed_dim=self.word_embed_dim,
            use_one_hot_embeddings=use_one_hot_embeddings)

        self.projection = PredLogProbs(
            batch_size=config.batch_size * config.beam_width,
            seq_length=1,
            width=config.vocab_size,
            compute_type=config.compute_type)

        self.seq_length = config.max_decode_length
        self.decoder = GNMTDecoder(config,
                                   is_training=False,
                                   infer_beam_width=config.beam_width)

        self.ones_like = P.OnesLike()
        self.shape = P.Shape()

        self.create_att_paddings_from_input_paddings = CreateAttentionPaddingsFromInputPaddings(config,
                                                                                                is_training=False)
        self.expand = P.ExpandDims()
        self.multiply = P.Mul()

        ones = np.ones(shape=(config.max_decode_length, config.max_decode_length))
        self.future_mask = Tensor(np.tril(ones), dtype=mstype.float32)

        self.cast_compute_type = SaturateCast(dst_type=compute_type)

        self.transpose = P.Transpose()
        self.transpose_orders = (1, 0, 2)

    def construct(self, input_ids, enc_states, enc_attention_mask, decoder_hidden_state=None):
        """
        Get log probs.

        Args:
            input_ids: [batch_size * beam_width, m]
            enc_states: [batch_size * beam_width, T, D]
            enc_attention_mask: [batch_size * beam_width, T]
            decoder_hidden_state: [decoder_layers_nums, 2, batch_size * beam_width, hidden_size].

        Returns:
            Tensor, the log_probs. [batch_size * beam_width, 1, vocabulary_size]
        """

        # process embedding. input_embedding: [batch_size * beam_width, m, D], embedding_tables: [V, D]
        input_embedding, _ = self.embedding_lookup(input_ids)
        input_embedding = self.cast_compute_type(input_embedding)

        input_shape = self.shape(input_ids)
        input_len = input_shape[1]
        # [m, batch_size * beam_width, D]
        input_embedding = self.transpose(input_embedding, self.transpose_orders)
        enc_states = self.transpose(enc_states, self.transpose_orders)

        # decoder_output: [m, batch_size*beam_width, V], scores:[b, t_q, t_k], all_decoder_state:[4,2,b*beam_width,D]
        decoder_output, all_decoder_state, scores = self.decoder(input_embedding, enc_states, enc_attention_mask,
                                                                 decoder_hidden_state)
        # [batch_size * beam_width, m, v]
        decoder_output = self.transpose(decoder_output, self.transpose_orders)

        # take the last step, [batch_size * beam_width, 1, V]
        decoder_output = decoder_output[:, (input_len - 1):input_len, :]

        # projection and log_prob
        log_probs = self.projection(decoder_output)

        # [batch_size * beam_width, 1, vocabulary_size]
        return log_probs, all_decoder_state, scores
