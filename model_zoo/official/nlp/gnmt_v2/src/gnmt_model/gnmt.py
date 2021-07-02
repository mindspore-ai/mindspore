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
"""GNMTv2 network."""
import copy

from mindspore import nn
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype

from .embedding import EmbeddingLookup
from .create_attn_padding import CreateAttentionPaddingsFromInputPaddings
from .beam_search import BeamSearchDecoder, TileBeam
from .encoder import GNMTEncoder
from .decoder import GNMTDecoder
from .decoder_beam_infer import BeamDecoderStep
from .components import SaturateCast


class GNMT(nn.Cell):
    """
    GNMT with encoder and decoder.

    In GNMT, we define T = src_max_len, T' = tgt_max_len.

    Args:
        config: Model config.
        is_training (bool): Whether is training.
        use_one_hot_embeddings (bool): Whether use one-hot embedding.

    Returns:
        Tuple[Tensor], network outputs.
    """

    def __init__(self,
                 config,
                 is_training: bool = False,
                 use_one_hot_embeddings: bool = False,
                 use_positional_embedding: bool = True,
                 compute_type=mstype.float32):
        super(GNMT, self).__init__()

        self.input_mask_from_dataset = config.input_mask_from_dataset
        self.max_positions = config.seq_length
        self.attn_embed_dim = config.hidden_size

        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_dropout_prob = 0.0
        self.is_training = is_training
        self.num_layers = config.num_hidden_layers
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.vocab_size = config.vocab_size
        self.seq_length = config.seq_length
        self.batch_size = config.batch_size
        self.max_decode_length = config.max_decode_length
        self.word_embed_dim = config.hidden_size

        self.beam_width = config.beam_width

        self.transpose = P.Transpose()
        self.transpose_orders = (1, 0, 2)
        self.embedding_lookup = EmbeddingLookup(
            is_training=self.is_training,
            vocab_size=self.vocab_size,
            embed_dim=self.word_embed_dim,
            use_one_hot_embeddings=use_one_hot_embeddings)

        self.gnmt_encoder = GNMTEncoder(config, is_training)

        if self.is_training:
            # use for train.
            self.gnmt_decoder = GNMTDecoder(config, is_training)
        else:
            # use for infer.
            self.expand = P.ExpandDims()
            self.multiply = P.Mul()
            self.reshape = P.Reshape()
            self.create_att_paddings_from_input_paddings = CreateAttentionPaddingsFromInputPaddings(config,
                                                                                                    is_training=False)
            self.tile_beam = TileBeam(beam_width=config.beam_width)
            self.cast_compute_type = SaturateCast(dst_type=compute_type)

            beam_decoder_cell = BeamDecoderStep(config, use_one_hot_embeddings=use_one_hot_embeddings)
            # link beam_search after decoder
            self.beam_decoder = BeamSearchDecoder(
                batch_size=config.batch_size,
                seq_length=config.seq_length,
                vocab_size=config.vocab_size,
                decoder=beam_decoder_cell,
                beam_width=config.beam_width,
                decoder_layers_nums=config.num_hidden_layers,
                length_penalty_weight=config.length_penalty_weight,
                hidden_size=config.hidden_size,
                max_decode_length=config.max_decode_length)
            self.beam_decoder.add_flags(loop_can_unroll=True)
        self.shape = P.Shape()

    def construct(self, source_ids, source_mask=None, target_ids=None):
        """
        Construct network.

        In this method, T = src_max_len, T' = tgt_max_len.

        Args:
            source_ids (Tensor): Source sentences with shape (N, T).
            source_mask (Tensor): Source sentences padding mask with shape (N, T),
                where 0 indicates padding position.
            target_ids (Tensor): Target sentences with shape (N, T').

        Returns:
            Tuple[Tensor], network outputs.
        """

        # Process source sentences. src_embeddings:[N, T, D].
        src_embeddings, _ = self.embedding_lookup(source_ids)
        # T, N, D
        inputs = self.transpose(src_embeddings, self.transpose_orders)
        # encoder. encoder_outputs: [T, N, D]
        encoder_outputs = self.gnmt_encoder(inputs)

        # decoder.
        if self.is_training:
            # training
            # process target input sentences. N, T, D
            tgt_embeddings, _ = self.embedding_lookup(target_ids)
            # T, N, D
            tgt_embeddings = self.transpose(tgt_embeddings, self.transpose_orders)
            # cell: [T,N,D].
            cell, _, _ = self.gnmt_decoder(tgt_embeddings,
                                           encoder_outputs,
                                           attention_mask=source_mask)
            # decoder_output: (N, T', V).
            decoder_outputs = self.transpose(cell, self.transpose_orders)
            out = decoder_outputs
        else:
            # infer
            # encoder_output:  [T, N, D] -> [N, T, D].
            beam_encoder_output = self.transpose(encoder_outputs, self.transpose_orders)
            # bean search for encoder output, [N*beam_width, T, D]
            beam_encoder_output = self.tile_beam(beam_encoder_output)

            # (N*beam_width, T)
            beam_enc_attention_pad = self.tile_beam(source_mask)

            predicted_ids = self.beam_decoder(beam_encoder_output, beam_enc_attention_pad)
            predicted_ids = self.reshape(predicted_ids, (-1, self.max_decode_length))
            out = predicted_ids

        return out
