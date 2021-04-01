# Copyright 2021 Huawei Technologies Co., Ltd
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
"""seq2seq model"""
import copy
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype

from config.config import Seq2seqConfig
from .embedding import EmbeddingLookup
from .beam_search import BeamSearchDecoder, TileBeam
from .encoder import Seq2seqEncoder
from .decoder import Seq2seqDecoder
from .components import SaturateCast
from .decoder_beam_infer import BeamDecoderStep


class Seq2seqModel(nn.Cell):
    """
    Seq2seq with encoder and decoder.

    Args:
        config (Seq2seqConfig): Model config.
        is_training (bool): Whether is training.
        use_one_hot_embeddings (bool): Whether use one-hot embedding.

    Returns:
        Tuple[Tensor], network outputs.
    """
    def __init__(self,
                 config: Seq2seqConfig,
                 is_training: bool = False,
                 use_one_hot_embeddings: bool = False,
                 compute_type=mstype.float32):
        super(Seq2seqModel, self).__init__()

        config = copy.deepcopy(config)

        self.is_training = is_training
        self.vocab_size = config.vocab_size
        self.seq_length = config.seq_length
        self.batch_size = config.batch_size
        self.max_decode_length = config.max_decode_length
        self.word_embed_dim = config.hidden_size
        self.hidden_size = config.hidden_size
        self.beam_width = config.beam_width
        self.expand = P.ExpandDims()
        self.state_concat = P.Concat(axis=0)
        self.transpose = P.Transpose()
        self.transpose_orders = (1, 0, 2)

        self.embedding_lookup = EmbeddingLookup(
            is_training=self.is_training,
            vocab_size=self.vocab_size,
            embed_dim=self.word_embed_dim,
            use_one_hot_embeddings=use_one_hot_embeddings)

        self.seq2seq_encoder = Seq2seqEncoder(config, is_training)

        if self.is_training:
            # use for train.
            self.seq2seq_decoder = Seq2seqDecoder(config, is_training)

        else:
            # use for infer.
            self.reshape = P.Reshape()
            self.tile_beam = TileBeam(beam_width=config.beam_width)
            self.cast_compute_type = SaturateCast(dst_type=compute_type)

            beam_decoder_cell = BeamDecoderStep(config,
                                                use_one_hot_embeddings=use_one_hot_embeddings)
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
        _, state = self.seq2seq_encoder(inputs)

        decoder_init_state = self.state_concat((self.expand(state, 0), self.expand(state, 0)))
        decoder_init_state = self.state_concat((decoder_init_state, decoder_init_state))

        # decoder.
        if self.is_training:
            # training
            # process target input sentences. N, T, D
            tgt_embeddings, _ = self.embedding_lookup(target_ids)
            # T, N, D
            tgt_embeddings = self.transpose(tgt_embeddings, self.transpose_orders)
            # cell: [T,N,D].
            cell, _ = self.seq2seq_decoder(tgt_embeddings, decoder_init_state)
            # decoder_output: (N, T', V).
            decoder_outputs = self.transpose(cell, self.transpose_orders)
            out = decoder_outputs
        else:
            #infer
            beam_state = self.transpose(state, self.transpose_orders)
            # bean search for state, [N*beam_width, 2, D]
            beam_state = self.tile_beam(beam_state)
            beam_state = self.transpose(beam_state, self.transpose_orders)
            #[2, N*beam_width, D]
            predicted_ids = self.beam_decoder(beam_state)
            predicted_ids = self.reshape(predicted_ids, (-1, self.max_decode_length))
            out = predicted_ids

        return out
