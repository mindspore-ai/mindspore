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
"""Decoder of Seq2seq."""
import copy
import numpy as np

from mindspore.common.initializer import Uniform
from mindspore import nn, Tensor
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype

from config.config import Seq2seqConfig
from .dynamic_rnn import DynamicRNNNet


class Seq2seqDecoder(nn.Cell):
    """
    Implements of decoder.

    Args:
        decoder_layers (int): Decoder layers.
        intermediate_size (int): Hidden size of FFN.
        initializer_range (float): Initial range. Default: 0.02.
        dropout_prob (float): Dropout rate between layers. Default: 0.1.
        hidden_act (str): Non-linear activation function in FFN. Default: "relu".
        compute_type (mstype): Mindspore data type. Default: mstype.float32.

    Returns:
        Tensor, shape of (N, T', D).
    """
    def __init__(self,
                 config: Seq2seqConfig,
                 is_training: bool,
                 use_one_hot_embeddings: bool = False,
                 initializer_range=0.1,
                 infer_beam_width=1,
                 compute_type=mstype.float16):

        super(Seq2seqDecoder, self).__init__()

        config = copy.deepcopy(config)

        if not is_training:
            config.hidden_dropout_prob = 0.0

        self.is_training = is_training
        self.num_layers = config.num_hidden_layers
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.vocab_size = config.vocab_size
        self.seq_length = config.max_decode_length
        # batchsize* beam_width for beam_search.
        self.batch_size = config.batch_size * infer_beam_width
        self.word_embed_dim = config.hidden_size
        self.hidden_size = config.hidden_size
        self.transpose = P.Transpose()
        self.transpose_orders = (1, 0, 2)
        self.reshape = P.Reshape()
        self.concat = P.Concat(axis=-1)
        self.oneslike = P.OnesLike()
        self.state_concat = P.Concat(axis=0)
        self.all_decoder_state = Tensor(np.zeros([self.num_layers, 2, self.batch_size, config.hidden_size]),
                                        mstype.float32)

        decoder_layers = []
        for _ in range(0, self.num_layers):
            layer = DynamicRNNNet(
                seq_length=self.seq_length,
                batchsize=self.batch_size,
                word_embed_dim=self.word_embed_dim,
                hidden_size=self.word_embed_dim)
            decoder_layers.append(layer)

        self.decoder_layers = nn.CellList(decoder_layers)
        self.dropout = nn.Dropout(keep_prob=1.0 - config.hidden_dropout_prob)
        self.classifier = nn.Dense(config.hidden_size,
                                   config.vocab_size,
                                   has_bias=True,
                                   weight_init=Uniform(initializer_range),
                                   bias_init=Uniform(initializer_range)).to_float(compute_type)
        self.cast = P.Cast()
        self.shape_op = P.Shape()
        self.expand = P.ExpandDims()
        self.squeeze = P.Squeeze(0)

    def construct(self, tgt_embeddings, decoder_init_state=None):
        """Decoder."""
        # tgt_embeddings: [T',N,D], state: [2,N,D]
        query_shape = self.shape_op(tgt_embeddings)
        if decoder_init_state is None:
            hidden_state = self.all_decoder_state
        else:
            hidden_state = decoder_init_state

        decoder_outputs = self.dropout(tgt_embeddings)
        decoder_outputs, state_0 = self.decoder_layers[0](decoder_outputs,
                                                          self.squeeze(hidden_state[0:1, :, :, :]))
        all_decoder_state = self.expand(state_0, 0)

        for i in range(1, self.num_layers):
            decoder_outputs = self.dropout(decoder_outputs)
            decoder_outputs, state = self.decoder_layers[i](decoder_outputs,
                                                            self.squeeze(hidden_state[i:i+1, :, :, :]))
            all_decoder_state = self.state_concat((all_decoder_state, self.expand(state, 0)))

        decoder_outputs = self.reshape(decoder_outputs, (-1, self.word_embed_dim))

        if self.is_training:
            decoder_outputs = self.cast(decoder_outputs, mstype.float16)
        decoder_outputs = self.classifier(decoder_outputs)
        if self.is_training:
            decoder_outputs = self.cast(decoder_outputs, mstype.float32)

        # [m, batch_size * beam_width, V]
        decoder_outputs = self.reshape(decoder_outputs, (query_shape[0], query_shape[1], self.vocab_size))

        return decoder_outputs, all_decoder_state
