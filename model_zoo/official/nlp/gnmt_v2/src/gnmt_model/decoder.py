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
"""Decoder of GNMT."""
import numpy as np

from mindspore.common.initializer import Uniform
from mindspore import nn, Tensor
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype

from .dynamic_rnn import DynamicRNNNet
from .create_attention import RecurrentAttention


class GNMTDecoder(nn.Cell):
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
                 config,
                 is_training: bool,
                 use_one_hot_embeddings: bool = False,
                 initializer_range=0.1,
                 infer_beam_width=1,
                 compute_type=mstype.float16):
        super(GNMTDecoder, self).__init__()

        self.is_training = is_training
        self.attn_embed_dim = config.hidden_size
        self.num_layers = config.num_hidden_layers
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.vocab_size = config.vocab_size
        self.seq_length = config.max_decode_length
        # batchsize* beam_width for beam_search.
        self.batch_size = config.batch_size * infer_beam_width
        self.word_embed_dim = config.hidden_size
        self.transpose = P.Transpose()
        self.transpose_orders = (1, 0, 2)
        self.reshape = P.Reshape()
        self.concat = P.Concat(axis=-1)
        self.state_concat = P.Concat(axis=0)
        self.all_decoder_state = Tensor(np.zeros([self.num_layers, 2, self.batch_size, config.hidden_size]),
                                        mstype.float32)

        decoder_layers = []
        for i in range(0, self.num_layers):
            if i == 0:
                # the inputs is [T,D,N]
                scaler = 1
            else:
                # the inputs is [T,D,2N]
                scaler = 2
            layer = DynamicRNNNet(seq_length=self.seq_length,
                                  batchsize=self.batch_size,
                                  word_embed_dim=scaler * self.word_embed_dim,
                                  hidden_size=self.word_embed_dim)
            decoder_layers.append(layer)
        self.decoder_layers = nn.CellList(decoder_layers)

        self.att_rnn = RecurrentAttention(rnn=self.decoder_layers[0],
                                          is_training=is_training,
                                          input_size=self.word_embed_dim,
                                          context_size=self.attn_embed_dim,
                                          hidden_size=self.attn_embed_dim,
                                          num_layers=1,
                                          dropout=config.attention_dropout_prob)

        self.dropout = nn.Dropout(keep_prob=1.0 - config.hidden_dropout_prob)

        self.classifier = nn.Dense(config.hidden_size,
                                   config.vocab_size,
                                   has_bias=True,
                                   weight_init=Uniform(initializer_range),
                                   bias_init=Uniform(initializer_range)).to_float(compute_type)
        self.cast = P.Cast()
        self.shape_op = P.Shape()
        self.expand = P.ExpandDims()

    def construct(self, tgt_embeddings, encoder_outputs, attention_mask=None,
                  decoder_init_state=None):
        """Decoder."""
        # tgt_embeddings: [T',N,D], encoder_outputs: [T,N,D], attention_mask: [N,T].
        query_shape = self.shape_op(tgt_embeddings)
        if decoder_init_state is None:
            hidden_state = self.all_decoder_state
        else:
            hidden_state = decoder_init_state
        # x:[t_q,b,D], attn:[t_q,b,D], scores:[b, t_q, t_k], state_0:[2,b,D].
        x, attn, state_0, scores = self.att_rnn(decoder_embedding=tgt_embeddings, context_key=encoder_outputs,
                                                attention_mask=attention_mask, rnn_init_state=hidden_state[0, :, :, :])
        x = self.concat((x, attn))
        x = self.dropout(x)
        decoder_outputs, state_1 = self.decoder_layers[1](x, hidden_state[1, :, :, :])

        all_decoder_state = self.state_concat((self.expand(state_0, 0), self.expand(state_1, 0)))

        for i in range(2, self.num_layers):
            residual = decoder_outputs
            decoder_outputs = self.concat((decoder_outputs, attn))

            decoder_outputs = self.dropout(decoder_outputs)
            # 1st unidirectional layer. encoder_outputs: [T,N,D]
            decoder_outputs, decoder_state = self.decoder_layers[i](decoder_outputs, hidden_state[i, :, :, :])
            decoder_outputs += residual
            all_decoder_state = self.state_concat((all_decoder_state, self.expand(decoder_state, 0)))

        # [m, batch_size * beam_width, D]
        decoder_outputs = self.reshape(decoder_outputs, (-1, self.word_embed_dim))
        if self.is_training:
            decoder_outputs = self.cast(decoder_outputs, mstype.float16)
        decoder_outputs = self.classifier(decoder_outputs)
        if self.is_training:
            decoder_outputs = self.cast(decoder_outputs, mstype.float32)
        # [m, batch_size * beam_width, V]
        decoder_outputs = self.reshape(decoder_outputs, (query_shape[0], query_shape[1], self.vocab_size))
        # all_decoder_state:[4,2,b,D]
        return decoder_outputs, all_decoder_state, scores
