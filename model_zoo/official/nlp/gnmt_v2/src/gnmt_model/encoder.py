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
"""Encoder of GNMT."""
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype

from config.config import GNMTConfig
from .dynamic_rnn import DynamicRNNNet


class GNMTEncoder(nn.Cell):
    """
    Implements of GNMT encoder.

    Args:
        config (GNMTConfig): Configuration of GNMT network.
        is_training (bool): Whether to train.
        compute_type (mstype): Mindspore data type.

    Returns:
        Tensor, shape of (N, T, D).
    """

    def __init__(self,
                 config: GNMTConfig,
                 is_training: bool,
                 compute_type=mstype.float32):
        super(GNMTEncoder, self).__init__()
        self.input_mask_from_dataset = config.input_mask_from_dataset
        self.max_positions = config.seq_length
        self.attn_embed_dim = config.hidden_size

        self.num_layers = config.num_hidden_layers
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.vocab_size = config.vocab_size
        self.seq_length = config.seq_length
        self.batch_size = config.batch_size
        self.word_embed_dim = config.hidden_size

        self.transpose = P.Transpose()
        self.transpose_orders = (1, 0, 2)
        self.reshape = P.Reshape()
        self.concat = P.Concat(axis=-1)
        encoder_layers = []
        for i in range(0, self.num_layers + 1):
            if i == 2:
                # the bidirectional layer's output is [T,D,2N]
                scaler = 2
            else:
                # the rest layer's output is [T,D,N]
                scaler = 1
            layer = DynamicRNNNet(seq_length=self.seq_length,
                                  batchsize=self.batch_size,
                                  word_embed_dim=scaler * self.word_embed_dim,
                                  hidden_size=self.word_embed_dim)
            encoder_layers.append(layer)
        self.encoder_layers = nn.CellList(encoder_layers)
        self.reverse_v2 = P.ReverseV2(axis=[0])
        self.dropout = nn.Dropout(keep_prob=1.0 - config.hidden_dropout_prob)

    def construct(self, inputs):
        """Encoder."""
        inputs = self.dropout(inputs)
        # bidirectional layer, fwd_encoder_outputs: [T,N,D]
        fwd_encoder_outputs, _ = self.encoder_layers[0](inputs)

        # the input need reverse.
        inputs_r = self.reverse_v2(inputs)
        bak_encoder_outputs, _ = self.encoder_layers[1](inputs_r)
        # the result need reverse.
        bak_encoder_outputs = self.reverse_v2(bak_encoder_outputs)

        # bi_encoder_outputs: [T,N,2D]
        bi_encoder_outputs = self.concat((fwd_encoder_outputs, bak_encoder_outputs))

        # 1st unidirectional layer. encoder_outputs: [T,N,D]
        bi_encoder_outputs = self.dropout(bi_encoder_outputs)
        encoder_outputs, _ = self.encoder_layers[2](bi_encoder_outputs)
        # Build all the rest unidi layers of encoder
        for i in range(3, self.num_layers + 1):
            residual = encoder_outputs
            encoder_outputs = self.dropout(encoder_outputs)
            # [T,N,D] -> [T,N,D]
            encoder_outputs_o, _ = self.encoder_layers[i](encoder_outputs)
            encoder_outputs = encoder_outputs_o + residual

        return encoder_outputs
