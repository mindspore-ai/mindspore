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
"""Encoder of Seq2seq."""
import copy
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype

from config.config import Seq2seqConfig
from .dynamic_rnn import DynamicRNNNet

class Seq2seqEncoder(nn.Cell):
    """
    Implements of Seq2seq encoder.

    Args:
        config (Seq2seqConfig): Configuration of Seq2seq network.
        is_training (bool): Whether to train.
        compute_type (mstype): Mindspore data type.

    Returns:
        Tensor, shape of (2, T, D).
    """

    def __init__(self,
                 config: Seq2seqConfig,
                 is_training: bool,
                 compute_type=mstype.float32):
        super(Seq2seqEncoder, self).__init__()

        config = copy.deepcopy(config)

        if not is_training:
            config.hidden_dropout_prob = 0.0

        self.num_layers = config.num_hidden_layers
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.seq_length = config.seq_length
        self.batch_size = config.batch_size
        self.word_embed_dim = config.hidden_size

        encoder_layers = []
        for _ in range(0, self.num_layers):
            layer = DynamicRNNNet(seq_length=self.seq_length,
                                  batchsize=self.batch_size,
                                  word_embed_dim=self.word_embed_dim,
                                  hidden_size=self.word_embed_dim)
            encoder_layers.append(layer)

        self.encoder_layers = nn.CellList(encoder_layers)
        self.dropout = nn.Dropout(keep_prob=1.0 - config.hidden_dropout_prob)
        self.reverse_v2 = P.ReverseV2(axis=[0])

    def construct(self, inputs):
        """Encoder."""
        inputs_r = self.reverse_v2(inputs)
        encoder_outputs = inputs_r
        state = 0

        for i in range(0, self.num_layers):
            encoder_outputs = self.dropout(encoder_outputs)
            # [T,N,D] -> [T,N,D]
            encoder_outputs, state = self.encoder_layers[i](encoder_outputs)

        return encoder_outputs, state
