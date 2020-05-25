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
"""LSTM."""
import math

import numpy as np

from mindspore import Parameter, Tensor, nn
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P


def init_lstm_weight(
        input_size,
        hidden_size,
        num_layers,
        bidirectional,
        has_bias=True):
    """Initialize lstm weight."""
    num_directions = 1
    if bidirectional:
        num_directions = 2

    weight_size = 0
    gate_size = 4 * hidden_size
    for layer in range(num_layers):
        for _ in range(num_directions):
            input_layer_size = input_size if layer == 0 else hidden_size * num_directions
            weight_size += gate_size * input_layer_size
            weight_size += gate_size * hidden_size
            if has_bias:
                weight_size += 2 * gate_size

    stdv = 1 / math.sqrt(hidden_size)
    w_np = np.random.uniform(-stdv, stdv, (weight_size, 1, 1)).astype(np.float32)
    w = Parameter(initializer(Tensor(w_np), [weight_size, 1, 1]), name='weight')

    return w


# Initialize short-term memory (h) and long-term memory (c) to 0
def lstm_default_state(batch_size, hidden_size, num_layers, bidirectional):
    """init default input."""
    num_directions = 1
    if bidirectional:
        num_directions = 2

    h = Tensor(
        np.zeros((num_layers * num_directions, batch_size, hidden_size)).astype(np.float32))
    c = Tensor(
        np.zeros((num_layers * num_directions, batch_size, hidden_size)).astype(np.float32))
    return h, c


class SentimentNet(nn.Cell):
    """Sentiment network structure."""

    def __init__(self,
                 vocab_size,
                 embed_size,
                 num_hiddens,
                 num_layers,
                 bidirectional,
                 num_classes,
                 weight,
                 batch_size):
        super(SentimentNet, self).__init__()
        # Mapp words to vectors
        self.embedding = nn.Embedding(vocab_size,
                                      embed_size,
                                      embedding_table=weight)
        self.embedding.embedding_table.requires_grad = False
        self.trans = P.Transpose()
        self.perm = (1, 0, 2)
        self.encoder = nn.LSTM(input_size=embed_size,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               has_bias=True,
                               bidirectional=bidirectional,
                               dropout=0.0)
        w_init = init_lstm_weight(
            embed_size,
            num_hiddens,
            num_layers,
            bidirectional)
        self.encoder.weight = w_init
        self.h, self.c = lstm_default_state(batch_size, num_hiddens, num_layers, bidirectional)

        self.concat = P.Concat(1)
        if bidirectional:
            self.decoder = nn.Dense(num_hiddens * 4, num_classes)
        else:
            self.decoder = nn.Dense(num_hiddens * 2, num_classes)

    def construct(self, inputs):
        # input：(64,500,300)
        embeddings = self.embedding(inputs)
        embeddings = self.trans(embeddings, self.perm)
        output, _ = self.encoder(embeddings, (self.h, self.c))
        # states[i] size(64,200)  -> encoding.size(64,400)
        encoding = self.concat((output[0], output[1]))
        outputs = self.decoder(encoding)
        return outputs
