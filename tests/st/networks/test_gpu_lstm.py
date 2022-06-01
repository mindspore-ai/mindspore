# Copyright 2019 Huawei Technologies Co., Ltd
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

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn.optim import Momentum
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


def InitialLstmWeight(input_size, hidden_size, num_layers, bidirectional, has_bias=False):
    num_directions = 1
    if bidirectional:
        num_directions = 2

    weight_size = 0
    gate_size = 4 * hidden_size
    for layer in range(num_layers):
        for d in range(num_directions):
            input_layer_size = input_size if layer == 0 else hidden_size * num_directions
            weight_size += gate_size * input_layer_size
            weight_size += gate_size * hidden_size
            if has_bias:
                weight_size += 2 * gate_size

    w_np = np.ones([weight_size, 1, 1]).astype(np.float32) * 0.01

    w = Parameter(initializer(Tensor(w_np), w_np.shape), name='w')

    h = Parameter(initializer(
        Tensor(np.ones((num_layers * num_directions, batch_size, hidden_size)).astype(np.float32)),
        [num_layers * num_directions, batch_size, hidden_size]), name='h')

    c = Parameter(initializer(
        Tensor(np.ones((num_layers * num_directions, batch_size, hidden_size)).astype(np.float32)),
        [num_layers * num_directions, batch_size, hidden_size]), name='c')

    return h, c, w


class SentimentNet(nn.Cell):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 bidirectional, weight, labels, batch_size):
        super(SentimentNet, self).__init__()
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_size = batch_size

        self.embedding = nn.Embedding(vocab_size, embed_size, use_one_hot=False, embedding_table=Tensor(weight))
        self.embedding.embedding_table.requires_grad = False
        self.trans = P.Transpose()
        self.perm = (1, 0, 2)
        self.h, self.c, self.w = InitialLstmWeight(embed_size, num_hiddens, num_layers, bidirectional)
        self.encoder = P.LSTM(input_size=embed_size, hidden_size=self.num_hiddens,
                              num_layers=num_layers, has_bias=False,
                              bidirectional=self.bidirectional, dropout=0.0)
        self.concat = P.Concat(2)
        if self.bidirectional:
            self.decoder = nn.Dense(num_hiddens * 4, labels)
        else:
            self.decoder = nn.Dense(num_hiddens * 2, labels)

        self.slice1 = P.Slice()
        self.slice2 = P.Slice()
        self.reshape = P.Reshape()

        self.num_direction = 1
        if bidirectional:
            self.num_direction = 2

    def construct(self, inputs):
        embeddings = self.embedding(inputs)
        embeddings = self.trans(embeddings, self.perm)
        output, hidden = self.encoder(embeddings, self.h, self.c, self.w)

        output0 = self.slice1(output, (0, 0, 0), (1, 64, 200))
        output1 = self.slice2(output, (499, 0, 0), (1, 64, 200))
        encoding = self.concat((output0, output1))
        encoding = self.reshape(encoding, (self.batch_size, self.num_hiddens * self.num_direction * 2))
        outputs = self.decoder(encoding)
        return outputs


batch_size = 64


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_LSTM():
    num_epochs = 5
    embed_size = 100
    num_hiddens = 100
    num_layers = 2
    bidirectional = True
    labels = 2
    vocab_size = 252193
    max_len = 500

    weight = np.ones((vocab_size + 1, embed_size)).astype(np.float32)

    net = SentimentNet(vocab_size=(vocab_size + 1), embed_size=embed_size,
                       num_hiddens=num_hiddens, num_layers=num_layers,
                       bidirectional=bidirectional, weight=weight,
                       labels=labels, batch_size=batch_size)

    learning_rate = 0.1
    momentum = 0.9

    optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate, momentum)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_with_criterion = WithLossCell(net, criterion)
    train_network = TrainOneStepCell(net_with_criterion, optimizer)  # optimizer
    train_network.set_train()

    train_features = Tensor(np.ones([64, max_len]).astype(np.int32))
    train_labels = Tensor(np.ones([64,]).astype(np.int32)[0:64])
    losses = []
    for epoch in range(num_epochs):
        loss = train_network(train_features, train_labels)
        losses.append(loss)
        print("loss:", loss.asnumpy())
    assert (losses[-1].asnumpy() < 0.01)
