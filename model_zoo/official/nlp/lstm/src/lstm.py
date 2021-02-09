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

from mindspore import Tensor, nn, context, Parameter, ParameterTuple
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P
import mindspore.ops.functional as F
import mindspore.common.dtype as mstype

STACK_LSTM_DEVICE = ["CPU"]


# Initialize short-term memory (h) and long-term memory (c) to 0
def lstm_default_state(batch_size, hidden_size, num_layers, bidirectional):
    """init default input."""
    num_directions = 2 if bidirectional else 1
    h = Tensor(np.zeros((num_layers * num_directions, batch_size, hidden_size)).astype(np.float32))
    c = Tensor(np.zeros((num_layers * num_directions, batch_size, hidden_size)).astype(np.float32))
    return h, c


def stack_lstm_default_state(batch_size, hidden_size, num_layers, bidirectional):
    """init default input."""
    num_directions = 2 if bidirectional else 1

    h_list = c_list = []
    for _ in range(num_layers):
        h_list.append(Tensor(np.zeros((num_directions, batch_size, hidden_size)).astype(np.float32)))
        c_list.append(Tensor(np.zeros((num_directions, batch_size, hidden_size)).astype(np.float32)))
    h, c = tuple(h_list), tuple(c_list)
    return h, c

def stack_lstm_default_state_ascend(batch_size, hidden_size, num_layers, bidirectional):
    """init default input."""

    h_list = c_list = []
    for _ in range(num_layers):
        h_fw = Tensor(np.zeros((1, batch_size, hidden_size)).astype(np.float16))
        c_fw = Tensor(np.zeros((1, batch_size, hidden_size)).astype(np.float16))
        h_i = [h_fw]
        c_i = [c_fw]

        if bidirectional:
            h_bw = Tensor(np.zeros((1, batch_size, hidden_size)).astype(np.float16))
            c_bw = Tensor(np.zeros((1, batch_size, hidden_size)).astype(np.float16))
            h_i.append(h_bw)
            c_i.append(c_bw)

        h_list.append(h_i)
        c_list.append(c_i)

    h, c = tuple(h_list), tuple(c_list)
    return h, c


class StackLSTM(nn.Cell):
    """
    Stack multi-layers LSTM together.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 has_bias=True,
                 batch_first=False,
                 dropout=0.0,
                 bidirectional=False):
        super(StackLSTM, self).__init__()
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.transpose = P.Transpose()

        # direction number
        num_directions = 2 if bidirectional else 1

        # input_size list
        input_size_list = [input_size]
        for i in range(num_layers - 1):
            input_size_list.append(hidden_size * num_directions)

        # layers
        layers = []
        for i in range(num_layers):
            layers.append(nn.LSTMCell(input_size=input_size_list[i],
                                      hidden_size=hidden_size,
                                      has_bias=has_bias,
                                      batch_first=batch_first,
                                      bidirectional=bidirectional,
                                      dropout=dropout))

        # weights
        weights = []
        for i in range(num_layers):
            # weight size
            weight_size = (input_size_list[i] + hidden_size) * num_directions * hidden_size * 4
            if has_bias:
                bias_size = num_directions * hidden_size * 4
                weight_size = weight_size + bias_size

            # numpy weight
            stdv = 1 / math.sqrt(hidden_size)
            w_np = np.random.uniform(-stdv, stdv, (weight_size, 1, 1)).astype(np.float32)

            # lstm weight
            weights.append(Parameter(initializer(Tensor(w_np), w_np.shape), name="weight" + str(i)))

        #
        self.lstms = layers
        self.weight = ParameterTuple(tuple(weights))

    def construct(self, x, hx):
        """construct"""
        if self.batch_first:
            x = self.transpose(x, (1, 0, 2))
        # stack lstm
        h, c = hx
        hn = cn = None
        for i in range(self.num_layers):
            x, hn, cn, _, _ = self.lstms[i](x, h[i], c[i], self.weight[i])
        if self.batch_first:
            x = self.transpose(x, (1, 0, 2))
        return x, (hn, cn)

class LSTM_Ascend(nn.Cell):
    """ LSTM in Ascend. """

    def __init__(self, bidirectional=False):
        super(LSTM_Ascend, self).__init__()
        self.bidirectional = bidirectional
        self.dynamic_rnn = P.DynamicRNN(forget_bias=0.0)
        self.reverseV2 = P.ReverseV2(axis=[0])
        self.concat = P.Concat(2)

    def construct(self, x, h, c, w_f, b_f, w_b=None, b_b=None):
        """construct"""
        x = F.cast(x, mstype.float16)
        if self.bidirectional:
            y1, h1, c1, _, _, _, _, _ = self.dynamic_rnn(x, w_f, b_f, None, h[0], c[0])
            r_x = self.reverseV2(x)
            y2, h2, c2, _, _, _, _, _ = self.dynamic_rnn(r_x, w_b, b_b, None, h[1], c[1])
            y2 = self.reverseV2(y2)

            output = self.concat((y1, y2))
            hn = self.concat((h1, h2))
            cn = self.concat((c1, c2))
            return output, (hn, cn)

        y1, h1, c1, _, _, _, _, _ = self.dynamic_rnn(x, w_f, b_f, None, h[0], c[0])
        return y1, (h1, c1)

class StackLSTMAscend(nn.Cell):
    """ Stack multi-layers LSTM together. """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 has_bias=True,
                 batch_first=False,
                 dropout=0.0,
                 bidirectional=False):
        super(StackLSTMAscend, self).__init__()
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.transpose = P.Transpose()

        # input_size list
        input_size_list = [input_size]
        for i in range(num_layers - 1):
            input_size_list.append(hidden_size * 2)

        #weights, bias and layers init
        weights_fw = []
        weights_bw = []
        bias_fw = []
        bias_bw = []

        stdv = 1 / math.sqrt(hidden_size)
        for i in range(num_layers):
            # forward weight init
            w_np_fw = np.random.uniform(-stdv,
                                        stdv,
                                        (input_size_list[i] + hidden_size, hidden_size * 4)).astype(np.float32)
            w_fw = Parameter(initializer(Tensor(w_np_fw), w_np_fw.shape), name="w_fw_layer" + str(i))
            weights_fw.append(w_fw)
            # forward bias init
            if has_bias:
                b_fw = np.random.uniform(-stdv, stdv, (hidden_size * 4)).astype(np.float32)
                b_fw = Parameter(initializer(Tensor(b_fw), b_fw.shape), name="b_fw_layer" + str(i))
            else:
                b_fw = np.zeros((hidden_size * 4)).astype(np.float32)
                b_fw = Parameter(initializer(Tensor(b_fw), b_fw.shape), name="b_fw_layer" + str(i))
            bias_fw.append(b_fw)

            if bidirectional:
                # backward weight init
                w_np_bw = np.random.uniform(-stdv,
                                            stdv,
                                            (input_size_list[i] + hidden_size, hidden_size * 4)).astype(np.float32)
                w_bw = Parameter(initializer(Tensor(w_np_bw), w_np_bw.shape), name="w_bw_layer" + str(i))
                weights_bw.append(w_bw)

                # backward bias init
                if has_bias:
                    b_bw = np.random.uniform(-stdv, stdv, (hidden_size * 4)).astype(np.float32)
                    b_bw = Parameter(initializer(Tensor(b_bw), b_bw.shape), name="b_bw_layer" + str(i))
                else:
                    b_bw = np.zeros((hidden_size * 4)).astype(np.float32)
                    b_bw = Parameter(initializer(Tensor(b_bw), b_bw.shape), name="b_bw_layer" + str(i))
                bias_bw.append(b_bw)

        # layer init
        self.lstm = LSTM_Ascend(bidirectional=bidirectional).to_float(mstype.float16)

        self.weight_fw = ParameterTuple(tuple(weights_fw))
        self.weight_bw = ParameterTuple(tuple(weights_bw))
        self.bias_fw = ParameterTuple(tuple(bias_fw))
        self.bias_bw = ParameterTuple(tuple(bias_bw))

    def construct(self, x, hx):
        """construct"""
        x = F.cast(x, mstype.float16)
        if self.batch_first:
            x = self.transpose(x, (1, 0, 2))
        # stack lstm
        h, c = hx
        hn = cn = None
        for i in range(self.num_layers):
            if self.bidirectional:
                x, (hn, cn) = self.lstm(x,
                                        h[i],
                                        c[i],
                                        self.weight_fw[i],
                                        self.bias_fw[i],
                                        self.weight_bw[i],
                                        self.bias_bw[i])
            else:
                x, (hn, cn) = self.lstm(x, h[i], c[i], self.weight_fw[i], self.bias_fw[i])
        if self.batch_first:
            x = self.transpose(x, (1, 0, 2))
        x = F.cast(x, mstype.float32)
        hn = F.cast(x, mstype.float32)
        cn = F.cast(x, mstype.float32)
        return x, (hn, cn)

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

        if context.get_context("device_target") in STACK_LSTM_DEVICE:
            # stack lstm by user
            self.encoder = StackLSTM(input_size=embed_size,
                                     hidden_size=num_hiddens,
                                     num_layers=num_layers,
                                     has_bias=True,
                                     bidirectional=bidirectional,
                                     dropout=0.0)
            self.h, self.c = stack_lstm_default_state(batch_size, num_hiddens, num_layers, bidirectional)
        elif context.get_context("device_target") == "GPU":
            # standard lstm
            self.encoder = nn.LSTM(input_size=embed_size,
                                   hidden_size=num_hiddens,
                                   num_layers=num_layers,
                                   has_bias=True,
                                   bidirectional=bidirectional,
                                   dropout=0.0)
            self.h, self.c = lstm_default_state(batch_size, num_hiddens, num_layers, bidirectional)
        else:
            self.encoder = StackLSTMAscend(input_size=embed_size,
                                           hidden_size=num_hiddens,
                                           num_layers=num_layers,
                                           has_bias=True,
                                           bidirectional=bidirectional)
            self.h, self.c = stack_lstm_default_state_ascend(batch_size, num_hiddens, num_layers, bidirectional)

        self.concat = P.Concat(1)
        self.squeeze = P.Squeeze(axis=0)
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
        encoding = self.concat((self.squeeze(output[0:1:1]), self.squeeze(output[499:500:1])))
        outputs = self.decoder(encoding)
        return outputs
