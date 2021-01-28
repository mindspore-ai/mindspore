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
#" ============================================================================
"""
Seq2Seq_OCR model.

"""
import mindspore.nn as nn
import mindspore.ops.operations as P
import mindspore.common.dtype as mstype

from src.cnn import CNN
from src.gru import GRU
from src.lstm import LSTM
from src.weight_init import lstm_default_state


class BidirectionalLSTM(nn.Cell):
    """Bidirectional LSTM with a Dense layer

    Args:
        batch_size(int): batch size of input data
        input_size(int): Size of time sequence
        hidden_size(int): the hidden size of LSTM layers
        output_size(int): the output size of the dense layer
    """
    def __init__(self, batch_size, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True).to_float(mstype.float16)
        self.h, self.c = lstm_default_state(batch_size, hidden_size, bidirectional=True)
        self.embedding = nn.Dense(hidden_size * 2, output_size).to_float(mstype.float16)
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.cast = P.Cast()

    def construct(self, inputs):
        inputs = self.cast(inputs, mstype.float16)
        recurrent, _ = self.rnn(inputs, (self.h, self.c))
        T, b, h = self.shape(recurrent)
        t_rec = self.reshape(recurrent, (T * b, h))
        output = self.embedding(t_rec)
        output = self.reshape(output, (T, b, -1))
        return output


class AttnDecoderRNN(nn.Cell):
    """Attention Decoder Structure with a one-layer GRU

    Args:
        hidden_size(int): the hidden size
        output_size(int): the output size
        max_length(iht): max time step of the decoder
        dropout_p(float): dropout probability, default is 0.1
    """
    def __init__(self, hidden_size, output_size, max_length, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Dense(in_channels=self.hidden_size * 2, out_channels=self.max_length).to_float(mstype.float16)
        self.attn_combine = nn.Dense(in_channels=self.hidden_size * 2,
                                     out_channels=self.hidden_size).to_float(mstype.float16)
        self.dropout = nn.Dropout(keep_prob=1.0 - self.dropout_p)
        self.gru = GRU(hidden_size, hidden_size).to_float(mstype.float16)
        self.out = nn.Dense(in_channels=self.hidden_size, out_channels=self.output_size).to_float(mstype.float16)
        self.transpose = P.Transpose()
        self.concat = P.Concat(axis=2)
        self.concat1 = P.Concat(axis=1)
        self.softmax = P.Softmax(axis=1)
        self.relu = P.ReLU()
        self.log_softmax = P.LogSoftmax(axis=1)
        self.bmm = P.BatchMatMul()
        self.unsqueeze = P.ExpandDims()
        self.squeeze = P.Squeeze(1)
        self.squeeze1 = P.Squeeze(0)
        self.cast = P.Cast()

    def construct(self, inputs, hidden, encoder_outputs):
        embedded = self.embedding(inputs)
        embedded = self.transpose(embedded, (1, 0, 2))
        embedded = self.dropout(embedded)
        embedded = self.cast(embedded, mstype.float16)

        embedded_concat = self.concat((embedded, hidden))
        embedded_concat = self.squeeze1(embedded_concat)
        attn_weights = self.softmax(self.attn(embedded_concat))
        attn_weights = self.unsqueeze(attn_weights, 1)
        perm_encoder_outputs = self.transpose(encoder_outputs, (1, 0, 2))
        attn_applied = self.bmm(attn_weights, perm_encoder_outputs)
        attn_applied = self.squeeze(attn_applied)
        embedded_squeeze = self.squeeze1(embedded)

        output = self.concat1((embedded_squeeze, attn_applied))
        output = self.attn_combine(output)
        output = self.unsqueeze(output, 0)
        output = self.relu(output)

        gru_hidden = self.squeeze1(hidden)
        output, hidden, _, _, _, _ = self.gru(output, gru_hidden)
        output = self.squeeze1(output)
        output = self.log_softmax(self.out(output))

        return output, hidden, attn_weights


class Encoder(nn.Cell):
    """Encoder with a CNN and two BidirectionalLSTM layers

    Args:
        batch_size(int): batch size of input data
        conv_out_dim(int): the output dimension of the cnn layer
        hidden_size(int): the hidden size of LSTM layers
    """
    def __init__(self, batch_size, conv_out_dim, hidden_size):
        super(Encoder, self).__init__()
        self.cnn = CNN(int(conv_out_dim/4))
        self.lstm1 = BidirectionalLSTM(batch_size, conv_out_dim, hidden_size, hidden_size).to_float(mstype.float16)
        self.lstm2 = BidirectionalLSTM(batch_size, hidden_size, hidden_size, hidden_size).to_float(mstype.float16)
        self.transpose = P.Transpose()
        self.cast = P.Cast()
        self.split = P.Split(axis=3, output_num=4)
        self.concat = P.Concat(axis=1)

    def construct(self, inputs):
        inputs = self.cast(inputs, mstype.float32)
        (x1, x2, x3, x4) = self.split(inputs)
        conv1 = self.cnn(x1)
        conv2 = self.cnn(x2)
        conv3 = self.cnn(x3)
        conv4 = self.cnn(x4)
        conv = self.concat((conv1, conv2, conv3, conv4))
        conv = self.transpose(conv, (2, 0, 1))
        output = self.lstm1(conv)
        output = self.lstm2(output)
        return output


class Decoder(nn.Cell):
    """Decoder

     Args:
        hidden_size(int): the hidden size
        output_size(int): the output size
        max_length(iht): max time step of the decoder
        dropout_p(float): dropout probability, default is 0.1
    """
    def __init__(self, hidden_size, output_size, max_length, dropout_p=0.1):
        super(Decoder, self).__init__()
        self.decoder = AttnDecoderRNN(hidden_size, output_size, max_length, dropout_p)

    def construct(self, inputs, hidden, encoder_outputs):
        return self.decoder(inputs, hidden, encoder_outputs)
