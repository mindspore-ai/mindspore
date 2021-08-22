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
"""Seq2Seq construction"""
import numpy as np
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops.operations as P
import mindspore.common.dtype as mstype
from src.gru import BidirectionGRU, GRU
from src.weight_init import dense_default_state

class Attention(nn.Cell):
    '''
    Attention model
    '''
    def __init__(self, config):
        super(Attention, self).__init__()
        self.text_len = config.max_length
        self.attn = nn.Dense(in_channels=config.hidden_size * 3,
                             out_channels=config.hidden_size).to_float(mstype.float16)
        self.fc = nn.Dense(config.hidden_size, 1, has_bias=False).to_float(mstype.float16)
        self.expandims = P.ExpandDims()
        self.tanh = P.Tanh()
        self.softmax = P.Softmax()
        self.tile = P.Tile()
        self.transpose = P.Transpose()
        self.concat = P.Concat(axis=2)
        self.squeeze = P.Squeeze(axis=2)
        self.cast = P.Cast()
    def construct(self, hidden, encoder_outputs):
        '''
        Attention construction

        Args:
            hidden(Tensor): hidden state
            encoder_outputs(Tensor): the output of encoder

        Returns:
            Tensor, attention output
        '''
        hidden = self.expandims(hidden, 1)
        hidden = self.tile(hidden, (1, self.text_len, 1))
        encoder_outputs = self.transpose(encoder_outputs, (1, 0, 2))
        out = self.concat((hidden, encoder_outputs))
        out = self.attn(out)
        energy = self.tanh(out)
        attention = self.fc(energy)
        attention = self.squeeze(attention)
        attention = self.cast(attention, mstype.float32)
        attention = self.softmax(attention)
        attention = self.cast(attention, mstype.float16)
        return attention

class Encoder(nn.Cell):
    '''
    Encoder model

    Args:
        config: config of network
    '''
    def __init__(self, config, is_training=True):
        super(Encoder, self).__init__()
        self.hidden_size = config.hidden_size
        self.vocab_size = config.src_vocab_size
        self.embedding_size = config.encoder_embedding_size
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.rnn = BidirectionGRU(config, is_training=is_training).to_float(mstype.float16)
        self.fc = nn.Dense(2*self.hidden_size, self.hidden_size).to_float(mstype.float16)
        self.shape = P.Shape()
        self.transpose = P.Transpose()
        self.p = P.Print()
        self.cast = P.Cast()
        self.text_len = config.max_length
        self.squeeze = P.Squeeze(axis=0)
        self.tanh = P.Tanh()

    def construct(self, src):
        '''
        Encoder construction

        Args:
            src(Tensor): source sentences

        Returns:
            output(Tensor): output of rnn
            hidden(Tensor): output hidden
        '''
        embedded = self.embedding(src)
        embedded = self.transpose(embedded, (1, 0, 2))
        embedded = self.cast(embedded, mstype.float16)
        output, hidden = self.rnn(embedded)
        hidden = self.fc(hidden)
        hidden = self.tanh(hidden)
        return output, hidden

class Decoder(nn.Cell):
    '''
    Decoder model

    Args:
        config: config of network
    '''
    def __init__(self, config, is_training=True):
        super(Decoder, self).__init__()
        self.hidden_size = config.hidden_size
        self.vocab_size = config.trg_vocab_size
        self.embedding_size = config.decoder_embedding_size
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.rnn = GRU(config, is_training=is_training).to_float(mstype.float16)
        self.text_len = config.max_length
        self.shape = P.Shape()
        self.transpose = P.Transpose()
        self.p = P.Print()
        self.cast = P.Cast()
        self.concat = P.Concat(axis=2)
        self.squeeze = P.Squeeze(axis=0)
        self.expandims = P.ExpandDims()
        self.log_softmax = P.LogSoftmax(axis=1)
        weight, bias = dense_default_state(self.embedding_size+self.hidden_size*3, self.vocab_size)
        self.fc = nn.Dense(self.embedding_size+self.hidden_size*3, self.vocab_size,
                           weight_init=weight, bias_init=bias).to_float(mstype.float16)
        self.attention = Attention(config)
        self.bmm = P.BatchMatMul()
        self.dropout = nn.Dropout(0.7)
        self.expandims = P.ExpandDims()
    def construct(self, inputs, hidden, encoder_outputs):
        '''
        Decoder construction

        Args:
            inputs(Tensor): decoder input
            hidden(Tensor): hidden state
            encoder_outputs(Tensor): encoder output

        Returns:
            pred_prob(Tensor): decoder predict probility
            hidden(Tensor): hidden state
        '''
        embedded = self.embedding(inputs)
        embedded = self.transpose(embedded, (1, 0, 2))
        embedded = self.cast(embedded, mstype.float16)
        attn = self.attention(hidden, encoder_outputs)
        attn = self.expandims(attn, 1)
        encoder_outputs = self.transpose(encoder_outputs, (1, 0, 2))
        weight = self.bmm(attn, encoder_outputs)
        weight = self.transpose(weight, (1, 0, 2))
        emd_con = self.concat((embedded, weight))
        output, hidden = self.rnn(emd_con)
        out = self.concat((embedded, output, weight))
        out = self.squeeze(out)
        hidden = self.squeeze(hidden)
        prediction = self.fc(out)
        prediction = self.dropout(prediction)
        prediction = self.cast(prediction, mstype.float32)
        prediction = self.cast(prediction, mstype.float32)
        pred_prob = self.log_softmax(prediction)
        pred_prob = self.expandims(pred_prob, 0)
        return pred_prob, hidden

class Seq2Seq(nn.Cell):
    '''
    Seq2Seq model

    Args:
        config: config of network
    '''
    def __init__(self, config, is_training=True):
        super(Seq2Seq, self).__init__()
        if is_training:
            self.batch_size = config.batch_size
        else:
            self.batch_size = config.eval_batch_size
        self.encoder = Encoder(config, is_training=is_training)
        self.decoder = Decoder(config, is_training=is_training)
        self.expandims = P.ExpandDims()
        self.dropout = nn.Dropout()
        self.shape = P.Shape()
        self.concat = P.Concat(axis=0)
        self.argmax = P.ArgMaxWithValue(axis=1, keep_dims=True)
        self.squeeze = P.Squeeze(axis=0)
        self.sos = Tensor(np.ones((self.batch_size, 1))*2, mstype.int32)
        self.select = P.Select()
        self.text_len = config.max_length

    def construct(self, encoder_inputs, decoder_inputs, teacher_force):
        '''
        Seq2Seq construction

        Args:
            encoder_inputs(Tensor): encoder input sentences
            decoder_inputs(Tensor): decoder input sentences
            teacher_force(Tensor): teacher force flag

        Returns:
            outputs(Tensor): total predict probility
        '''
        decoder_input = self.sos
        encoder_output, hidden = self.encoder(encoder_inputs)
        decoder_hidden = hidden
        decoder_outputs = ()
        for i in range(1, self.text_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_output)
            decoder_outputs += (decoder_output,)
            if self.training:
                decoder_input_force = decoder_inputs[::, i:i+1]
                decoder_input_top1, _ = self.argmax(self.squeeze(decoder_output))
                decoder_input = self.select(teacher_force, decoder_input_force, decoder_input_top1)
            else:
                decoder_input, _ = self.argmax(self.squeeze(decoder_output))
        outputs = self.concat(decoder_outputs)
        return outputs
