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
"""Create attention block."""
import mindspore.common.dtype as mstype
from mindspore import nn

from .attention import BahdanauAttention


class RecurrentAttention(nn.Cell):
    """
    Constructor for the RecurrentAttention.

    Args:
        input_size: number of features in input tensor.
        context_size: number of features in output from encoder.
        hidden_size: internal hidden size.
        num_layers: number of layers in LSTM.
        dropout: probability of dropout (on input to LSTM layer).
        initializer_range: range for the uniform initializer.

    Returns:
        Tensor, shape (N, T, D).
    """

    def __init__(self,
                 rnn,
                 is_training=True,
                 input_size=1024,
                 context_size=1024,
                 hidden_size=1024,
                 num_layers=1,
                 dropout=0.2,
                 initializer_range=0.1):
        super(RecurrentAttention, self).__init__()
        self.dropout = nn.Dropout(keep_prob=1.0 - dropout)
        self.rnn = rnn
        self.attn = BahdanauAttention(is_training=is_training,
                                      query_size=hidden_size,
                                      key_size=hidden_size,
                                      num_units=hidden_size,
                                      normalize=True,
                                      initializer_range=initializer_range,
                                      compute_type=mstype.float16)

    def construct(self, decoder_embedding, context_key, attention_mask=None, rnn_init_state=None):
        # decoder_embedding: [t_q,N,D]
        # context: [t_k,N,D]
        # attention_mask: [N,t_k]
        # [t_q,N,D]
        decoder_embedding = self.dropout(decoder_embedding)
        rnn_outputs, rnn_state = self.rnn(decoder_embedding, rnn_init_state)
        # rnn_outputs:[t_q,b,D], attn_outputs:[t_q,b,D], scores:[b, t_q, t_k], rnn_state:tuple([2,b,D]).
        attn_outputs, scores = self.attn(query=rnn_outputs, keys=context_key, attention_mask=attention_mask)
        return rnn_outputs, attn_outputs, rnn_state, scores
