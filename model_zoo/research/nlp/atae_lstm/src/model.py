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
"""AttentionLSTM Model"""
import numpy as np

import mindspore
from mindspore import nn, Parameter, Tensor
from mindspore import ops as P
from mindspore import ParameterTuple
from mindspore.common import dtype as mstype

from .model_utils.rnns import LSTM


class AttentionLstm(nn.Cell):
    """Model structure"""
    def __init__(self, config, weight, is_train=True):
        super(AttentionLstm, self).__init__()

        self.dim_word = config.dim_word
        self.dimh = config.dim_hidden
        self.dim_aspect = config.dim_aspect
        self.vocab_size = config.vocab_size
        self.grained = config.grained
        self.aspect_num = config.aspect_num
        self.embedding_table = weight
        self.is_train = is_train
        self.dropout_prob = config.dropout_prob

        self.dropout = nn.Dropout(keep_prob=1 - self.dropout_prob)
        self.mask = Tensor(np.random.uniform(size=(1, self.dimh)) > self.dropout_prob)

        self.embedding_word = nn.Embedding(vocab_size=self.vocab_size,
                                           embedding_size=self.dim_word,
                                           embedding_table=self.embedding_table)

        self.embedding_aspect = nn.Embedding(vocab_size=self.aspect_num,
                                             embedding_size=self.dim_aspect)

        self.dim_lstm_para = self.dim_word + self.dim_aspect

        self.init_state = Tensor(np.zeros((1, 1, self.dimh)).astype(np.float16))

        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.matmul = P.MatMul()
        self.expand = P.ExpandDims()
        self.cast = P.Cast()
        self.tanh = P.Tanh()
        self.tile = P.Tile()
        self.softmax_0 = P.Softmax(axis=0)
        self.softmax_1 = P.Softmax(axis=1)
        self.concat_0 = P.Concat(axis=0)
        self.concat_1 = P.Concat(axis=1)
        self.concat_2 = P.Concat(axis=2)
        self.squeeze_0 = P.Squeeze(axis=0)
        self.transpose = P.Transpose()
        self.transpose_orders = (1, 0, 2)
        self.trans_matrix = (1, 0)

        u = lambda x: 1 / np.sqrt(x)
        e = u(self.dimh)
        self.w = Parameter(Tensor(np.zeros((self.dimh + self.dim_aspect, 1)).astype(np.float16)))
        self.Ws = Parameter(Tensor(np.random.uniform(-e, e, (self.dimh, self.grained)).astype(np.float16)))
        self.bs = Parameter(Tensor(np.zeros((1, self.grained)).astype(np.float16)))
        self.Wh = Parameter(Tensor(np.random.uniform(-e, e, (self.dimh, self.dimh)).astype(np.float16)))
        self.Wv = Parameter(Tensor(np.random.uniform(-e, e, (self.dim_aspect, self.dim_aspect)).astype(np.float16)))
        self.Wp = Parameter(Tensor(np.random.uniform(-e, e, (self.dimh, self.dimh)).astype(np.float16)))
        self.Wx = Parameter(Tensor(np.random.uniform(-e, e, (self.dimh, self.dimh)).astype(np.float16)))

        self.lstm = LSTM(self.dim_lstm_para, self.dimh, batch_first=True, has_bias=True)

        self.params = ParameterTuple((self.Wv, self.Wh, self.Ws, self.bs, self.w, self.Wp, self.Wx))

    def construct(self, x, x_len, aspect):
        """
        shape:
            x: (1, N)  int32
            aspct: (1) int32
            x_len: (1) int32
        """
        # x: x shape: (1, N, 300)  aspect: (1, 100)   sloution: (1,3)
        x = self.embedding_word(x)
        aspect = self.embedding_aspect(aspect)

        x = self.cast(x, mstype.float16)
        aspect = self.cast(aspect, mstype.float16)

        # aspect: (1, N, 100)
        aspect = self.expand(aspect, 0)
        aspect_vector = self.tile(aspect, (1, x.shape[1], 1))

        lstm_input = self.concat_2((x, aspect_vector))

        h_0 = self.init_state
        c_0 = self.init_state

        output, (h_n, _) = self.lstm(lstm_input, (h_0, c_0), x_len)

        # H: [N, 300]
        H = self.squeeze_0(output)
        # h_n [1, 1, 300] - > [1, 300]
        h_n = self.reshape(h_n, (1, 300))

        # Wh_H.size = (N, 300)
        H_Wh = self.matmul(H, self.Wh)
        # a_Wv.size = (N, 100)
        a_Wv = self.matmul(self.reshape(aspect_vector, (-1, self.dim_aspect)), self.Wv)
        # M.size = (N, 400)
        H_Wh = self.cast(H_Wh, mindspore.float32)
        a_Wv = self.cast(a_Wv, mindspore.float32)
        M = self.tanh(self.concat_1((H_Wh, a_Wv)))
        # tmp.size = (N, 1)
        M = self.cast(M, mindspore.float16)
        tmp = self.matmul(M, self.w)
        tmp = self.reshape(tmp, (1, -1))
        # alpha.size = (1, N)
        tmp = self.cast(tmp, mindspore.float32)
        alpha = self.softmax_1(tmp)
        # r.size = (1, 300)
        alpha = self.cast(alpha, mindspore.float16)
        r = self.matmul(alpha, H)

        r_Wp = self.matmul(r, self.Wp)
        h_Wx = self.matmul(h_n, self.Wx)
        # h_star.size = (1, 300)
        r_Wp = self.cast(r_Wp, mindspore.float32)
        h_Wx = self.cast(h_Wx, mindspore.float32)
        h_star = self.tanh(r_Wp + h_Wx)
        # dropout
        h_star = h_star * self.mask / (1 - self.dropout_prob)

        # y.size = (1, self.grained)
        h_star = self.cast(h_star, mindspore.float16)
        y_hat = self.matmul(h_star, self.Ws) + self.bs
        y_hat = self.cast(y_hat, mindspore.float32)
        y = self.softmax_1(y_hat)
        # y.size = (1, self.grained)
        return y
