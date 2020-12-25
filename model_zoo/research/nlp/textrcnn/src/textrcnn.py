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
"""model textrcnn"""
import numpy as np

import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.parameter import Parameter
from mindspore import Tensor
from mindspore.common import dtype as mstype


class textrcnn(nn.Cell):
    """class textrcnn"""

    def __init__(self, weight, vocab_size, cell, batch_size):
        super(textrcnn, self).__init__()
        self.num_hiddens = 512
        self.embed_size = 300
        self.num_classes = 2
        self.batch_size = batch_size
        k = (1 / self.num_hiddens) ** 0.5

        self.embedding = nn.Embedding(vocab_size, self.embed_size, embedding_table=weight)
        self.embedding.embedding_table.requires_grad = False
        self.cell = cell

        self.cast = P.Cast()

        self.h1 = Tensor(np.zeros(shape=(self.batch_size, self.num_hiddens)).astype(np.float16))
        self.c1 = Tensor(np.zeros(shape=(self.batch_size, self.num_hiddens)).astype(np.float16))

        if cell == "lstm":
            self.lstm = P.DynamicRNN(forget_bias=0.0)
            self.w1_fw = Parameter(
                np.random.uniform(-k, k, (self.embed_size + self.num_hiddens, 4 * self.num_hiddens)).astype(
                    np.float16), name="w1_fw")
            self.b1_fw = Parameter(np.random.uniform(-k, k, (4 * self.num_hiddens)).astype(np.float16),
                                   name="b1_fw")
            self.w1_bw = Parameter(
                np.random.uniform(-k, k, (self.embed_size + self.num_hiddens, 4 * self.num_hiddens)).astype(
                    np.float16), name="w1_bw")
            self.b1_bw = Parameter(np.random.uniform(-k, k, (4 * self.num_hiddens)).astype(np.float16),
                                   name="b1_bw")
            self.h1 = Tensor(np.zeros(shape=(1, self.batch_size, self.num_hiddens)).astype(np.float16))
            self.c1 = Tensor(np.zeros(shape=(1, self.batch_size, self.num_hiddens)).astype(np.float16))

        if cell == "vanilla":
            self.rnnW_fw = nn.Dense(self.num_hiddens, self.num_hiddens)
            self.rnnU_fw = nn.Dense(self.embed_size, self.num_hiddens)
            self.rnnW_bw = nn.Dense(self.num_hiddens, self.num_hiddens)
            self.rnnU_bw = nn.Dense(self.embed_size, self.num_hiddens)

        if cell == "gru":
            self.rnnWr_fw = nn.Dense(self.num_hiddens + self.embed_size, self.num_hiddens)
            self.rnnWz_fw = nn.Dense(self.num_hiddens + self.embed_size, self.num_hiddens)
            self.rnnWh_fw = nn.Dense(self.num_hiddens + self.embed_size, self.num_hiddens)
            self.rnnWr_bw = nn.Dense(self.num_hiddens + self.embed_size, self.num_hiddens)
            self.rnnWz_bw = nn.Dense(self.num_hiddens + self.embed_size, self.num_hiddens)
            self.rnnWh_bw = nn.Dense(self.num_hiddens + self.embed_size, self.num_hiddens)
            self.ones = Tensor(np.ones(shape=(self.batch_size, self.num_hiddens)).astype(np.float16))
            self.rnnWr_fw.to_float(mstype.float16)
            self.rnnWz_fw.to_float(mstype.float16)
            self.rnnWh_fw.to_float(mstype.float16)
            self.rnnWr_bw.to_float(mstype.float16)
            self.rnnWz_bw.to_float(mstype.float16)
            self.rnnWh_bw.to_float(mstype.float16)

        self.transpose = P.Transpose()
        self.reduce_max = P.ReduceMax()
        self.expand_dims = P.ExpandDims()
        self.concat = P.Concat()

        self.reshape = P.Reshape()
        self.left_pad_tensor = Tensor(np.zeros((1, self.batch_size, self.num_hiddens)).astype(np.float16))
        self.right_pad_tensor = Tensor(np.zeros((1, self.batch_size, self.num_hiddens)).astype(np.float16))
        self.output_dense = nn.Dense(self.num_hiddens * 1, 2)
        self.concat0 = P.Concat(0)
        self.concat2 = P.Concat(2)
        self.concat1 = P.Concat(1)
        self.text_rep_dense = nn.Dense(2 * self.num_hiddens + self.embed_size, self.num_hiddens)
        self.mydense = nn.Dense(self.num_hiddens, 2)
        self.drop_out = nn.Dropout(keep_prob=0.7)
        self.tanh = P.Tanh()
        self.sigmoid = P.Sigmoid()
        self.slice = P.Slice()
        self.text_rep_dense.to_float(mstype.float16)
        self.mydense.to_float(mstype.float16)
        self.output_dense.to_float(mstype.float16)

    def construct(self, x):
        """class construction"""
        # x: bs, sl
        output_fw = x
        output_bw = x

        if self.cell == "vanilla":
            x = self.embedding(x)  # bs, sl, emb_size
            x = self.cast(x, mstype.float16)
            x = self.transpose(x, (1, 0, 2))  # sl, bs, emb_size
            x = self.drop_out(x)  # sl,bs, emb_size

            h1_fw = self.cast(self.h1, mstype.float16)  # bs, num_hidden
            h1_fw = self.tanh(self.rnnW_fw(h1_fw) + self.rnnU_fw(x[0, :, :]))  # bs, num_hidden
            output_fw = self.expand_dims(h1_fw, 0)  # 1, bs, num_hidden

            for i in range(1, F.shape(x)[0]):
                h1_fw = self.tanh(self.rnnW_fw(h1_fw) + self.rnnU_fw(x[i, :, :]))  # 1, bs, num_hidden
                h1_after_expand_fw = self.expand_dims(h1_fw, 0)
                output_fw = self.concat((output_fw, h1_after_expand_fw))  # 2/3/4.., bs, num_hidden
            output_fw = self.cast(output_fw, mstype.float16)  # sl, bs, num_hidden

            h1_bw = self.cast(self.h1, mstype.float16)  # bs, num_hidden
            h1_bw = self.tanh(self.rnnW_bw(h1_bw) + self.rnnU_bw(x[F.shape(x)[0] - 1, :, :]))  # bs, num_hidden
            output_bw = self.expand_dims(h1_bw, 0)  # 1, bs, num_hidden

            for i in range(F.shape(x)[0] - 2, -1, -1):
                h1_bw = self.tanh(self.rnnW_bw(h1_bw) + self.rnnU_bw(x[i, :, :]))  # 1, bs, num_hidden
                h1_after_expand_bw = self.expand_dims(h1_bw, 0)
                output_bw = self.concat((h1_after_expand_bw, output_bw))  # 2/3/4.., bs, num_hidden
            output_bw = self.cast(output_bw, mstype.float16)  # sl, bs, num_hidden

        if self.cell == "gru":
            x = self.embedding(x)  # bs, sl, emb_size
            x = self.cast(x, mstype.float16)
            x = self.transpose(x, (1, 0, 2))  # sl, bs, emb_size
            x = self.drop_out(x)  # sl,bs, emb_size

            h_fw = self.cast(self.h1, mstype.float16)

            h_x_fw = self.concat1((h_fw, x[0, :, :]))
            r_fw = self.sigmoid(self.rnnWr_fw(h_x_fw))
            z_fw = self.sigmoid(self.rnnWz_fw(h_x_fw))
            h_tilde_fw = self.tanh(self.rnnWh_fw(self.concat1((r_fw * h_fw, x[0, :, :]))))
            h_fw = (self.ones - z_fw) * h_fw + z_fw * h_tilde_fw
            output_fw = self.expand_dims(h_fw, 0)

            for i in range(1, F.shape(x)[0]):
                h_x_fw = self.concat1((h_fw, x[i, :, :]))
                r_fw = self.sigmoid(self.rnnWr_fw(h_x_fw))
                z_fw = self.sigmoid(self.rnnWz_fw(h_x_fw))
                h_tilde_fw = self.tanh(self.rnnWh_fw(self.concat1((r_fw * h_fw, x[i, :, :]))))
                h_fw = (self.ones - z_fw) * h_fw + z_fw * h_tilde_fw
                h_after_expand_fw = self.expand_dims(h_fw, 0)
                output_fw = self.concat((output_fw, h_after_expand_fw))
            output_fw = self.cast(output_fw, mstype.float16)

            h_bw = self.cast(self.h1, mstype.float16)  # bs, num_hidden

            h_x_bw = self.concat1((h_bw, x[F.shape(x)[0] - 1, :, :]))
            r_bw = self.sigmoid(self.rnnWr_bw(h_x_bw))
            z_bw = self.sigmoid(self.rnnWz_bw(h_x_bw))
            h_tilde_bw = self.tanh(self.rnnWh_bw(self.concat1((r_bw * h_bw, x[F.shape(x)[0] - 1, :, :]))))
            h_bw = (self.ones - z_bw) * h_bw + z_bw * h_tilde_bw
            output_bw = self.expand_dims(h_bw, 0)
            for i in range(F.shape(x)[0] - 2, -1, -1):
                h_x_bw = self.concat1((h_bw, x[i, :, :]))
                r_bw = self.sigmoid(self.rnnWr_bw(h_x_bw))
                z_bw = self.sigmoid(self.rnnWz_bw(h_x_bw))
                h_tilde_bw = self.tanh(self.rnnWh_bw(self.concat1((r_bw * h_bw, x[i, :, :]))))
                h_bw = (self.ones - z_bw) * h_bw + z_bw * h_tilde_bw
                h_after_expand_bw = self.expand_dims(h_bw, 0)
                output_bw = self.concat((h_after_expand_bw, output_bw))
            output_bw = self.cast(output_bw, mstype.float16)
        if self.cell == 'lstm':
            x = self.embedding(x)  # bs, sl, emb_size
            x = self.cast(x, mstype.float16)
            x = self.transpose(x, (1, 0, 2))  # sl, bs, emb_size
            x = self.drop_out(x)  # sl,bs, emb_size

            h1_fw_init = self.h1  # bs, num_hidden
            c1_fw_init = self.c1  # bs, num_hidden

            _, output_fw, _, _, _, _, _, _ = self.lstm(x, self.w1_fw, self.b1_fw, None, h1_fw_init, c1_fw_init)
            output_fw = self.cast(output_fw, mstype.float16)  # sl, bs, num_hidden

            h1_bw_init = self.h1  # bs, num_hidden
            c1_bw_init = self.c1  # bs, num_hidden
            _, output_bw, _, _, _, _, _, _ = self.lstm(x, self.w1_bw, self.b1_bw, None, h1_bw_init, c1_bw_init)
            output_bw = self.cast(output_bw, mstype.float16)  # sl, bs, hidden

        c_left = self.concat0((self.left_pad_tensor, output_fw[:F.shape(x)[0] - 1]))  # sl, bs, num_hidden
        c_right = self.concat0((output_bw[1:], self.right_pad_tensor))  # sl, bs, num_hidden
        output = self.concat2((c_left, self.cast(x, mstype.float16), c_right))  # sl, bs, 2*num_hidden+emb_size
        output = self.cast(output, mstype.float16)

        output_flat = self.reshape(output, (F.shape(x)[0] * self.batch_size, 2 * self.num_hiddens + self.embed_size))
        output_dense = self.text_rep_dense(output_flat)  # sl*bs, num_hidden
        output_dense = self.tanh(output_dense)  # sl*bs, num_hidden
        output = self.reshape(output_dense, (F.shape(x)[0], self.batch_size, self.num_hiddens))  # sl, bs, num_hidden
        output = self.reduce_max(output, 0)  # bs, num_hidden
        outputs = self.cast(self.mydense(output), mstype.float16)  # bs, num_classes
        return outputs
