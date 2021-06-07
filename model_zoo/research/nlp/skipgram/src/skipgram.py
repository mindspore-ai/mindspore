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
"""
skipgram network
"""

import os
import numpy as np

import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import Uniform


class SkipGram(nn.Cell):
    """Skip gram model of word2vec.

    Attributes:
        vocab_size: Vocabulary size.
        emb_dimension: Embedding dimension.
        c_emb: Embedding for center word.
        n_emb: Embedding for neighbor word.
    """

    def __init__(self, vocab_size, emb_dimension):
        """Initialize model parameters.

        Apply for two embedding layers.
        Initialize layer weight.

        Args:
            vocab_size: Vocabulary size.
            emb_dimension: Embedding dimension.

        Returns:
            None
        """
        super(SkipGram, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dimension = emb_dimension
        self.c_emb = nn.Embedding(vocab_size, emb_dimension, embedding_table=Uniform(0.5/emb_dimension))
        self.n_emb = nn.Embedding(vocab_size, emb_dimension, embedding_table=Uniform(0))
        # Operators (stateless)
        self.mul = ops.Mul()
        self.sum = ops.ReduceSum(keep_dims=False)
        self.logsigmoid = nn.LogSigmoid()
        self.expand_dims = ops.ExpandDims()
        self.squeeze = ops.Squeeze()

        self.transpose = ops.Transpose()
        self.perm = (0, 2, 1)

        self.cast = ops.Cast()

    def construct(self, center_word, pos_word, neg_words):
        """Forward network construction.

        Args:
            center_word: center word ids.
            pos_word: positive word ids.
            neg_words: negative samples' word ids.

        Returns:
            loss.
        """
        emb_u = self.c_emb(center_word)  # (batch_size, emb_dim)
        emb_v = self.n_emb(pos_word)
        score = self.mul(emb_u, emb_v)  # (batch_size, emb_dim)
        score = self.sum(score, 1)   # (batch_size, )
        score = self.logsigmoid(score)

        neg_emb_v = self.n_emb(neg_words)  # (batch_size, neg_num, emb_dim)
        neg_emb_v = self.transpose(neg_emb_v, self.perm)  # (batch_size, emb_dim, neg_num)
        emb_u2 = self.expand_dims(emb_u, 2)  # (batch_size, emb_dim, 1)

        neg_score = self.mul(neg_emb_v, emb_u2) # (batch_size, emb_dim, neg_num)
        neg_score = self.transpose(neg_score, self.perm)  # (batch_size, neg_num, emb_dim)
        neg_score = self.sum(neg_score, 2) # (batch_size, neg_num)

        neg_score = self.logsigmoid(-1 * neg_score)
        neg_score = self.sum(neg_score, 1)  # (batch_size, )
        loss = self.cast(-(score + neg_score), mstype.float32)
        return loss

    def save_w2v_emb(self, dir_path, id2word):
        """Save word2vec embeddings to file.

        Args:
            id2word: map wid to word.
            filename: file name.
        Returns:
            None.
        """
        w2v_emb = dict()
        parameters = []
        for item in self.c_emb.get_parameters():
            parameters.append(item)
        emb_mat = parameters[0].asnumpy()

        for wid, emb in enumerate(emb_mat):
            word = id2word[wid]
            w2v_emb[word] = emb
        np.save(os.path.join(dir_path, 'w2v_emb.npy'), w2v_emb)
