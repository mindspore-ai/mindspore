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
"""Word embedding for gnmt."""
import numpy as np

import mindspore.common.dtype as mstype
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter


class EmbeddingLookup(nn.Cell):
    """
    Embeddings lookup table with a fixed dictionary and size.

    Args:
        is_training (bool): Whether to train.
        vocab_size (int): Size of the dictionary of embeddings.
        embed_dim (int): The size of word embedding.
        initializer_range (int): The initialize range of parameters.
        use_one_hot_embeddings (bool): Whether use one-hot embedding. Default: False.
    """

    def __init__(self,
                 is_training,
                 vocab_size,
                 embed_dim,
                 initializer_range=0.1,
                 use_one_hot_embeddings=False):

        super(EmbeddingLookup, self).__init__()
        self.is_training = is_training
        self.embedding_dim = embed_dim
        self.vocab_size = vocab_size
        self.use_one_hot_embeddings = use_one_hot_embeddings

        init_weight = np.random.normal(-initializer_range, initializer_range, size=[vocab_size, embed_dim])
        self.embedding_table = Parameter(Tensor(init_weight, mstype.float32))
        self.expand = P.ExpandDims()
        self.gather = P.Gather()
        self.one_hot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.array_mul = P.MatMul()
        self.reshape = P.Reshape()
        self.get_shape = P.Shape()
        self.cast = P.Cast()

    def construct(self, input_ids):
        """
        Construct network.

        Args:
            input_ids (Tensor): A batch of sentences with shape (N, T).

        Returns:
            Tensor, word embeddings with shape (N, T, D)
        """
        _shape = self.get_shape(input_ids)  # (N, T).
        _batch_size = _shape[0]
        _max_len = _shape[1]
        if self.is_training:
            embedding_table = self.cast(self.embedding_table, mstype.float16)
        else:
            embedding_table = self.embedding_table

        flat_ids = self.reshape(input_ids, (_batch_size * _max_len,))
        if self.use_one_hot_embeddings:
            one_hot_ids = self.one_hot(flat_ids, self.vocab_size, self.on_value, self.off_value)
            if self.is_training:
                one_hot_ids = self.cast(one_hot_ids, mstype.float16)
            output_for_reshape = self.array_mul(
                one_hot_ids, embedding_table)
        else:
            output_for_reshape = self.gather(embedding_table, flat_ids, 0)

        output = self.reshape(output_for_reshape, (_batch_size, _max_len, self.embedding_dim))
        if self.is_training:
            output = self.cast(output, mstype.float32)
            embedding_table = self.cast(embedding_table, mstype.float32)
        return output, embedding_table
