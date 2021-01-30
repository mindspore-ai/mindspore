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
"""Embedding."""
import numpy as np
import mindspore.common.dtype as mstype
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter


class EmbeddingLookup(nn.Cell):
    """Embeddings lookup table with a fixed dictionary and size."""

    def __init__(self,
                 vocab_size,
                 embed_dim,
                 use_one_hot_embeddings=False):
        """
        Embeddings lookup table with a fixed dictionary and size.

        Args:
            vocab_size (int): Size of the dictionary of embeddings.
            embed_dim (int): The size of word embedding.
            use_one_hot_embeddings (bool): Whether use one-hot embedding. Default: False.
        """
        super(EmbeddingLookup, self).__init__()
        self.embedding_dim = embed_dim
        self.vocab_size = vocab_size
        self.use_one_hot_embeddings = use_one_hot_embeddings

        init_weight = np.random.normal(0, embed_dim ** -0.5, size=[vocab_size, embed_dim]).astype(np.float32)
        # 0 is Padding index, thus init it as 0.
        init_weight[0, :] = 0
        self.embedding_table = Parameter(Tensor(init_weight))
        self.expand = P.ExpandDims()
        self.gather = P.Gather()
        self.one_hot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.array_mul = P.MatMul()
        self.reshape = P.Reshape()
        self.get_shape = P.Shape()

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

        flat_ids = self.reshape(input_ids, (_batch_size * _max_len,))
        if self.use_one_hot_embeddings:
            one_hot_ids = self.one_hot(flat_ids, self.vocab_size, self.on_value, self.off_value)
            output_for_reshape = self.array_mul(
                one_hot_ids, self.embedding_table)
        else:
            output_for_reshape = self.gather(self.embedding_table, flat_ids, 0)

        output = self.reshape(output_for_reshape, (_batch_size, _max_len, self.embedding_dim))
        return output, self.embedding_table
