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
"""Embedding."""
import numpy as np

import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter

from src.weight_init import normal_weight


class EmbeddingLookup(nn.Cell):
    """
    A embeddings lookup table with a fixed dictionary and size.

    Args:
        batch_size (int): Batch size of input dataset.
        seq_length (int): Length of input tensor sequence.
        vocab_size (int): Size of the dictionary of embeddings.
        embedding_dim (int): The size of each embedding vector.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form.
        config: The config of networks.
        compute_type (:class:`mindspore.dtype`): Compute type.
    """

    def __init__(self,
                 batch_size,
                 seq_length,
                 vocab_size,
                 embedding_dim,
                 config=None,
                 use_one_hot_embeddings=True,
                 compute_type=mstype.float16):
        super(EmbeddingLookup, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.embedding_table = Parameter(normal_weight([vocab_size, embedding_dim], embedding_dim),
                                         name='embedding_table')
        self.embedding_table.parallel_optimizer = False
        self.shape_flat = (-1,)
        self.gather = P.GatherV2().shard(((1, 1), (config.dp,)))

        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.cast = P.Cast()

        self.less = P.Less().shard(((config.dp, 1), (config.dp, 1)))
        self.greaterequal = P.GreaterEqual().shard(((config.dp, 1), (config.dp, 1)))
        self.logicalor = P.LogicalOr().shard(((config.dp, 1), (config.dp, 1)))
        self.zero = Tensor(np.zeros((self.batch_size, self.seq_length), dtype=np.int32))
        self.zero_2 = Tensor(np.zeros((self.batch_size, self.seq_length, 1), dtype=np.int32))
        self.start = Tensor(np.zeros((self.batch_size, self.seq_length), dtype=np.int32))
        self.end = Tensor(np.array([[self.vocab_size]], dtype=np.int32))
        self.expanddim_first = P.ExpandDims().shard(((config.dp, 1),))
        self.expanddim = P.ExpandDims().shard(((config.dp, 1),))
        self.tile_in_mask = P.Tile().shard(((config.dp, 1, 1),))

        self.tile_2 = P.Tile().shard(((1, 1),))
        self.tile = P.Tile().shard(((config.dp, 1, 1),))
        self.select = P.Select().shard(
            ((config.dp, 1, config.mp), (config.dp, 1, config.mp), (config.dp, 1, config.mp)))
        self.mask_select = P.Select().shard(((config.dp, 1), (config.dp, 1), (config.dp, 1)))
        self.sub = P.Sub().shard(((config.dp, 1), (config.dp, 1)))
        self.get_dtype = P.DType()

    def construct(self, input_ids):
        """
        get embedding according to input_ids.
        """
        input_less = self.less(input_ids, self.start)
        ends = self.end
        ends = self.tile_2(ends, (self.batch_size, self.seq_length))
        input_greater = self.greaterequal(input_ids, ends)
        input_mask = self.logicalor(input_less, input_greater)
        masked_input = self.sub(input_ids, self.start)
        # [batchsize, seq_length]
        masked_input = self.mask_select(input_mask, self.zero, self.cast(masked_input, mstype.int32))

        input_shape = self.shape(masked_input)
        flat_ids = self.reshape(masked_input, self.shape_flat)

        flat_ides = self.cast(flat_ids, mstype.int32)
        output_for_reshape = self.gather(self.embedding_table, flat_ides, 0)

        out_shape = input_shape + (self.embedding_dim,)
        output = self.reshape(output_for_reshape, out_shape)
        # [batchsize, seq_length]
        input_masks = self.expanddim(input_mask, -1)
        input_mask_tile = self.tile_in_mask(self.cast(input_masks, mstype.int32), (1, 1, self.embedding_dim))
        zero_expand = self.zero_2
        zero_tiled = self.tile(zero_expand, (1, 1, self.embedding_dim))
        zero_tiled = self.cast(zero_tiled, self.get_dtype(output))
        output = self.select(self.cast(input_mask_tile, mstype.bool_), zero_tiled, output)
        output = self.cast(output, mstype.float32)
        return output, self.embedding_table


class EmbeddingPostprocessor(nn.Cell):
    """
    Positional embeddings.

    Args:
        max_seq_length (int): the length of input sequence.
        embedding_dim (int): The size of each embedding vector.
        config: The config of networks.
        use_one_hot_embeddings (bool): Whether use one-hot embedding.
        compute_type (:class:`mindspore.dtype`): Compute type. Default: mstype.float16.
     """

    def __init__(self,
                 max_seq_length,
                 embedding_dim,
                 config=None,
                 use_one_hot_embeddings=True,
                 compute_type=mstype.float16):
        super(EmbeddingPostprocessor, self).__init__()
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.compute_type = compute_type
        self.position_embedding_table = Parameter(normal_weight([max_seq_length, embedding_dim], embedding_dim),
                                                  name='position_embeddings')
        self.position_embedding_table.parallel_optimizer = False
        self.shape_flat = (-1,)
        self.gather = P.GatherV2().shard(((1, 1), (config.dp,)))

        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.cast = P.Cast()
        position_ids = np.expand_dims(np.arange(config.seq_length * 1), 0)
        self.position_ids = Tensor(np.tile(position_ids, (config.batch_size, 1)), dtype=mstype.int64)

    def construct(self, input_ids=None):
        r"""
        get embedding according to input_ids.
        """
        if input_ids is None:
            input_ids = self.position_ids

        input_shape = self.shape(input_ids)
        flat_ids = self.reshape(input_ids, self.shape_flat)

        output_for_reshape = self.gather(self.position_embedding_table, flat_ids, 0)

        out_shape = input_shape + (self.embedding_dim,)
        output = self.reshape(output_for_reshape, out_shape)
        return output
