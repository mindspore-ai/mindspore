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
"""Positional Embedding."""
import numpy as np
from mindspore import nn
from mindspore import Tensor
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P


def position_encoding(length, depth,
                      min_timescale=1,
                      max_timescale=1e4):
    """
    Create Tensor of sinusoids of different frequencies.

    Args:
        length (int): Length of the Tensor to create, i.e. Number of steps.
        depth (int): Dimensions of embedding.
        min_timescale (float): Minimum time scale.
        max_timescale (float): Maximum time scale.

    Returns:
        Tensor of shape (T, D)
    """
    depth = depth // 2
    positions = np.arange(length, dtype=np.float32)
    log_timescale_increment = (np.log(max_timescale / min_timescale) / (depth - 1))
    inv_timescales = min_timescale * np.exp(
        np.arange(depth, dtype=np.float32) * -log_timescale_increment)
    scaled_time = np.expand_dims(positions, 1) * np.expand_dims(inv_timescales, 0)
    # instead of using SIN and COS interleaved
    # it's  the same to first use SIN then COS
    # as they are applied to the same position
    x = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    return x


class PositionalEmbedding(nn.Cell):
    """
    Add positional info to word embeddings.

    Args:
        embedding_size (int): Size of word embedding.
        max_position_embeddings (int): Maximum step in this model.

    Returns:
        Tensor, shape of (N, T, D).
    """

    def __init__(self,
                 embedding_size,
                 max_position_embeddings=512):
        super(PositionalEmbedding, self).__init__()
        self.add = P.Add()
        self.expand_dims = P.ExpandDims()
        self.position_embedding_table = Tensor(
            position_encoding(max_position_embeddings, embedding_size),
            mstype.float32
        )
        self.gather = P.Gather()
        self.get_shape = P.Shape()

    def construct(self, word_embeddings):
        input_shape = self.get_shape(word_embeddings)
        input_len = input_shape[1]
        position_embeddings = self.position_embedding_table[0:input_len:1, ::]
        position_embeddings = self.expand_dims(position_embeddings, 0)
        output = self.add(word_embeddings, position_embeddings)
        return output
