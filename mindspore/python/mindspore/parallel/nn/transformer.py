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
Note:
    Transformer Networks. This is interface that is subject to change or deletion.
"""
from __future__ import absolute_import
from mindspore.nn.transformer.transformer import AttentionMask, VocabEmbedding, MultiHeadAttention, FeedForward, \
    TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer, Transformer, \
    TransformerOpParallelConfig, \
    EmbeddingOpParallelConfig, TransformerRecomputeConfig, \
    default_transformer_config, default_embedding_parallel_config, default_dpmp_config, default_moe_config, \
    default_transformer_recompute_config

__all__ = ["AttentionMask", "VocabEmbedding", "MultiHeadAttention", "FeedForward", "TransformerEncoder",
           "TransformerDecoder", "TransformerEncoderLayer", "TransformerDecoderLayer", "Transformer",
           "TransformerOpParallelConfig", "EmbeddingOpParallelConfig", "TransformerRecomputeConfig",
           "default_transformer_config", "default_embedding_parallel_config", "default_dpmp_config",
           "default_moe_config", "default_transformer_recompute_config"]
