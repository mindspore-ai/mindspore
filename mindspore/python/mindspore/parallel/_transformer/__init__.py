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
NOTE:
    Transformer Networks.
    This is an experimental interface that is subject to change or deletion.
"""
from __future__ import absolute_import

from mindspore.parallel._transformer.transformer import AttentionMask, VocabEmbedding, MultiHeadAttention, \
    FeedForward, TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer, \
    Transformer, TransformerOpParallelConfig, EmbeddingOpParallelConfig, TransformerRecomputeConfig
from mindspore.parallel._transformer.moe import MoEConfig
from mindspore.parallel._transformer.layers import FixedSparseAttention
from mindspore.parallel._transformer.loss import CrossEntropyLoss
from mindspore.parallel._transformer.op_parallel_config import OpParallelConfig

__all__ = []
__all__.extend(transformer.__all__)
__all__.extend(loss.__all__)
__all__.extend(op_parallel_config.__all__)
__all__.extend(layers.__all__)
__all__.extend(moe.__all__)
