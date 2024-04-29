# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Paged Attention Manager for inference."""
import math
from typing import Optional
import numpy as np

import mindspore.common.dtype as mstype
from mindspore import nn, Parameter
from mindspore import ops as P
from mindspore.common.tensor import Tensor


class PagedAttentionMgr(nn.Cell):
    """Paged Attention Manager."""

    def __init__(self,
                 n_heads,
                 head_dim,
                 hidden_size,
                 n_kv_heads: Optional[int] = None,
                 block_size=16,
                 num_blocks=256,
                 compute_dtype=mstype.float16,
                 ):
        super().__init__()
        self.is_first_iteration = True
        self.n_head = n_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.n_kv_heads = n_kv_heads
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.dtype = compute_dtype

        self.scale_value = 1 / math.sqrt(self.head_dim)

        self.reshape = P.Reshape().add_prim_attr("skip_redistribution", True)
        self.transpose = P.Transpose()

        kv_shape = (num_blocks, self.block_size, self.n_kv_heads, self.head_dim)
        self.key_cache = Parameter(Tensor(np.zeros(kv_shape), compute_dtype), name="key_cache", requires_grad=False)
        self.value_cache = Parameter(Tensor(np.zeros(kv_shape), compute_dtype), name="value_cache", requires_grad=False)
        # if is_paged_attention_v2():
        self.reshape_and_cache = P.auto_generate.ReshapeAndCache()
        self.paged_attention = P.auto_generate.PagedAttention(self.n_head, self.scale_value,
                                                              self.n_kv_heads)
        self.paged_attention_with_alibi = P.auto_generate.PagedAttentionMask(self.n_head,
                                                                             self.scale_value,
                                                                             self.n_kv_heads)
    def construct(self, key, value, slot_mapping):
        """The forward compute of KVCache for Paged Attention."""
        _, _, n_kv_head, head_dim = self.key_cache.shape
        tmp_key = self.transpose(key, (0, 2, 1, 3))  # [b, n, s/1, d] -> [b, s/1, n, d]
        tmp_key = self.reshape(tmp_key, (-1, n_kv_head, head_dim))

        tmp_value = self.transpose(value, (0, 2, 1, 3))  # [b, n, s/1, d] -> [b, s/1, n, d]
        tmp_value = self.reshape(tmp_value, (-1, n_kv_head, head_dim))

        key_out = self.reshape_and_cache(tmp_key, tmp_value, self.key_cache, self.value_cache, slot_mapping)

        return key_out

    def paged_attn(self, query, batch_valid_length, block_tables):
        """The forward compute of Paged Attention."""
        query_pa = self.reshape(self.transpose(query, (0, 2, 1, 3)), (-1, self.n_head, self.head_dim))
        pa_out = self.paged_attention(query_pa, self.key_cache, self.value_cache, block_tables, batch_valid_length)
        attention = self.reshape(pa_out, (-1, 1, self.hidden_size))
        return attention

    def paged_attn_with_alibi(self, query, batch_valid_length, block_tables, alibi_tensor):
        """The forward compute of KVCache for Paged Attention with alibi tensor."""
        query_pa = self.reshape(self.transpose(query, (0, 2, 1, 3)), (-1, self.n_head, self.head_dim))
        pa_out = self.paged_attention_with_alibi(query_pa, self.key_cache, self.value_cache,
                                                 block_tables, batch_valid_length, alibi_tensor)
        attention = self.reshape(pa_out, (-1, 1, self.hidden_size))
        return attention

    def shard(self, parallel_config):
        """The shard strategy."""
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.transpose.shard(((dp, mp, 1, 1),))
        self.reshape_and_cache.shard(((dp, mp, 1), (dp, mp, 1), (1, 1, mp, 1), (1, 1, mp, 1), (1,)))
        self.paged_attention.shard(((dp, mp, 1), (1, 1, mp, 1), (1, 1, mp, 1), (dp, 1), (dp,)))
        self.paged_attention_with_alibi.shard(((dp, mp, 1), (1, 1, mp, 1), (1, 1, mp, 1), (dp, 1), (dp,)))
