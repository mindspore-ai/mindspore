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
"""KVCache Manager for inference."""
import numpy as np
from mindspore.common.tensor import Tensor
from mindspore import nn, Parameter, ops
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P


class KVCacheMgr(nn.Cell):
    """KVCache Manager."""
    def __init__(self,
                 n_head,
                 head_dim,
                 max_batch_size=8,
                 max_seq_length=4096,
                 compute_dtype=mstype.float16,
                 is_dynamic=False,
                 use_kvcache_op=True,
                 is_flexible_shape=False):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        self.dtype = compute_dtype
        self.use_kvcache_op = use_kvcache_op
        self.is_dynamic = is_dynamic
        self.is_flexible_shape = is_flexible_shape
        self.is_first_iteration = True

        self.cache_length_tensor = Tensor([max_batch_size * max_seq_length], dtype=mstype.int32)
        self.cache_pad_tensor = Tensor([3], dtype=mstype.int64)
        self.seq_length_tensor = Tensor([max_seq_length], dtype=mstype.int32)
        self.seq_length_tensor_pad = Tensor([max_seq_length, 3], dtype=mstype.int64)
        self.seqlen_axis_tensor_pad = Tensor([2, 3], dtype=mstype.int64)
        self.pad_before = Tensor([0, 0, 0, 0, 0], mstype.int32)
        self.pad_after = Tensor([0, 0], mstype.int32)
        self.pad_zero = Tensor(0, compute_dtype)

        if self.use_kvcache_op:
            # pylint: disable=W0212
            self.prompt_kvcache = P._inner_ops.PromptKVCache()
            # pylint: disable=W0212
            self.decoder_kvcache = P._inner_ops.DecoderKVCache()
        else:
            self.add = P.Add()
            self.mul = P.Mul()
            self.assign = P.Assign()
        self.concat = P.Concat(axis=0)
        self.sub = P.Sub()
        self.div = P.Div()
        self.pad = P.PadV3()
        self.slice = P.StridedSlice()
        self.cast = P.Cast()
        self.shape = P.Shape()
        self.reshape = P.Reshape().add_prim_attr("skip_redistribution", True)

        kv_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.key_past = Parameter(Tensor(np.zeros(kv_shape), compute_dtype), name="key_past", requires_grad=False)
        self.value_past = Parameter(Tensor(np.zeros(kv_shape), compute_dtype), name="value_past", requires_grad=False)

    def shard(self, parallel_config):
        """shard"""
        dp = parallel_config.data_parallel
        mp = parallel_config.model_parallel
        self.pad.shard(((dp, mp, 1, 1), (1,), ()))
        self.slice.shard(((dp, mp, 1, 1),))
        if self.use_kvcache_op:
            self.prompt_kvcache.shard(((dp, mp, 1, 1), (dp, mp, 1, 1), (dp,), (1,), (1,), (1,), (1,)))
            self.decoder_kvcache.shard(((dp, mp, 1, 1), (dp, mp, 1, 1), (dp,), (1,), (1,), (1,), (1,)))
        else:
            self.add.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))
            self.mul.shard(((dp, mp, 1, 1), (dp, 1, 1, 1)))
            self.assign.shard(((dp, mp, 1, 1), (dp, mp, 1, 1)))

    def padding(self, key, value, seq_length):
        """padding key, value"""
        pad_length = self.sub(self.seq_length_tensor, seq_length)
        # calculate padding parameter: (0, 0),(0,0),(0,pad_length),(0,0), append values of 'pad_length' in axis
        pad_config = self.concat((self.pad_before, pad_length, self.pad_after))
        key_padding = self.pad(key, pad_config, self.pad_zero)
        value_padding = self.pad(value, pad_config, self.pad_zero)
        return key_padding, value_padding

    def trimming(self, key, value, zactivate_len, batch_size):
        """tramming key, value"""
        if self.is_flexible_shape:
            key = self.reshape(key, (batch_size, self.n_head, -1, self.head_dim))
            value = self.reshape(value, (batch_size, self.n_head, -1, self.head_dim))
        if zactivate_len is not None:
            act_len = self.shape(zactivate_len)[0]
            key = self.slice(key, (0, 0, 0, 0), (batch_size, self.n_head, act_len, self.head_dim), (1, 1, 1, 1))
            value = self.slice(value, (0, 0, 0, 0), (batch_size, self.n_head, act_len, self.head_dim), (1, 1, 1, 1))
        elif not self.is_flexible_shape:
            key = self.slice(key, (0, 0, 0, 0),
                             (batch_size, self.n_head, self.max_seq_length, self.head_dim), (1, 1, 1, 1))
            value = self.slice(value, (0, 0, 0, 0),
                               (batch_size, self.n_head, self.max_seq_length, self.head_dim), (1, 1, 1, 1))
        return key, value

    def auto_caching(self, key_update, value_update, batch_valid_length, seq_length_tensor_pad, batch_index_pad=None):
        """use kvcache op to cache key, value"""
        # key_update shape: [real_bs, n_head, max_seqlen, head_dim]
        if self.is_first_iteration:
            batch_valid_length = batch_valid_length * 0
            self.prompt_kvcache(self.key_past, key_update, batch_valid_length, batch_index_pad,
                                self.seqlen_axis_tensor_pad, seq_length_tensor_pad, seq_length_tensor_pad)
            self.prompt_kvcache(self.value_past, value_update, batch_valid_length, batch_index_pad,
                                self.seqlen_axis_tensor_pad, seq_length_tensor_pad, seq_length_tensor_pad)
            return None

        key_cache = self.key_past
        value_cache = self.value_past
        key_update = self.decoder_kvcache(self.key_past, key_update, batch_valid_length, batch_index_pad,
                                          self.seqlen_axis_tensor_pad, seq_length_tensor_pad, seq_length_tensor_pad)
        value_update = self.decoder_kvcache(self.value_past, value_update, batch_valid_length, batch_index_pad,
                                            self.seqlen_axis_tensor_pad, seq_length_tensor_pad, seq_length_tensor_pad)
        key_cache = ops.depend(key_cache, key_update)
        value_cache = ops.depend(value_cache, value_update)
        return key_cache, value_cache

    def manual_caching(self, key_update, value_update, valid_length_vector, batch_size):
        """use assign to cache key, value"""
        # key_update shape: [real_bs, n_head, 1, head_dim]
        if self.is_first_iteration:
            if self.is_dynamic:
                self.assign(self.key_past,
                            self.reshape(key_update, (self.max_batch_size, self.n_head, -1, self.head_dim)))
                self.assign(self.value_past,
                            self.reshape(value_update, (self.max_batch_size, self.n_head, -1, self.head_dim)))
            else:
                self.assign(self.key_past, self.mul(key_update, valid_length_vector))
                self.assign(self.value_past, self.mul(value_update, valid_length_vector))
            return None

        if self.is_dynamic:
            key = self.add(self.reshape(self.key_past, (batch_size, self.n_head, -1, self.head_dim)),
                           self.mul(key_update, valid_length_vector))
            value = self.add(self.reshape(self.value_past, (batch_size, self.n_head, -1, self.head_dim)),
                             self.mul(value_update, valid_length_vector))
            self.assign(self.key_past,
                        self.reshape(key, (self.max_batch_size, self.n_head, -1, self.head_dim)))
            self.assign(self.value_past,
                        self.reshape(value, (self.max_batch_size, self.n_head, -1, self.head_dim)))
        else:
            key = self.add(self.key_past, self.mul(key_update, valid_length_vector))
            value = self.add(self.value_past, self.mul(value_update, valid_length_vector))
            self.assign(self.key_past, key)
            self.assign(self.value_past, value)
        # key shape: [real_bs, n_head, max_cache_len // real_bs, head_dim]
        return key, value

    def construct(self, key, value, kvcache_inputs=None):
        """The forward compute of KVCacheMgr."""
        # TODO: add inputs check
        batch_valid_length, zactivate_len, batch_index_pad, seq_length_tensor_pad = kvcache_inputs
        if not self.use_kvcache_op:
            batch_valid_length = self.cast(batch_valid_length, self.dtype)
        batch_size, _, seq_length, _ = self.shape(key)
        if self.is_first_iteration:
            if self.is_dynamic:
                key_padding, value_padding = self.padding(key, value, seq_length=seq_length)
            else:
                key_padding, value_padding = key, value
            if self.use_kvcache_op:
                self.auto_caching(key_padding, value_padding, batch_valid_length,
                                  seq_length_tensor_pad, batch_index_pad)
            else:
                self.manual_caching(key_padding, value_padding, batch_valid_length, batch_size=batch_size)
        else:
            if self.use_kvcache_op:
                key, value = self.auto_caching(key, value, batch_valid_length,
                                               seq_length_tensor_pad, batch_index_pad)
            else:
                key, value = self.manual_caching(key, value, batch_valid_length, batch_size=batch_size)
            key, value = self.trimming(key, value, zactivate_len, batch_size=batch_size)

        return key, value


class KVCachePreprocess(nn.Cell):
    """KVCache Manager."""
    def __init__(self,
                 max_batch_size=8,
                 max_seq_length=4096,
                 is_dynamic=False,
                 use_kvcache_op=False,
                 is_flexible_shape=False,
                 use_paged_attention=False
                 ):
        super().__init__()
        self.is_dynamic = is_dynamic
        self.use_kvcache_op = use_kvcache_op
        self.is_flexible_shape = is_flexible_shape
        self.use_paged_attention = use_paged_attention

        self.max_cache_length = max_batch_size * max_seq_length
        range_len = self.max_cache_length if self.is_flexible_shape else max_seq_length
        self.range = Tensor(np.arange(range_len).reshape((1, 1, -1)), mstype.int32)
        self.cache_length_tensor = Tensor([max_batch_size * max_seq_length], dtype=mstype.int32)
        self.cache_pad_tensor = Tensor([3], dtype=mstype.int64)
        self.seq_length_tensor = Tensor([max_seq_length], dtype=mstype.int32)
        self.seq_length_tensor_pad = Tensor([max_seq_length, 3], dtype=mstype.int64)
        self.is_first_iteration = True

        self.slice = P.StridedSlice()
        self.reshape = P.Reshape().add_prim_attr("skip_redistribution", True)
        self.equal = P.Equal().shard(((1, 1, 1), (1, 1, 1)))
        self.less = P.Less().shard(((1, 1, 1), (1, 1, 1)))
        self.expand_dims = P.ExpandDims().shard(((1, 1, 1),))
        self.div = P.Div()
        self.concat = P.Concat(axis=0)

    def construct(self, batch_size, batch_valid_length=None, batch_index=None, zactivate_len=None,
                  block_tables=None, slot_mapping=None):
        """precompute kvcache inputs"""
        if self.use_paged_attention:
            cur_pos = batch_valid_length + 1
            kvcache_inputs = (cur_pos, block_tables, slot_mapping)
            return kvcache_inputs

        seq_range = self.range
        if self.is_dynamic and self.is_flexible_shape and not self.use_kvcache_op:
            seq_range = self.slice(seq_range, (0, 0, 0), (1, 1, self.max_cache_length // batch_size), (1, 1, 1))

        if self.use_kvcache_op:
            if batch_index is None:
                batch_index = ops.arange(0, batch_size, 1)
            batch_index_pad = self.concat((batch_index, self.cache_pad_tensor))
            seq_length_tensor_pad = self.get_seq_length_tensor_pad(batch_size=batch_size)
            batch_valid_length = self.cast(self.reshape(batch_valid_length, (-1,)), mstype.int64)
            kvcache_inputs = (batch_valid_length, zactivate_len, batch_index_pad, seq_length_tensor_pad)
        else:
            if self.is_first_iteration:
                valid_length_vector = self.less(seq_range, self.reshape(batch_valid_length, (-1, 1, 1)))
            else:
                valid_length_vector = self.equal(seq_range, self.reshape(batch_valid_length, (-1, 1, 1)))
            valid_length_vector = self.expand_dims(valid_length_vector, 3)
            kvcache_inputs = (valid_length_vector, zactivate_len, None, None)
        return kvcache_inputs

    def get_seq_length_tensor_pad(self, batch_size):
        """get seq_length_tensor_pad"""
        if self.is_flexible_shape:
            max_seq_length = self.div(self.cache_length_tensor, batch_size).astype(mstype.int64)
            return self.concat((max_seq_length, self.cache_pad_tensor))
        return self.seq_length_tensor_pad
