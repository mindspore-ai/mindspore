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

import numpy as np
import mindspore as ms
from mindspore.nn import Cell
from mindspore import context, Tensor, Parameter
from mindspore.ops.auto_generate import PagedAttention
from parallel.utils.utils import ParallelValidator, compile_net
import pytest
import math


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")

def generate_inputs(bs=1, seq_len=1, heads=40, head_dims=128, block_size=16, max_seq=2048):
    query = Parameter(Tensor(np.ones([bs, seq_len, heads*head_dims]), dtype=ms.float16), "query")
    key_cache = Parameter(Tensor(np.ones([max_seq, block_size, heads, head_dims]), dtype=ms.float16), "key_cache")
    value_cache = Parameter(Tensor(np.ones([max_seq, block_size, heads, head_dims]), dtype=ms.float16), "value_cache")
    block_tables = Parameter(Tensor(np.ones([bs, max_seq]), dtype=ms.int32), "block_tables")
    context_lens = Parameter(Tensor(np.ones([max_seq]), dtype=ms.int32), "context_lens")
    return query, key_cache, value_cache, block_tables, context_lens

class PagedAttentionNet(Cell):
    def __init__(self, mp=None, strategy=None):
        super(PagedAttentionNet, self).__init__()
        self.n_head_no_use = 40
        self.head_dim_no_use = 128
        self.scale_value_no_use = 1 / math.sqrt(self.head_dim_no_use)
        self.n_kv_head_no_use = 40
        self.paged_attention = PagedAttention(self.n_head_no_use, self.scale_value_no_use, self.n_kv_head_no_use)
        if strategy is not None:
            self.paged_attention.shard(strategy)
        elif mp is not None:
            strategy = ((1, 1, mp), (1, 1, mp, 1), (1, 1, mp, 1), (1, 1), (1,))
            self.paged_attention.shard(strategy)

    def construct(self, query, key_cache, value_cache, block_tables, context_lens):
        return self.paged_attention(query, key_cache, value_cache, block_tables, context_lens)


def test_paged_attention_semi_auto_parallel():
    """
    Feature: test paged_attention semi auto parallel
    Description: semi auto parallel with strategy
    Expectation: compile success and shape is as expected.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    mp = 8
    net = PagedAttentionNet(mp)

    bs, seq_len, heads, head_dim, blk_size, max_seq = 1, 1, 40, 128, 16, 2048
    net_inputs = generate_inputs(bs, seq_len, heads, head_dim, blk_size, max_seq)
    net.set_inputs(*net_inputs)

    phase = compile_net(net, *net_inputs)
    validator = ParallelValidator(net, phase)
    assert validator.check_parameter_shape('query', [bs, seq_len, heads * head_dim // mp])
    assert validator.check_parameter_shape('key_cache', [max_seq, blk_size, heads // mp, head_dim])
    assert validator.check_parameter_shape('value_cache', [max_seq, blk_size, heads // mp, head_dim])
    assert validator.check_parameter_shape('block_tables', [bs, max_seq])
    assert validator.check_parameter_shape('context_lens', [max_seq])


def test_paged_attention_standalone():
    """
    Feature: test paged_attention standalone
    Description: StandAlone
    Expectation: compile success
    """
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="stand_alone")
    net = PagedAttentionNet()

    bs, seq_len, heads, head_dim, blk_size, max_seq = 1, 1, 40, 128, 16, 2048
    net_inputs = generate_inputs(bs, seq_len, heads, head_dim, blk_size, max_seq)
    net.set_inputs(*net_inputs)
    compile_net(net, *net_inputs)


def test_paged_attention_strategy_error():
    """
    Feature: test paged_attention parallel with wrong strategy
    Description: semi auto parallel with strategy
    Expectation: error and catch
    """
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    mp = 4
    mp2 = 8
    strategies = [
        ((1, mp, 1), (1, 1, mp2, 1), (1, 1, mp, 1), (1, 1), (1,)),
        ((1, mp2, 1), (1, 1, mp, 1), (1, 1, mp, 1), (1, 1), (1,)),
        ((1, mp, 1), (1, 1, mp, 1), (1, 1, mp, 1), (1, 2), (1,))
    ]
    for strategy in strategies:
        net = PagedAttentionNet(strategy=strategy)
        with pytest.raises(RuntimeError):
            bs, seq_len, heads, head_dim, blk_size, max_seq = 1, 1, 40, 128, 16, 2048
            net_inputs = generate_inputs(bs, seq_len, heads, head_dim, blk_size, max_seq)
            net.set_inputs(*net_inputs)
            compile_net(net, *net_inputs)
    context.reset_auto_parallel_context()
